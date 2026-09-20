"""R-G1: 因果 G — content(768@50fps) + f0/energy(86fps) -> mel80(172fps)。

`current/vc_eg.md` のラダー第 1 段。E と独立に「因果 G が成立するか」だけを見る
（失敗の帰属を狭く保つ）。教師 content は data/*_feat に焼き込み済み。

    uv run python train_vc_g.py --steps 20000 --tag diag_g1        # 切り分け
    uv run python train_vc_g.py --steps 200000 --tag g1_full       # 本走行（フルコーパス）
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import ship_front as SF
from causal_mel import causal_mel
from v2f import FreqNorm

ROOT = Path(__file__).resolve().parent.parent
FEATS = [ROOT / "data/female_real_feat", ROOT / "data/female_tts_feat"]
MEL_FPS = 44100 / SF.HOP_A            # 172.27
CV_FPS = 50.0
F0_FPS = 44100 / 512                  # 86.13


class CausalBlock(nn.Module):
    def __init__(self, dim: int, k: int = 3, dil: int = 1):
        super().__init__()
        self.k, self.dil = k, dil
        self.dw = nn.Conv1d(dim, dim, k, dilation=dil)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim * 3)
        self.pw2 = nn.Linear(dim * 3, dim)

    def forward(self, x):                             # [B, C, T]
        r = x
        x = F.pad(x, ((self.k - 1) * self.dil, 0))
        x = self.dw(x).transpose(1, 2)
        x = self.pw2(F.gelu(self.pw1(self.norm(x)))).transpose(1, 2)
        return r + x


class G1(nn.Module):
    """content+f0+energy -> mel80。全て左パディング＝先読み 0。"""

    def __init__(self, dim: int = 256, layers: int = 6):
        super().__init__()
        self.dim, self.layers = dim, layers
        self.inp = nn.Conv1d(768 + 2, dim, 3)         # 左 pad は forward で
        self.blocks = nn.ModuleList(
            [CausalBlock(dim, 3, 2 ** (i // 2)) for i in range(layers)])
        self.out = nn.Linear(dim, SF.N_MEL)

    @property
    def ctx(self) -> int:
        return 2 + sum((3 - 1) * (2 ** (i // 2)) for i in range(self.layers))

    def forward(self, x):                             # [B, 770, T] -> [B, 80, T]
        x = self.inp(F.pad(x, (2, 0)))
        for b in self.blocks:
            x = b(x)
        return self.out(x.transpose(1, 2)).transpose(1, 2)

    def arch(self) -> dict:
        return {"arch": "g1", "dim": self.dim, "L": self.layers, "ctx": self.ctx}


class AdaIN(nn.Module):
    """speaker emb -> 各 block の per-ch affine（identity 注入の勝ち筋 m2）。"""

    def __init__(self, dim: int, emb: int = 192):
        super().__init__()
        self.proj = nn.Linear(emb, dim * 2)

    def forward(self, s, x):                          # s [B, emb], x [B, C, T]
        h = self.proj(s)
        g, b = h.chunk(2, dim=-1)
        return x * (1 + g[:, :, None]) + b[:, :, None]


class GS(G1):
    """G1 + AdaIN speaker 条件（V2-1）。emb は事前計算の固定ベクトル（推論時定数）。"""

    def __init__(self, dim: int = 256, layers: int = 6, emb: int = 192):
        super().__init__(dim, layers)
        self.emb_dim = emb
        self.adains = nn.ModuleList([AdaIN(dim, emb) for _ in range(layers)])

    def forward(self, x, s=None):                     # [B,770,T],[B,192] -> [B,80,T]
        x = self.inp(F.pad(x, (2, 0)))
        for b, a_ in zip(self.blocks, self.adains):
            x = b(x)
            if s is not None:
                x = a_(s, x)
        return self.out(x.transpose(1, 2)).transpose(1, 2)

    def arch(self) -> dict:
        d = super().arch()
        d["arch"] = "gs"
        d["emb"] = self.emb_dim
        return d


def resample_to(x: torch.Tensor, t: int, src_fps: float) -> torch.Tensor:
    """因果リサンプル: mel フレーム t が使ってよい最新の src フレームを取る。"""
    idx = ((torch.arange(t, dtype=torch.float64) + 1.0) * src_fps / MEL_FPS - 1.0)
    idx = idx.floor().clamp(min=0, max=x.shape[-1] - 1).long()
    return x[..., idx]


def feat_util(d, mel):
    t = mel.shape[-1]
    c = resample_to(d["content"].T.float(), t, CV_FPS)              # [768, T]
    f0 = resample_to(d["f0"].float()[None], t, F0_FPS)              # [1, T]
    en = resample_to(d["energy"].float()[None], t, F0_FPS)
    f0 = torch.log(f0.clamp(min=50.0) / 200.0)
    en = torch.log(en.clamp(min=1e-4))
    return torch.cat([c, f0, en], 0)


def load_wav(f: Path) -> torch.Tensor:
    """44.1kHz mono float32。soundfile 直読（librosa と全コーパスでビット一致・~900 倍速）。"""
    import soundfile as sf_
    w, sr = sf_.read(str(f), dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    if sr != 44100:
        import librosa
        w = librosa.resample(w, orig_sr=sr, target_sr=44100)
    return torch.from_numpy(w)


def wav_path_of(d) -> Path:
    wp = ROOT / str(d["path"]).lstrip("./")
    return wp if wp.exists() else Path(str(d["path"]))


def load_item(f: Path):
    d = torch.load(f, map_location="cpu")
    w = load_wav(wav_path_of(d)) * 32768.0           # 学習系の慣習スケール
    mel = causal_mel(w, n_fft=SF.NFFT_A, hop=SF.HOP_A, num_mels=SF.N_MEL, sr=44100)[0]
    return d, mel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=344)   # 2 秒 @172fps
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ft", action="store_true",
                    help="fine-tune: OneCycle を使わず定数 LR（収束済み ckpt の再加熱を防ぐ）")
    ap.add_argument("--every", type=int, default=1000)
    ap.add_argument("--snap", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-utts", type=int, default=0,
                    help="diag_ 専用。0 = フルコーパス")
    ap.add_argument("--only-spk", type=str, default=None,
                    help="voice cartridge FT: この話者だけで学習（diag_/cart_ タグ限定）。"
                         "eval はその話者の末尾 5 発話")
    ap.add_argument("--through-v", type=float, default=0.0,
                    help="凍結 V を通した音声域 mrstft の重み（texture fine-tune）")
    ap.add_argument("--cipt", type=float, default=0.0,
                    help="CIPT: 学習中に男声→target 変換を実行し、V 出力の ECAPA を"
                         "目標重心へ引く重み（cipt-cross-identity-plan の再現）")
    ap.add_argument("--cipt-spk", type=str, default=None,
                    help="CIPT の目標話者（female_tts_feat の話者名）")
    ap.add_argument("--cipt-batch", type=int, default=4)
    ap.add_argument("--cipt-crop", type=int, default=256,
                    help="変換バッチの解析フレーム長（256≒1.49s、ECAPA が安定する長さ）")
    ap.add_argument("--cipt-margin", type=float, default=0.6,
                    help="identity の hinge: cos がこの線を越えたら圧を止める"
                         "（w5 直行は cos 0.907 まで突っ走り CER canary が発火した）")
    ap.add_argument("--cipt-content", type=float, default=1.0,
                    help="変換バッチの content 錨: E-student(pred mel) ≈ 入力 content")
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--content-from", type=str, default="teacher",
                    help="teacher | E ckpt へのパス（R-G2: 学生 content で学習）")
    ap.add_argument("--spk-cond", action="store_true",
                    help="V2-1: GS（AdaIN speaker 条件）。emb=data/ecapa_spk_mean_full.pt")
    ap.add_argument("--pairs", type=float, default=0.0,
                    help="V2-3: same-text 実ペア監督の重み（data/vctk_pairs/*.pt）")
    ap.add_argument("--pairs-dir", type=str, default=str(ROOT / "data/vctk_pairs"))
    ap.add_argument("--tag", type=str, required=True)
    a = ap.parse_args()
    if a.limit_utts and not a.tag.startswith("diag_"):
        sys.exit("部分集合は diag_ タグでのみ許可（CLAUDE.md Data 規則）")
    if a.only_spk and not (a.tag.startswith("diag_") or a.tag.startswith("cart_")):
        sys.exit("--only-spk は diag_/cart_ タグ限定（cartridge=フル土台からの適応）")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(a.seed)
    files: list[Path] = []
    for root in FEATS:
        for spk in sorted(root.iterdir()):
            if spk.is_dir():
                files += sorted(spk.glob("*.pt"))
    rng.shuffle(files)
    # held-out: 末尾 24 話者ぶんを検証に（学習に混ぜない）
    spk_all = sorted({f.parent.name for f in files})
    held = set(spk_all[-24:])
    tr = [f for f in files if f.parent.name not in held]
    ev = [f for f in files if f.parent.name in held][:24]
    if a.only_spk:
        mine = sorted(f for f in files if f.parent.name == a.only_spk)
        if len(mine) < 10:
            sys.exit(f"--only-spk {a.only_spk}: {len(mine)} 発話しか無い")
        tr, ev = mine[:-5], mine[-5:]
    if a.limit_utts:
        tr = tr[:a.limit_utts]
    print(f"  train {len(tr)} utts / {len(spk_all) - 24} spk   eval {len(ev)}", flush=True)

    student = None
    if a.content_from != "teacher":
        from train_vc_e import E1
        ek = torch.load(a.content_from, map_location=dev)
        student = E1(dim=ek["args"]["dim"], layers=ek["args"]["L"],
                     look=ek["args"].get("look", 0)).to(dev).eval()
        student.load_state_dict(ek["net"])
        for q in student.parameters():
            q.requires_grad_(False)
        print(f"  content: E-student {a.content_from}"
              f"（cos {ek.get('eval_cos', '?')}）", flush=True)

    spk_emb = None
    if a.spk_cond:
        spk_emb = torch.load(ROOT / "data/ecapa_spk_mean_full.pt",
                             map_location="cpu", weights_only=False)
        print(f"  spk-cond: {len(spk_emb)} speakers (ECAPA mean, frozen consts)",
              flush=True)
    pair_items = None
    if a.pairs > 0:
        pdir = Path(a.pairs_dir)
        pair_items = sorted(pdir.glob("*.pt"))
        if not pair_items:
            sys.exit(f"--pairs: {pdir} にペア .pt が無い")
        print(f"  pairs: {len(pair_items)} same-text 実ペア（重み {a.pairs}）", flush=True)
    net = (GS(dim=a.dim, layers=a.layers).to(dev) if a.spk_cond
           else G1(dim=a.dim, layers=a.layers).to(dev))
    if a.resume:
        rk = torch.load(a.resume, map_location=dev)
        net.load_state_dict(rk["net"])
        print(f"  resume from {a.resume}（eval-L1 {rk.get('eval_l1')}）", flush=True)

    vnet = None
    W_lin = None
    if a.through_v > 0:
        import v2f as V2
        from rddsp_gpu import mel_to_linear, mrstft
        vtag = Path("/tmp/current_tag").read_text().strip()
        vk = torch.load(ROOT / f"results/{vtag}/{vtag}_best.pt", map_location=dev)
        vnet = V2.V2F(cin=4, ch=vk["args"]["ch"], layers=vk["args"]["L"],
                      norm=vk["args"]["norm"]).to(dev).eval()
        vnet.load_state_dict(vk["net"])
        for q in vnet.parameters():
            q.requires_grad_(False)
        W_lin = mel_to_linear(dev, nbin=SF.NFFT_S // 2 + 1)
        globals()["_mrstft"] = mrstft
        print(f"  through-V: {vtag} best（凍結・texture fine-tune）", flush=True)

    cipt_on = a.cipt > 0
    if cipt_on:
        import math as _m
        if not a.cipt_spk:
            sys.exit("--cipt には --cipt-spk が必要")
        if a.content_from == "teacher":
            sys.exit("--cipt には --content-from（E ckpt）が必要"
                     "（出荷経路＝student content で変換分布を一致させる）")
        # 凍結 V（through-V と共用。未ロードならここでロード）
        if a.through_v <= 0:
            import v2f as V2
            from rddsp_gpu import mel_to_linear
            vtag = Path("/tmp/current_tag").read_text().strip()
            vk = torch.load(ROOT / f"results/{vtag}/{vtag}_best.pt", map_location=dev)
            vnet = V2.V2F(cin=4, ch=vk["args"]["ch"], layers=vk["args"]["L"],
                          norm=vk["args"]["norm"]).to(dev).eval()
            vnet.load_state_dict(vk["net"])
            for q in vnet.parameters():
                q.requires_grad_(False)
            W_lin = mel_to_linear(dev, nbin=SF.NFFT_S // 2 + 1)
        # 凍結 ECAPA（微分可能経路: train_cipt2 の実証レシピ）
        from speechbrain.inference.speaker import EncoderClassifier
        sb = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", savedir="/tmp/sb_ecapa",
            run_opts={"device": dev})
        sbm = sb.mods.to(dev)
        for q in sbm.parameters():
            q.requires_grad_(False)

        def ecapa_emb(w16):
            fe = sbm.compute_features(w16)
            fe = sbm.mean_var_norm(fe, torch.ones(w16.shape[0], device=dev))
            e_ = sbm.embedding_model(fe, torch.ones(w16.shape[0], device=dev)).squeeze(1)
            return e_ / (e_.norm(dim=-1, keepdim=True) + 1e-6)

        def to16(y):
            return F.interpolate(y.unsqueeze(1), scale_factor=16000 / 44100.0,
                                 mode="linear", align_corners=False).squeeze(1)

        def _med_f0(fs_):
            vs = []
            for f_ in fs_:
                d_ = torch.load(f_, map_location="cpu", weights_only=False)
                v_ = d_["f0"][d_["f0"] > 50]
                if len(v_):
                    vs.append(v_)
            return float(torch.cat(vs).median())

        tgt_dir = ROOT / "data/female_tts_feat" / a.cipt_spk
        tfs = sorted(tgt_dir.glob("*.pt"))
        cen = []
        with torch.no_grad():
            for f_ in tfs[:20]:
                d_ = torch.load(f_, map_location="cpu", weights_only=False)
                cen.append(ecapa_emb(to16(load_wav(wav_path_of(d_))[None].to(dev)))[0])
        cen = F.normalize(torch.stack(cen).mean(0), dim=-1)
        tgt_f0 = _med_f0(tfs[:40])
        mroot = ROOT / "data/male_feat"
        male_files, male_ratio = [], {}
        for sd_ in sorted(p_ for p_ in mroot.iterdir() if p_.is_dir()):
            fs_ = sorted(sd_.glob("*.pt"))
            if not fs_:
                continue
            male_files += fs_
            sh_ = round(12.0 * _m.log2(tgt_f0 / _med_f0(fs_[:20])))
            male_ratio[sd_.name] = 2.0 ** (sh_ / 12.0)
        rng_c = random.Random(a.seed + 7)
        conv_cache: dict = {}

        def conv_item(f_):
            """eval_x と同一の変換系列（causal_f0・[-1,1]RMS energy）＋シフト済み励起。"""
            if f_ not in conv_cache:
                if len(conv_cache) > 60:
                    conv_cache.pop(next(iter(conv_cache)))
                d_ = torch.load(f_, map_location="cpu", weights_only=False)
                gt_ = load_wav(wav_path_of(d_)) * 32768.0
                n_ = gt_.shape[-1]
                mel_ = SF.mel(gt_)
                t_ = mel_.shape[-1]
                ratio_ = male_ratio[f_.parent.name]
                f0c_, _v = SF.causal_f0(gt_)
                f0s_ = torch.where(f0c_ > 50, f0c_ * ratio_, f0c_)
                lf0_ = torch.log(f0s_.clamp(min=50.0) / 200.0)[:t_]
                hop_ = 512
                nf_ = n_ // hop_
                rms_ = torch.sqrt(((gt_[: nf_ * hop_] / 32768.0)
                                   .reshape(nf_, hop_) ** 2).mean(-1) + 1e-12)
                en_ = torch.log(resample_to(rms_[None], t_, F0_FPS)[0].clamp(min=1e-4))
                g_ = torch.Generator().manual_seed(0)
                z_ = torch.randn(n_, generator=g_)
                phi_, f0u_ = SF.phase_of(f0s_, n_)
                nyq = 22050.0
                p64, f64 = phi_.double(), f0u_.double()
                K0 = (nyq / f64.clamp(min=1e-6)).floor().clamp(max=float(SF.KMAX))
                K = (K0 - ((K0.float() * f0u_ >= nyq) & (K0 >= 1)).double()
                     + (((K0 + 1).float() * f0u_ < nyq) & (K0 + 1 <= SF.KMAX)).double()
                     ).clamp(min=0.0, max=float(SF.KMAX))
                s64 = torch.sin(p64 / 2)
                imp = torch.where(s64.abs() > 1e-9,
                                  torch.sin((K + 0.5) * p64) / (2 * s64) - 0.5, K).float()
                exc_ = 0.7 * (imp / K.float().clamp(min=1.0).sqrt()) + 0.3 * z_
                E_ = SF.cstft(exc_.to(dev), SF.NFFT_S, SF.HOP_S)
                conv_cache[f_] = (mel_, lf0_, en_[:t_], E_)
            return conv_cache[f_]

        print(f"  CIPT: 目標 {a.cipt_spk}（f0~{tgt_f0:.0f}Hz）  男声 {len(male_files)} utts  "
              f"shift {sorted(set(round(12*_m.log2(r)) for r in male_ratio.values()))}", flush=True)
    print(f"  params {sum(p.numel() for p in net.parameters())/1e6:.2f}M  ctx {net.ctx}",
          flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.99))
    # ⚠ --resume で OneCycle を新規構築すると最大 LR へ再加熱して収束済みモデルを壊す
    #   （diag_g1_tv: eval-L1 1.17→1.34、warmup と同時系列）。fine-tune は定数 LR。
    sch = (None if a.ft else
           torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps))

    out_dir = ROOT / "results" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cache: dict = {}
    ecache: dict = {}

    def exc_cache(f: Path):
        if f not in ecache:
            if len(ecache) > 400:
                ecache.pop(next(iter(ecache)))
            d_, mel_ = get(f)
            gt = load_wav(wav_path_of(d_)) * 32768.0
            n_ = gt.shape[-1]
            f0_, _v = SF.causal_f0(gt)
            g_ = torch.Generator().manual_seed(0)
            z_ = torch.randn(n_, generator=g_)
            phi_, f0u_ = SF.phase_of(f0_, n_)
            # Dirichlet 閉形式（f64）。K は明示ループの f32 判定 `k*f0u < nyq` と
            # 完全一致させる（境界候補 2 本を f32 で数え直す）。正規化後の残差は
            # 2.1e-4（f32 加算順序ノイズ水準）、196 倍速。
            nyq = 22050.0
            p64, f64 = phi_.double(), f0u_.double()
            K0 = (nyq / f64.clamp(min=1e-6)).floor().clamp(max=float(SF.KMAX))
            K = (K0 - ((K0.float() * f0u_ >= nyq) & (K0 >= 1)).double()
                 + (((K0 + 1).float() * f0u_ < nyq) & (K0 + 1 <= SF.KMAX)).double()
                 ).clamp(min=0.0, max=float(SF.KMAX))
            s64 = torch.sin(p64 / 2)
            imp = torch.where(s64.abs() > 1e-9,
                              torch.sin((K + 0.5) * p64) / (2 * s64) - 0.5, K).float()
            exc = 0.7 * (imp / K.float().clamp(min=1.0).sqrt()) + 0.3 * z_
            E_ = SF.cstft(exc.to(dev), SF.NFFT_S, SF.HOP_S)
            ecache[f] = (E_, gt)
        return ecache[f]

    def get(f: Path):
        if f not in cache:
            if len(cache) > 3000:
                cache.pop(next(iter(cache)))
            try:
                cache[f] = load_item(f)
            except Exception:                          # noqa: BLE001
                cache[f] = None
        return cache[f]

    feat_of = feat_util

    def evaluate() -> float:
        net.eval()
        tot = 0.0
        n = 0
        with torch.no_grad():
            for f in ev:
                it = get(f)
                if it is None:
                    continue
                d, mel = it
                x = feat_of(d, mel).to(dev)[None]
                if student is not None:
                    with torch.no_grad():
                        x[:, :768] = student(mel.to(dev)[None])
                s_ = (spk_emb[d["speaker"]][None].to(dev)
                      if spk_emb is not None and d.get("speaker") in spk_emb else None)
                y = net(x, s_)[0].cpu() if s_ is not None else net(x)[0].cpu()
                m = min(y.shape[-1], mel.shape[-1])
                tot += float((y[:, :m] - mel[:, :m]).abs().mean())
                n += 1
        net.train()
        return tot / max(n, 1)

    t0 = time.time()
    best = 1e9
    step = 0
    while step < a.steps:
        xs, ys, ms, es, ss = [], [], [], [], []
        while len(xs) < a.batch:
            fsel = rng.choice(tr)
            it = get(fsel)
            if it is None:
                continue
            d, mel = it
            t = mel.shape[-1]
            if t <= a.crop + net.ctx + 4:
                continue
            s = rng.randrange(net.ctx, t - a.crop)
            x = feat_of(d, mel)
            xs.append(x[:, s - net.ctx: s + a.crop])
            ys.append(mel[:, s: s + a.crop])
            ms.append(mel[:, s - net.ctx: s + a.crop])
            ss.append(spk_emb[d["speaker"]]
                      if spk_emb is not None and d.get("speaker") in spk_emb
                      else torch.zeros(192))
            if a.through_v > 0:
                es.append((fsel, s))
        step += 1
        xb = torch.stack(xs).to(dev)
        yb = torch.stack(ys).to(dev)
        sb = torch.stack(ss).to(dev) if spk_emb is not None else None
        if student is not None:
            # ⚠ 学生 content に**その場で**置き換える（教師列と同じ座標系）。
            #   mel は crop 済みなので学生もそのまま crop 区間で走らせる。
            with torch.no_grad():
                mel_in = torch.stack(ms).to(dev)
                xb[:, :768] = student(mel_in)
        pred = net(xb, sb)[:, :, net.ctx:] if sb is not None else net(xb)[:, :, net.ctx:]
        loss = (pred - yb).abs().mean() + 0.5 * F.mse_loss(pred, yb)
        if pair_items is not None:
            # V2-3: same-text 実ペア直接監督。source(male) front 特徴から
            # G(content, f0_tgt, en_tgt) -> warp した target mel を L1。
            # G 入力の f0/energy は target 側（変換後の prosody を与える想定）。
            pxs, pys, pss = [], [], []
            tries = 0
            while len(pxs) < max(2, a.batch // 2) and tries < a.batch * 6:
                tries += 1
                pd = torch.load(rng.choice(pair_items), map_location="cpu",
                                weights_only=False)
                ms, mt = pd["mel_src"], pd["mel_tgt_warp"]
                t_ = min(ms.shape[-1], mt.shape[-1])
                if t_ <= a.crop + net.ctx + 4:
                    continue
                s_ = rng.randrange(net.ctx, t_ - a.crop)
                f0t = pd.get("f0_tgt_warp", pd["f0_tgt"])
                lf0t = torch.log(f0t.clamp(min=50.0) / 200.0)
                pxs.append((ms[:, s_ - net.ctx: s_ + a.crop], lf0t, s_))
                pys.append(mt[:, s_: s_ + a.crop])
                if spk_emb is not None:
                    pss.append(spk_emb.get(pd["female"], torch.zeros(192)))
            if pxs:
                # content は student で source mel から抽出（推論経路一致）
                mel_in_p = torch.stack([p[0] for p in pxs]).to(dev)
                with torch.no_grad():
                    c_p = student(mel_in_p)
                lf0s = []
                for p in pxs:
                    mseg = p[0]
                    n_ = mseg.shape[-1]
                    lf0s.append(p[1][p[2]: p[2] + n_])
                lf0_b = torch.stack([
                    F.pad(s_, (0, pxs[i][0].shape[-1] - s_.shape[-1]))
                    for i, s_ in enumerate(lf0s)])[:, None]
                en_b = torch.zeros_like(lf0_b)
                xp = torch.cat([c_p, lf0_b.to(dev), en_b.to(dev)], 1)
                sp_b = (torch.stack(pss).to(dev)
                        if spk_emb is not None else None)
                pred_p = (net(xp, sp_b)[:, :, net.ctx:] if sp_b is not None
                          else net(xp)[:, :, net.ctx:])
                yp = torch.stack(pys).to(dev)
                loss = loss + a.pairs * (pred_p - yp).abs().mean()
                globals()["_prun"] = (globals().get("_prun", 0.0) * 0.98
                                      + float((pred_p - yp).abs().mean()) * 0.02)
        if a.through_v > 0:
            # G→mel→(凍結 V)→音声 の mrstft。励起 STFT E は G 非依存なので
            # 発話ごとにキャッシュ（重いのは初回だけ）。
            # V forward を 1 バッチに畳む（逐次 16 回だと GPU 起動律速で 2.2s/step）。
            fvs, gts = [], []
            for bi_, (f_, s_) in enumerate(es):
                E_, gt_w = exc_cache(f_)
                # crop の合成フレーム範囲（解析 s..s+crop → 合成 2s..2(s+crop)）
                t0_, t1_ = 2 * s_, 2 * (s_ + a.crop)
                nseg = (t1_ - t0_) * SF.HOP_S
                g_ = gt_w[t0_ * SF.HOP_S: t0_ * SF.HOP_S + nseg]
                if t1_ > E_.shape[-1] or g_.shape[-1] < nseg:
                    continue
                ml_syn = SF.to_frames(W_lin @ (pred[bi_] - SF.V_MEL_ADAPT),
                                      2 * pred.shape[-1])
                H_ = (ml_syn - SF.MEL_REF).exp()
                P_ = E_[:, t0_:t1_] * H_[:, : t1_ - t0_]
                fvs.append(torch.cat([ml_syn[None, :, : t1_ - t0_], P_.real[None],
                                      P_.imag[None],
                                      torch.log(P_.abs()[None] + 1e-5)], 0))
                gts.append(g_)
            if fvs:
                o_ = vnet(torch.stack(fvs))
                S_ = torch.complex(o_[:, 0], o_[:, 1])
                y_ = SF.cistft(S_, gts[0].shape[-1])
                # V 出力は [-1,1] スケール -> gt も揃える
                loss = loss + a.through_v * _mrstft(
                    y_, torch.stack(gts).to(dev) / 32768.0)
        if cipt_on:
            # 男声→target 変換バッチ: G 入力は eval_x と同一構成。V を通して ECAPA を
            # 目標重心へ引く（出力側同一性監督＝許可された表現監督。VC teacher ではない）
            fvs_c = []
            tries = 0
            while len(fvs_c) < a.cipt_batch and tries < a.cipt_batch * 8:
                tries += 1
                f_ = rng_c.choice(male_files)
                mel_, lf0_, en_, E_ = conv_item(f_)
                t_ = min(mel_.shape[-1], lf0_.shape[-1], en_.shape[-1])
                if t_ <= a.cipt_crop + net.ctx + 4:
                    continue
                s_ = rng_c.randrange(net.ctx, t_ - a.cipt_crop)
                if 2 * (s_ + a.cipt_crop) > E_.shape[-1]:
                    continue
                xin = torch.zeros(768 + 2, s_ + a.cipt_crop - (s_ - net.ctx))
                seg = slice(s_ - net.ctx, s_ + a.cipt_crop)
                xin[768] = lf0_[seg]
                xin[769] = en_[seg]
                fvs_c.append((f_, s_, xin, mel_[:, seg], E_))
            if fvs_c:
                mel_in_c = torch.stack([m_ for _, _, _, m_, _ in fvs_c]).to(dev)
                xb_c = torch.stack([x_ for _, _, x_, _, _ in fvs_c]).to(dev)
                with torch.no_grad():
                    xb_c[:, :768] = student(mel_in_c)
                s_c = (spk_emb[a.cipt_spk][None].expand(xb_c.shape[0], -1).to(dev)
                       if spk_emb is not None and a.cipt_spk in spk_emb else None)
                pred_c = (net(xb_c, s_c)[:, :, net.ctx:] if s_c is not None
                          else net(xb_c)[:, :, net.ctx:])
                fv_c = []
                _VA = SF.V_MEL_ADAPT
                for i_, (f_, s_, _x, _m, E_) in enumerate(fvs_c):
                    ml_ = SF.to_frames(W_lin @ (pred_c[i_] - _VA),
                                       2 * pred_c.shape[-1])
                    H_ = (ml_ - SF.MEL_REF).exp()
                    t0_, t1_ = 2 * s_, 2 * (s_ + a.cipt_crop)
                    P_ = E_[:, t0_:t1_] * H_
                    fv_c.append(torch.cat([ml_[None], P_.real[None],
                                           P_.imag[None],
                                           torch.log(P_.abs()[None] + 1e-5)], 0))
                o_c = vnet(torch.stack(fv_c))
                S_c = torch.complex(o_c[:, 0], o_c[:, 1])
                y_c = SF.cistft(S_c, 2 * a.cipt_crop * SF.HOP_S)
                emb_c = ecapa_emb(to16(y_c))    # V 出力は既に [-1,1] スケール
                cos_c = (emb_c * cen[None]).sum(-1)
                cid = F.relu(a.cipt_margin - cos_c).mean()
                loss = loss + a.cipt * cid
                if a.cipt_content > 0:
                    # content 錨（勾配は pred_c 側のみ・student は凍結済み）。
                    # 錨が無いと identity 圧が非音声解へ逃げる（CER canary 発火の帰属）
                    c_pred = student(pred_c)
                    tgt_c = xb_c[:, :768, net.ctx:].detach()
                    ccons = (1.0 - F.cosine_similarity(
                        c_pred, tgt_c, dim=1)).mean()
                    loss = loss + a.cipt_content * ccons
                    globals()["_cc_run"] = (globals().get("_cc_run", 0.0) * 0.98
                                            + float(ccons) * 0.02)
                globals()["_cid_run"] = (globals().get("_cid_run", 0.0) * 0.98
                                         + float(cos_c.mean()) * 0.02)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        if sch is not None:
            sch.step()
        if step % a.every == 0 or step == a.steps:
            evl = evaluate()
            cid_s = (f"  cos {globals()['_cid_run']:.4f}"
                     if "_cid_run" in globals() else "")
            cid_s += (f"  ccons {globals()['_cc_run']:.4f}"
                      if "_cc_run" in globals() else "")
            print(f"  step {step:6d}  loss {float(loss):.4f}  eval-L1 {evl:.4f}"
                  f"{cid_s}  ({time.time()-t0:.0f}s)", flush=True)
            ck = {"net": net.state_dict(), "args": net.arch(), "step": step,
                  "eval_l1": evl, "cli": vars(a)}
            torch.save(ck, out_dir / f"{a.tag}_last.pt")
            if evl < best:
                best = evl
                torch.save(ck, out_dir / f"{a.tag}_best.pt")
            if a.snap and step % a.snap == 0:
                torch.save(ck, out_dir / f"{a.tag}_s{step}.pt")
    print(f"\n{a.tag}: best eval-L1 {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
