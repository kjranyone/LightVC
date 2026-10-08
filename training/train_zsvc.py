"""ZS-VC の学習。事前登録 results/<tag>/prereg.yaml。

  段 R(step ≤ r_steps): 女声フルコーパスの自己再構成。入力 mel に周波数伸縮 α・利得の摂動(内容側だけ)、参照 = 同じ話者の別発話。
      損失 = L1(出力 mel, 正解 mel)+ ContentVec 補助(内容 → 768 の cos)
  段 C(以降): 段 R に加えて交差 CIPT(記憶 cipt-cross-identity-plan の成功要点):
      男声の内容 × 女声の参照 × 女声音域へ写した f0 → 出力 mel → nvoc(凍結・微分可能・調波源なし)→ 波形
      損失 += w_id·(1 − cos(ECAPA(出力), ECAPA(参照)))+ w_art·(1 − cos(ContentVec(出力), ContentVec(男声入力)))
             + w_adv·LSGAN(mel 判別器: 本物 = 女声の正解 mel、偽物 = 交差の出力 mel)
  ECAPA・ContentVec は損失にだけ使い推論経路に入らない。

    CUDA_VISIBLE_DEVICES=0 uv run python train_zsvc.py --tag zsvc1
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import zsvc as Z
import train_nvoc as TN

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data/artic_feat"
SEG = 57600
T = SEG // N.HOP
CTX = N.WIN - N.HOP
W0F = 20
REF_MAX = 144000
MALE_HELD = 4


def logf0_stats(f0: np.ndarray) -> tuple[float, float]:
    v = f0[f0 > 0]
    if len(v) < 20:
        return 0.0, 0.0
    lv = np.log(v)
    return float(np.median(lv)), float(max(np.std(lv), 0.05))


def male_index(vctk: bool = False) -> tuple[list, list]:
    spk = sorted(p.name for p in (FEAT / "tts_male_ja").iterdir())
    held = spk[-MALE_HELD:]
    items = {s: [(z, ROOT / "data/male_tts_corpus" / s / (z.stem + ".wav")) for z in sorted((FEAT / "tts_male_ja" / s).glob("*.npz"))] for s in spk}
    train = [x for s in spk if s not in held for x in items[s]]
    if vctk:
        vc = ROOT / "data/vctk/VCTK-Corpus/VCTK-Corpus/wav48"
        info = (vc.parent / "speaker-info.txt").read_text().splitlines()[1:]
        males = sorted("p" + r.split()[0] for r in info if len(r.split()) > 2 and r.split()[2] == "M")
        for i, m in enumerate(males):
            if i % 3 == 0 or not (vc / m).is_dir():
                continue
            train += [(None, w) for w in sorted((vc / m).glob("*.wav"))]
    return train, [x for s in held for x in items[s]]


class FemaleDS(torch.utils.data.IterableDataset):
    def __init__(self, spk: dict, keys: list, seed: int, f0_adv: int = 0):
        self.spk, self.keys, self.seed, self.f0_adv = spk, keys, seed, f0_adv
        self.flat = [(k, i) for k in keys for i in range(len(spk[k]))]

    def __iter__(self):
        from train_ddsp_vc import load48
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 1009 + (wi.id if wi else 0))
        while True:
            k, i = self.flat[rng.randrange(len(self.flat))]
            fz, fw, fc = self.spk[k][i]
            try:
                x = load48(fw)
                if len(x) < SEG + 8 * N.HOP:
                    continue
                u = rng.randrange(4, (len(x) - SEG) // N.HOP)
                s = u * N.HOP
                xin = x[s - CTX:s + SEG]
                if np.sqrt((xin[CTX:] ** 2).mean()) < 1e-3:
                    continue
                f0 = np.load(fz)["f0"].astype(np.float32)[u + self.f0_adv:u + self.f0_adv + T]
                f0 = np.pad(f0, (0, T - len(f0)))
                cv = np.zeros((SEG // 960, 768), np.float32)
                has_cv = 0.0
                if fc.exists():
                    c = torch.load(fc, map_location="cpu", weights_only=False)["content"].numpy()
                    j0 = s // 960
                    if j0 + SEG // 960 <= len(c):
                        cv, has_cv = c[j0:j0 + SEG // 960].astype(np.float32), 1.0
                others = [j for j in range(len(self.spk[k])) if j != i]
                jr = rng.choice(others)
                xr = load48(self.spk[k][jr][1])
                mu_t, sd_t = logf0_stats(np.load(self.spk[k][jr][0])["f0"])
                if sd_t == 0.0:
                    mu_t, sd_t = logf0_stats(f0)
                if len(xr) > REF_MAX:
                    r0 = rng.randrange(0, len(xr) - REF_MAX)
                    xr = xr[r0:r0 + REF_MAX]
                ref = np.zeros(REF_MAX, np.float32)
                ref[:len(xr)] = xr
                g = 2.0 ** rng.uniform(-0.5, 0.5)
                g = min(g, 0.99 / max(float(np.abs(xin).max()), 1e-6))
                yield (torch.from_numpy(xin * g), torch.from_numpy(f0), torch.from_numpy(cv), has_cv, torch.from_numpy(ref),
                       len(xr), torch.tensor([mu_t, sd_t], dtype=torch.float32))
            except Exception:
                continue


class MaleDS(torch.utils.data.IterableDataset):
    def __init__(self, items: list, seed: int):
        self.items, self.seed = items, seed

    def __iter__(self):
        from train_ddsp_vc import load48
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 7919 + (wi.id if wi else 0))
        while True:
            fz, fw = self.items[rng.randrange(len(self.items))]
            try:
                x = load48(fw)
                if len(x) < SEG + 8 * N.HOP:
                    continue
                if fz is None:
                    import artic_dsp as AD
                    f0a = AD.causal_yin(x.astype(np.float64), voi_max=0.45)[0].astype(np.float32)
                else:
                    f0a = np.load(fz)["f0"].astype(np.float32)
                import f0_fix as FX
                f0a = FX.fix_f0(f0a)[0]
                mu_s, sd_s = logf0_stats(f0a)
                if sd_s == 0.0:
                    continue
                u = rng.randrange(4, (len(x) - SEG) // N.HOP)
                s = u * N.HOP
                xin = x[s - CTX:s + SEG]
                if np.sqrt((xin[CTX:] ** 2).mean()) < 1e-3:
                    continue
                f0 = np.pad(f0a[u:u + T], (0, max(0, T - len(f0a[u:u + T]))))
                yield torch.from_numpy(xin), torch.from_numpy(f0), torch.tensor([mu_s, sd_s], dtype=torch.float32)
            except Exception:
                continue


def acf_pitch_loss(wav: torch.Tensor, f0: torch.Tensor, frame0: int) -> torch.Tensor:
    """出力波形が目標 f0 の周期で周期的かを罰する: 各フレーム(窓 1024)の正規化自己相関 r(τ) を τ = SR/f0 で読み、有声フレームの 1 − r(τ) の平均。
    wav のサンプル i はフレーム frame0 + i // HOP に対応。"""
    W = 1024
    n = wav.shape[-1]
    Tn = (n - W) // N.HOP
    fr = wav.unfold(-1, W, N.HOP)[:, :Tn] * torch.hann_window(W, device=wav.device)
    X = torch.fft.rfft(fr, n=2 * W)
    r = torch.fft.irfft(X.abs() ** 2, n=2 * W)[..., :W]
    r = r / r[..., :1].clamp(min=1e-8)
    ff = f0[:, frame0 + W // (2 * N.HOP):frame0 + W // (2 * N.HOP) + Tn]
    Tn = min(Tn, ff.shape[-1])
    r, ff = r[:, :Tn], ff[:, :Tn]
    tau = (N.SR / ff.clamp(min=50.0)).clamp(max=W - 2)
    lo = tau.floor().long()
    fr_ = tau - lo
    rt = torch.gather(r, 2, lo[..., None])[..., 0] * (1 - fr_) + torch.gather(r, 2, (lo + 1)[..., None])[..., 0] * fr_
    v = (ff > 0).float()
    return ((1 - rt) * v).sum() / v.sum().clamp(min=1)


def map_f0(f0: torch.Tensor, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    lf = torch.log(f0.clamp(min=1.0))
    m = tgt[:, :1] + (lf - src[:, :1])
    return torch.where(f0 > 0, torch.exp(m), torch.zeros_like(f0))


class MelD(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torch.nn.utils.parametrizations import weight_norm as wn
        self.layers = torch.nn.ModuleList([wn(torch.nn.Conv2d(1, 32, (5, 5), padding=2)),
                                           wn(torch.nn.Conv2d(32, 64, (5, 5), stride=(2, 1), padding=2)),
                                           wn(torch.nn.Conv2d(64, 64, (5, 5), stride=(2, 1), padding=2)),
                                           wn(torch.nn.Conv2d(64, 64, (5, 5), stride=(2, 1), padding=2)),
                                           wn(torch.nn.Conv2d(64, 1, (3, 3), padding=1))])

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        h = m[:, None]
        for i, l in enumerate(self.layers):
            h = l(h)
            if i < len(self.layers) - 1:
                h = F.leaky_relu(h, 0.2)
        return h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="zsvc1")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--r_steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--w_id", type=float, default=5.0)
    ap.add_argument("--w_art", type=float, default=2.0)
    ap.add_argument("--w_adv", type=float, default=0.5)
    ap.add_argument("--w_rip", type=float, default=0.0, help="交差出力の帯域別の縞の強さを目標話者の実 mel に合わせる")
    ap.add_argument("--w_adv_r", type=float, default=0.0, help="自己再構成の出力 mel にも判別器の敵対損失")
    ap.add_argument("--w_pitch", type=float, default=0.0)
    ap.add_argument("--init", default=None, help="重み(model/ema/D/opt)の初期値")
    ap.add_argument("--cv", action="store_true", help="内容 = ContentVec の予測(勾配停止)")
    ap.add_argument("--env_keep", type=int, default=0, help="内容の入力 mel を周波数方向 DCT の低次だけに(倍音の縞を消す)")
    ap.add_argument("--vctk", action="store_true", help="交差の男声に VCTK 実男声を加える")
    ap.add_argument("--f0_adv", type=int, default=0, help="女声の自己再構成で f0 条件を k フレーム先取り(出力フレームの本当のピッチに揃える・推論は因果 f0 のまま)")
    ap.add_argument("--rip", action="store_true", help="予測 mel = 包絡 + 調波性 × f0 の倍音の縞(決定的)")
    ap.add_argument("--pr_st", type=float, default=0.0, help="内容の入力で細かい構造(倍音の縞)だけを ±pr_st 半音ずらす(包絡はそのまま)")
    ap.add_argument("--pr_keep", type=int, default=12)
    ap.add_argument("--pr_amp", type=float, default=1.0, help="内容入力の細かい構造の振幅を U[pr_amp, 1] 倍(入力の縞の強さから調波性を読ませない)")
    ap.add_argument("--train_voc", action="store_true", help="出力部(nvoc)も一緒に学習し、女声の自己再構成を波形で監督する")
    ap.add_argument("--voc_lr", type=float, default=1e-4)
    ap.add_argument("--w_wave", type=float, default=1.0)
    ap.add_argument("--w_wgan", type=float, default=0.0, help="自己再構成の波形に GAN(新規 MPD + MRD-log・48kHz)")
    ap.add_argument("--vocoder", default=str(ROOT / "results/nvoc5r2/last.pt"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"prereg: results/{a.tag}/prereg.yaml", flush=True)
    from train_ddsp_vc import index, ecapa_model
    import eval_zsvc as EZ
    spk, tr, ev = index()
    males, males_held = male_index(a.vctk)
    print(f"female train speakers {len(tr)} | male train utts {len(males)} | male held utts {len(males_held)}", flush=True)
    fl = iter(torch.utils.data.DataLoader(FemaleDS(spk, tr, 3, a.f0_adv), batch_size=a.bs, num_workers=a.workers, pin_memory=True,
                                          persistent_workers=True, prefetch_factor=4))
    ml = iter(torch.utils.data.DataLoader(MaleDS(males, 5), batch_size=a.bs, num_workers=max(2, a.workers // 2), pin_memory=True,
                                          persistent_workers=True, prefetch_factor=4))
    model = Z.ZSVC(cv=a.cv, env_keep=a.env_keep, rip=a.rip).to(dev)
    D = MelD().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.8, 0.99))
    dopt = torch.optim.AdamW(D.parameters(), lr=a.lr, betas=(0.8, 0.99))
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    voc = N.NVoc().to(dev)
    voc.load_state_dict(torch.load(a.vocoder, map_location=dev, weights_only=False)["ema"])
    voc.eval()
    voc.requires_grad_(a.train_voc)
    vopt = torch.optim.AdamW(voc.parameters(), lr=a.voc_lr, betas=(0.8, 0.99)) if a.train_voc else None
    vema = {k: v.detach().clone() for k, v in voc.state_dict().items()}
    evv = N.NVoc().to(dev)
    evv.load_state_dict(vema)
    evv.eval()
    from train_s1_1 import MEL_SPECS, build_mel, logmel_l1, mrstft as mrstft_w
    wD = None
    if a.w_wgan > 0:
        from types import SimpleNamespace
        from bigvgan.discriminators import MultiPeriodDiscriminator
        wmpd = MultiPeriodDiscriminator(SimpleNamespace(mpd_reshapes=[2, 3, 5, 7, 11], use_spectral_norm=False, discriminator_channel_mult=1)).to(dev)
        wmrd = TN.MRDLog().to(dev)
        wD = list(wmpd.parameters()) + list(wmrd.parameters())
        wdopt = torch.optim.AdamW(wD, lr=2e-4, betas=(0.8, 0.99))
    wmels = [build_mel(nf, hm, nm).to(dev) for nf, hm, nm in MEL_SPECS]
    emb = ecapa_model(dev)
    from transformers import HubertModel
    cvm = HubertModel.from_pretrained("lengyue233/content-vec-best").to(dev).eval()
    cvm.requires_grad_(False)
    step = 0
    if a.init:
        st = torch.load(a.init, map_location=dev, weights_only=False)
        if "model" in st:
            model.load_state_dict(st["model"])
            ema = st["ema"]
            opt.load_state_dict(st["opt"])
            D.load_state_dict(st["D"])
            dopt.load_state_dict(st["dopt"])
        else:
            miss = model.load_state_dict(st["ema"], strict=False)
            print("init missing keys", miss.missing_keys, "unexpected", miss.unexpected_keys, flush=True)
            ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print("init from", a.init, "step", st["step"], flush=True)
    if a.resume and (out / "last.pt").exists():
        st = torch.load(out / "last.pt", map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        ema = st["ema"]
        opt.load_state_dict(st["opt"])
        D.load_state_dict(st["D"])
        dopt.load_state_dict(st["dopt"])
        step = int(st["step"])
    evm = Z.ZSVC(cv=a.cv, env_keep=a.env_keep, rip=a.rip).to(dev)
    evset = EZ.build_evalset(spk, ev, males_held, a.f0_adv)
    log = open(out / "train.jsonl", "a")

    def evaluate() -> None:
        evm.load_state_dict(ema)
        evv.load_state_dict(vema)
        r = {"step": step, **EZ.evaluate(evm, evv, emb, evset, dev)}
        print("eval", json.dumps(r), flush=True)
        log.write(json.dumps(r) + "\n")
        log.flush()
        model.train()

    def melctx(x: torch.Tensor) -> torch.Tensor:
        return N.NVoc.mel_ctx(model, x)

    def cv16(w: torch.Tensor) -> torch.Tensor:
        return cvm(AF.resample(w, N.SR, 16000)).last_hidden_state

    if step == 0:
        evaluate()
    acc: dict = {}
    t0 = time.time()
    total = 30 if a.smoke else a.steps
    n_bad = 0
    model.train()
    while step < total:
        step += 1
        xin, f0, cv, has_cv, ref, nref, tstat = (t.to(dev, non_blocking=True) if torch.is_tensor(t) else t for t in next(fl))
        has_cv = has_cv.to(dev).float()
        with torch.no_grad():
            mel = melctx(xin)
            B = mel.shape[0]
            alpha = torch.exp(torch.empty(B, device=dev).uniform_(math.log(0.8), math.log(1.25)))
            mel_p = Z.warp_mel(mel, alpha, model.centers) + torch.empty(B, 1, 1, device=dev).uniform_(-1.4, 1.4)
            if a.pr_st > 0:
                env = Z.lifter_env(mel_p, model.dct, a.pr_keep)
                rr = 2.0 ** (torch.empty(B, device=dev).uniform_(-a.pr_st, a.pr_st) / 12)
                amp = torch.empty(B, 1, 1, device=dev).uniform_(a.pr_amp, 1.0)
                mel_p = env + amp * Z.warp_mel(mel_p - env, rr, model.centers)
            rmel = N.NVoc.mel_ctx(model, torch.cat([torch.zeros(B, CTX, device=dev), ref], -1))
            rmask = (torch.arange(rmel.shape[-1], device=dev)[None] < (nref.to(dev) // N.HOP)[:, None]).float()
        s = model.spk(rmel, rmask)
        y, parts = model(mel_p, model.level(mel), f0, s, return_parts=True)
        lm = (y - mel)[..., W0F:].abs().mean()
        auxp = F.avg_pool1d(parts["cvpred"], 4).transpose(1, 2) if a.cv else model.content.aux(F.avg_pool1d(parts["content"], 4).transpose(1, 2))
        aux = ((1 - F.cosine_similarity(auxp, cv, dim=-1)).mean(-1) * has_cv).sum() / has_cv.sum().clamp(min=1)
        loss = 15 * lm + aux
        rec = {"lm": lm, "aux": aux}
        if a.train_voc:
            exf = torch.stack([torch.zeros(B, SEG, device=dev), torch.randn(B, SEG, device=dev)], 1)
            wr = voc.generate(y, exf)[:, N.HOP * W0F:]
            tw = xin[:, CTX - N.DELAY + N.HOP * W0F:CTX + SEG - N.DELAY]
            wl = 15 * logmel_l1(wr[:, None], tw[:, None], wmels) + 2 * mrstft_w(wr[:, None], tw[:, None])
            loss = loss + a.w_wave * wl
            rec["wave"] = wl
            if wD is not None:
                off = random.randrange(0, wr.shape[-1] - 16384)
                gg = (0.95 / tw.abs().amax(-1, keepdim=True).clamp(min=1e-3)).clamp(max=20.0)
                rw, fw = (tw * gg)[:, off:off + 16384], (wr * gg)[:, off:off + 16384]
                wdopt.zero_grad(set_to_none=True)
                a1, b1, _, _ = wmpd(rw[:, None], fw.detach()[:, None])
                a2, b2, _, _ = wmrd(rw, fw.detach())
                wdl = sum(((p.float() - 1) ** 2).mean() + (q.float() ** 2).mean() for p, q in zip(a1 + a2, b1 + b2))
                if torch.isfinite(wdl):
                    wdl.backward()
                    torch.nn.utils.clip_grad_norm_(wD, 500.0)
                    wdopt.step()
                for p_ in wD:
                    p_.requires_grad_(False)
                _, b1, f1r, f1g = wmpd(rw[:, None], fw[:, None])
                _, b2, f2r, f2g = wmrd(rw, fw)
                for p_ in wD:
                    p_.requires_grad_(True)
                wadv = sum(((q.float() - 1) ** 2).mean() for q in b1 + b2)
                wfm = sum(F.l1_loss(u.detach().float(), v.float()) for A, Bb in zip(f1r + f2r, f1g + f2g) for u, v in zip(A, Bb))
                loss = loss + a.w_wgan * (wadv + 2 * wfm)
                rec.update({"wadv": wadv, "wfm": wfm, "wdl": wdl})
        if step > a.r_steps:
            xm, f0m, sstat = (t.to(dev, non_blocking=True) for t in next(ml))
            with torch.no_grad():
                melm = melctx(xm)
            f0t = map_f0(f0m, sstat, tstat)
            yc, pm = model(melm, model.level(melm), f0t, s, return_parts=True)
            dopt.zero_grad(set_to_none=True)
            dl = ((D(mel[..., W0F:]) - 1) ** 2).mean() + (D(yc.detach()[..., W0F:]) ** 2).mean()
            if a.w_adv_r > 0:
                dl = dl + (D(y.detach()[..., W0F:]) ** 2).mean()
            dl.backward()
            dopt.step()
            exc = torch.stack([torch.zeros(B, SEG, device=dev), torch.randn(B, SEG, device=dev)], 1)
            voc.requires_grad_(False)
            wav = voc.generate(yc, exc)[:, N.HOP * W0F:]
            with torch.no_grad():
                e_ref = emb(ref[:, :min(REF_MAX, int(nref.min()))])
                c_in = cv16(xm[:, CTX + N.HOP * W0F - N.DELAY:CTX + SEG - N.DELAY])
            idl = (1 - (emb(wav) * e_ref).sum(-1)).mean()
            c_out = cv16(wav)
            n = min(c_out.shape[1], c_in.shape[1])
            art = (1 - F.cosine_similarity(c_out[:, :n], c_in[:, :n], dim=-1)).mean()
            adv = ((D(yc[..., W0F:]) - 1) ** 2).mean()
            if a.w_rip > 0:
                def ripstd(mm, vmask):
                    r = mm - Z.lifter_env(mm, model.dct, 12)
                    w = vmask[:, None, :].float()
                    return ((r ** 2 * w).sum(-1) / w.sum(-1).clamp(min=1)).clamp(min=1e-6).sqrt()
                bmask = (model.centers >= 300) & (model.centers <= 5000)
                rp = ripstd(yc[..., W0F:], f0t[..., W0F:] > 0)[:, bmask]
                with torch.no_grad():
                    rt = ripstd(mel[..., W0F:], f0[..., W0F:] > 0)[:, bmask]
                okm = ((f0t[..., W0F:] > 0).sum(-1) > 10) & ((f0[..., W0F:] > 0).sum(-1) > 10)
                ripl = ((rp - rt).abs().mean(-1) * okm).sum() / okm.sum().clamp(min=1)
                loss = loss + a.w_rip * ripl
                rec["rip"] = ripl
            if a.w_adv_r > 0:
                advr = ((D(y[..., W0F:]) - 1) ** 2).mean()
                loss = loss + a.w_adv_r * advr
                rec["adv_r"] = advr
            if a.cv:
                with torch.no_grad():
                    c_m = cv16(xm[:, CTX:CTX + SEG])
                pa = F.avg_pool1d(pm["cvpred"], 4).transpose(1, 2)
                k = min(pa.shape[1], c_m.shape[1])
                aux_m = (1 - F.cosine_similarity(pa[:, :k], c_m[:, :k], dim=-1)).mean()
                loss = loss + aux_m
                rec["aux_m"] = aux_m
            loss = loss + a.w_id * idl + a.w_art * art + a.w_adv * adv
            rec.update({"id": idl, "art": art, "adv": adv, "dl": dl})
            voc.requires_grad_(a.train_voc)
            if a.w_pitch > 0:
                pl = acf_pitch_loss(wav, f0t, W0F)
                loss = loss + a.w_pitch * pl
                rec["pitch"] = pl
        opt.zero_grad(set_to_none=True)
        if vopt is not None:
            vopt.zero_grad(set_to_none=True)
        bad = not torch.isfinite(loss)
        if not bad:
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            bad = not torch.isfinite(gn)
        if bad:
            print("non-finite at", step, flush=True)
            opt.zero_grad(set_to_none=True)
            n_bad += 1
            if n_bad >= 20:
                raise RuntimeError("non-finite 20 steps in a row")
            continue
        n_bad = 0
        opt.step()
        if vopt is not None:
            torch.nn.utils.clip_grad_norm_(voc.parameters(), 500.0)
            vopt.step()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(0.999).add_(v.detach(), alpha=0.001)
            if vopt is not None:
                for k, v in voc.state_dict().items():
                    if v.dtype.is_floating_point:
                        vema[k].mul_(0.999).add_(v.detach(), alpha=0.001)
        for k, v in rec.items():
            acc.setdefault(k, []).append(float(v))
        if step % 100 == 0 or a.smoke:
            r = {"step": step, "phase": "C" if step > a.r_steps else "R", "min": round((time.time() - t0) / 60, 1)}
            r.update({k: round(float(np.mean(v)), 4) for k, v in acc.items()})
            acc = {}
            print(json.dumps(r), flush=True)
            log.write(json.dumps(r) + "\n")
            log.flush()
        if step % a.eval_every == 0 or (a.smoke and step == total):
            evaluate()
            torch.save({"model": model.state_dict(), "ema": ema, "opt": opt.state_dict(), "D": D.state_dict(),
                        "dopt": dopt.state_dict(), "step": step}, out / "last.tmp")
            (out / "last.tmp").replace(out / "last.pt")
            (out / "snap").mkdir(exist_ok=True)
            torch.save({"ema": ema, "voc_ema": vema if vopt is not None else None, "step": step, "cv": a.cv, "env_keep": a.env_keep, "rip": a.rip},
                       out / "snap" / f"ema_{step // 1000}k.pt")
    print("done", step, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
