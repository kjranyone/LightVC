"""C3: 出力側の同一性監督で変換器 G を学習する(current/converter.md §3c・prereg 必須)。
経路(全て因果・出荷ゲートで未来不変性を起動時に検査): 男声 x → C1(単位の事後 p_t)・F1(f0・有声)・F2(c0)凍結 → 表 T[tgt] から ê_t = T[argmax p_t](因果の 5 フレーム平均)→ G(恒等初期化)→ 条件 → 出力部(凍結・EMA)→ y
f0 の写像の定数 med = ソース話者ごとの有声 log f0 の中央値(較正値・製品は最初の数秒で較正して固定)。
損失: λ_id·(1 − cos(ECAPA(y), e_tgt)) + λ_c·KL(p_src ‖ q(y)) + λ_Δ·(Δ² + 隣の差²)+ λ_p·(周期性の補正)²。WavLM-SV は学習に使わず監視だけ(ずる = fooling の検出)。
    RVOC_MANIFEST は使わない。  uv run python train_c3.py --tag c3_1 --steps 20000
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).parent))
import c1_content as CC
import f0est as FE
import idloss as ID
import nvoc as N
import pae as PA
import rvoc as R
import train_c1 as T1
import train_f0est as TF
import lvl as LV
import train_rvoc as TR

ROOT = Path(__file__).resolve().parent.parent
HOP = N.HOP
CTX_S, SEG_S = 1.5, 2.0
NCTX, NSEG = int(CTX_S * 200), int(SEG_S * 200)
NF = NCTX + NSEG
D_C = 24
W_SM = 5


def smooth3(E: np.ndarray, w: float = 0.25) -> np.ndarray:
    e = np.pad(E, ((0, 0), (1, 1)), mode="edge")
    return w * e[:, :-2] + (1 - 2 * w) * e[:, 1:-1] + w * e[:, 2:]


def male_rows_spk() -> list:
    """(wav, f0 npy, sr, dur, 話者キー)。f0est の台帳の男声 train。"""
    m = json.loads((ROOT / "data/f0est_hi/manifest.json").read_text())
    return [(r["wav"], str(ROOT / r["f0"]), r["sr"], r["dur"], r["src"] + "/" + r["spk"]) for r in m["rows"] if r["ok"] and r["split"] == "train" and r["src"] in ("vctk_m", "tts_m")]


def speaker_med(rows: list) -> dict:
    """話者ごとの有声 log f0 の中央値(全ファイルの f0 から)。"""
    acc: dict = {}
    for wav, f0p, sr, dur, key in rows:
        f = np.load(f0p)
        acc.setdefault(key, []).append(np.log(f[f > 0]))
    return {k: float(np.median(np.concatenate(v))) for k, v in acc.items()}


class SrcDS(torch.utils.data.IterableDataset):
    """男声の区間 = 左文脈 1.5s + 2.0s(ファイルの外は 0)・利得 ±3dB。med = 話者の較正値。c0 は GPU の F2 が作る(CheapTrick は使わない)。"""

    def __init__(self, rows: list, meds: dict, seed: int):
        self.rows, self.meds, self.seed = rows, meds, seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 131 + (wi.id if wi else 0))
        while True:
            wav, f0p, sr, dur, key = self.rows[rng.randrange(len(self.rows))]
            try:
                n48 = int(dur * N.SR)
                if n48 < 2 * N.SR:
                    continue
                s_seg = rng.randrange(0, int(n48 - SEG_S * N.SR)) // HOP * HOP
                a = s_seg - NCTX * HOP
                lo, hi = max(a, 0), a + NF * HOP
                x = TR.read_span(wav, sr, lo, hi - lo)
                x = np.concatenate([np.zeros(lo - a, np.float32), x]) if lo > a else x
                if len(x) < NF * HOP:
                    x = np.pad(x, (0, NF * HOP - len(x)))
                x = x[:NF * HOP].astype(np.float32)
                if np.sqrt((x[NCTX * HOP:] ** 2).mean()) < 1e-3:
                    continue
                g = min(2.0 ** rng.uniform(-0.5, 0.5), 0.99 / max(float(np.abs(x).max()), 1e-6))
                yield torch.from_numpy((x * g).astype(np.float32)), torch.tensor(self.meds[key], dtype=torch.float32)
            except Exception:
                continue


class Blk(nn.Module):
    def __init__(self, ch: int, dil: int, d_film: int):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 3, dilation=dil)
        self.c2 = nn.Conv1d(ch, ch, 1)
        self.film = nn.Linear(d_film, 2 * ch)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.lp = 2 * dil

    def forward(self, x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.pad(F.leaky_relu(x, 0.1), (self.lp, 0)))
        sc, sh = self.film(f).chunk(2, -1)
        h = h * (1 + sc[..., None]) + sh[..., None]
        return x + self.c2(F.leaky_relu(h, 0.1))


class ConvG(nn.Module):
    """因果 TCN。入力 [ê(24)・p_t(K)・log f0_out・c0・有声](フレーム 200fps)+ FiLM(目標の要約)→ Δc(24)・Δ周期性(4)。出力の重みは 0 で初期化(恒等)。"""

    def __init__(self, K: int, ch: int = 128, dils: tuple = (1, 2, 4, 8, 16, 32), d_film: int = 24 + 1 + 4, m_basis: int = 0):
        super().__init__()
        self.cfg = {"K": K, "ch": ch, "dils": list(dils), "d_film": d_film, "m_basis": m_basis}
        self.m = m_basis
        self.register_buffer("basis", torch.zeros(D_C, max(m_basis, 1)))
        self.register_buffer("zscale", torch.ones(max(m_basis, 1)))
        self.register_buffer("film_mu", torch.zeros(d_film))
        self.register_buffer("film_sd", torch.ones(d_film))
        self.register_buffer("dscale", torch.full((D_C,), 0.5))
        self.register_buffer("pscale", torch.full((4,), 0.3))
        self.inp = nn.Conv1d(D_C + K + 3, ch, 1)
        self.blks = nn.ModuleList(Blk(ch, d, d_film) for d in dils)
        self.out = nn.Conv1d(ch, (m_basis if m_basis else D_C) + 4, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, e_hat, p, lf, c0, v, film):
        film = (film - self.film_mu) / self.film_sd
        x = self.inp(torch.cat([e_hat, p, lf, c0, v], 1))
        for b in self.blks:
            x = b(x, film)
        o = self.out(F.leaky_relu(x, 0.1))
        act = ((c0 - 0.5) / 1.0).clamp(0, 1)
        if self.m:
            z = self.zscale[None, :, None] * torch.tanh(o[:, :self.m])
            dc = torch.einsum("km,bmt->bkt", self.basis, z) * act
            return dc, self.pscale[None, :, None] * torch.tanh(o[:, self.m:])
        return self.dscale[None, :, None] * torch.tanh(o[:, :D_C]) * act, self.pscale[None, :, None] * torch.tanh(o[:, D_C:])


def causal_avg(x: torch.Tensor, w: int) -> torch.Tensor:
    """[B, C, T] を過去 w フレームの平均(因果・先頭は複製)。"""
    return F.avg_pool1d(F.pad(x, (w - 1, 0), mode="replicate"), w, 1)


def build_cond(c0n, e_s, dc, lf, v, per, dper):
    """c0n [B,1,T]((c0 − C0_SIL)/10)・e_s [B,24,T]((c/10) の単位)・dc [B,24,T]・lf [B,1,T]・v [B,1,T]・per [B,4,T](目標の平均)・dper [B,4,T] → cond [B,31,T]。"""
    per_o = (per + dper).clamp(0, 1.5) * v
    return torch.cat([c0n, e_s + dc, lf, v, per_o], 1)


def augment(y: torch.Tensor) -> torch.Tensor:
    """微分できる出力の摂動(検証器の近道を断つ): 利得 ±6dB・対数周波数の緩い EQ(±3dB)・雑音 SNR 25〜40dB。"""
    B, L = y.shape
    g = 10 ** ((torch.rand(B, 1, device=y.device) * 12 - 6) / 20)
    Y = torch.fft.rfft(y, dim=-1)
    f = torch.linspace(0, 1, Y.shape[-1], device=y.device)[None]
    a = (torch.rand(B, 3, device=y.device) * 2 - 1) * 3.0
    eq_db = a[:, :1] * (f - 0.5) * 2 + a[:, 1:2] * torch.sin(2 * torch.pi * 2 * f) + a[:, 2:3] * torch.cos(2 * torch.pi * 3 * f)
    y = torch.fft.irfft(Y * 10 ** (eq_db / 20), n=L, dim=-1) * g
    snr = torch.rand(B, 1, device=y.device) * 15 + 25
    nz = torch.randn_like(y) * y.detach().pow(2).mean(-1, keepdim=True).sqrt() * 10 ** (-snr / 20)
    return y + nz


def basis_from_targets(Tt: torch.Tensor, m: int, idx: torch.Tensor) -> tuple:
    """学習話者の表 Tt [S, 24, K](条件の単位)から、単位ごとの話者間の差(単位の平均を引く)の PCA の上位 m 成分 → (basis [24, m], 成分の標準偏差 [m])。"""
    T = Tt[idx]
    D = (T - T.mean(0, keepdim=True)).permute(0, 2, 1).reshape(-1, T.shape[1])
    sub = D[torch.randperm(D.shape[0], device=D.device)[:200000]]
    U, S_, Vh = torch.linalg.svd(sub, full_matrices=False)
    return Vh[:m].T.contiguous(), (S_[:m] / sub.shape[0] ** 0.5)


def atomic_save(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def anchor_from_c1(p: torch.Tensor, n_cv: int) -> torch.Tensor:
    """C1 の事後 p [B, K, NF](200fps・フレーム t の窓は (t+1)·240 サンプルで終わり、教師の時刻は窓の終わり − 960 サンプル = 240(t − 3))を、区間(NCTX·HOP サンプルから)の
    ContentVec のフレーム j(中心 960j + 600 サンプル)に合わせる: 240(t − 3) = 240·NCTX + 960j + 600 → t = NCTX + 4j + 5.5 → t = 5 と 6 の平均(2 巡目のレビューで 1 フレーム早い版を修正)。→ [B, n_cv, K]"""
    j = torch.arange(n_cv, device=p.device)
    t0 = NCTX + 4 * j + 5
    t1 = NCTX + 4 * j + 6
    return 0.5 * (p[..., t0.clamp(max=p.shape[-1] - 1)] + p[..., t1.clamp(max=p.shape[-1] - 1)]).transpose(1, 2)


class Chain(nn.Module):
    """推論経路: x [B, L](48kHz・左文脈つき)→ 条件 [B, 31, n]・f0_out・Δ。C1・F1・F2・出力部の前の全てが因果。G 以外は凍結。"""

    def __init__(self, c1, mfront, Cb, f1, f1front, f2, G, Tt, mu, per_t, summ, dev):
        super().__init__()
        self.c1, self.mfront, self.f1, self.f1front, self.f2, self.G = c1, mfront, f1, f1front, f2, G
        for nm, t in (("Cb", Cb), ("Tt", Tt), ("mu", mu), ("per_t", per_t), ("summ", summ)):
            self.register_buffer(nm, t, persistent=False)
        self.dev = dev
        self.rho = 1.0
        self.onehot = False
        self.c0_avg = 3
        self.w_sm = 3

    def prep(self, x, idx, med, hard_override=None, drop=0.0):
        """凍結の前段(C1・F1・F2・表の引き): G に渡す入力一式。学習(c4)と推論で同じ経路。"""
        B, L = x.shape
        n = L // HOP
        with torch.no_grad():
            xp = torch.cat([torch.zeros(B, T1.PRE, device=x.device), x], 1)
            p = self.c1(self.mfront(xp))[..., -n:].softmax(1)
            xx = torch.cat([torch.zeros(B, FE.CTX48, device=x.device), x], 1)
            v_, p_ = self.f1(self.f1front(xx, n))
            f0s = FE.decode(v_, p_, 0.5)
            vo = (f0s > 0).float()
            lf0s = torch.log(f0s.clamp(min=1.0))
            f0o = torch.where(f0s > 0, torch.exp(self.rho * (lf0s - med[:, None]) + self.mu[idx][:, None]), torch.zeros_like(f0s))
            lf = torch.where(f0o > 0, torch.log(f0o.clamp(min=1.0) / 200.0), torch.zeros_like(f0o))[:, None]
            hard = p.argmax(1) if hard_override is None else hard_override
            if drop > 0:
                hard = torch.where(torch.rand(hard.shape, device=hard.device) < drop, torch.randint(0, p.shape[1], hard.shape, device=hard.device), hard)
            e_hat = torch.stack([self.Tt[idx[b]][:, hard[b]] for b in range(B)])
            e_s = causal_avg(e_hat, self.w_sm)
            c0n = self.f2(self.mfront(torch.cat([torch.zeros(B, N.WIN - N.HOP, device=x.device), x], 1)))[:, None]
            if self.c0_avg > 1:
                c0n = causal_avg(c0n, self.c0_avg)
            pm = self.per_t[idx][:, :, None].expand(-1, -1, n)
            p_in = F.one_hot(hard, p.shape[1]).permute(0, 2, 1).float() if self.onehot else p
        return {"e_s": e_s, "p": p_in, "lf": lf, "c0n": c0n, "vo": vo[:, None], "film": self.summ[idx], "pm": pm, "f0o": f0o, "p_soft": p, "hard": hard}

    def forward(self, x, idx, med, zero=False, hard_override=None):
        q = self.prep(x, idx, med, hard_override)
        dc, dp = self.G(q["e_s"], q["p"], q["lf"], q["c0n"], q["vo"], q["film"])
        if zero:
            dc, dp = dc * 0, dp * 0
        cond = build_cond(q["c0n"], q["e_s"], dc, q["lf"], q["vo"], q["pm"], dp)
        return cond, q["f0o"], dc, dp, q["p_soft"]


def ship_gate(chain: Chain, dev: str) -> bool:
    """変換器の推論経路(C1・F1・F2・表の引き・G)の未来不変性(実音声・1 bit)と静的遅延台帳。CPU のコピーで測る。"""
    import copy
    import ship_check as SC
    ch = copy.deepcopy(chain).cpu().eval()
    ch.dev = "cpu"
    idx = torch.zeros(1, dtype=torch.long)
    med = torch.full((1,), 4.9)

    def fn(x: torch.Tensor) -> torch.Tensor:
        n = len(x) // HOP
        with torch.no_grad():
            cond, f0o, dc, dp, p = ch(x[:n * HOP].float()[None], idx, med)
        return torch.cat([cond[0], f0o])

    la = SC.future_invariance(fn, hop=N.HOP, n=2 * N.SR, sr=N.SR, quantity=False, n_edit=150, male=2)
    return SC.ledger([("変換器の推論経路(C1・F1・F2・表の引き・G)実音声", la), ("出力部 DELAY 240", 5.0), ("ブロック HOP 240", 5.0)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--l_id", type=float, default=1.0)
    ap.add_argument("--l_c", type=float, default=0.5)
    ap.add_argument("--l_d", type=float, default=0.1)
    ap.add_argument("--l_p", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--nce", action="store_true", help="同一性の損失を、目標の話者を多数の話者の中から当てる InfoNCE にする(cos の損失の代わり)")
    ap.add_argument("--verifiers", default="ecapa", help="学習に使う検証器: ecapa / ecapa,wavlm")
    ap.add_argument("--aug", action="store_true", help="検証器に通す前に出力へ EQ・ゲイン・雑音の摂動(微分できる)")
    ap.add_argument("--basis", type=int, default=0, help="Δc を学習話者の表の差の主成分 m 個の部分空間に限る(0 = 24 次元そのまま)")
    ap.add_argument("--n_neg", type=int, default=96)
    ap.add_argument("--temp", type=float, default=0.07)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--targets", nargs="+", default=[str(ROOT / "data/c3/targets.npz")], help="目標の npz(複数なら連結: 参照の長さ・加工の変種)")
    ap.add_argument("--vctk_rep", type=int, default=1, help="VCTK の目標の行を何倍に増やすか(16 人と少ないので)")
    ap.add_argument("--rvoc", default=str(ROOT / "results/rvoc3hi/snap/ema_330k.pt"))
    ap.add_argument("--f1", default=str(ROOT / "results/f0est3/last.pt"))
    ap.add_argument("--f2", default=str(ROOT / "results/f2_2/last.pt"))
    ap.add_argument("--c1", default=str(ROOT / "results/c1_1"))
    ap.add_argument("--g_ch", type=int, default=128)
    ap.add_argument("--g_dils", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--n_targets", type=int, default=0, help="過学習ゲート: 学習する目標の話者数をこれに限る(0 = 全部)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    if not a.smoke and not (out / "prereg.yaml").exists():
        print(f"results/{a.tag}/prereg.yaml が無い: 起動しない", flush=True)
        return 1
    print(f"prereg: results/{a.tag}/prereg.yaml", flush=True)
    sr_ = torch.load(a.rvoc, map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=sr_["cfg"]["ch"], kernels=tuple(sr_["cfg"]["kernels"]), dils=tuple(sr_["cfg"]["dils"]), d_cond=sr_["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(sr_["ema"])
    gen.eval().requires_grad_(False)
    sm_w = TR.ckpt_env_smooth(sr_, None)
    sf1 = torch.load(a.f1, map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev)
    f1.load_state_dict(sf1["net"])
    f1.eval().requires_grad_(False)
    f1front = FE.Front().to(dev)
    sf2 = torch.load(a.f2, map_location="cpu", weights_only=False)
    f2 = LV.Lvl(sf2["cfg"]["ch"], tuple(sf2["cfg"]["dils"])).to(dev)
    f2.load_state_dict(sf2["net"])
    f2.eval().requires_grad_(False)
    c1d = Path(a.c1)
    st1 = torch.load(c1d / "last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev)
    c1.load_state_dict(st1["net"])
    c1.eval().requires_grad_(False)
    mfront = CC.MelFront().to(dev)
    Cb = torch.load(c1d / "codebook.pt").to(dev)
    K = Cb.shape[0]
    ecapa, wsv, cvec = ID.Ecapa(dev), ID.WavlmSV(dev), ID.ContentVec(dev)
    zs = [np.load(p) for p in a.targets]
    Z = {k: np.concatenate([z[k] for z in zs]) for k in zs[0].files}
    if a.vctk_rep > 1:
        isv = np.array([s_ == "vctk" for s_ in Z["src"].tolist()])
        Z = {k: np.concatenate([v] + [v[isv]] * (a.vctk_rep - 1)) for k, v in Z.items()}
    print("目標の行", len(Z["spk"]), "話者", len(set(Z["spk"].tolist())), "src", {s_: int((Z["src"] == s_).sum()) for s_ in sorted(set(Z["src"].tolist()))}, flush=True)
    Tt = torch.from_numpy(Z["T"].astype(np.float32)).to(dev) / 10.0
    mu = torch.from_numpy(Z["mu"]).to(dev)
    per_t = torch.from_numpy(Z["per"]).to(dev)
    e_ec = torch.from_numpy(Z["e_ecapa"]).to(dev)
    e_wv = torch.from_numpy(Z["e_wavlm"]).to(dev)
    S = Tt.shape[0]
    uniq = sorted(set(Z["spk"].tolist()))
    rs = np.random.default_rng(0).permutation(len(uniq))
    val_spk = {uniq[i] for i in rs[:max(20, len(uniq) // 20)]}
    isval = np.array([sp in val_spk for sp in Z["spk"].tolist()])
    val_idx = torch.from_numpy(np.where(isval)[0]).to(dev)
    tr_np = np.where(~isval)[0]
    if a.n_targets:
        keep = sorted({sp for sp in Z["spk"][tr_np].tolist()})[:a.n_targets]
        tr_np = np.array([i for i in tr_np if Z["spk"][i] in set(keep)])
    tr_idx = torch.from_numpy(tr_np).to(dev)
    summ = torch.cat([Tt.mean(2), mu[:, None], per_t], 1)
    print("targets", S, "train", len(tr_idx), "val", len(val_idx), flush=True)
    rows = male_rows_spk()
    meds = speaker_med(rows)
    print("source files", len(rows), "speakers", len(meds), flush=True)
    G = ConvG(K, a.g_ch, tuple(a.g_dils), m_basis=a.basis).to(dev)
    if a.basis:
        Bm, sd = basis_from_targets(Tt, a.basis, tr_idx)
        G.basis.copy_(Bm)
        G.zscale.copy_(1.5 * sd)
        print("Δ の部分空間: 成分", a.basis, "標準偏差", [round(float(v), 3) for v in sd[:6]], "…", flush=True)
    G.film_mu.copy_(summ[tr_idx].mean(0))
    G.film_sd.copy_(summ[tr_idx].std(0).clamp(min=1e-3))
    G.dscale.copy_(1.5 * Tt.mean(2)[tr_idx].std(0))
    G.pscale.copy_(1.5 * per_t[tr_idx].std(0).clamp(min=0.02))
    print("G params (M)", round(sum(p.numel() for p in G.parameters()) / 1e6, 3), "Δ scale (cond units) min/med/max", [round(float(v), 3) for v in (G.dscale.min(), G.dscale.median(), G.dscale.max())], flush=True)
    chain = Chain(c1, mfront, Cb, f1, f1front, f2, G, Tt, mu, per_t, summ, dev)
    if not a.smoke and not ship_gate(chain, dev):
        print("出荷ゲート FAIL: 起動しない", flush=True)
        return 1
    opt = torch.optim.AdamW(G.parameters(), lr=a.lr, betas=(0.8, 0.99), weight_decay=0.0)
    step0 = 0
    if a.resume and (out / "last.pt").exists():
        sd = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        G.load_state_dict(sd["G"])
        opt.load_state_dict(sd["opt"])
        step0 = sd["step"]
        print("resume from", step0, flush=True)
    loader = iter(torch.utils.data.DataLoader(SrcDS(rows, meds, 5 + step0), batch_size=a.bs, num_workers=a.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4))
    log = open(out / "train.jsonl", "a")
    n_seg_s = NSEG * HOP
    o = NCTX * HOP

    def fwd(x, med, idx, zero=False, seed=None):
        cond, f0o, dc, dp, p = chain(x, idx, med, zero)
        gsd = torch.Generator().manual_seed(seed) if seed is not None else None
        with torch.no_grad():
            exc = TR.excitation(f0o, x.shape[1], gsd)
        y = gen(cond, exc)
        ys = y[:, o + N.DELAY:o + n_seg_s]
        xs = x[:, o:o + n_seg_s - N.DELAY]
        return ys, xs, dc, dp, p

    def content(ys, xs, p):
        qy = (cvec(ys) @ Cb.T / tau).log_softmax(-1)
        ps = anchor_from_c1(p, qy.shape[1])
        return F.kl_div(qy, ps, reduction="none").sum(-1).mean()

    def evaluate() -> dict:
        """検証の目標(学習に使わない話者)で G あり / Δ = 0(P0 の因果版)を同じ入力・同じ励振の雑音で比べる: ECAPA・WavLM-SV の目標への cos と検証の目標の中での順位。"""
        G.eval()
        res = {k_: [] for k_ in ("id", "wv", "id0", "wv0", "top1", "top1_0", "rk", "rk0", "wtop1", "wtop1_0", "kl", "kl0")}
        with torch.no_grad():
            for it in range(6):
                x, med = (t.to(dev, non_blocking=True) for t in next(loader))
                j = torch.randint(len(val_idx), (x.shape[0],), device=dev)
                idx = val_idx[j]
                for zero in (False, True):
                    ys, xs, _, _, p = fwd(x, med, idx, zero, seed=1000 + it)
                    ee, ww = ecapa(ys), wsv(ys)
                    sim = ee @ e_ec[val_idx].T
                    simw = ww @ e_wv[val_idx].T
                    sfx = "0" if zero else ""
                    res["id" + sfx].append((ee * e_ec[idx]).sum(-1).mean().item())
                    res["wv" + sfx].append((ww * e_wv[idx]).sum(-1).mean().item())
                    res["top1" + ("_0" if zero else "")].append((sim.argmax(1) == j).float().mean().item())
                    res["wtop1" + ("_0" if zero else "")].append((simw.argmax(1) == j).float().mean().item())
                    res["rk" + sfx].append(((sim > sim.gather(1, j[:, None])).sum(1).float() + 1).mean().item())
                    res["kl" + sfx].append(content(ys, xs, p).item())
        G.train()
        return {k_: round(float(np.mean(v__)), 4) for k_, v__ in res.items()}

    tau = st1.get("tau", 0.05)
    acc: dict = {}
    t0 = time.time()
    total = step0 + 20 if a.smoke else a.steps
    for step in range(step0 + 1, total + 1):
        x, med = (t.to(dev, non_blocking=True) for t in next(loader))
        B = x.shape[0]
        idx = tr_idx[torch.randint(len(tr_idx), (B,), device=dev)]
        ys, xs, dc, dp, p = fwd(x, med, idx)
        yv = augment(ys) if a.aug else ys
        l_id = 0.0
        if a.nce:
            neg = tr_idx[torch.randint(len(tr_idx), (a.n_neg,), device=dev)]
            bank = torch.cat([idx, neg])
            lab = torch.arange(B, device=dev)
            if "ecapa" in a.verifiers:
                l_id = l_id + F.cross_entropy(ecapa(yv) @ e_ec[bank].T / a.temp, lab)
            if "wavlm" in a.verifiers:
                l_id = l_id + F.cross_entropy(wsv(yv) @ e_wv[bank].T / a.temp, lab)
        else:
            if "ecapa" in a.verifiers:
                l_id = l_id + (1 - (ecapa(yv) * e_ec[idx]).sum(-1)).mean()
            if "wavlm" in a.verifiers:
                l_id = l_id + (1 - (wsv(yv) * e_wv[idx]).sum(-1)).mean()
        l_c = content(ys, xs, p)
        dcs = dc[..., NCTX:] / (G.dscale[None, :, None] if not a.basis else (G.basis * G.zscale[None]).abs().sum(1)[None, :, None].clamp(min=1e-3))
        l_d = (dcs ** 2).mean() + ((dcs[..., 1:] - dcs[..., :-1]) ** 2).mean() * 10
        l_p = ((dp[..., NCTX:] / G.pscale[None, :, None]) ** 2).mean()
        loss = a.l_id * l_id + a.l_c * l_c + a.l_d * l_d + a.l_p * l_p
        if step == step0 + 1:
            gi = torch.autograd.grad(l_id, list(G.parameters()), retain_graph=True, allow_unused=True)
            gc = torch.autograd.grad(l_c, list(G.parameters()), retain_graph=True, allow_unused=True)
            ni = float(torch.sqrt(sum((g_ ** 2).sum() for g_ in gi if g_ is not None)))
            nc = float(torch.sqrt(sum((g_ ** 2).sum() for g_ in gc if g_ is not None)))
            print(f"勾配ノルム(Δ = 0 の初期): ∂l_id {ni:.4g}  ∂l_c {nc:.4g}  比 l_c/l_id {nc / max(ni, 1e-12):.2f}(重み込み: {a.l_c * nc / max(a.l_id * ni, 1e-12):.2f})", flush=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0)
        if not torch.isfinite(gn):
            print("非有限の勾配: スキップ", flush=True)
            continue
        opt.step()
        rec = {"id": l_id, "c": l_c, "d": l_d, "p": l_p, "gn": gn}
        if step % 50 == 0:
            with torch.no_grad():
                rec["wv_cos"] = (wsv(ys) * e_wv[idx]).sum(-1).mean()
                rec["dc_rms_over_scale"] = dcs.pow(2).mean().sqrt()
        for k_, v__ in rec.items():
            acc.setdefault(k_, []).append(float(v__.detach()))
        if step % 100 == 0 or a.smoke:
            r = {"step": step, "min": round((time.time() - t0) / 60, 1), **{k_: round(float(np.mean(v__)), 4) for k_, v__ in acc.items()}}
            print(json.dumps(r), flush=True)
            log.write(json.dumps(r) + "\n")
            log.flush()
            acc = {}
        if step % a.eval_every == 0 or (a.smoke and step == total) or step == step0 + 1:
            ev = evaluate()
            print("eval", json.dumps({"step": step, **ev}), flush=True)
            log.write(json.dumps({"step": step, "eval": ev}) + "\n")
            log.flush()
            atomic_save({"G": G.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": G.cfg, "sm_w": sm_w, "W_SM": W_SM}, out / "last.pt")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
