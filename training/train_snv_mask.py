"""paper_snv rev32 §5 M23 の前段 v3 の時間周波数の重み(段 I17): 前段 v3 の計算は変えず、その入力のエネルギーに学んだ重み m(t, f) ∈ [0, 1] を掛ける。
重み = 2048 点の左寄せの窓(前段 v3 の周期性と同じ格子・1025 本・ホップ 240・窓は (t+1)H で終わる = 先読み 0)の上で、前段 D と同じ因果の畳み込み(train_snv_sep の Front と Block・出力だけ 1025 の重み・初期は 1 に近い)が出す。
features(M, x) / features_with_mask(x, m) = 有声と高さは前段 v3 のまま・周期性は前段 v3 の式を m² ⊙ P で・包絡は CheapTrick の mel の帯の振幅に帯の利得 √((m² ⊙ P)·fb / (P·fb)) を掛けて対数と DCT = m = 1 で前段 v3 と 1 bit 同じ。
m は振幅の重み。網はパワーの比 |S|²/(|S|²+|N|²) を当てるので、features は網の出力の平方根を m にする(パワーに掛かるのは比そのもの)。
学習 = train_snv_sep の Mix(--fam i15 の族・声の成分 S と残り N を分けて持つ)で、当てる先 = 理想の比の重み |S|² / (|S|² + |N|²)・損失 = 乱れた音のパワーの 0.3 乗で重みを付けた L1。入力側だけ・当てる先は元の音の成分から。
    cd training && uv run --no-sync python train_snv_mask.py --tag snv_i17_mask --prereg results/snv_i17/prereg.yaml [--fam i15] [--steps 20000] [--bs 16] [--smoke]
paper_snv rev34 §5 M24(段 I19・--target det): 同じ網が、足された出来事の音(手拍子・叩く・擦る = 族の events)がビンのパワーで声と残りの背景の和を越えるかを判定する(当てる先 = [|E|² > |S|² + |N − E|²]・乱れた音のパワーの 0.3 乗で重みを付けた 2 値の交差エントロピー)。
features は判定 ≥ 0.5 のビンを外し(重み 0)、他のビンは重み 1 のまま = 判定が出ない所は前段 v3 と 1 bit 同じ。保存の cfg の kind = det で features と load が切り替わる。
paper_snv rev38 §5 M27(段 I23・--target gate): 同じ網の形で出力 1 本の門 a(t) を学ぶ(当てる先 = [フレームの音楽と出来事の音のパワー > 0.01 ×(声 + 床 + 雑音の背景)]・2 値の交差エントロピー)。保存の cfg の kind = gate・mask = 使う重みの保存・dec = myin。
features は門が開いたフレームだけ重みを掛け、有声と高さを重みで濾した YIN で決める。閉じたフレームは重み 1 = 前段 v3 と 1 bit 同じ。
cfg の hyst = [on, off](paper_snv rev39・段 I24)= 門の確率が 0.5 以上で on フレーム続いたら開き、0.5 未満で off フレーム続いたら閉じる(因果の持続の事前)。
重みの網に dec = "myin" を付けると(paper_snv §9 O13・段 I19 の後の切り分け)、有声と高さを snv_p0_front.f0_track_myin(YIN の低い帯の区間とスペクトルに重みを掛けてから候補を作る・重みが全て 1 のフレームは元のまま)で決める。
--target ratio_nf(paper_snv rev36 の候補): 比の当てる先で床を声の側に数える = |S + F|² / (|S + F|² + |N − F|²)(F = 族の床・操作者の録音の一部として外さない)。features は ratio と同じ。
--dil d1,d2,…(段 P50): 重みの網の因果の畳み込みの膨らみの並び(既定 train_snv_sep.DIL = 受容野 257 フレーム)。--crop N: 学習の切り出しのフレーム数(既定 800 = 4 s・検証の束は 800 のまま)。
--band λ(paper_snv rev91 §5 M35・段 P37 step b): 損失に、前段 v3 の包絡の mel の帯ごとの、重みを通したパワーの比の対数の誤差 |log(g_θ + 1e-4) − log(g* + 1e-4)|(g = (重み ⊙ P)·fb / P·fb・g* は理想の比の重み)の平均を λ 倍で足す = 符号の入力(対数の包絡)の誤差に揃えた項。
--eval_only <ckpt>: 保存した重みの網を同じ検証の束で測って終わる(l1w・帯の誤差 lb・足された音と声の残り)。
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_

import pae as PA
import snv_p0_front as P
import train_snv_p1 as T
import train_snv_sep as SEP

ROOT = Path(__file__).resolve().parent.parent
H = SEP.H
NB = SEP.NFFT // 2 + 1
B0 = 3.0


class MaskNet(nn.Module):
    def __init__(self, ch: int = 256, b0: float = B0, nb: int = NB, dil: tuple = SEP.DIL) -> None:
        super().__init__()
        self.front = SEP.Front()
        self.inp = nn.Conv1d(SEP.D_IN, ch, 1)
        self.blocks = nn.ModuleList([SEP.Block(ch, d) for d in dil])
        self.ln = nn.LayerNorm(ch)
        self.head = nn.Conv1d(ch, nb, 1)
        with torch.no_grad():
            self.head.weight.mul_(0.01)
            self.head.bias.fill_(b0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(self.front(x))
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln(h.transpose(1, 2)).transpose(1, 2))


def power(w: torch.Tensor) -> torch.Tensor:
    Tn = w.shape[-1] // H
    fr = F_.pad(w.float(), (SEP.NFFT - H, 0)).unfold(-1, SEP.NFFT, H)[..., :Tn, :]
    win = torch.hann_window(SEP.NFFT, periodic=False, device=w.device)
    return torch.fft.rfft(fr * win, n=SEP.NFFT).abs().pow(2)


BAND_EPS = 1e-4
_FB: dict = {}


def band_err(m: torch.Tensor, irm: torch.Tensor, px: torch.Tensor) -> torch.Tensor:
    if px.device not in _FB:
        _FB[px.device] = torch.from_numpy(PA._np_consts()[0]).float().to(px.device)
    fb = _FB[px.device]
    den = px @ fb.T + 1e-12
    return (torch.log((m * px) @ fb.T / den + BAND_EPS) - torch.log((irm * px) @ fb.T / den + BAND_EPS)).abs()


def loss_fn(logit: torch.Tensor, b: dict, band: float = 0.0) -> tuple[torch.Tensor, dict]:
    m = torch.sigmoid(logit).transpose(1, 2)
    ps, pn, px = power(b["s"]), power(b["n"]), power(b["x"])
    irm = ps / (ps + pn + 1e-10)
    w = (px + 1e-10) ** 0.3
    w = w / w.mean(dim=(1, 2), keepdim=True)
    l = (w * (m - irm).abs()).mean()
    if band <= 0.0:
        return l, {"l1w": float(l)}
    lb = band_err(m, irm, px).mean()
    return l + band * lb, {"l1w": float(l), "lb": float(lb)}


def loss_det(logit: torch.Tensor, b: dict) -> tuple[torch.Tensor, dict]:
    z = logit.transpose(1, 2)
    ps, pe, pr, px = power(b["s"]), power(b["e"]), power(b["n"] - b["e"]), power(b["x"])
    lab = (pe > ps + pr).float()
    w = (px + 1e-10) ** 0.3
    w = w / w.mean(dim=(1, 2), keepdim=True)
    l = (w * F_.binary_cross_entropy_with_logits(z, lab, reduction="none")).mean()
    return l, {"bce": float(l)}


GATE_K = 0.01


def gate_label(b: dict) -> torch.Tensor:
    fs = lambda w: power(w).sum(-1)
    ps, pe, pf, pb = fs(b["s"]), fs(b["e"]), fs(b["f"]), fs(b["b"])
    bt = b["bt"].float()[:, None]
    return ((pe + bt * pb) > GATE_K * (ps + pf + (1 - bt) * pb)).float()


def loss_gate(logit: torch.Tensor, b: dict) -> tuple[torch.Tensor, dict]:
    l = F_.binary_cross_entropy_with_logits(logit[:, 0], gate_label(b))
    return l, {"bce": float(l)}


class Gated(nn.Module):
    def __init__(self, gate: nn.Module, mask: nn.Module) -> None:
        super().__init__()
        self.gate, self.mask = gate, mask
        self.kind = "gate"
        self.dec = None
        self.hyst = None
        self.thr = 0.5
        self.gamma = 1.0


def hysteresis(p: torch.Tensor, on: int, off: int, thr: float = 0.5) -> torch.Tensor:
    out = torch.zeros(len(p), dtype=torch.bool)
    state, c_on, c_off = False, 0, 0
    for t, v in enumerate((p >= thr).tolist()):
        c_on, c_off = (c_on + 1, 0) if v else (0, c_off + 1)
        if not state and c_on >= on:
            state = True
        elif state and c_off >= off:
            state = False
        out[t] = state
    return out


def weight(M: nn.Module, z: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(z)
    return (p < 0.5).double() if getattr(M, "kind", "ratio") == "det" else p.sqrt()


@torch.no_grad()
def features(M: nn.Module, x: np.ndarray, dev: str) -> dict:
    Tn = P.n_frames(len(x))
    xt = torch.from_numpy(np.asarray(x[: Tn * H], np.float32))
    if getattr(M, "kind", "ratio") == "gate":
        xin = xt[None].to(dev)
        p = torch.sigmoid(M.gate(xin)[0, 0].double()).cpu()[:Tn]
        op = hysteresis(p, *M.hyst, M.thr) if M.hyst else p >= M.thr
        mm = torch.sigmoid(M.mask(xin)[0].double()).transpose(0, 1).cpu()[:Tn].sqrt()
        if M.gamma != 1.0:
            mm = mm.pow(M.gamma)
        m = torch.where(op[:, None], mm, torch.ones_like(mm))
        u = features_with_mask(x, m, P.f0_track_myin(xt, m)[0] if M.dec == "myin" else None)
        return u | {"gate_open": float(op.double().mean()), "gate": op.numpy().astype(np.float32), "gate_p": p.numpy().astype(np.float32)}
    m = weight(M, M(xt[None].to(dev))[0].double()).transpose(0, 1).cpu()[:Tn]
    return features_with_mask(x, m, P.f0_track_myin(xt, m)[0] if getattr(M, "dec", None) == "myin" else None)


@torch.no_grad()
def features_with_mask(x: np.ndarray, m: torch.Tensor, f0: torch.Tensor | None = None) -> dict:
    Tn = P.n_frames(len(x))
    x = np.asarray(x[: Tn * H], np.float32)
    xt = torch.from_numpy(x)
    if f0 is None:
        f0, _ = P.f0_track(xt)
    f0n = f0.cpu().numpy().astype(np.float64)
    m = m.double()[:Tn]
    pw = P.power_p(xt, Tn)
    pm = pw * m.pow(2)
    per = P.periodicity_from_power(pm, f0).cpu().numpy()
    fb, _ = PA._np_consts()
    num, den = pm.numpy() @ fb.T, pw.numpy() @ fb.T
    g = np.where(den > 0, np.sqrt(np.divide(num, den, out=np.ones_like(num), where=den > 0)), 1.0)
    env = P.envelope_from_amp(P.envelope_amp(x, f0n) * g)
    y = T.to_y(env, per, f0n)
    v = (f0n > 0).astype(np.float32)
    s, n = T.voiced_mean(env, f0n)
    lf = np.where(f0n > 0, np.log(np.maximum(f0n, 1.0) / 200.0), 0.0).astype(np.float32)
    return {"y": y, "lf": lf, "v": v, "f0": f0n.astype(np.float32), "vs": s.astype(np.float32), "vn": n, "mask_mean": float(m.mean())}


def load(path: Path, dev: str) -> nn.Module:
    st = torch.load(path, map_location="cpu", weights_only=False)
    if st["cfg"].get("kind") == "gate":
        G = MaskNet(st["cfg"]["ch"], 0.0, 1).to(dev).eval()
        G.load_state_dict(st["M"])
        out = Gated(G, load(ROOT / st["cfg"]["mask"], dev)).eval()
        out.dec = st["cfg"].get("dec")
        out.hyst = st["cfg"].get("hyst")
        out.thr = float(st["cfg"].get("thr", 0.5))
        out.gamma = float(st["cfg"].get("gamma", 1.0))
        return out
    M = MaskNet(st["cfg"]["ch"], dil=tuple(st["cfg"].get("dil", SEP.DIL))).to(dev).eval()
    M.load_state_dict(st["M"])
    M.kind = st["cfg"].get("kind", "ratio")
    return M


def unit_tests() -> None:
    import librosa
    M = MaskNet(32).eval()
    with torch.no_grad():
        M.head.bias.fill_(40.0)
        M.head.weight.zero_()
    x = librosa.load(str(ROOT / "data/snv_e1/ef000/t00.wav"), sr=P.SR, mono=True)[0]
    Tn = P.n_frames(len(x))
    x = x[: Tn * H]
    a, ref = features(M, x, "cpu"), P.analyze(x)
    yr = T.to_y(ref["env"], ref["per"], ref["f0"])
    assert float(np.abs(a["y"] - yr).max()) == 0.0, ("重み 1 で前段 v3 と違う", float(np.abs(a["y"] - yr).max()))
    Md = MaskNet(32, -40.0).eval()
    with torch.no_grad():
        Md.head.weight.zero_()
    Md.kind = "det"
    assert float(np.abs(features(Md, x, "cpu")["y"] - yr).max()) == 0.0, "判定なしで前段 v3 と違う"
    Gc = MaskNet(32, -40.0, 1).eval()
    with torch.no_grad():
        Gc.head.weight.zero_()
    Wr = MaskNet(32).eval()
    Gw = Gated(Gc, Wr)
    Gw.dec = "myin"
    ug = features(Gw, x, "cpu")
    assert float(np.abs(ug["y"] - yr).max()) == 0.0 and float(np.abs(ug["f0"] - ref["f0"]).max()) == 0.0, "門が閉じて前段 v3 と違う"
    M2 = MaskNet(32).eval()
    z = torch.randn(1, 600 * H) * 0.1
    z2 = z.clone()
    z2[:, 301 * H:] = torch.randn(1, z.shape[1] - 301 * H)
    with torch.no_grad():
        o, o2 = M2(z), M2(z2)
    assert float((o[..., :301] - o2[..., :301]).abs().max()) < 1e-5, "重みの網に先読みがある"
    assert float((o[..., 301:] - o2[..., 301:]).abs().max()) > 1e-6, "未来の書き換えで後ろが動かない(検査が無効)"
    g = torch.Generator().manual_seed(0)
    px, irm = torch.rand(2, 50, NB, generator=g) + 0.01, torch.rand(2, 50, NB, generator=g)
    assert float(band_err(irm, irm, px).max()) == 0.0, "理想の重みで帯の誤差が 0 でない"
    e1, e9 = float(band_err(torch.ones_like(irm), irm * 0.01, px).mean()), float(band_err(torch.full_like(irm, 0.1), irm * 0.01, px).mean())
    assert e1 > e9 > 1.0, ("帯の誤差が深さの不足を罰しない", e1, e9)
    zz = torch.zeros(2, NB, 50, requires_grad=True)
    lf, _ = loss_fn(zz, {"s": torch.randn(2, 50 * H, generator=g) * 0.1, "n": torch.randn(2, 50 * H, generator=g) * 0.1, "x": torch.randn(2, 50 * H, generator=g) * 0.14}, 0.1)
    lf.backward()
    assert bool(torch.isfinite(zz.grad).all()) and float(zz.grad.abs().sum()) > 0, "帯の項の勾配"
    print("unit tests ok(重み 1 で前段 v3 と同じ・先読み 0)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--ch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prereg", default="results/snv_i17/prereg.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fam", default="i15", choices=sorted(SEP.FAMS))
    ap.add_argument("--target", default="ratio", choices=["ratio", "det", "ratio_nf", "gate"])
    ap.add_argument("--gate_mask", default="results/snv_i21_mask_nf/last.pt")
    ap.add_argument("--dec", default="myin")
    ap.add_argument("--band", type=float, default=0.0)
    ap.add_argument("--eval_only", default="")
    ap.add_argument("--dil", default="")
    ap.add_argument("--crop", type=int, default=SEP.CROP)
    a = ap.parse_args()
    det = a.target == "det"
    gate = a.target == "gate"
    parts = "ev" if a.target in ("det", "ratio_nf", "gate") else True
    split_nf = (lambda b: b | {"s": b["s"] + b["f"], "n": b["n"] - b["f"]}) if a.target == "ratio_nf" else (lambda b: b)
    lossf = loss_gate if gate else loss_det if det else (lambda z, bb: loss_fn(z, bb, a.band))
    fam = SEP.FAMS[a.fam]
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    if not a.smoke and not (ROOT / a.prereg).exists():
        print(f"{a.prereg} が無い: 起動しない", flush=True)
        return 1
    print("prereg", ROOT / a.prereg, "paper current/paper_snv.md", flush=True)
    torch.manual_seed(a.seed)
    unit_tests()
    split = json.loads((ROOT / "results/snv_i8_m13/split.json").read_text())
    tr, va = SEP.utterances(split["train"], fast=bool(fam.get("fast"))), SEP.utterances(split["val"])
    bs_ = SEP.bg_split()
    (out / "split.json").write_text(json.dumps(split))
    (out / "bg_split.json").write_text(json.dumps(bs_, indent=0))
    ptr, pdv = SEP.make_pools(bs_, fam)
    print("fam", a.fam, fam, "utts train", len(tr), "val", len(va), "bg train", {k: len(v) for k, v in ptr.items()}, "dev", {k: len(v) for k, v in pdv.items()}, flush=True)
    dev = a.device
    dil = tuple(int(v) for v in a.dil.split(",")) if a.dil else SEP.DIL
    M = MaskNet(a.ch, -2.0, 1, dil).to(dev) if gate else MaskNet(a.ch, -3.0 if det else B0, NB, dil).to(dev)
    M.kind = a.target if det else "ratio"
    print("mask params", sum(p.numel() for p in M.parameters()), "受容野", 1 + 2 * sum(dil), "frames", "切り出し", a.crop, "frames", flush=True)
    steps = 200 if a.smoke else a.steps
    opt = torch.optim.AdamW(M.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    dl = torch.utils.data.DataLoader(SEP.Mix(tr, ptr, a.seed, steps * a.bs, fam=fam, parts=parts, crop=a.crop), batch_size=a.bs, num_workers=a.workers, worker_init_fn=lambda i: torch.set_num_threads(1),
                                     persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers else None, drop_last=True)
    vspecs = SEP.val_specs(va, pdv, 8 if a.smoke else 24, fam)
    vbatches = list(torch.utils.data.DataLoader(SEP.Mix(va, pdv, 999, len(vspecs), vspecs, fam, parts=parts), batch_size=6, num_workers=2))

    @torch.no_grad()
    def evaluate() -> dict:
        M.eval()
        conds = [s["cond"] for s in vspecs]
        rows: dict = {}
        k = 0
        for vb in vbatches:
            b = split_nf({kk: x.to(dev) for kk, x in vb.items()})
            z = M(b["x"])
            if gate:
                pg, lg = torch.sigmoid(z[:, 0]) >= 0.5, gate_label(b) > 0.5
                for i in range(b["x"].shape[0]):
                    c = conds[k]
                    k += 1
                    r = rows.setdefault(c, {"open": [], "label": [], "acc": [], "false_open": []})
                    r["open"].append(float(pg[i].float().mean()))
                    r["label"].append(float(lg[i].float().mean()))
                    r["acc"].append(float((pg[i] == lg[i]).float().mean()))
                    r["false_open"].append(float((pg[i] & ~lg[i]).float().sum() / max(1.0, float((~lg[i]).float().sum()))))
                continue
            m = torch.sigmoid(z).transpose(1, 2)
            ps, pn, px = power(b["s"]), power(b["n"]), power(b["x"])
            irm = ps / (ps + pn + 1e-10)
            w = (px + 1e-10) ** 0.3
            if det:
                pe = power(b["e"])
                lab = pe > ps + power(b["n"] - b["e"])
            for i in range(b["x"].shape[0]):
                c = conds[k]
                k += 1
                if det:
                    r = rows.setdefault(c, {"bce": [], "det_frac": [], "lab_frac": [], "ev_kept_db": [], "voice_lost": [], "fp_w": []})
                    d = m[i] >= 0.5
                    r["bce"].append(float((w[i] * F_.binary_cross_entropy_with_logits(z[i].transpose(0, 1), lab[i].float(), reduction="none")).sum() / w[i].sum()))
                    r["det_frac"].append(float(d.float().mean()))
                    r["lab_frac"].append(float(lab[i].float().mean()))
                    r["ev_kept_db"].append(float(10 * torch.log10(((~d) * pe[i]).sum() / (pe[i].sum() + 1e-10) + 1e-10)) if float(pe[i].sum()) > 0 else 0.0)
                    r["voice_lost"].append(float((d * ps[i]).sum() / (ps[i].sum() + 1e-10)))
                    r["fp_w"].append(float((d & ~lab[i]).float().mul(px[i]).sum() / (px[i].sum() + 1e-10)))
                    continue
                r = rows.setdefault(c, {"l1w": [], "lb": [], "mask_mean": [], "noise_kept": [], "voice_kept": []})
                r["l1w"].append(float((w[i] * (m[i] - irm[i]).abs()).sum() / w[i].sum()))
                r["lb"].append(float(band_err(m[i], irm[i], px[i]).mean()))
                r["mask_mean"].append(float(m[i].mean()))
                e = m[i]
                r["noise_kept"].append(float((e * pn[i]).sum() / (pn[i].sum() + 1e-10)))
                r["voice_kept"].append(float((e * ps[i]).sum() / (ps[i].sum() + 1e-10)))
        M.train()
        return {c: {kk: round(float(np.median(x)), 4) for kk, x in r.items()} for c, r in rows.items()}

    if a.eval_only:
        M.load_state_dict(torch.load(ROOT / a.eval_only, map_location="cpu", weights_only=False)["M"])
        print("val", a.eval_only, json.dumps(evaluate()), flush=True)
        return 0
    hist, t0, it = [], time.time(), iter(dl)
    M.train()
    for step in range(1, steps + 1):
        b = {k: x.to(dev, non_blocking=True) for k, x in next(it).items()}
        loss, _ = lossf(M(b["x"]), split_nf(b))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(M.parameters(), 1.0))
        opt.step()
        sched.step()
        if step % 100 == 0 or step == 1:
            print(f"step {step} loss {float(loss):.4f} gn {gn:.2f} {(time.time() - t0) / step:.3f}s/step", flush=True)
        if step % 2000 == 0 or step == steps:
            ev = evaluate()
            hist.append({"step": step, "val": ev})
            print("val", step, json.dumps(ev), flush=True)
            st = {"M": M.state_dict(), "cfg": {"ch": a.ch, "dil": list(dil), "crop": a.crop, "nfft": SEP.NFFT, "hop": H, "fam": a.fam, "b0": -2.0 if gate else -3.0 if det else B0, "band": a.band, "kind": "gate" if gate else "det" if det else "ratio", "target": a.target} | ({"mask": a.gate_mask, "dec": a.dec} if gate else {}), "step": step, "tag": a.tag, "prereg": a.prereg}
            torch.save(st, out / "last.pt.tmp")
            (out / "last.pt.tmp").replace(out / "last.pt")
            if step == 10000:
                torch.save(st, out / "step10k.pt")
            (out / "train_log.json").write_text(json.dumps(hist))
    print("done", steps, f"{time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
