"""D392: 中間の帯(1〜3 Hz)の包絡の動きを、内容で条件づけた分布(平均 μ と大きさ σ)から引く推論の手の探査(学習なし・生成器は候補 C のまま)。
D364(固定の統計の乱数 = 定常部でも揺れて悪化)・D366(平均の予測 = 不動)の間: σ が内容で変われば、動きは音の移り変わりに集まり定常部では静かになる見込み。
予測器 = 因果な文脈の符号器(CtxEnc 96 ch・入力 [事後・c0n・log f0・有声・表の包絡])→ [μ 8・log σ 8](中間の帯の主成分 8 = results/diag_ppgvoc24/mid_stats.npz)をガウスの NLL で学習(e2e/fem・保留を除く)。
推論: m = μ + γ σ ñ(ñ = 中間の帯の形の単位分散の乱数・因果な最小位相 FIR・共通の乱数)→ 基底で 24 次元 → 包絡の行に足す(遅い揺らぎの後)。γ = 0(平均だけ)/ 0.5 / 1・種 0〜3・SR / FF / FF2。
    uv run python d392_midprior.py --ckpt ../results/diag_ppgvoc29/product_005000_m0.pt --out ../results/conv_p0/d392
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P


def feats(c: np.ndarray, lg: np.ndarray, T: np.ndarray, dev: str) -> tuple:
    p = torch.from_numpy(lg.T.copy()).float().to(dev)[None].softmax(1)
    ct = torch.from_numpy(c.T.copy()).float().to(dev)[None]
    base = torch.einsum("dk,bkt->bdt", torch.from_numpy(T).float().to(dev), p) / 10
    x = torch.cat([p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], base], 1)
    r = ct[:, 1:25] - base
    mid = F.avg_pool1d(F.pad(r, (15, 14), mode="replicate"), 30, 1) - F.avg_pool1d(F.pad(r, (50, 49), mode="replicate"), 100, 1)
    return x, mid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", default=str(P.ROOT / "results/diag_ppgvoc24/mid_stats.npz"))
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--gammas", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(392); random.seed(392)
    O = Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    z = np.load(a.stats)
    Bm = torch.from_numpy(z["B"]).float().to(dev)
    h = P.minphase_fir(torch.from_numpy(z["amp"]).float()).double().numpy()
    hg = 1.0 / np.sqrt((h ** 2).sum(1))
    E2 = P.H / "e2e"
    held = P.e2e_held()
    hs = {x[0] for x in held}
    TD = P.ROOT / "data/rvoc_tables_real"
    tr = []
    for m in json.loads((E2 / "meta.json").read_text())["fem"]:
        f = E2 / "fem" / f"{m['key']}.cond"
        if m["spk"] in hs or not f.exists() or not (TD / f"{m['spk']}.npz").exists():
            continue
        c, lg = P.load_dump(E2 / "fem", m["key"])
        if len(c) >= 900:
            tr.append((c, lg, np.load(TD / f"{m['spk']}.npz")["T"]))
    enc = P.CtxEnc(96).to(dev)
    enc.out = torch.nn.Conv1d(96, 16, 1).to(dev)
    torch.nn.init.zeros_(enc.out.weight)
    torch.nn.init.zeros_(enc.out.bias)
    opt = torch.optim.AdamW(enc.parameters(), lr=3e-4)
    for step in range(1, a.steps + 1):
        xs, ys = [], []
        for _ in range(16):
            c, lg, T = random.choice(tr)
            o = random.randrange(0, len(c) - 800)
            x, mid = feats(c[o:o + 800], lg[o:o + 800], T, dev)
            xs.append(x)
            ys.append(torch.einsum("dk,bdt->bkt", Bm, mid))
        x, y = torch.cat(xs), torch.cat(ys)
        out = enc(x)
        mu, ls = out[:, :8], out[:, 8:].clamp(-6, 3)
        nll = (0.5 * ((y - mu) / ls.exp()) ** 2 + ls)[..., 255:].mean()
        opt.zero_grad(set_to_none=True)
        nll.backward()
        opt.step()
        if step % 1000 == 0:
            print("step", step, "nll", round(float(nll), 4), flush=True)
    enc.eval()
    sr, _ = P.eval_items(held, "table", dev)
    with torch.no_grad():
        r2s, cors = [], []
        for nm, c, lg, spk in sr:
            x, mid = feats(c, lg, spk[0], dev)
            y = torch.einsum("dk,bdt->bkt", Bm, mid)
            out = enc(x)
            mu, sg = out[:, :8], out[:, 8:].clamp(-6, 3).exp()
            r2s.append(float(1 - ((y - mu) ** 2).sum() / (y ** 2).sum()))
            cors.append(float(np.corrcoef(sg.flatten().cpu().numpy(), (y - mu).abs().flatten().cpu().numpy())[0, 1]))
    print("held SR: R² of μ %.3f · corr(σ, |m − μ|) %.3f" % (np.mean(r2s), np.mean(cors)), flush=True)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    P.slow_from_ckpt(st)
    g = P.init_gen(dev, "table")
    g.load_state_dict(st["ema"])
    g.eval()
    its = {"sr": sr, "devff": P.ff_items(["devff"]), "devff2": P.ff_items(["devff2"])}
    groups: dict = {k: [] for k in its}
    with torch.no_grad():
        for gm in a.gammas:
            for sd in (a.seeds if gm > 0 else a.seeds[:1]):
                P.SLOW["seed"] = sd
                tag = f"G{gm}_r{sd}"
                for s_, lst in its.items():
                    od = O / s_ / tag
                    od.mkdir(parents=True, exist_ok=True)
                    for nm, c, lg, spk in lst:
                        x, _ = feats(c, lg, spk[0], dev)
                        out = enc(x)
                        mu, sg = out[:, :8], out[:, 8:].clamp(-6, 3).exp()
                        Tn = mu.shape[-1]
                        L = h.shape[1]
                        e = P.splitmix_normals(sd + Tn + 4242, (Tn + L - 1) * 8).reshape(Tn + L - 1, 8).T
                        nz = np.stack([np.convolve(e[j], h[j], "valid") * hg[j] for j in range(8)])
                        m = mu + gm * sg * torch.from_numpy(nz).float().to(dev)[None]
                        cond, exc, s2 = P.cond_exc(g, c, lg, spk, dev)
                        cond = torch.cat([cond[:, :1], cond[:, 1:25] + torch.einsum("dk,bkt->bdt", Bm, m), cond[:, 25:]], 1)
                        y = g(cond, exc, s2)[0].cpu().numpy()[N.DELAY:]
                        sf.write(od / f"{nm}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
                    groups[s_].append(f"{tag}={od}/*.wav")
                print("rendered", tag, flush=True)
    rows = {}
    for s_ in its:
        r = subprocess.run([sys.executable, str(Path(__file__).parent / "eval_v1perp.py"), "--groups", *groups[s_], "--out", str(O / f"{s_}_v1perp.json")], capture_output=True, text=True)
        if r.returncode:
            print(r.stderr[-1500:], flush=True)
        j = json.loads((O / f"{s_}_v1perp.json").read_text())
        rows[s_] = {k: v["v1perp"]["logit"] for k, v in j.items()}
    (O / "summary.json").write_text(json.dumps(rows, indent=1))
    for gm in a.gammas:
        mm = [np.mean([v for k, v in rows[s_].items() if k.startswith(f"G{gm}_")]) for s_ in its]
        print(f"γ {gm}:", " ".join(f"{s_} {x:.3f}" for s_, x in zip(its, mm)), f"| 3-set {np.mean(mm):.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
