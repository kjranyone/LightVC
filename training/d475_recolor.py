"""D475(results/conv_p0/d475/prereg.yaml): 操作者の遅い揺らぎ(OPSLOW)を、成分ごとに 目標の参照の標準偏差 / 操作者の参照の標準偏差 倍(0.5〜2)して目標の揺らぎの大きさへ色づけし直す。
定常状態(D467 の前置きの書き出し eval_<組>_pre)の dev2・dev3・devff2 で OP15(色づけなし)と OP15RC を描画し、元の 8 s を <out>/<組>/<版>/ に。
    uv run python d475_recolor.py --ckpt ../results/diag_ppgvoc43/product_002500_op15.pt --out ../results/conv_p0/d475
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P
from d273_data import BIN, job
from d274_evaldump import COM

PRE = 12.0


def comp_std(cond: Path, T: np.ndarray) -> np.ndarray:
    c = np.fromfile(cond, "<f4").reshape(-1, 31)
    lg = np.fromfile(str(cond).replace(".cond", ".front"), "<f4").reshape(-1, 224)[:, :200]
    n = min(len(c), len(lg))
    c, lg = c[:n], lg[:n]
    p = torch.from_numpy(lg).float().softmax(-1).T[None]
    r = torch.from_numpy(c[:, 1:25].T.copy()).float()[None] - torch.einsum("dk,bkt->bdt", torch.from_numpy(T).float(), p) / 10
    w = (P.SLOW["B"].T @ P.opslow_causal(r, P.OPSLOW["tau"])[0]).numpy()
    v = c[:, 0] > 1.0
    v[:200] = False
    return w[:, v].std(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="+", default=["dev2", "dev3", "devff2"])
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    P.slow_from_ckpt(st)
    assert P.OPSLOW["on"]
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    n0 = int(PRE * N.SR)
    rep = {}
    for s in a.sets:
        S = P.H / s
        orf = P.H / "e2e" / f"opref_{s}"
        orf.mkdir(parents=True, exist_ok=True)
        pairs = [r for r in json.loads((S / "devset.json").read_text())["pairs"] if r.get("kind", "FF") != "AA"]
        jobs = [([BIN, "artic", "--enroll", S / "d85_work" / f"en_{r['ti']:03d}"] + COM + ["-i", S / "d56" / "_work" / f"op_{r['pi']:04d}.wav"], orf / f"op_{r['pi']:04d}.cond", orf / f"op_{r['pi']:04d}.front") for r in pairs]
        with Pool(3) as pl:
            list(pl.imap_unordered(job, jobs))
        ed = P.H / "e2e" / f"eval_{s}_pre"
        ratios = []
        for r in pairs:
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            if not (ed / f"{k}.cond").exists():
                continue
            T_op = P.op_tab(S, k)
            st_ = comp_std(P.H / "e2e" / f"eval_{s}" / f"ref_{r['ti']:03d}.cond", P.load_enroll(S / "d85_work" / f"en_{r['ti']:03d}")[0])
            so_ = comp_std(orf / f"op_{r['pi']:04d}.cond", T_op)
            ratio = np.clip(st_ / np.maximum(so_, 1e-6), 0.5, 2.0)
            ratios.append(ratio)
            c, lg = P.load_dump(ed, k)
            p = P.ppg_form(torch.from_numpy(lg.T.copy()).float().to(dev)[None], P.PPG)
            ct = torch.from_numpy(c.T.copy()).float().to(dev)[None]
            stt = P.spk_tensor(P.load_enroll(S / "d85_work" / f"en_{r['ti']:03d}"), dev)
            z = P.OPSLOW["gain"] * P.opslow_causal(ct[:, 1:25] - torch.einsum("dk,bkt->bdt", torch.from_numpy(T_op).float().to(dev), p) / 10, P.OPSLOW["tau"])
            B = P.SLOW["B"].to(dev)
            zrc = torch.einsum("dk,nkt->ndt", B, torch.from_numpy(ratio).float().to(dev)[None, :, None] * torch.einsum("dk,ndt->nkt", B, z))
            f0 = torch.where(ct[:, 26] > 0.5, 200.0 * torch.exp(ct[:, 25]), torch.zeros_like(ct[:, 25]))
            exc = P.TR.excitation(f0, ct.shape[-1] * N.HOP, torch.Generator().manual_seed(0))
            with torch.no_grad():
                for v, zz in (("OP15", z), ("OP15RC", zrc)):
                    y = g(g.make(p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], stt, zz), exc, stt)[0].cpu().numpy()[N.DELAY:]
                    y = y[n0:n0 + 8 * N.SR]
                    (O / s / v).mkdir(parents=True, exist_ok=True)
                    sf.write(O / s / v / f"{k}_{v}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
        R = np.array(ratios)
        rep[s] = {"n": len(R), "ratio_median_by_comp": np.round(np.median(R, 0), 3).tolist(), "frac_clipped": round(float(((R <= 0.5) | (R >= 2.0)).mean()), 3)}
        print(s, rep[s], flush=True)
    (O / "ratios.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
