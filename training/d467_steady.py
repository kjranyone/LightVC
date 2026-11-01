"""D467(10-21 査読 R2 への対応): 遅い揺らぎの版を製品の定常状態で比べる。操作者の参照の末尾 12 s を入力の前に置いて Rust の条件を書き出し(移動平均の初期値の重み 約 5%)、
候補(既定 D3 = diag_ppgvoc43)で描画して元の 8 s だけを評価に回す。版 = OFF(揺らぎなし)・RND10C / RND15C(因果な乱数 σ 1.0 / 1.5)・OP10 / OP15(操作者の揺らぎ 利得 1.0 / 1.5)・CP(写し合成・女声 → 女声の組だけ)。
組 = 利得の選択に使っていない dev2・dev3(男声 → 女声)・devff2(女声 → 女声・学習の外の操作者)。評価は d467_paired_v1.py(組ごとに判別器 1 つ・同じ発話の対の差)。
    uv run python d467_steady.py --ckpt ../results/diag_ppgvoc43/product_002500_m0.pt --out ../results/conv_p0/d467
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
from d6_unit_cov import load48

PRE = 12.0
VARS = ("OFF", "RND10C", "RND15C", "OP10", "OP15")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="+", default=["dev2", "dev3", "devff2"])
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if st.get("cvin"):
        P.cvin_setup(st["cvin"] if Path(st["cvin"]).is_absolute() else str(Path(__file__).parent / st["cvin"]), dev)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    n0 = int(PRE * N.SR)
    for s in a.sets:
        S = P.H / s
        W = P.H / s / "d467"
        W.mkdir(parents=True, exist_ok=True)
        ed = P.H / "e2e" / f"eval_{s}_pre"
        ed.mkdir(parents=True, exist_ok=True)
        pairs = json.loads((S / "devset.json").read_text())["pairs"]
        pairs = [r for r in pairs if r.get("kind", "FF") != "AA" and (S / "d56" / f"p{r['pi']:04d}_t{r['ti']:03d}_SRC.wav").exists()]
        if s == "devvctkm":
            pairs = [r for r in pairs if r["kind"].startswith("M")]
        jobs = []
        for r in pairs:
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            w = W / f"{k}.wav"
            if not w.exists():
                opw = S / "d56" / "_work" / f"op_{r['pi']:04d}.wav"
                if not opw.exists() and (S / "d56" / "_work" / f"ref_{r.get('spk', '')}.wav").exists():
                    opw = S / "d56" / "_work" / f"ref_{r['spk']}.wav"
                pre = load48(opw if opw.exists() else sorted((S / "d56" / "_work").glob("ref_*.wav"))[0])[-n0:]
                pre = np.pad(pre, (n0 - len(pre), 0))
                sf.write(w, np.concatenate([pre, load48(S / "d56" / f"{k}_SRC.wav")[: 8 * N.SR]]).astype(np.float32), N.SR)
            jobs.append(([BIN, "artic", "--enroll", S / "d85_work" / f"en_{r['ti']:03d}", "--operator-enroll", S / "d85_work" / f"op_{r['pi']:04d}", "--med-from-operator"] + COM + ["-i", w], ed / f"{k}.cond", ed / f"{k}.front"))
        with Pool(3) as pl:
            list(pl.imap_unordered(job, jobs))
        print("dumped", s, len(jobs), flush=True)
        if s.startswith("devff"):
            import rvoc as R
            import train_rvoc as TR
            s_ = torch.load(P.INIT / "snap/ema_330k.pt", map_location="cpu", weights_only=False)
            gcp = R.RVoc(ch=s_["cfg"]["ch"], kernels=tuple(s_["cfg"]["kernels"]), dils=tuple(s_["cfg"]["dils"]), d_cond=s_["cfg"]["d_cond"]).to(dev)
            gcp.load_state_dict(s_["ema"])
            gcp.eval()
        for v in VARS + (("CP",) if s.startswith("devff") else ()):
            od = O / s / v
            od.mkdir(parents=True, exist_ok=True)
            if v == "CP":
                pass
            elif v == "OFF":
                st["slow_sigma"], st["slow_causal"], st["opslow_tau"] = 1.0, True, 0.0
                P.slow_from_ckpt(st)
                P.SLOW["on"] = False
            elif v.startswith("RND"):
                st["slow_sigma"], st["slow_causal"], st["opslow_tau"] = (1.0 if v == "RND10C" else 1.5), True, 0.0
                P.slow_from_ckpt(st)
            else:
                st["slow_sigma"], st["slow_causal"], st["opslow_tau"], st["opslow_gain"] = 1.0, True, 4.0, (1.0 if v == "OP10" else 1.5)
                P.slow_from_ckpt(st)
            for r in pairs:
                k = f"p{r['pi']:04d}_t{r['ti']:03d}"
                c, lg = P.load_dump(ed, k)
                if v == "CP":
                    with torch.no_grad():
                        cond = torch.from_numpy(c.T.astype(np.float32))[None].to(dev)
                        f0 = torch.where(cond[:, 26] > 0.5, 200.0 * torch.exp(cond[:, 25]), torch.zeros_like(cond[:, 25]))
                        y = gcp(cond, TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
                else:
                    cvf = None
                    if P.CVIN["on"]:
                        import train_c1 as TC
                        x_ = load48(W / f"{k}.wav").astype(np.float32)
                        cvf = P.cvin_frames(torch.from_numpy(TC.prime(x_))[None].to(dev), len(lg))
                    y = P.render(g, c, lg, P.load_enroll(S / "d85_work" / f"en_{r['ti']:03d}"), dev, T_op=P.op_tab(S, k) if P.OPSLOW["on"] else None, cvf=cvf)
                y = y[n0:n0 + 8 * N.SR]
                sf.write(od / f"{k}_{v}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
            print("rendered", s, v, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
