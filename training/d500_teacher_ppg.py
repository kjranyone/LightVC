"""D500(診断・内容の情報の律速の切り分け): 候補(P43OP)の C1 の事後を、C1 の教師(ContentVec → 200 単位のコードブック・cos/τ の softmax・非因果・15 ms の遅れの規約は同じ)の事後に差し替えて描画する。
C1 の日本語女声での教師との一致は 49%(c1_1)= 生徒の忠実度が明瞭さの律速なら教師の事後で CER が写し合成へ近づく。表(登録)は C1 の硬い単位のまま。非因果 = 診断のみ。
版 = FF_TEA(目標の表)・SELF_TEA(操作者の表)。CER は d23_cer で別に測る。
    uv run python d500_teacher_ppg.py --ckpt ../results/diag_ppgvoc43/product_002500_m0.pt --out ../results/conv_p0/d500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_c1 as TC
import train_ppgvoc as P
from d6_unit_cov import load48

PRE = 12.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="devff23x")
    ap.add_argument("--c1", default="../results/c1_1")
    a = ap.parse_args()
    dev = "cuda"
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    st["slow_sigma"], st["slow_causal"], st["opslow_tau"], st["opslow_gain"] = 1.0, True, 4.0, 1.5
    P.slow_from_ckpt(st)
    c1 = torch.load(Path(a.c1) / "last.pt", map_location="cpu", weights_only=False)
    teacher = TC.Teacher(dev, torch.load(Path(a.c1) / "codebook.pt", map_location="cpu"), float(c1.get("tau", 0.05)))
    S = P.H / a.set
    ed = P.H / "e2e" / f"eval_{a.set}_pre"
    n0 = int(PRE * N.SR)
    O = Path(a.out)
    n, agree = 0, []
    for r in json.loads((S / "devset.json").read_text())["pairs"]:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        if not (ed / f"{k}.cond").exists():
            continue
        c, lg = P.load_dump(ed, k)
        x = load48(S / "d467" / f"{k}.wav").astype(np.float32)
        tp = TC.teacher_post(teacher, x, dev)[0].T.cpu().numpy()
        T = min(len(tp), len(lg))
        c, lg, tp = c[:T], lg[:T], tp[:T]
        agree.append(float((tp[n0 // N.HOP:].argmax(1) == lg[n0 // N.HOP:].argmax(1)).mean()))
        ltp = np.log(np.clip(tp, 1e-8, None)).astype(np.float32)
        for mode in ("FF", "SELF"):
            spk = P.load_enroll(S / "d85_work" / (f"en_{r['ti']:03d}" if mode == "FF" else f"op_{r['pi']:04d}"))
            y = P.render(g, c, ltp, spk, dev, T_op=P.op_tab(S, k))[n0:n0 + 8 * N.SR]
            od = O / f"{mode}_TEA"
            od.mkdir(parents=True, exist_ok=True)
            sf.write(od / f"{k}_{mode}TEA.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
        n += 1
    print("rendered", n, "C1-teacher argmax agreement (eval 8 s)", round(float(np.mean(agree)), 3), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
