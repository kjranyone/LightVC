"""D558(推論だけ・男声 → 女声のこもり): 内容 → 波形の生成器の音量の条件 c0n を、操作者ごとの登録の定数で女声の水準へずらす(D185 の --c0-shift の内容 → 波形の版)。
ずらし = 4.40(実女声の入力の発声中の c0n の平均・D557)− 操作者の登録の音声(書き出しの前置き 12 s = 操作者の参照の末尾)の発声中(c0n > 0.5)の平均。無音(c0n ≤ 0.05)には足さない。
TTS 男声の c0n は女声より約 1.0 低く(D557)、生成器は低い音量 = 柔らかく暗い声と学んでいる = こもりの候補。P43OP(+ 任意で V70 の書き出し)で描画し d109 で測る。
    uv run python d558_c0shift_ppg.py --ckpt ../results/diag_ppgvoc43/product_002500_m0.pt --sets dev2 dev3 devreal --name P43S --out ../results/conv_p0/d558
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
import train_ppgvoc as P

PRE = 12.0
FEM = 4.40


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--cfg", default="", help="書き出しの設定(V70 など・空 = 製品の既定)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tlevel", default="", help="行き先を目標本人の参照の水準に(d560_target_level.py の json・D560)。空 = 実女声の平均 4.40")
    ap.add_argument("--tscale", type=float, default=1.0, help="行き先 = 4.40 + tscale·(目標の水準 − 4.40)(--tlevel のとき)")
    ap.add_argument("--opgain", type=float, default=1.5, help="操作者の遅い揺らぎの利得(製品 1.5)")
    ap.add_argument("--rndslow", action="store_true", help="操作者の遅い揺らぎを使わず乱数の遅い揺らぎ(σ 1.0・因果・腕 43 以前の製品の型)")
    ap.add_argument("--cvnoise", type=float, default=0.0, help="推論で C2 の特徴に足す正規雑音の σ(腕 52 の学習と同じ 0.5 なら分布の内・D563)")
    ap.add_argument("--f0k", type=float, default=1.0, help="診断(D563・話し方を変える = 製品不可): 有声の log f0 の動きを前置きの有声の中央値のまわりで κ 倍")
    ap.add_argument("--c0k", type=float, default=1.0, help="診断(D563): ずらした後の c0n の動きを 4.40 のまわりで κ 倍(発声中)")
    a = ap.parse_args()
    tl = json.loads(Path(a.tlevel).read_text()) if a.tlevel else {}
    dev = "cuda"
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if st.get("cvin"):
        P.cvin_setup(st["cvin"] if Path(st["cvin"]).is_absolute() else str(Path(__file__).parent / st["cvin"]), dev)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch") or 192))
    g = P.init_gen(dev, "table")
    g.load_state_dict(st["ema"])
    g.eval()
    st["slow_sigma"], st["slow_causal"], st["opslow_tau"], st["opslow_gain"] = 1.0, True, (None if a.rndslow else 4.0), a.opgain
    P.slow_from_ckpt(st)
    n0 = int(PRE * N.SR)
    info = {}
    for s in a.sets:
        S = P.H / s
        ed = P.H / "e2e" / (f"eval_{s}_pre" + (f"_{a.cfg}" if a.cfg else ""))
        od = Path(a.out) / a.name / s
        od.mkdir(parents=True, exist_ok=True)
        sh = []
        for r in json.loads((S / "devset.json").read_text())["pairs"]:
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            if not (ed / f"{k}.cond").exists():
                continue
            c, lg = P.load_dump(ed, k)
            pre = c[:2400, 0]
            mu = float(pre[pre > 0.5].mean()) if (pre > 0.5).sum() > 50 else FEM
            tv = tl[s]["level"].get(str(r["ti"])) if tl else None
            d = (FEM + a.tscale * (tv - FEM) if tv is not None else FEM) - mu
            sh.append(d)
            c = c.copy()
            c[:, 0] = np.where(c[:, 0] > 0.05, c[:, 0] + d, c[:, 0])
            if a.c0k != 1.0:
                c[:, 0] = np.where(c[:, 0] > 0.05, FEM + a.c0k * (c[:, 0] - FEM), c[:, 0])
            if a.f0k != 1.0:
                v = c[:, 26] > 0.5
                m = float(np.median(c[:2400][v[:2400], 25])) if v[:2400].sum() > 50 else float(np.median(c[v, 25]))
                c[:, 25] = np.where(v, m + a.f0k * (c[:, 25] - m), c[:, 25])
            cvf = None
            if P.CVIN["on"]:
                import train_c1 as TC
                from d6_unit_cov import load48
                x_ = load48(S / "d467" / f"{k}.wav").astype(np.float32)
                cvf = P.cvin_frames(torch.from_numpy(TC.prime(x_))[None].to(dev), len(lg))
                if a.cvnoise > 0:
                    cvf = cvf + a.cvnoise * torch.randn(cvf.shape, generator=torch.Generator().manual_seed(r["pi"] * 1000 + r["ti"])).to(dev)
            y = P.render(g, c, lg, P.load_enroll(S / "d85_work" / f"en_{r['ti']:03d}"), dev, T_op=P.op_tab(S, k), cvf=cvf)[n0:n0 + 8 * N.SR]
            sf.write(od / f"{k}_{a.name}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
        info[s] = {"n": len(sh), "shift_median": round(float(np.median(sh)), 3), "shift_p10_p90": [round(float(np.percentile(sh, 10)), 3), round(float(np.percentile(sh, 90)), 3)]}
        print(s, info[s], flush=True)
    (Path(a.out) / f"{a.name}_info.json").write_text(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
