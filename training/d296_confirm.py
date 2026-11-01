"""D296: 確認の段(current/ml_method.md §3 段 2 の 5): 見当づけ(dev)で基準に迫った腕を dev2・dev3(各 60 組)で鳴らし、同じ実行で製品・kNN-VC と v1⊥ を比べる。
Rust の書き出し = d274_evaldump.py --set devN・目標の表 = <devN>/d85_work/en_TTT。
    uv run python d296_confirm.py --ckpts A=<last_ema.pt> B=<last_ema.pt> --sets dev2 dev3 --out ../results/conv_p0/d296
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--sets", nargs="+", default=["dev2", "dev3"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--f0ms", action="store_true", help="log f0 を平均と幅で目標へ写す: lf' = μ_t + (lf − μ_o)·σ_t / σ_o(μ・σ = 操作者 / 目標の登録の参照の有声の log f0・登録の定数・D321)")
    ap.add_argument("--sets_dev", action="store_true")
    ap.add_argument("--c0shift", type=float, default=0.0, help="c0n += α·(目標の参照の有声の c0n の中央値 − 操作者の参照の有声の c0n の中央値)(登録の定数・D401: c0n は入力の高域の水準を運ぶ)")
    ap.add_argument("--vgate", type=float, default=0.0, help="内容から有声を補正: 女声の単位ごとの有声の割合(lightvc_scratch/e2e/unit_voiced_rate_female.npy・学習の女声の固定の表)が閾より低い単位のフレームを無声に(D323)")
    ap.add_argument("--pbias", type=float, default=0.0, help="事後の話者ごとの偏りの補正の強さ(logit += a·(log 平均事後_目標の参照 − log 平均事後_操作者の参照)・登録の定数・D316)")
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    for spec in a.ckpts:
        name, path = spec.split("=", 1)
        st = torch.load(path, map_location="cpu", weights_only=False)
        P.PPG = st.get("ppg", "soft")
        P.CTX["on"] = bool(st.get("ctx", False))
        P.CTX["ch"] = int(st.get("ctx_ch", 192))
        P.CTX["adapt"] = bool(st.get("adapt", False))
        P.slow_from_ckpt(st)
        g = P.init_gen(dev, st.get("spk", "table"))
        g.load_state_dict(st["ema"])
        g.eval()
        for s in a.sets:
            ed = P.H / "e2e" / f"eval_{s}"
            od = O / s / name
            od.mkdir(parents=True, exist_ok=True)
            n = 0
            for f in sorted(ed.glob("p00*.cond")):
                ti = int(f.stem.split("_t")[1][:3])
                c, lg = P.load_dump(ed, f.stem)
                if a.f0ms:
                    pi = int(f.stem[1:5])

                    def ms(fr: Path) -> tuple:
                        cc = np.fromfile(fr, "<f4").reshape(-1, 31)
                        v_ = cc[:, 26] > 0.5
                        return float(cc[v_, 25].mean()), float(cc[v_, 25].std())
                    mt, st_ = ms(ed / f"ref_{ti:03d}.cond")
                    mo, so = ms(P.H / "e2e" / f"opref_{s}" / f"op_{pi:04d}.cond")
                    c = c.copy()
                    vo = c[:, 26] > 0.5
                    emu = lambda d_: float(json.loads((d_ / "enroll.json").read_text())["mu"])
                    lf_o = c[vo, 25] - (emu(P.H / s / "d85_work" / f"en_{ti:03d}") - emu(P.H / s / "d85_work" / f"op_{pi:04d}"))
                    c[vo, 25] = mt + (lf_o - mo) * (st_ / max(so, 1e-3))
                if a.c0shift != 0:
                    pi = int(f.stem[1:5])

                    def c0med(fr: Path) -> float:
                        cc = np.fromfile(fr, "<f4").reshape(-1, 31)
                        return float(np.median(cc[cc[:, 26] > 0.5, 0]))
                    c = c.copy()
                    c[:, 0] = c[:, 0] + a.c0shift * (c0med(ed / f"ref_{ti:03d}.cond") - c0med(P.H / "e2e" / f"opref_{s}" / f"op_{pi:04d}.cond"))
                if a.vgate > 0:
                    rf = np.load(P.H / "e2e" / "unit_voiced_rate_female.npy")
                    u = lg.argmax(1)
                    gate = np.nan_to_num(rf[u], nan=1.0) < a.vgate
                    c = c.copy()
                    c[gate, 26] = 0.0
                    c[gate, 25] = 0.0
                if a.pbias > 0:
                    pi = int(f.stem[1:5])
                    pb = lambda fr: np.log(torch.from_numpy(np.fromfile(fr, "<f4").reshape(-1, 224)[:, :200]).softmax(1).numpy().mean(0) + 1e-6)
                    lg = lg + a.pbias * (pb(ed / f"ref_{ti:03d}.front") - pb(P.H / "e2e" / f"opref_{s}" / f"op_{pi:04d}.front"))[None, :].astype(np.float32)
                sf.write(od / f"{f.stem}_{name}.wav", P.render(g, c, lg, P.load_enroll(P.H / s / "d85_work" / f"en_{ti:03d}"), dev, T_op=P.op_tab(P.H / s, f.stem) if P.OPSLOW["on"] else None), N.SR)
                n += 1
            print("rendered", name, s, n, flush=True)
    for s in a.sets:
        groups = [f"{spec.split('=')[0]}={O / s / spec.split('=')[0]}/p00*.wav" for spec in a.ckpts]
        import glob as _g
        for nm_, pat_ in (("PROD", f"{P.H / s / 'd114'}/p00*_G03S10ZH.wav"), ("KNN", f"{P.H / s / 'd135_real'}/p00*_KNN.wav"), ("SRC", f"{P.H / s / 'd56'}/p00*_SRC.wav")):
            if _g.glob(pat_):
                groups.append(f"{nm_}={pat_}")
        r = subprocess.run([sys.executable, str(Path(__file__).parent / "eval_v1perp.py"), "--groups", *groups, "--out", str(O / f"{s}_v1perp.json")], capture_output=True, text=True)
        print(s, r.stdout[-3000:], r.stderr[-1500:] if r.returncode else "", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
