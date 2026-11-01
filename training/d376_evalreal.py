"""D376: 実男声(外郎売・devreal)→ 女声(dev の目標 30 人 × 2 = 60 組)で、ckpt ごとに描画し v1⊥(入力そのもの SRC と並べる)と CER を測る。
揺らぎの設定は ckpt のまま(product_* = 因果な共通の乱数 σ 1.5)。
    uv run python d376_evalreal.py --ckpts PPG18=../results/diag_ppgvoc18/last_ema.pt A=... B=... --out ../results/conv_p0/d376
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--set", default="devreal")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pbias", type=float, default=0.0, help="事後の話者ごとの偏りの補正(D316 と同じ・登録の定数): logit += a·(log 平均事後_目標の参照 − log 平均事後_操作者の参照)・操作者の参照 = e2e/opref_<set>/op_uiro.front(D391)")
    ap.add_argument("--vgate", type=float, default=0.0, help="内容から有声を補正(d296_confirm.py と同じ・D323 / D382): 女声の単位ごとの有声の割合が閾より低い単位のフレームを無声に")
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    S = P.H / a.set
    ed = P.H / "e2e" / f"eval_{a.set}"
    rows = json.loads((S / "devset.json").read_text())["pairs"]
    groups = [f"SRC={S / 'd56'}/p00*_SRC.wav"]
    cer = O / "cer_in"
    cer.mkdir(parents=True, exist_ok=True)
    for f in sorted((S / "d56").glob("p00*_SRC.wav")):
        (cer / f.name).unlink(missing_ok=True)
        (cer / f.name).symlink_to(f)
    for spec in a.ckpts:
        name, ck = spec.split("=", 1)
        st = torch.load(ck, map_location="cpu", weights_only=False)
        P.PPG = st.get("ppg", "soft")
        P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
        P.slow_from_ckpt(st)
        g = P.init_gen(dev, st.get("spk", "table"))
        g.load_state_dict(st["ema"])
        g.eval()
        od = O / name
        od.mkdir(parents=True, exist_ok=True)
        for r in rows:
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            c, lg = P.load_dump(ed, k)
            if a.vgate > 0:
                import numpy as np
                rf = np.load(P.H / "e2e" / "unit_voiced_rate_female.npy")
                gate = np.nan_to_num(rf[lg.argmax(1)], nan=1.0) < a.vgate
                c = c.copy()
                c[gate, 26] = 0.0
                c[gate, 25] = 0.0
            if a.pbias > 0:
                import numpy as np
                pb = lambda fr: np.log(torch.from_numpy(np.fromfile(fr, "<f4").reshape(-1, 224)[:, :200]).softmax(1).numpy().mean(0) + 1e-6)
                lg = lg + a.pbias * (pb(ed / f"ref_{r['ti']:03d}.front") - pb(P.H / "e2e" / f"opref_{a.set}" / "op_uiro.front"))[None, :].astype(np.float32)
            y = P.render(g, c, lg, P.load_enroll(S / "d85_work" / f"en_{r['ti']:03d}"), dev, T_op=P.op_tab(S, k) if P.OPSLOW["on"] else None)
            sf.write(od / f"{k}_CV.wav", y, N.SR)
            (cer / f"{k}_{name}.wav").unlink(missing_ok=True)
            (cer / f"{k}_{name}.wav").symlink_to((od / f"{k}_CV.wav").resolve())
        groups.append(f"{name}={od}/*_CV.wav")
        print("rendered", name, len(rows), "slow", P.SLOW["on"], P.SLOW.get("sigma"), P.SLOW.get("causal"), flush=True)
    T = Path(__file__).parent
    r = subprocess.run([sys.executable, str(T / "eval_v1perp.py"), "--groups", *groups, "--out", str(O / "v1perp.json")], capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-1500:] if r.returncode else "", flush=True)
    r = subprocess.run(["uv", "run", "python", str(T / "d23_cer.py"), "--dirs", str(cer), "--names", a.set, "--out", str(O / "cer.json"), "--device", "cuda"], capture_output=True, text=True, cwd=T)
    print([l for l in r.stdout.splitlines() if a.set in l][-1:] if r.returncode == 0 else r.stderr[-1500:], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
