import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import snv_p0_front as P
import snv_p1_eval as EV

ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> None:
    out = ROOT / sys.argv[1]
    src = ROOT / "data/snv_p_feat"
    (out / "a0").mkdir(parents=True, exist_ok=True)
    for q in src.iterdir():
        if q.name != "a0" and not (out / q.name).exists():
            os.symlink(q.resolve(), out / q.name)
    st = Path(str(out).replace("data/snv_p_feat", "data/snv_p_feat_st", 1))
    if not st.exists():
        os.symlink((ROOT / "data/snv_p_feat_st").resolve(), st)
    tg = json.loads((EV.A1 / "convert_meta.json").read_text())["targets"][:4]
    redo = {q.stem for q in (src / "a0").glob("*.npz")} if len(sys.argv) > 2 and sys.argv[2] == "all" else {f"{v}_{j}" for v in tg for j in (0, 1, 2)}
    dmax = 0.0
    for q in sorted((src / "a0").iterdir()):
        dst = out / "a0" / q.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        if q.suffix == ".npz" and q.stem in redo:
            d = dict(np.load(q))
            x = librosa.load(EV.wav_path(f"a0/{q.stem}"), sr=P.SR, mono=True)[0].astype(np.float32)
            f = P.analyze(x[: P.n_frames(len(x)) * P.H])
            n = d["env"].shape[1]
            assert f["env"].shape[1] == n, (q.name, f["env"].shape, n)
            dmax = max(dmax, float(np.abs(f["env"] - d["env"]).mean()))
            np.savez(dst, **(d | {"f0": f["f0"].astype(d["f0"].dtype), "env": f["env"].astype(d["env"].dtype), "per": f["per"].astype(d["per"].dtype)}))
            print(q.stem, "f0 の中央値", round(float(np.median(f["f0"][f["f0"] > 0])), 1), "包絡の差の平均", round(float(np.abs(f["env"] - d["env"]).mean()), 4), "有声の一致", round(float(((f["f0"] > 0) == (d["f0"] > 0)).mean()), 4), flush=True)
        else:
            os.symlink(q.resolve(), dst)
    print("done", out, "redo", len(redo), "包絡の差の平均の最大", round(dmax, 4))


if __name__ == "__main__":
    main()
