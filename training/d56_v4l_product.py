"""D56: 新しい男声(Irodori-TTS-v4-Large・data/male_tts_v4l)を元の声にして、製品の条件(評価プールの女声 --n_tg 人 × 2 文・Rust の製品経路)を作り直す。
元の文 = 同じ話者の 2〜3 文をつないだ 8s(錨 t10 を除く)。操作者の登録 = その話者の元の文に使っていない文 22s 以上(Rust artic-enroll・周期性 75 パーセンタイルつき)。
目標の登録 = D53 の Rust 登録(<s>/d53_en/en_TTT・周期性 75 パーセンタイルつき)。全条件に周期性 75 パーセンタイルと高さの基準 = 操作者の中央値を入れる。
  L1COPY = 男声のまま製品経路(写し合成)・L2PITCH = 高さだけ目標へ・L3VTLN = 声道長の伸縮 α 1.3(表なし)・L4FULL = 伸縮つき話者差 γ 0.7・TAB = 柔らかい引き(γ 0)。
出力 = d4b の形式(cen・pNNNN_tNNN_<cond>・SRC)。
    uv run python d56_v4l_product.py --s <scratchpad> --out_dir <scratchpad>/d56 --n_tg 30
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target/release/lightvc-app"
CORPUS = ROOT / "data/male_tts_v4l"


def run(cmd: list) -> None:
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"FAIL {cmd} {r.stderr[-500:]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_tg", type=int, default=30)
    ap.add_argument("--per", type=int, default=2)
    a = ap.parse_args()
    import nvoc as N
    import zsvc as Z
    S = Path(a.s)
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    work = od / "_work"; work.mkdir(exist_ok=True)
    man = json.loads((CORPUS / "manifest.json").read_text())
    spks = sorted(s for s, v in man["speakers"].items() if len(v["files"]) >= 30)
    assert spks, "コーパスが未完成"
    Dm = Z.dct_mat(N.N_MEL).numpy().astype(np.float64)
    fb = N.mel_fb().numpy()
    fc = fb.argmax(1) * (N.SR / 2) / (fb.shape[1] - 1)
    B = Dm[1:25]
    I = np.zeros((N.N_MEL, N.N_MEL))
    src_ = np.clip(fc / 1.3, fc[0], fc[-1])
    jj = np.clip(np.searchsorted(fc, src_) - 1, 0, N.N_MEL - 2)
    ww = np.clip((src_ - fc[jj]) / np.maximum(fc[jj + 1] - fc[jj], 1e-9), 0, 1)
    I[np.arange(N.N_MEL), jj] = 1 - ww
    I[np.arange(N.N_MEL), jj + 1] += ww
    W = B @ I @ B.T
    rd = lambda d: (np.fromfile(d / "enroll.bin", dtype="<f4").reshape(24, -1).astype(np.float64), json.loads((d / "enroll.json").read_text()))

    def wr(d: Path, T: np.ndarray, j: dict) -> None:
        d.mkdir(parents=True, exist_ok=True)
        T.astype("<f4").tofile(d / "enroll.bin")
        (d / "enroll.json").write_text(json.dumps(j))

    rng = random.Random(9)
    pi = 0
    for ti in range(a.n_tg):
        endir = S / "d53_en" / f"en_{ti:03d}"
        Tt, jt = rd(endir)
        shutil.copy(S / "d37" / f"cen_{ti:03d}.wav", od / f"cen_{ti:03d}.wav")
        for j in range(a.per):
            spk = spks[(ti * a.per + j) % len(spks)]
            files = sorted(k for k in man["speakers"][spk]["files"] if k != "t10")
            rng.shuffle(files)
            used, xs, tl = [], [], 0.0
            for k in files:
                if tl >= 8.0:
                    break
                x = load48(CORPUS / spk / f"{k}.wav"); xs.append(x); used.append(k); tl += len(x) / 48000
            srcp = od / f"p{pi:04d}_t{ti:03d}_SRC.wav"
            sf.write(srcp, np.concatenate(xs)[: 8 * 48000].astype(np.float32), 48000)
            opd = work / f"op_{pi:04d}"
            if not (opd / "enroll.json").exists():
                rest = [k for k in sorted(man["speakers"][spk]["files"]) if k not in used]
                ys, tot = [], 0.0
                for k in rest:
                    if tot >= 22.0:
                        break
                    y = load48(CORPUS / spk / f"{k}.wav"); ys.append(y); tot += len(y) / 48000
                assert tot >= 20.0, f"{spk} の操作者の参照が足りない {tot:.1f}s"
                opw = work / f"op_{pi:04d}.wav"
                sf.write(opw, np.concatenate(ys).astype(np.float32), 48000)
                run([BIN, "artic-enroll", "-r", opw, "-o", opd])
            To, jo = rd(opd)
            base = {"D": 24, "K": To.shape[1], "counts": []}
            cm, cf, vt = work / f"cm_{pi:04d}", work / f"cf_{pi:04d}", work / f"vt_{pi:04d}"
            wr(cm, To, {**base, "mu": jo["mu"], "per_mean": jo["per_mean"], "per_p75": jo["per_p75"]})
            wr(cf, To, {**base, "mu": jt["mu"], "per_mean": jt["per_mean"], "per_p75": jt["per_p75"]})
            WT = W @ To
            wr(vt, WT + (Tt - WT).mean(1, keepdims=True), {**base, "mu": jt["mu"], "per_mean": jt["per_mean"], "per_p75": jt["per_p75"]})
            com = ["--mix", "0", "--operator-enroll", opd, "--per-p75", "--med-from-operator", "--warp"]
            conds = {"L1COPY": [cm, "--alpha", "1.0", "--resid", "1"], "L2PITCH": [cf, "--alpha", "1.0", "--resid", "1"],
                     "L3VTLN": [vt, "--alpha", "1.3", "--resid", "1"], "L4FULL": [endir, "--alpha", "1.3", "--resid", "0.7"],
                     "TAB": [endir, "--alpha", "1.3", "--resid", "0"]}
            for name, (en, *args) in conds.items():
                run([BIN, "artic", "-i", srcp, "-o", od / f"p{pi:04d}_t{ti:03d}_{name}.wav", "--enroll", en] + com + list(args))
            (work / f"src_{pi:04d}.json").write_text(json.dumps({"spk": spk, "caption": man["speakers"][spk]["caption"], "used": used}, ensure_ascii=False))
            pi += 1
        print("target", ti, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
