"""段 P114(学習なし): 「複数の基音が共存するフレーム」を数える。参照の高さ f0(pYIN)の半分の周波数の奇数倍(f0/2・3f0/2・…)= 倍の周期の成分の強さを、f0 の倍音に対する比(SHR・dB)で測る。
通常の発話の中ほど / 出だしと終わりの 6 フレーム / 音素の無い発声(笑い・悲鳴・喘ぎ・泣き)で、SHR が −10 dB を超えるフレームの率と、前段の高さの読み違えの率を並べる。
    cd training && uv run --no-sync python snv_diag/p114_shr.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import snv_p0_front as P

ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> None:
    vs = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())["val"]
    voices = [v for v in vs if v.startswith("em")][:6] + [v for v in vs if v.startswith("ef")][:6]
    acc: dict = {}
    def add(key: str, shr: np.ndarray, bad: np.ndarray) -> None:
        a = acc.setdefault(key, {"n": 0, "sub": 0, "bad": 0, "bad_sub": 0, "bad_nosub": 0, "nosub": 0})
        s = shr > -10.0
        a["n"] += len(shr); a["sub"] += int(s.sum()); a["bad"] += int(bad.sum()); a["bad_sub"] += int((bad & s).sum()); a["bad_nosub"] += int((bad & ~s).sum()); a["nosub"] += int((~s).sum())
    for v in voices:
        items = [("speech", ROOT / f"data/snv_e1/{v}/t0{j}.wav") for j in (3, 6)] + [(c, ROOT / f"data/snv_e1nv/{v}/{c}_0.wav") for c in ("laugh", "scream", "moan", "cry")]
        for kind, p in items:
            x = librosa.load(p, sr=P.SR, mono=True)[0].astype(np.float32)
            x = x[: P.n_frames(len(x)) * P.H]
            r, vf, _ = librosa.pyin(librosa.resample(x, orig_sr=P.SR, target_sr=16000), fmin=60, fmax=1600 if kind != "speech" else 700, sr=16000, frame_length=1024, hop_length=80, center=True)
            f = P.f0_track(torch.from_numpy(x))[0].numpy()
            n = min(len(f), len(r))
            r, vf, f = np.nan_to_num(r[:n]), vf[:n], f[:n]
            S = P.power_spec(torch.from_numpy(x))[:n]
            shr = P.odd_even(S, (r / 2)[:, None])[:, 0]
            ok = vf & np.isfinite(shr) & (r > 0)
            bad = (f > 0) & (np.abs(np.log2(np.maximum(f, 1) / np.maximum(r, 1))) > 0.2)
            t = 0
            edge = np.zeros(n, bool)
            while t < n:
                if vf[t]:
                    s = t
                    while t < n and vf[t]:
                        t += 1
                    edge[s : s + 6] = True
                    edge[max(s, t - 6) : t] = True
                else:
                    t += 1
            if kind == "speech":
                add("通常の発話・中ほど", shr[ok & ~edge], bad[ok & ~edge])
                add("通常の発話・出だしと終わり", shr[ok & edge], bad[ok & edge])
            else:
                add({"laugh": "笑い", "scream": "悲鳴", "moan": "喘ぎ", "cry": "泣き"}[kind], shr[ok], bad[ok])
    for k, a in acc.items():
        print(k, {"フレーム": a["n"], "倍の周期の成分が強い率": round(a["sub"] / max(1, a["n"]), 3), "前段の読み違えの率": round(a["bad"] / max(1, a["n"]), 3), "読み違え(成分が強い所)": round(a["bad_sub"] / max(1, a["sub"]), 3), "読み違え(弱い所)": round(a["bad_nosub"] / max(1, a["nosub"]), 3)}, flush=True)
    (ROOT / "results/snv_p110/shr.json").write_text(json.dumps(acc, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
