"""段 P114(学習なし): 「倍の周期の成分が強い」と数えたフレームの中身を、元の音の自己相関で分ける。
倍 = 周期 2 T0 の相関が T0 より 0.1 以上高い(周期が倍になっている)・雑 = T0 も 2 T0 も相関 0.3 未満(声というより息・雑音)・周 = それ以外(T0 で周期的)。種類ごとの率と、写しの周期の構造のずれ。
    cd training && SNV_ESM=1 SNV_DEV=cuda [SNV_RCK=...] uv run --no-sync python snv_diag/p114_class.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import p98_probe as PB
import p114_copy as C
import snv_p0_front as P
import snv_p1_eval as EV

ROOT = Path(__file__).resolve().parent.parent.parent


def peaks(x: np.ndarray, r: np.ndarray, n: int) -> np.ndarray:
    out = np.full((2, n), np.nan, np.float32)
    for t in range(4, n - 4):
        if r[t] <= 0:
            continue
        T0 = int(round(EV.SR / r[t]))
        c = t * 240 + 120
        a, b = c - 3 * T0, c + 3 * T0
        if a < 0 or b + 2 * T0 + 4 > len(x) or T0 < 16:
            continue
        s0 = x[a:b]
        for i, L in enumerate((T0, 2 * T0)):
            best = -1.0
            for o in range(-3, 4):
                s1 = x[a + L + o : b + L + o]
                best = max(best, float((s0 * s1).sum() / (np.sqrt((s0 * s0).sum() * (s1 * s1).sum()) + 1e-12)))
            out[i, t] = best
    return out


def main() -> None:
    gen = EV.load_gen()
    vs = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())["val"]
    voices = [v for v in vs if v.startswith("em")][:4] + [v for v in vs if v.startswith("ef")][:4]
    acc: dict = {}
    for v in voices:
        for kind, p in [("通常の発話", ROOT / f"data/snv_e1/{v}/t03.wav")] + [("音素の無い発声", ROOT / f"data/snv_e1nv/{v}/{c}_0.wav") for c in ("laugh", "moan", "cry", "scream")]:
            u = PB.utt(librosa.load(p, sr=EV.SR, mono=True)[0].astype(np.float32))
            x, n = u["wav"], len(u["f0"])
            r, vf, _ = librosa.pyin(librosa.resample(x, orig_sr=EV.SR, target_sr=16000), fmin=60, fmax=1600 if kind != "通常の発話" else 700, sr=16000, frame_length=1024, hop_length=80, center=True)
            r = np.where(vf, np.nan_to_num(r), 0.0)[:n].astype(np.float32)
            r = np.concatenate([r, np.zeros(n - len(r), np.float32)])
            S = P.power_spec(torch.from_numpy(x))[:n]
            shr = P.odd_even(S, (r / 2)[:, None])[:, 0]
            pk = peaks(x, r, n)
            f0 = u["f0"].astype(np.float32)
            vv = (f0 > 0).astype(np.float32)
            lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
            cond = torch.from_numpy(np.concatenate([u["y"][:25], lf[None], vv[None], np.clip(u["y"][25:29], 0, 1.5) * vv[None]], 0))[None].to(EV.DEV)
            cond = torch.cat([EV.smooth_env(gen, cond[:, :25]), cond[:, 25:]], 1)
            w = np.asarray(EV.draw(gen, cond, torch.from_numpy(f0)[None].to(EV.DEV), x, f0), np.float32)
            w = np.concatenate([w[gen.delay :], np.zeros(gen.delay, np.float32)])[: len(x)]
            w = np.concatenate([w, np.zeros(max(0, len(x) - len(w)), np.float32)])
            e = C.acf_err(x, w, r, n)
            pw = peaks(w, r, n)
            ok = (r > 0) & np.isfinite(pk[0]) & np.isfinite(e) & np.isfinite(shr) & np.isfinite(pw[0])
            dbl = ok & (pk[1] - pk[0] > 0.1)
            nz = ok & ~dbl & (pk[0] < 0.3) & (pk[1] < 0.3)
            per = ok & ~dbl & ~nz
            for cls, m in (("倍(周期が倍)", dbl), ("雑(息・雑音)", nz), ("周(T0 で周期的)", per)):
                a = acc.setdefault((kind, cls), {"n": 0, "strong": 0, "err": 0.0, "orig_gap": 0.0, "copy_gap": 0.0})
                a["n"] += int(m.sum()); a["strong"] += int((m & (shr > -10)).sum()); a["err"] += float(e[m].sum())
                a["orig_gap"] += float((pk[1] - pk[0])[m].sum()); a["copy_gap"] += float((pw[1] - pw[0])[m].sum())
    tot = {k: sum(a["n"] for (kk, _), a in acc.items() if kk == k) for k in ("通常の発話", "音素の無い発声")}
    out = {}
    for (k, cls), a in sorted(acc.items()):
        n = max(1, a["n"])
        out[f"{k}|{cls}"] = {"率": round(a["n"] / tot[k], 3), "うち SHR が強いと数えた率": round(a["strong"] / n, 2), "写しのずれ": round(a["err"] / n, 3), "元の音の(2T0 − T0)の相関の差": round(a["orig_gap"] / n, 3), "写しの同じ差": round(a["copy_gap"] / n, 3)}
        print(k, cls, json.dumps(out[f"{k}|{cls}"], ensure_ascii=False), flush=True)
    tag = Path(__import__("os").environ.get("SNV_RCK", "rvoc3hi")).parts[-3] if __import__("os").environ.get("SNV_RCK") else "rvoc3hi"
    (ROOT / f"results/snv_p114/class_{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
