"""D6: 元の声の単位(C1 の硬い argmax)が、目標の表(参照 20s)の『観測済みのセル』に当たる割合(学習なし)。
目標 = 日本語 held の女声(conv_ja と同じ 20 人・参照 20s)。元の声の群: 目標自身の別発話 / 別の日本語女声 / 日本語男声(TTS 男声) / VCTK 英語男声 / VCTK 英語女声。
指標: 活性フレームのうち、目標の参照で観測回数 ≥1 / ≥5 のセルに当たる割合・単位分布の JS 距離。
    uv run python d6_unit_cov.py --ladder <scratchpad>/r4_spk --out ../results/conv_p0/d6_unit_cov.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parent.parent
TTS_LEAK = {"fe9565ca1f33bf20", "ffb9b5647612b32b"}


def load48(path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != 48000:
        g = gcd(sr, 48000)
        x = resample_poly(x, 48000 // g, sr // g).astype(np.float32)
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref_sec", type=float, default=20.0)
    a = ap.parse_args()
    import c1_content as CC
    import conv_c0 as C0
    import nvoc as N
    import train_c1 as T1
    import train_rvoc as TR
    dev = "cuda"
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev)
    c1.load_state_dict(st1["net"]); c1.eval()
    mfront = CC.MelFront().to(dev)
    K = torch.load(ROOT / "results/c1_1/codebook.pt").shape[0]

    @torch.no_grad()
    def units(x: np.ndarray) -> np.ndarray:
        n = len(x) // N.HOP
        u = c1(mfront(torch.from_numpy(T1.prime(x))[None].to(dev)))[..., -n:].argmax(1)[0].cpu().numpy()
        fr = x[: n * N.HOP].reshape(n, N.HOP)
        e = 10 * np.log10((fr ** 2).mean(1) + 1e-10)
        act = e > e.max() - 40
        return u[act[: len(u)]]

    held = [s for s in sorted(TR.held_speakers()) if s != "unknown" and s not in TTS_LEAK and (ROOT / "female-dataset" / s).is_dir()]
    tg = []
    for s in held:
        ws = sorted((ROOT / "female-dataset" / s).glob("*.wav"))
        ref, tot = [], 0.0
        for w in ws:
            if tot >= a.ref_sec:
                break
            ref.append(w); tot += sf.info(str(w)).duration
        rest = [w for w in ws if w not in ref]
        if not rest:
            continue
        xr = np.concatenate([load48(w) for w in ref])[: int(max(a.ref_sec, 25) * 48000)]
        xc = np.concatenate([load48(w) for w in rest])[: 25 * 48000]
        tg.append({"spk": s, "cnt": np.bincount(units(xr), minlength=K), "self": units(xc)})
    S = len(tg)
    print("targets", S, flush=True)
    J = json.loads((Path(a.ladder) / "jobs.json").read_text())
    vm = [np.concatenate([units(load48(C0.utt(m, u))) for u in C0.SENTS if C0.utt(m, u).exists()]) for m in J["males"][:10]]
    vf = [np.concatenate([units(load48(C0.utt(f, u))) for u in C0.SENTS if C0.utt(f, u).exists()]) for f in J["fems"][:10]]
    mrows = [r for r in json.loads((ROOT / "data/f0est_hi/manifest.json").read_text())["rows"] if r["ok"] and r["src"] == "tts_m"]
    rng = random.Random(0)
    by: dict = {}
    for r in mrows:
        by.setdefault(r["spk"], []).append(r)
    jm = []
    for sp in sorted(by)[:10]:
        rs = rng.sample(by[sp], min(4, len(by[sp])))
        jm.append(np.concatenate([units(load48(r["wav"])) for r in rs]))

    def cov(u: np.ndarray, cnt: np.ndarray) -> tuple:
        c = cnt[u]
        return float((c >= 1).mean()), float((c >= 5).mean())

    def js(u: np.ndarray, cnt: np.ndarray) -> float:
        p = np.bincount(u, minlength=K) + 0.5
        q = cnt + 0.5
        p, q = p / p.sum(), q / q.sum()
        m = (p + q) / 2
        return float(0.5 * (p * np.log(p / m)).sum() + 0.5 * (q * np.log(q / m)).sum())

    groups = {"ja_female_self": lambda i: [tg[i]["self"]], "ja_female_other": lambda i: [tg[(i + 1) % S]["self"], tg[(i + 7) % S]["self"]],
              "ja_male_tts": lambda i: [jm[i % len(jm)]], "vctk_male": lambda i: [vm[i % len(vm)]], "vctk_female": lambda i: [vf[i % len(vf)]]}
    rep: dict = {"n_targets": S, "ref_sec": a.ref_sec, "units_observed_in_ref_mean": float(np.mean([(t["cnt"] > 0).sum() for t in tg]))}
    for g_, fn in groups.items():
        c1s, c5s, jss = [], [], []
        for i in range(S):
            for u in fn(i):
                x1, x5 = cov(u, tg[i]["cnt"])
                c1s.append(x1); c5s.append(x5); jss.append(js(u, tg[i]["cnt"]))
        rep[g_] = {"hit_obs>=1": round(float(np.mean(c1s)), 3), "hit_obs>=5": round(float(np.mean(c5s)), 3), "js": round(float(np.mean(jss)), 3)}
        print(g_, rep[g_], flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
