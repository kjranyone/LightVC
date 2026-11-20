"""段 P17(paper_snv rev45 §6・学習なし・信号だけ): 帯域の台帳。開発の組・学習の族・生成した声の模型・JVNV・試験の登録の音の、帯域ごとの水準(1〜4 kHz 比 dB・Welch 4096)と切りの位置
(300〜3000 Hz の山から 60 dB 下を最後に上回る周波数)。開発の組は話者ごとの値と、P8 の話者の CER(results/snv_p15/rows.json・有効な源・k 7)との順位相関も出す。
出力 = results/snv_p17/band_ledger.json。
    cd training && OMP_NUM_THREADS=2 uv run --no-sync python snv_diag/p17_band_ledger.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import welch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
BANDS = {"b8_11": (8000, 11000), "b11_13": (11500, 13000), "b13_16": (13000, 16000), "b16_24": (16000, 23500)}


def bw(path: Path) -> dict:
    x, sr = sf.read(str(path), dtype="float32")
    x = x.mean(1) if x.ndim > 1 else x
    f, p = welch(x, sr, nperseg=4096, noverlap=2048)
    db = 10 * np.log10(p + 1e-20)
    ref = 10 * np.log10(p[(f >= 1000) & (f < 4000)].mean() + 1e-20)
    return {"sr": int(sr), "fc60": float(f[np.where(db > db[(f > 300) & (f < 3000)].max() - 60)[0].max()])} | {k: float(10 * np.log10(p[(f >= lo) & (f < hi)].mean() + 1e-20) - ref) for k, (lo, hi) in BANDS.items()}


def pop(paths: list) -> dict:
    d = [bw(q) for q in paths]
    return {"n": len(d), "sr": sorted({x["sr"] for x in d})} | {k: [round(float(q), 1) for q in np.percentile([x[k] for x in d], [5, 25, 50, 75, 95])] for k in ["fc60"] + list(BANDS)}


def main() -> int:
    random.seed(0)
    man = json.loads((ROOT / "data/cv22_ja/dev_manifest.json").read_text())
    dv = json.loads((ROOT / "results/snv_d5/dev_rows_devA.json").read_text())
    rows = json.loads((ROOT / "results/snv_p15/rows.json").read_text())["rows"]
    out: dict = {"note": "帯域の水準は 1〜4 kHz 比 dB の 5 / 25 / 50 / 75 / 95 %・fc60 は Hz", "pop": {}}
    dev = {sp: [x for x in man["speakers"] if x["split"] == sp][:n] for sp, n in (("devA", 64), ("devB", 32))}
    for sp, S in dev.items():
        out["pop"][sp] = pop([ROOT / f"data/snv_p_feat/cv/{s['speaker']}_{c['name']}.wav" for s in S for c in s["src"][:2]]) | {"mp3_sr": {str(k): sum(1 for s in S if s.get("mp3_sr") == k) for k in ([32000], [48000])}}
    out["pop"]["E1 の女声"] = pop(random.sample(sorted((ROOT / "data/snv_e1").glob("ef*/t01.wav")), 60))
    out["pop"]["E1 の男声"] = pop(random.sample(sorted((ROOT / "data/snv_e1").glob("em*/t01.wav")), 60))
    out["pop"]["生成した対照の声 g0_c1"] = pop(sorted((ROOT / "data/snv_d5").glob("*_g0_c1/t01.wav")))
    out["pop"]["生成した低い声 w_c2"] = pop(sorted((ROOT / "data/snv_d5").glob("*_w_c2/t01.wav")))
    out["pop"]["JVNV の男声"] = pop(sorted((ROOT / "data/jvnv/jvnv_v1").glob("M*/*/regular/*_01.wav"))[:24])
    out["pop"]["試験の登録"] = pop(sorted((ROOT / "data/snv_p_feat/real").glob("*_enroll.wav")))
    L = {s["speaker"]: {k: float(np.median([bw(ROOT / f"data/snv_p_feat/cv/{s['speaker']}_{c['name']}.wav")[k] for c in s["src"][:2] + s["enroll"][:2]])) for k in ["fc60"] + list(BANDS)} for s in dev["devA"]}
    valid = lambda r: dv["ref"][r["src"]]["text_vs_orig"] <= 0.1
    cross: dict = {}
    slf: dict = {}
    for r in rows["base4"]:
        if valid(r) and r["k"] == 7:
            cross.setdefault(r["spk"], []).append(r["cer"])
    for r in rows["self"]:
        if valid(r):
            slf.setdefault(r["spk"], []).append(r["cer"])
    S = [s for s in L if s in cross and s in slf]
    c, e = np.array([np.mean(cross[s]) for s in S]), np.array([np.mean(slf[s]) for s in S])
    out["devA_speakers"] = {"n": len(S), "rho": {k: {"cross": round(float(spearmanr([L[s][k] for s in S], c)[0]), 3), "self": round(float(spearmanr([L[s][k] for s in S], e)[0]), 3),
                                                     "cross_minus_self": round(float(spearmanr([L[s][k] for s in S], c - e)[0]), 3)} for k in ["fc60"] + list(BANDS)}, "ledger": L}
    (ROOT / "results/snv_p17").mkdir(exist_ok=True)
    (ROOT / "results/snv_p17/band_ledger.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "devA_speakers"} | {"rho": out["devA_speakers"]["rho"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
