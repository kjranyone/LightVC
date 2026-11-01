"""D690(腕 64 の検査 I の探針・学習前): Irodori-TTS-v4-Large の日本語の男声(対照 = 同じ作り方の女声)と VCTK の別の話者の録音が、生成器の入力の条件(C2 の平均・幅 + 事後)の空間で、学習の女声(きれい)から本物の男声の入力の側へどれだけ寄るか。
判別器 A = D689 と同じ(同じ種・同じ標本 = D689 の値を再現)・判別器 B = 本物の男声を外郎売だけにした版(外郎売の半分で学習・残りの半分で読む)。規則は results/diag_ppgvoc64/prereg.yaml。
    uv run python d690_tts_cover.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

import c1_content as CC
import d689_pert_cover as D9
import nvoc as N
import train_ppgvoc as TP

ROOT = Path(__file__).resolve().parent.parent
H = D9.H
OUT = ROOT / "results/diag_ppgvoc64"
VCTK = ROOT / "data/vctk/VCTK-Corpus/VCTK-Corpus"
GAP = 0.15


def clips(files: list, n: int) -> list:
    out, cur, tl = [], [], 0.0
    for f in files:
        x = D9.load48(str(f))
        cur += [x, np.zeros(int(GAP * N.SR), np.float32)]
        tl += len(x) / N.SR + GAP
        if tl >= D9.SEC:
            out.append(np.concatenate(cur))
            cur, tl = [], 0.0
            if len(out) >= n:
                break
    return out


def tts_group(d: Path, pre: str, n: int) -> list:
    return [x for s in sorted(d.glob(f"{pre}*")) if s.is_dir() for x in clips(sorted(s.glob("t*.wav")), n)]


def vctk_group(sex: str, n_spk: int, excl: set) -> list:
    info = [l.split() for l in (VCTK / "speaker-info.txt").read_text().splitlines()[1:] if l.strip()]
    spk = sorted(f"p{r[0]}" for r in info if r[2] == sex and f"p{r[0]}" not in excl and (VCTK / "wav48" / f"p{r[0]}").is_dir())
    spk = sorted(random.Random(690).sample(spk, min(n_spk, len(spk))))
    return [x for s in spk for x in clips(sorted((VCTK / "wav48" / s).glob("*.wav"))[30:], 2)]


def main() -> int:
    dev = "cuda"
    OUT.mkdir(parents=True, exist_ok=True)
    TP.cvin_setup(str(ROOT / "results/c2_1/last.pt"), dev)
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1n = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev)
    c1n.load_state_dict(st1["net"])
    c1n.eval().requires_grad_(False)
    mfc = CC.MelFront().to(dev)
    F = lambda x: D9.feats(x, c1n, mfc, dev)
    rows, _, _ = TP.load_rows({h[0] for h in TP.e2e_held()}, True)
    rng = random.Random(689)
    pick = [r for r in rng.sample(rows, 600) if r[3] >= D9.SEC][:240]
    tr, te = pick[:160], pick[160:]
    clean_tr = np.stack([F(D9.load48(r[0])) for r in tr])
    clean_te = np.stack([F(D9.load48(r[0])) for r in te])
    male = []
    vspk = set()
    for s, kd in (("d487_uiro", None), ("devvctkm", "ME")):
        for r in json.loads((H / s / "devset.json").read_text())["pairs"]:
            if s == "devvctkm":
                vspk.add(r["spk"])
            if kd is None or r["kind"] == kd:
                male.append((s, F(D9.load48(str(H / s / "d56" / f"p{r['pi']:04d}_t{r['ti']:03d}_SRC.wav")))))
    mi = list(range(len(male)))
    random.Random(1).shuffle(mi)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def fit(Xc: np.ndarray, Xm: np.ndarray):
        X = np.concatenate([Xc, Xm])
        y = np.r_[np.zeros(len(Xc)), np.ones(len(Xm))]
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=0.1, max_iter=5000).fit(sc.transform(X), y)
        return lambda Z: clf.decision_function(sc.transform(np.atleast_2d(Z)))

    dA = fit(clean_tr, np.stack([male[i][1] for i in mi[: len(mi) // 2]]))
    mA_te = [male[i] for i in mi[len(mi) // 2:]]
    ui = [i for i in mi if male[i][0] == "d487_uiro"]
    dB = fit(clean_tr, np.stack([male[i][1] for i in ui[: len(ui) // 2]]))
    mB_te = np.stack([male[i][1] for i in ui[len(ui) // 2:]])
    groups = {
        "tts_v4l_pair_male": tts_group(ROOT / "data/pair_tts_v4l", "pm", 2),
        "tts_v4l_pair_female": tts_group(ROOT / "data/pair_tts_v4l", "pf", 2),
        "tts_v4l_eval_male": tts_group(ROOT / "data/male_tts_v4l", "m", 1),
        "vctk_other_male": vctk_group("M", 20, vspk),
        "vctk_other_female": vctk_group("F", 20, vspk),
    }
    feats = {k: np.stack([F(x) for x in v]) for k, v in groups.items()}
    rep: dict = {"n": {k: len(v) for k, v in groups.items()}}
    for nm, dfn, mref in (("A_d689", dA, np.stack([z for _, z in mA_te])), ("B_uiro_only", dB, mB_te)):
        c_med = float(np.median(dfn(clean_te)))
        m_med = float(np.median(dfn(mref)))
        mid = (c_med + m_med) / 2
        r = {"clean_test": round(c_med, 3), "male_test": round(m_med, 3)}
        if nm == "A_d689":
            for s in ("d487_uiro", "devvctkm"):
                r[f"male_test_{s}"] = round(float(np.median(dfn(np.stack([z for s_, z in mA_te if s_ == s])))), 3)
        for k, Z in feats.items():
            d = dfn(Z)
            r[k] = {"median": round(float(np.median(d)), 3), "frac_of_gap": round((float(np.median(d)) - c_med) / (m_med - c_med), 3),
                    "past_mid": round(float((d > mid).mean()), 3)}
        rep[nm] = r
    print(json.dumps(rep, indent=1, ensure_ascii=False), flush=True)
    (OUT / "d690_tts_cover.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
