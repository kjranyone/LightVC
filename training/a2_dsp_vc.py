"""A-2 学習なしの男→女変換(物理骨格)の成分別寄与を、補助モデルの数値で測る(耳の前の参考値・採用判定ではない)。

条件(すべて因果・出力遅延 D は S0-7b の D* か既定 10ms):
  identity      LPC 分析合成(恒等)
  pitch         残差の因果 PSOLA で ST 半音上げるだけ
  vtl           包絡を α 倍に伸縮するだけ(A-1 の一様 α)
  pitch_vtl     両方(一様 α)
  pitch_vtl_nu  両方(A-1 の帯域別 α を滑らかにつないだ α(f))
指標:
  p_female  ECAPA(speechbrain spkrec-ecapa-voxceleb)埋め込みのロジスティック回帰(VCTK 実音声・評価話者を除く男女で学習)
  secs_src  変換前との ECAPA コサイン(話者性がどれだけ動いたか)
  cer       Whisper base(VCTK は正解テキスト・英語)
  df0_st    harvest の中央 f0 の変化(半音)

    CUDA_VISIBLE_DEVICES= uv run python a2_dsp_vc.py --st 10   # results/artic_a2/
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile

sys.path.insert(0, str(Path(__file__).parent))
import artic_dsp as D
from train_dec2 import load48

ROOT = Path(__file__).resolve().parent.parent
VC = ROOT / "data/vctk/VCTK-Corpus/VCTK-Corpus"
OUT = ROOT / "results/artic_a2"


def spk_lists():
    import s0_artic as S
    info = (VC / "speaker-info.txt").read_text().splitlines()[1:]
    males = sorted("p" + r.split()[0] for r in info if len(r.split()) > 2 and r.split()[2] == "M")
    females = sorted("p" + r.split()[0] for r in info if len(r.split()) > 2 and r.split()[2] == "F")
    test_s05b = {Path(p).parent.name for p in S.vctk_male_test()}
    dev = {"p226", "p227", "p232"}
    pool = [m for m in males if m not in test_s05b and m not in dev]
    eval_m = pool[::3][:10]
    clf_m = [m for m in males if m not in eval_m]
    clf_f = females
    return eval_m, clf_m, clf_f


def ecapa():
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    m = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                       savedir=str(ROOT / "pretrained_models/spkrec-ecapa-voxceleb"),
                                       run_opts={"device": "cpu"})

    def emb(x48: np.ndarray) -> np.ndarray:
        import librosa
        y = librosa.resample(x48.astype(np.float64), orig_sr=D.SR, target_sr=16000).astype(np.float32)
        with torch.no_grad():
            v = m.encode_batch(torch.from_numpy(y)[None])[0, 0].numpy()
        return v / (np.linalg.norm(v) + 1e-9)
    return emb


def norm_text(s: str) -> str:
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def cer(ref: str, hyp: str) -> float:
    ref, hyp = ref.replace(" ", ""), hyp.replace(" ", "")
    if not ref:
        return 0.0
    dp = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(hyp) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev = cur
    return dp[-1] / len(ref)


def alpha_fn(bands: dict):
    """A-1 の帯域別 α(F1 域・F2 域・F3 域の中心)を対数周波数で区分線形につなぐ。範囲外は端の値。"""
    cen = np.array([np.sqrt(250 * 1000), np.sqrt(800 * 2800), np.sqrt(2200 * 4500)])
    val = np.array([bands["F1域 250-1000Hz"]["alpha_median"], bands["F2域 800-2800Hz"]["alpha_median"],
                    bands["F3域 2200-4500Hz"]["alpha_median"]])

    def f(freq: np.ndarray) -> np.ndarray:
        return np.interp(np.log(np.maximum(freq, 1.0)), np.log(cen), val)
    return f


def main() -> int:
    import whisper
    from sklearn.linear_model import LogisticRegression
    from s0_artic import harvest_f0
    ap = argparse.ArgumentParser()
    ap.add_argument("--st", type=float, default=10.0)
    ap.add_argument("--D_ms", type=float, default=None)
    a = ap.parse_args()
    a1 = json.loads((ROOT / "results/artic_a1/vtl_warp.json").read_text())
    alpha_u = a1["bands"]["全域 300-5000Hz"]["alpha_median"]
    anu = alpha_fn(a1["bands"])
    s7 = ROOT / "results/artic_s0/s07b.json"
    dms = a.D_ms if a.D_ms is not None else (json.loads(s7.read_text()).get("D_star_ms") if s7.exists() else None)
    dms = 10.0 if dms is None else dms
    Dn = int(round(dms * D.SR / 1000))
    eval_m, clf_m, clf_f = spk_lists()
    emb = ecapa()
    X, y = [], []
    for lab, spks in ((0, clf_m), (1, clf_f)):
        for s in spks:
            for u in (30, 31, 32):
                p = VC / f"wav48/{s}/{s}_{u:03d}.wav"
                if p.exists():
                    X.append(emb(load48(str(p)).astype(np.float64)))
                    y.append(lab)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(np.array(X), np.array(y))
    asr = whisper.load_model("base", device="cpu")
    conds = {"identity": (0.0, 1.0), "pitch": (a.st, 1.0), "vtl": (0.0, alpha_u),
             "pitch_vtl": (a.st, alpha_u), "pitch_vtl_nu": (a.st, anu)}
    rows = {c: [] for c in conds}
    for s in eval_m:
        for u in (40, 41):
            p = VC / f"wav48/{s}/{s}_{u:03d}.wav"
            t = VC / f"txt/{s}/{s}_{u:03d}.txt"
            if not (p.exists() and t.exists()):
                continue
            x = load48(str(p)).astype(np.float64)[: 8 * D.SR]
            ref = norm_text(t.read_text())
            e0 = emb(x)
            f0x, _ = harvest_f0(x)
            for c, (st, al) in conds.items():
                yv = D.dsp_convert(x, st, al, Dn)
                pk = float(np.abs(yv).max())
                yv = yv * (0.95 / pk if pk > 0.95 else 1.0)
                (OUT / c).mkdir(parents=True, exist_ok=True)
                soundfile.write(OUT / c / f"{s}_{u:03d}.wav", yv.astype(np.float32), D.SR)
                ev = emb(yv)
                hyp = norm_text(asr.transcribe(str(OUT / c / f"{s}_{u:03d}.wav"), language="en", fp16=False)["text"])
                f0y, _ = harvest_f0(np.clip(yv, -1, 1))
                df0 = (12 * np.log2(np.median(f0y[f0y > 0]) / np.median(f0x[f0x > 0]))
                       if (f0y > 0).sum() > 10 and (f0x > 0).sum() > 10 else None)
                rows[c].append({"utt": f"{s}_{u:03d}", "p_female": float(clf.predict_proba(ev[None])[0, 1]),
                                "secs_src": float(ev @ e0), "cer": cer(ref, hyp), "df0_st": df0})
            print(s, u, {c: {k: round(v, 3) for k, v in rows[c][-1].items() if isinstance(v, float)} for c in conds}, flush=True)
    summ = {c: {k: round(float(np.median([r[k] for r in rs if r[k] is not None])), 3)
                for k in ("p_female", "secs_src", "cer", "df0_st")} for c, rs in rows.items()}
    rep = {"st": a.st, "D_ms": dms, "alpha_uniform": alpha_u, "alpha_bands": {k: v["alpha_median"] for k, v in a1["bands"].items()},
           "eval_speakers": eval_m, "clf_train": {"male": len(clf_m), "female": len(clf_f), "n": len(y)},
           "summary_median": summ, "rows": rows,
           "note": "補助モデルの数値は耳の代わりにならない(本プロジェクトの実測: 代理指標の単独昇格禁止)。成分別の寄与の向きを見るための参考値。"}
    (OUT / f"a2_st{a.st:g}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
