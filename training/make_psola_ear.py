"""運ぶ方式の天井の耳(学習なし・オーナー承認 2026-10-02): 非因果 PSOLA(理想的な声門の時刻)で男声を +12 半音にした音に、加工の傷があるか。

NAM 路線(入力の波形を運び、回路は決定的な写像)が生きるかは、質を壊さないピッチ演算子があるかで決まる。因果 RRPS は音域移動だけで 4/4 不合格(spk_ear)。
ここでは上限 = 非因果 PSOLA(s0_artic.s05 と同じ実装: 因果 LPC p24 の残差・harvest f0 + 残差ピークのマーク・元の包絡で合成)を聞く。
候補(伏せ): ORIG(元の男声)・PSOLA12(+12 半音・包絡そのまま)・PSOLA12A(+12 半音 + LPC 包絡の声道長伸縮 α 1.14)・RRPS12(因果 RRPS +12・陰性の対照)。
判定 = 加工の傷(コーラス・ジリジリ・ざらつき・機械的・途切れ)の有無だけ。音量は試行内で RMS をそろえ共通の減衰・長さ 8s。

    uv run python make_psola_ear.py
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile

sys.path.insert(0, str(Path(__file__).parent))
import artic_dsp as D
from s0_artic import harvest_f0
from train_ddsp_vc import load48

ROOT = Path(__file__).resolve().parent.parent
EB = ROOT / "results/earbattery"
VC = ROOT / "data/vctk/VCTK-Corpus/VCTK-Corpus/wav48"
OUT = EB / "psola_ear"
SEC = 8.0


def utterances() -> list[tuple[str, Path]]:
    pick = [("VCTK p245", VC / "p245/p245_010.wav"), ("VCTK p251", VC / "p251/p251_015.wav"), ("VCTK p298", VC / "p298/p298_020.wav")]
    ja = sorted((ROOT / "data/male_tts_corpus").glob("*/*.wav"))
    pick.append(("JA TTS 男声", ja[len(ja) // 3]))
    return [(n, p) for n, p in pick if p.exists()]


def trim_active(x: np.ndarray, n: int) -> np.ndarray:
    import librosa
    y, _ = librosa.effects.trim(x, top_db=35)
    if len(y) < n:
        y = np.concatenate([y, np.zeros(n - len(y))])
    return y[:n]


def psola(x: np.ndarray, st: float, alpha: float = 1.0) -> np.ndarray:
    lar, a_sub, e = D.analyze(x, 24)
    f0, t = harvest_f0(x)
    segs = D.pitch_marks(e, t, f0)
    e1, _ = D.psola_shift(e, segs, 2 ** (st / 12))
    a = a_sub if alpha == 1.0 else D.coef_schedule(D.warp_lar(lar, alpha), len(x))
    return D.synthesize(e1, a)


def rrps(x: np.ndarray, st: float) -> np.ndarray:
    lar, a_sub, e = D.analyze(x, 24)
    f0, _ = D.causal_yin(x, voi_max=0.45)
    Dn = 480
    e1, _ = D.rrps(e, f0, 2 ** (st / 12), Dn, voiced=D.voiced_known(len(x), f0, Dn, D.F0_HOP))
    y = D.synthesize(e1, a_sub)
    return np.concatenate([y[Dn:], np.zeros(Dn)])


def main() -> int:
    from render_d1_ab import norm_trial
    OUT.mkdir(parents=True, exist_ok=True)
    n = int(SEC * 48000)
    key: dict = {"conditions": {"ORIG": "元の男声(加工なし)", "PSOLA12": "非因果 PSOLA +12 半音(包絡そのまま)",
                                "PSOLA12A": "非因果 PSOLA +12 半音 + 声道長伸縮 α 1.14", "RRPS12": "因果 RRPS +12 半音(陰性の対照)"},
                 "trials": {}}
    for i, (name, p) in enumerate(utterances()):
        x = trim_active(load48(p).astype(np.float64), n)
        clips = {"ORIG": x, "PSOLA12": psola(x, 12), "PSOLA12A": psola(x, 12, 1.14), "RRPS12": rrps(x, 12)}
        clips = {k: np.clip(np.nan_to_num(v), -4, 4)[:n] for k, v in clips.items()}
        normed = norm_trial(clips, clips["ORIG"])
        names = list(normed)
        tid = f"ps_{i + 1}"
        random.Random(tid).shuffle(names)
        td = OUT / tid
        td.mkdir(exist_ok=True)
        key["trials"][tid] = {"utt": name, "file": str(p), "map": {}}
        for j, nm in enumerate(names):
            soundfile.write(td / f"{'ABCD'[j]}.wav", normed[nm], 48000)
            key["trials"][tid]["map"]["ABCD"[j]] = nm
        print(tid, name, flush=True)
    (OUT / "_key_聴取後に開く.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
    (OUT / "_key_page.json").write_text(json.dumps({t: {"map": {L: "" for L in v["map"]}} for t, v in key["trials"].items()}, ensure_ascii=False))
    subprocess.run([sys.executable, "make_ab_page.py", "--dir", "psola_ear", "--order", "ps", "--save", "/save_psola", "--no-err",
                    "--key", "_key_page.json", "--title", "ピッチ操作の天井(加工の傷の有無)",
                    "--intro", "各試行は同じ男声の 4 本です(1 本は加工なしの元の録音)。3 本は声を 1 オクターブ高くしてあり、甲高く聞こえますが、"
                               "<b>声の高さ・声質の変化は判定に含めず、加工の傷(コーラス・ジリジリ・ざらつき・機械的・途切れ)が無ければ合格</b>としてください。"
                               "気づきはメモへ。音量はそろえてあります。"],
                   cwd=Path(__file__).parent, check=True)
    print("ear page ->", OUT / "listen.html", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
