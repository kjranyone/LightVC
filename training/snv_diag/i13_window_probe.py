"""段 I13 の後の切り分け(学習なし・CPU で決定的): 伴奏の 3 区間(namikawa_0〜2)の自己の reamp(i8_text_probe の real_self と同じ経路・構造と登録 = 先頭 30 s・k 0)を、語りの時間帯だけで書き起こす。
窓 = 元の音の large-v3 の語の時刻(段 I12 の測定)を ±0.15 s 広げたもの: namikawa_0 0〜1.8 / 2.8〜9.0 s・namikawa_1 0〜1.5 / 5.2〜9.2 s・namikawa_2 0〜2.9 / 3.9〜5.9 s。
full = 出力の全体(物差しと同じ)/ gate = 窓の外を 0 にした全体 / win = 窓ごとに切り出した書き起こしを繋いだもの。参照 = 写しの描画に同じ処理をした書き起こし(orig = 元の音の同じ処理の写しに対する CER)。
out_in_db = 窓の外(伴奏だけの時間帯)の RMS ÷ 窓の中の RMS。large-v3 は CPU の int8(物差しの GPU の float16 と細部が違いうるので full を物差しの値と並べる)。
    cd training && SNV_DEV=cpu uv run --no-sync python snv_diag/i13_window_probe.py snv_i13_e_l13 snv_i13_e_l16 snv_i12_e_r2 snv_i10_c_pr [threads=4] [out=results/snv_i13/window_probe.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nvoc as N
import snv_p1_eval as EV
import train_rvoc as TR
import train_snv_p1 as T

ROOT = EV.ROOT
DEV = EV.DEV
WIN = {"namikawa_0": ((0.0, 1.8), (2.8, 9.0)), "namikawa_1": ((0.0, 1.5), (5.2, 9.2)), "namikawa_2": ((0.0, 2.9), (3.9, 5.9))}
PAD = 0.15
RAMP = 0.01


def mask(n: int, sr: int, wins: tuple) -> np.ndarray:
    m = np.zeros(n, np.float32)
    r = int(RAMP * sr)
    for a, b in wins:
        i, j = max(0, int((a - PAD) * sr)), min(n, int((b + PAD) * sr))
        m[i:j] = 1.0
        if r and i > 0:
            m[i:i + r] *= np.linspace(0, 1, r, dtype=np.float32)[: len(m[i:i + r])]
        if r and j < n:
            m[j - r:j] *= np.linspace(1, 0, r, dtype=np.float32)[-len(m[j - r:j]):]
    return m


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    tags = [x for x in sys.argv[1:] if "=" not in x]
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, kana, load, to16
    asr = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=int(opt.get("threads", 4)))

    def tr16(w16: np.ndarray) -> str:
        segs, _ = asr.transcribe(w16.astype(np.float32), language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(x.text for x in segs))

    gen = EV.load_gen()

    @torch.no_grad()
    def copy(u: dict) -> np.ndarray:
        y = torch.from_numpy(u["y"])[None].to(DEV)
        lf = torch.from_numpy(u["lf"])[None].to(DEV)
        v = torch.from_numpy(u["v"])[None].to(DEV)
        cond = torch.cat([y[:, :25], lf[:, None], v[:, None], y[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
        f0 = torch.from_numpy(u["f0"])[None].to(DEV)
        return gen(cond, TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:].astype(np.float32)

    def views(w16: np.ndarray, wins: tuple) -> dict:
        m = mask(len(w16), 16000, wins)
        cut = [w16[max(0, int((a - PAD) * 16000)):int((b + PAD) * 16000)] for a, b in wins]
        rin, rout = np.sqrt(np.mean(w16[m > 0.5] ** 2) + 1e-12), np.sqrt(np.mean(w16[m < 0.5] ** 2) + 1e-12)
        return {"full": tr16(w16), "gate": tr16(w16 * m), "win": "".join(tr16(c) for c in cut), "out_in_db": round(float(20 * np.log10(rout / rin)), 2)}

    systems = [EV.System(t, gen) for t in tags]
    stat = [EV.utt("real/namikawa_enroll")]
    res: dict = {"tags": tags, "windows": WIN, "pad": PAD, "asr": "large-v3 cpu int8", "segments": {}}
    for seg, wins in WIN.items():
        u = EV.utt(f"real/{seg}")
        ref = views(to16(copy(u).astype(np.float64), EV.SR), wins)
        orig = views(to16(*load(T.FEAT / f"real/{seg}.wav")), wins)
        row: dict = {"copy": ref, "orig": orig | {f"cer_{k}": round(cer(ref[k], orig[k]), 4) for k in ("full", "gate", "win")}}
        print(seg, "copy", json.dumps(ref, ensure_ascii=False), flush=True)
        print(seg, "orig", json.dumps(row["orig"], ensure_ascii=False), flush=True)
        for s in systems:
            S = s.structure(f"self|{seg}", stat)
            o = views(to16(s.convert(u, stat, S, 0).astype(np.float64), EV.SR), wins)
            row[s.tag] = o | {f"cer_{k}": round(cer(ref[k], o[k]), 4) for k in ("full", "gate", "win")}
            print(seg, s.tag, json.dumps(row[s.tag], ensure_ascii=False), flush=True)
        res["segments"][seg] = row
    res["mean"] = {t: {k: round(float(np.mean([res["segments"][g][t][f"cer_{k}"] for g in WIN])), 4) for k in ("full", "gate", "win")} for t in tags + ["orig"]}
    print(json.dumps(res["mean"], ensure_ascii=False), flush=True)
    (ROOT / opt.get("out", "results/snv_i13/window_probe.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
