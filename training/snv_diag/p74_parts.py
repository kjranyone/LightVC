"""段 P74 step b(results/snv_p74/prereg.yaml・学習なし): 変換の経路の形を 1 つずつ替えた CER(用量に直して読む)。
形: self0 / self7 = 自己の経路(登録 = 源の話者自身)を k 0 / k 7 で描く・selfin0 = 自己の経路の構造に源の発話そのものも入れる(表に正解のフレームが在る)・
x7r1 / x7r3 / x7r6 = 女声 2 声へ k 7(目標の登録 1 / 3 / 6 発話)・x0r3 = 女声 2 声へ k 0・m0r3 = E1 の検証の男声 2 声(源の声と別)へ k 0。
self0eN / x7r3eN(en=N)= 操作者の登録を先頭 N 本にする(既定の形は 3 本)。
参照 = 源の元の音の書き起こし(1 で切る)。書き起こしの文字は保存も表示もしない(数値だけ)。
    cd training && SNV_DEV=cuda OMP_NUM_THREADS=2 uv run --no-sync python snv_diag/p74_parts.py sets=cv en=8 [sys=snv_p30_c] [nmax=64] out=results/snv_p74/parts_cv.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p18_align as AL
import snv_p0_front as P
import snv_p1_eval as EV

ROOT = EV.ROOT


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    names, tag, nmax = opt["sets"].split(","), opt.get("sys", "snv_p30_c"), int(opt.get("nmax", 64))
    en = int(opt["en"])
    out = ROOT / opt.get("out", "results/snv_p74/parts.json")
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, kana, to16
    asr = WhisperModel("large-v3", device=EV.DEV, compute_type="float16" if EV.DEV == "cuda" else "int8")
    tr = lambda w: kana("".join(s.text for s in asr.transcribe(to16(w.astype(np.float64), EV.SR).astype(np.float32), language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)[0]))
    sy = EV.System(tag, EV.load_gen())
    split = json.loads((ROOT / "results/snv_p19_x/split.json").read_text())
    sets = {nm: [x for x in its if x["valid"]][:nmax] for nm, its in AL.build_sets(names, 64, split).items()}
    tgts = json.loads((EV.A1 / "convert_meta.json").read_text())["targets"][:2]
    St = {(t, n): sy.structure(f"t|{t}|{n}", [EV.utt(f"a0/{t}_{j}") for j in range(n)]) for t in tgts for n in (1, 3, 6)}
    males = [v for v in split["val"] if v.startswith("em")]
    Sm = {v: sy.structure(f"m|{v}", [AL.T.load_utt(AL.T.FEAT / f"e1/{v}/{t}.npz") for t in ("t00", "t01", "t02")]) for v in males}

    def ref_of(it: dict) -> str:
        if "ref" in it:
            return it["ref"]
        path = it.get("path") or str(ROOT / "data/snv_e1" / it["src"].split("|")[0] / f"{it['src'].split('|')[1]}.wav")
        return tr(librosa.load(path, sr=P.SR, mono=True)[0].astype(np.float32))

    rows = []
    for nm, its in sets.items():
        for i, it in enumerate(its):
            ref, enr, u = ref_of(it), it["enr"][:3], it["u"]
            c = lambda S, k: round(min(cer(ref, tr(sy.convert(u, enr, S, k))), 1.0), 4)
            So = sy.structure(f"self|{nm}|{it['src']}", enr)
            Si = sy.structure(f"self|in|{nm}|{it['src']}", enr + [u])
            ms = [v for v in males if v != it["spk"]]
            ms = [ms[i % len(ms)], ms[(i + 2) % len(ms)]]
            en_ = it["enr"][:en]
            ce = lambda S, k: round(min(cer(ref, tr(sy.convert(u, en_, S, k))), 1.0), 4)
            rows.append({"set": nm, "spk": it["spk"], "src": it["src"], "n_enr": len(en_), "self0": c(So, 0), "self7": c(So, 7), "selfin0": c(Si, 0)}
                        | {f"x7r{n}": [c(St[(t, n)], 7) for t in tgts] for n in (1, 3, 6)} | {"x0r3": [c(St[(t, 3)], 0) for t in tgts], "m0r3": [c(Sm[v], 0) for v in ms]}
                        | {f"self0e{en}": ce(sy.structure(f"self|e{en}|{nm}|{it['src']}", en_), 0), f"x7r3e{en}": [ce(St[(t, 3)], 7) for t in tgts]})
        rs = [r for r in rows if r["set"] == nm]
        print(nm, len(rs), {k: round(float(np.mean([np.mean(r[k]) for r in rs])), 4) for k in rs[0] if k not in ("set", "spk", "src", "n_enr")}, flush=True)
        out.write_text(json.dumps({"prereg": "results/snv_p74/prereg.yaml(step_b)", "sys": tag, "tgts": tgts, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
