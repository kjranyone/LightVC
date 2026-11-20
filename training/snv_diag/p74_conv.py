"""段 P74 step a(results/snv_p74/prereg.yaml・学習なし): 系ごとの、自己の経路(k 0)と女声 2 声への経路(k 7)の CER。
組 = p18_align.build_sets の名前(cv・x:d_clean・x:d_rv3・x:d_rv6 など)。参照 = 源の元の音の書き起こし(1 で切る)。書き起こしの文字は保存も表示もしない(数値だけ)。
    cd training && SNV_DEV=cuda OMP_NUM_THREADS=2 uv run --no-sync python snv_diag/p74_conv.py sys=snv_p30_c,snv_p19_x sets=cv,x:d_rv6 out=results/snv_p74/conv_a.json
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
    names, tags, nmax = opt["sets"].split(","), opt["sys"].split(","), int(opt.get("nmax", 64))
    out = ROOT / opt.get("out", "results/snv_p74/conv.json")
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, kana, to16
    asr = WhisperModel("large-v3", device=EV.DEV, compute_type="float16" if EV.DEV == "cuda" else "int8")
    tr = lambda w: kana("".join(s.text for s in asr.transcribe(to16(w.astype(np.float64), EV.SR).astype(np.float32), language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)[0]))
    gen = EV.load_gen()
    split = json.loads((ROOT / "results/snv_p19_x/split.json").read_text())
    sets = {nm: [x for x in its if x["valid"]][:nmax] for nm, its in AL.build_sets(names, 64, split).items()}
    tgts = json.loads((EV.A1 / "convert_meta.json").read_text())["targets"][:2]
    refs: dict = {}

    def ref_of(nm: str, it: dict) -> str:
        if "ref" in it:
            return it["ref"]
        if (nm, it["src"]) not in refs:
            path = it.get("path") or str(ROOT / "data/snv_e1" / it["src"].split("|")[0] / f"{it['src'].split('|')[1]}.wav")
            refs[(nm, it["src"])] = tr(librosa.load(path, sr=P.SR, mono=True)[0].astype(np.float32))
        return refs[(nm, it["src"])]

    rows = []
    for tag in tags:
        sy = EV.System(tag, gen)
        St = {t: sy.structure(f"t|{t}", [EV.utt(f"a0/{t}_{j}") for j in (0, 1, 2)]) for t in tgts}
        for nm, its in sets.items():
            for it in its:
                ref, enr = ref_of(nm, it), it["enr"][:3]
                So = sy.structure(f"self|{nm}|{it['src']}", enr)
                rows.append({"sys": tag, "set": nm, "spk": it["spk"], "src": it["src"], "self": round(min(cer(ref, tr(sy.convert(it["u"], enr, So, 0))), 1.0), 4),
                             "x": [round(min(cer(ref, tr(sy.convert(it["u"], enr, St[t], 7))), 1.0), 4) for t in tgts]})
            rs = [r for r in rows if r["sys"] == tag and r["set"] == nm]
            print(tag, nm, len(rs), "self", round(float(np.mean([r["self"] for r in rs])), 4), "x", round(float(np.mean([np.mean(r["x"]) for r in rs])), 4), flush=True)
        out.write_text(json.dumps({"prereg": "results/snv_p74/prereg.yaml(step_a)", "tgts": tgts, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
