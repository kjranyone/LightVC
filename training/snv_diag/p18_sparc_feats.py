"""段 P18 C の準備(学習なし): 外の調音の物差し(SPARC multi の EMA・物差しとしてだけ)を、開発の組(devA の先頭 64 人の源 s0・s1 と登録 e0〜e7)・
生成した声の模型(data/snv_d5 の g0_c1・w_c2 の t00〜t02)・JVNV の男声 2 人(regular_01・02 の 6 感情)に書き出す。録音の鎖を通した版は、通す前の音の EMA をそのまま使う(同じ演技)。
出力 = data/snv_e1_feat/sparc/{cv,d5,jvnv}/<名前>.npz(snv_a2_sparc_feats.py と同じ形)。SPARC の依存は ../sparc_env(uv)。
    cd .. && ../sparc_env/bin/python training/snv_diag/p18_sparc_feats.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/snv_e1_feat/sparc"
EMOS = ("anger", "disgust", "fear", "happy", "sad", "surprise")


def jobs() -> list[tuple[str, Path]]:
    man = json.loads((ROOT / "data/cv22_ja/dev_manifest.json").read_text())
    out = [(f"cv/{s['speaker']}_{c['name']}", ROOT / f"data/snv_p_feat/cv/{s['speaker']}_{c['name']}.wav") for s in [x for x in man["speakers"] if x["split"] == "devA"][:64] for c in s["src"][:2] + s["enroll"]]
    dm = json.loads((ROOT / "data/snv_d5/manifest.json").read_text())
    out += [(f"d5/{v['voice']}_{t}", ROOT / f"data/snv_d5/{v['voice']}/{t}.wav") for v in dm["voices"] if (v["type"], v["cap"]) in (("g0", "c1"), ("w", "c2")) for t in ("t00", "t01", "t02")]
    out += [(f"jvnv/{sp}_{e}_regular_{j}", ROOT / f"data/jvnv/jvnv_v1/{sp}/{e}/regular/{sp}_{e}_regular_{j}.wav") for sp in ("M1", "M2") for e in EMOS for j in ("01", "02")]
    xp = ROOT / "data/snv_p73/sets.json"
    if xp.exists():
        out += sorted({(str(Path(w["ema"]).relative_to("data/snv_e1_feat/sparc").with_suffix("")), ROOT / w["wav"]) for vs in json.loads(xp.read_text()).values() for v in vs for w in v["src"] + v["enr"]
                       if w["ema"].startswith("data/snv_e1_feat/sparc/p73/")})
    return out


def main() -> int:
    from sparc import load_model
    m = load_model("multi", device="cuda")
    js = jobs()
    print("jobs", len(js), flush=True)
    for i, (name, p) in enumerate(js):
        q = OUT / f"{name}.npz"
        if q.exists():
            continue
        q.parent.mkdir(parents=True, exist_ok=True)
        c = m.encode(str(p))
        n = min(len(c["ema"]), len(c["pitch"]), len(c["loudness"]), len(c["periodicity"]))
        np.savez(q, ema=c["ema"][:n].astype(np.float32), pitch=c["pitch"][:n, 0].astype(np.float32), loudness=c["loudness"][:n, 0].astype(np.float32),
                 periodicity=c["periodicity"][:n, 0].astype(np.float32), pitch_stats=np.asarray(c["pitch_stats"], dtype=np.float32))
        if i % 200 == 0:
            print(i, name, flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
