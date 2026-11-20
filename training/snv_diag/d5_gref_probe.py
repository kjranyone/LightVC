"""段 D5(paper_snv rev41 §6・results/snv_d5/prereg.yaml lever_a・学習なし): 生成器の参照の型の下見の測定。training/snv_d5_gen.py の声(data/snv_d5/manifest.json)ごとに、
高さの中央値と参照の高さの差(半音)・文に対する CER(large-v3 の仮名)・UTMOS・同じ声の 3 発話の ECAPA の余弦の平均・元の E1 の声との余弦・開発の組(cv の devA)の最も近い声の余弦・台帳の量(per0・rate)を測る。
型(g0 対照・w・r)× 説明文(c1・c2)ごとにまとめ、予測 Pa_pitch・Pa_content・Pa_natural・Pa_consist を判定する。文そのものは出力に書かない。
    cd training && uv run --no-sync python snv_diag/d5_gref_probe.py [dir=data/snv_d5] [out=results/snv_d5/gref.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import d5_ledger as LD

ROOT = LD.ROOT
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    src = ROOT / opt.get("dir", "data/snv_d5")
    out = ROOT / opt.get("out", "results/snv_d5/gref.json")
    man = json.loads((ROOT / "data/snv_d5/manifest.json").read_text())
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, kana, to16
    from speechbrain.inference.speaker import EncoderClassifier
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16" if DEV == "cuda" else "int8")
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(DEV).eval()
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})
    w16 = lambda x: to16(x.astype(np.float64), LD.SR).astype(np.float32)

    def tr(y: np.ndarray) -> str:
        segs, _ = asr.transcribe(y, language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(s.text for s in segs))

    def emb(y: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(y).float()[None].to(DEV)).reshape(-1)
        return torch.nn.functional.normalize(e, dim=0).cpu().numpy()

    cvm = json.loads((ROOT / "data/cv22_ja/dev_manifest.json").read_text())
    Ecv = np.stack([emb(np.concatenate([w16(LD.load48(LD.FEAT / f"cv/{s['speaker']}_{c['name']}.wav")) for c in s["enroll"][:4]])) for s in cvm["speakers"] if s["split"] == "devA"])
    hz = lambda st: float(100 * 2 ** (st / 12))
    rows = {}
    ebase: dict = {}
    for v in man["voices"]:
        d = src / v["voice"]
        ws = [d / f"{f['t']}.wav" for f in v["files"]]
        if not all(w.exists() for w in ws):
            continue
        xs = [LD.load48(w) for w in ws]
        fs = [LD.feat_of(f"dir:{src.name}", v["voice"], w, x) for w, x in zip(ws, xs)]
        st = LD.feat_stats(fs)
        ys = [w16(x) for x in xs]
        es = [emb(y) for y in ys]
        b = v["base"]
        if b not in ebase:
            ebase[b] = emb(np.concatenate([w16(LD.load48(ROOT / f"data/snv_e1/{b}/t0{j}.wav")) for j in (3, 4, 5)]))
        ev = emb(np.concatenate(ys))
        f0 = hz(st["f0_med"]) if st else float("nan")
        rows[v["voice"]] = {"base": b, "type": v["type"], "cap": v["cap"], "ref_f0": v["ref_f0"], "f0": round(f0, 1), "d_st": round(float(12 * np.log2(f0 / v["ref_f0"])), 2),
                            "alpha_v": v.get("alpha_v"), "vtl_lpc": round(LD.vtl_of(xs, [f["f0"] for f in fs]), 2),
                            "cer": round(float(np.median([cer(kana(f["text"]), tr(y)) for f, y in zip(v["files"], ys)])), 4),
                            "utmos": round(float(np.mean([float(utmos(torch.from_numpy(y).float()[None].to(DEV), 16000).item()) for y in ys])), 3),
                            "consist": round(float(np.mean([es[i] @ es[j] for i in range(3) for j in range(i + 1, 3)])), 4), "cos_base": round(float(ev @ ebase[b]), 4),
                            "near_cv": round(float((Ecv @ ev).max()), 4)} | {k: round(float(st[k]), 3) for k in ("per0", "per1", "rate", "rate_lv", "f0_p5", "f0_span") if st}
        print(v["voice"], json.dumps(rows[v["voice"]], ensure_ascii=False), flush=True)
    summ = {}
    for typ, cap in (("g0", "c1"), ("w", "c1"), ("w", "c2"), ("r", "c1"), ("r", "c2")):
        R = [r for r in rows.values() if r["type"] == typ and r["cap"] == cap]
        if not R:
            continue
        hi = [r for r in R if r["ref_f0"] >= 93]
        summ[f"{typ}_{cap}"] = {"n": len(R), "abs_d_st_median": round(float(np.median([abs(r["d_st"]) for r in R])), 2), "abs_d_st_median_ref_ge95": round(float(np.median([abs(r["d_st"]) for r in hi])), 2) if hi else None,
                                "within_2st": round(float(np.mean([abs(r["d_st"]) <= 2 for r in R])), 3), "d_st_median": round(float(np.median([r["d_st"] for r in R])), 2),
                                "f0_median": round(float(np.median([r["f0"] for r in R])), 1), "f0_min": round(float(np.min([r["f0"] for r in R])), 1),
                                "cer_median": round(float(np.median([r["cer"] for r in R])), 4), "utmos": round(float(np.mean([r["utmos"] for r in R])), 3),
                                "consist": round(float(np.mean([r["consist"] for r in R])), 4), "cos_base": round(float(np.mean([r["cos_base"] for r in R])), 4),
                                "near_cv": round(float(np.median([r["near_cv"] for r in R])), 4), "per0": round(float(np.median([r["per0"] for r in R])), 3), "rate": round(float(np.median([r["rate"] for r in R])), 2)}
    g0 = summ.get("g0_c1")
    verdict = {}
    if g0:
        for k, s in summ.items():
            if k == "g0_c1":
                continue
            verdict[k] = {"Pa_pitch": bool(s["abs_d_st_median_ref_ge95"] is not None and s["abs_d_st_median_ref_ge95"] <= 2.0), "Pa_content": bool(s["cer_median"] <= g0["cer_median"] + 0.03),
                          "Pa_natural": bool(s["utmos"] >= g0["utmos"] - 0.20), "Pa_consist": bool(s["consist"] >= g0["consist"] - 0.05)}
    if any(r["type"] == "w" for r in rows.values()):
        W = [r for r in rows.values() if r["type"] == "w" and np.isfinite(r["vtl_lpc"])]
        G = {r["base"]: r["vtl_lpc"] for r in rows.values() if r["type"] == "g0"}
        pts = [(np.log(r["alpha_v"]), np.log(r["vtl_lpc"] / G[r["base"]])) for r in W if r["base"] in G and np.isfinite(G[r["base"]])]
        if len(pts) >= 6:
            verdict["Pa_vtl_slope_lpc(記録・物差しは較正を通っていない)"] = round(float(np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)[0]), 3)
    print(json.dumps(summ, ensure_ascii=False, indent=1), "\n", json.dumps(verdict, ensure_ascii=False, indent=1), flush=True)
    out.write_text(json.dumps({"prereg": "results/snv_d5/prereg.yaml", "summary": summ, "verdict": verdict, "rows": rows}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
