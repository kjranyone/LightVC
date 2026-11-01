"""E0 P2 の測定(results/snv_e0/prereg.yaml の p2): t* ごとの時刻の結合(A1)・X の高さ(M_pitch)・声(M_voice・ECAPA)・Y からの漏れ。
基準 = P1 の XI(本当の男声の生成・同じ文・同じ長さ)。
    uv run python snv_e0_p2_measure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soxr
import torch
from speechbrain.inference.speaker import EncoderClassifier

import c1_content as CC
import nvoc as N
import snv_e0_measure as M

P2 = M.OUT / "wav_p2"
PAIR = M.ROOT / "data/pair_tts_v4l"
KEYS = ("XP50", "XP70", "XP85")


def main() -> int:
    dev = "cuda"
    st1 = torch.load(M.ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1n = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev)
    c1n.load_state_dict(st1["net"])
    c1n.eval().requires_grad_(False)
    mfc = CC.MelFront().to(dev)
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="hf_models/spkrec-ecapa", run_opts={"device": dev})

    @torch.no_grad()
    def emb(x48: np.ndarray) -> np.ndarray:
        x = soxr.resample(x48, N.SR, 16000).astype(np.float32)
        e = ec.encode_batch(torch.from_numpy(x)[None].to(dev)).squeeze().cpu().numpy()
        return e / np.linalg.norm(e)

    its = json.loads((P2 / "items.json").read_text())
    rows = []
    for it in its:
        p = P2 / f"i{it['item']:03d}"
        if not Path(f"{p}_XP85.wav").exists():
            continue
        y = M.load(Path(f"{p}_Y.wav"))
        y1 = M.load(M.WAV / f"i{it['item']:03d}_Y.wav")
        xi = M.load(M.WAV / f"i{it['item']:03d}_XI.wav")
        xs = {k: M.load(Path(f"{p}_{k}.wav")) for k in KEYS}
        em, ef, ey = emb(M.load(PAIR / it["m"] / "t00.wav")), emb(M.load(PAIR / it["f"] / "t00.wav")), emb(y)
        qy = M.post(y, c1n, mfc, dev)
        r = {"item": it["item"], "style": it["style"] or "plain", "y_same_as_p1": float(np.abs(y - y1[: len(y)]).max()) if len(y1) == len(y) else -1.0}
        for k, x in {**xs, "XI": xi}.items():
            s = M.pair_stats(y, x, qy, M.post(x, c1n, mfc, dev))
            e = emb(x)
            s.update({"m_margin": float(e @ em - e @ ef), "cos_y": float(e @ ey)})
            r[k] = s
        rows.append(r)
    keys = (*KEYS, "XI")
    g = lambda rs, k, f: float(np.nanmedian([r[k][f] for r in rs]))
    summ = {}
    for st in ("all", *sorted({r["style"] for r in rows})):
        rs = rows if st == "all" else [r for r in rows if r["style"] == st]
        summ[st] = {k: {**M.pool(rs, k), "m_margin_med": g(rs, k, "m_margin"), "cos_y_med": g(rs, k, "cos_y")} for k in keys}
    a = summ["all"]
    xi = a["XI"]
    verdict = {}
    for k in KEYS:
        v = a[k]
        verdict[k] = {"A1": v["frac_lag_ok"] >= 0.9, "M_pitch": abs(12 * np.log2(v["f0x_med"] / xi["f0x_med"])) <= 2.0,
                      "M_voice": v["m_margin_med"] >= 0.8 * xi["m_margin_med"], "leak": v["cos_y_med"] <= xi["cos_y_med"] + 0.10}
        verdict[k]["all"] = all(verdict[k].values())
    out = {"n_items": len(rows), "summary": summ, "verdict": verdict,
           "PASS_tstar": next((k for k in KEYS if verdict[k]["all"]), None)}
    (M.OUT / "e0_p2_measure.json").write_text(json.dumps({**out, "rows": rows}, ensure_ascii=False, indent=1, default=float))
    print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
