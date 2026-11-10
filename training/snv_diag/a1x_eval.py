"""diag_snv_a1x の評価(results/diag_snv_a1x/prereg.yaml): 演奏 = 源の SPARC の EMA(物差しとしてだけ)→ C(目標の参照の構造・高さ = 源の高さ + 固定の k 半音)→ rvoc3hi。
M = 本物の男声 12 区間 → A0F 24 声 × k ∈ {0, 5, 7, 12}・ECAPA + LDA の 24 声中の 1 位(snv_p1_eval の test と同じ数え方)・g = (M − 0.061) / 0.574。
FF = A0F の j3 の EMA → 別の声の参照(72 組・k 0)の src_FF・M_FF(snv_diag/p3_ff_probe と同じ組)。CER = k 0 / 7 × 目標 4 声の large-v3 の仮名の CER(記録)。
    cd training && uv run --no-sync python snv_diag/a1x_eval.py [ckpt=results/diag_snv_a1x/last.pt] [out=results/diag_snv_a1x/eval.json]
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
from snv_diag.a1x_train import ema_of

ROOT = EV.ROOT
DEV = EV.DEV


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    st = torch.load(ROOT / opt.get("ckpt", "results/diag_snv_a1x/last.pt"), map_location="cpu", weights_only=False)
    C = T.Circ().to(DEV).eval()
    C.load_state_dict(st["C"])
    norm = (np.asarray(st["ema_norm"][0]), np.asarray(st["ema_norm"][1]))
    from snv_a0_measure import cer, kana, load, to16
    from snv_a0b_measure import proj
    from snv_a4_eval import cluster_ci
    from snv_n0_measure import trim
    from speechbrain.inference.speaker import EncoderClassifier
    gen = EV.load_gen()
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})
    Tz = np.load(ROOT / "data/snv_e1_feat/ecapa_lda_T.npz")
    Tt = (Tz["mu"], Tz["scalings"])

    def emb(w16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(w16).float()[None].to(DEV)).reshape(-1)
        return proj(torch.nn.functional.normalize(e, dim=0).cpu().numpy(), Tt)

    w16 = lambda w: to16(w.astype(np.float64), EV.SR).astype(np.float32)
    lw = lambda p: to16(*load(p)).astype(np.float32)
    meta = json.loads((EV.A1 / "convert_meta.json").read_text())
    tgts, srcs = meta["targets"], meta["sources"]
    Et = {v: emb(np.concatenate([lw(EV.A0 / f"wav/{v}_{j}.wav") for j in (0, 1, 2)])) for v in tgts}
    top = lambda e: max(tgts, key=lambda u: float(e @ Et[u]))
    tt = lambda x: torch.from_numpy(x)[None].to(DEV)
    S_cache: dict = {}

    @torch.no_grad()
    def structure(v: str) -> dict:
        if v not in S_cache:
            us = [(f"a0/{v}_{j}", EV.utt(f"a0/{v}_{j}")) for j in (0, 1, 2)]
            ry = np.concatenate([u["y"] for _, u in us], 1)
            ra = np.concatenate([ema_of(r, u["y"].shape[1], norm) for r, u in us], 1)
            S_cache[v] = C.structure(tt(ry), tt(ra), torch.ones((1, ry.shape[1]), dtype=torch.bool, device=DEV))
        return S_cache[v]

    @torch.no_grad()
    def convert(rel: str, v: str, k: int) -> np.ndarray:
        u = EV.utt(rel)
        n = u["y"].shape[1]
        vv = tt(u["v"])
        lf = tt(u["lf"]) + k * T.LN2_12 * vv
        yo = C(tt(ema_of(rel, n, norm)), lf, vv, structure(v))
        cond = torch.cat([yo[:, :25], lf[:, None], vv[:, None], yo[:, 25:29].clamp(0, 1.5) * vv[:, None]], 1)
        f0 = tt(u["f0"] * 2 ** (k / 12))
        return gen(cond, TR.excitation(f0, n * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:].astype(np.float32)

    res: dict = {"prereg": "results/diag_snv_a1x/prereg.yaml", "step": st["step"]}
    for k in (0, 5, 7, 12):
        hit, grp = [], []
        for s_ in srcs:
            for v in tgts:
                hit.append(top(emb(trim(w16(convert(f"real/{s_}", v, k))))) == v), grp.append(v)
        res[f"M_k{k}"] = round(float(np.mean(hit)), 4)
        print(f"M_k{k}", res[f"M_k{k}"], flush=True)
    res["M_mean_4k"] = round(float(np.mean([res[f"M_k{k}"] for k in (0, 5, 7, 12)])), 4)
    res["g"] = round((res["M_mean_4k"] - 0.061) / 0.574, 4)
    pairs = [(tgts[(i + d) % len(tgts)], v) for i, v in enumerate(tgts) for d in (1, 8, 15)]
    hit, src = [], []
    for u_, v in pairs:
        t1 = top(emb(trim(w16(convert(f"a0/{u_}_3", v, 0)))))
        hit.append(t1 == v), src.append(t1 == u_)
    res["FF"] = {"M_FF": round(float(np.mean(hit)), 4), "src_FF": round(float(np.mean(src)), 4)}
    print("FF", res["FF"], flush=True)
    from faster_whisper import WhisperModel
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16" if DEV == "cuda" else "int8")

    def tr(w: np.ndarray) -> str:
        segs, _ = asr.transcribe(w, language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(x.text for x in segs))

    ref_txt = {s_: tr(lw(EV.FEAT / f"real/{s_}.wav")) for s_ in srcs}
    res["cer_k0_k7"] = round(float(np.mean([cer(ref_txt[s_], tr(w16(convert(f"real/{s_}", v, k)))) for k in (0, 7) for s_ in srcs for v in tgts[:4]])), 4)
    res["P_a1"] = bool(res["g"] >= 0.25 and res["FF"]["src_FF"] <= 0.3)
    print(json.dumps(res), flush=True)
    (ROOT / opt.get("out", "results/diag_snv_a1x/eval.json")).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
