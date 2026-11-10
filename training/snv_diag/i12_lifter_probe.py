"""段 I12 の後の切り分け(学習なし): 推論の E の入力の包絡を低い次数だけに絞ったとき、(1) 試験の伴奏の区間が学習の族(rec2)の入力の分布に入るか、(2) 内容が残るか。
(1) 覆い = 有声のフレームの 1 近傍の距離の中央値(参照 = rec2 の学習の入力 400 本の版 + 元の音 130 本・i10_real_decomp mode=ood と同じ)を、包絡の次数 K と周期性 4・有声 1 で。
(2) 内容 = 写しの描画(描画器 rvoc3hi)で包絡の K 次から上を学習の参照の平均に置き換えた音の large-v3 の仮名の書き起こしを、置き換えない写しの書き起こしと比べた CER(開発 = 検証の Irodori 24 本・JVNV の M1・M2 12 本・参考 = 試験の 12 区間)。
    cd training && uv run --no-sync python snv_diag/i12_lifter_probe.py [ks=25,20,16,13,10] [out=results/snv_i12/lifter_probe.json]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i10_real_decomp as D
import i10_real_ood as O
import nvoc as N
import snv_p1_eval as EV
import train_rvoc as TR
import train_snv_p1 as T


def reference() -> list[dict]:
    rng = random.Random(0)
    tr = [p for v in D.split()["train"] for p in sorted((T.FEAT / "e1" / v).glob("t*.npz"))]
    pick = rng.sample(tr, 400)
    return [T.load_utt(D.REC / p.relative_to(T.FEAT).with_name(f"{p.stem}_k{i % 2}.npz")) for i, p in enumerate(pick)] + [T.load_utt(p) for p in pick[:130]]


def coverage(ref: list[dict], groups: dict, ks: list[int]) -> dict:
    out = {}
    for K in ks:
        dims = list(range(K)) + [25, 26, 27, 28, 29]
        R = torch.from_numpy(np.concatenate([O.xin_np(u)[:, dims][u["v"] > 0.5] for u in ref])).to(O.DEV)
        mu, sd = R.mean(0), R.std(0) + 1e-3
        Rn = (R - mu) / sd
        res = {}
        for g, us in groups.items():
            ds = [float(O.nn_dist((torch.from_numpy(O.xin_np(u)[:, dims][u["v"] > 0.5]).to(O.DEV) - mu) / sd, Rn).median()) for u in us]
            res[g] = round(float(np.median(ds)), 3)
        out[str(K)] = res
        print("coverage K", K, res, flush=True)
    return out


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    ks = [int(k) for k in opt.get("ks", "25,20,16,13,10").split(",")]
    D.REC = D.ROOT / opt.get("rec", "data/snv_p_feat_rec2")
    ref = reference()
    G = D.groups()
    real = [f"uirou_{i}" for i in range(6)] + [f"namikawa_{i}" for i in range(6)]
    groups = {"irodori_val": G["irodori_val"], "irodori_val_rec_k0": G["irodori_val_rec_k0"], "jvnv_M": G["jvnv_M"]} | {n: G[n] for n in real}
    res = {"coverage": coverage(ref, groups, ks)}
    ymean = np.concatenate([u["y"][:25] for u in ref], 1).mean(1)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, kana, to16
    asr = WhisperModel("large-v3", device=EV.DEV, compute_type="float16" if EV.DEV == "cuda" else "int8")
    tr = lambda w: kana("".join(x.text for x in asr.transcribe(to16(w.astype(np.float64), EV.SR).astype(np.float32), language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)[0]))
    gen = EV.load_gen()

    @torch.no_grad()
    def copy(u: dict, K: int) -> np.ndarray:
        yy = u["y"].copy()
        yy[K:25] = ymean[K:25, None]
        y = torch.from_numpy(yy)[None].to(EV.DEV)
        lf = torch.from_numpy(u["lf"])[None].to(EV.DEV)
        v = torch.from_numpy(u["v"])[None].to(EV.DEV)
        cond = torch.cat([y[:, :25], lf[:, None], v[:, None], y[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
        f0 = torch.from_numpy(u["f0"])[None].to(EV.DEV)
        return gen(cond, TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:].astype(np.float32)

    jv = sorted((T.FEAT / "jvnv").glob("M*_regular_0[1-6].npz"))[:12]
    sets = {"irodori_val": G["irodori_val"], "jvnv": [T.load_utt(p) for p in jv], "test_real": [G[n][0] for n in real]}
    names = {"irodori_val": [f"val{i}" for i in range(len(G["irodori_val"]))], "jvnv": [p.stem for p in jv], "test_real": real}
    rows = []
    for s, us in sets.items():
        for name, u in zip(names[s], us):
            ref_txt = tr(copy(u, 25))
            for K in ks:
                out = ref_txt if K == 25 else tr(copy(u, K))
                rows.append({"set": s, "name": name, "K": K, "cer": cer(ref_txt, out), "ref": ref_txt[:50], "out": out[:50]})
        for K in ks:
            c = [r["cer"] for r in rows if r["set"] == s and r["K"] == K]
            print("content", s, "K", K, "cer median", round(float(np.median(c)), 4), "mean", round(float(np.mean(c)), 4), flush=True)
    res["content"] = {s: {str(K): {"median": round(float(np.median([r["cer"] for r in rows if r["set"] == s and r["K"] == K])), 4),
                                   "mean": round(float(np.mean([r["cer"] for r in rows if r["set"] == s and r["K"] == K])), 4)} for K in ks} for s in sets}
    res["rows"] = rows
    (D.ROOT / opt.get("out", "results/snv_i12/lifter_probe.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
