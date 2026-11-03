"""N1(diag・学習なし): 参照に無い演奏への外挿。A2 の G を 2 つ(g = 静的な 1 本のベクトルの連続の回路・x = 参照を調音の状態で読む表)そのまま使い、
N0r の男 12 声の発声(7 種 × 2 本・N0c の分類器の学習に使っていない)を目標の女 24 声(参照 = A0 の普通の発話の登録だけ = 参照に叫び・喘ぎは無い)へ描画して比べる。
事前登録 = results/diag_snv_n1/prereg.yaml。演奏 = SPARC・描画 = BigVGAN の代役(非因果)なので diag。
条件: IN = 入力そのもの・V = 入力の mel の BigVGAN の写し合成・S_<m> = 本人の登録の構造で描画(+0 半音)・M_<m> = 目標 6 声へ(+7 半音・オーナーの固定の半音)。
    cd training && uv run --no-sync python snv_n1_eval.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from snv_a0_measure import DEV, ROOT, load, to16
from snv_a0b_measure import proj
from snv_a2_eval import vocoder
from snv_n0_measure import cat_feat, hf_share, trim
from snv_n0c_gen import CATS
from train_snv_a2 import FEAT, G, playing

OUT = ROOT / "results/diag_snv_n1"
N0R = ROOT / "results/snv_n0r/wav"
N0C = ROOT / "results/snv_n0c/wav"
A0 = ROOT / "results/snv_a0"
TAGS = {"g": "diag_snv_a2_g", "x": "diag_snv_a2_x"}
SEMI = 7
N_TGT = 6


def cluster_ci(d: np.ndarray, groups: np.ndarray, n: int = 4000) -> list:
    rng = np.random.default_rng(0)
    gs = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in gs}
    bs = [np.mean(np.concatenate([d[idx[g]] for g in rng.choice(gs, len(gs))])) for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "wav").mkdir(exist_ok=True)
    tgts = [f"pf{k:03d}" for k in range(24)]
    its = [it for it in json.loads((N0R / "items.json").read_text()) if it["sex"] == "M"]
    names = [f"{it['voice']}_{it['cat']}_{it['k']}" for it in its]
    cat_of = {n: it["cat"] for n, it in zip(names, its)}
    voice_of = {n: it["voice"] for n, it in zip(names, its)}
    pairs = {n: [tgts[(i + 4 * j) % 24] for j in range(N_TGT)] for i, n in enumerate(names)}

    voc = vocoder()

    def render(mel: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return voc(mel.to(DEV)).squeeze().cpu().numpy()

    wavs: dict = {}
    for n in names:
        wavs[("IN", n, None)] = load(N0R / f"{n}.wav")
        mel = np.load(FEAT / f"mel/n0r/{n}.npy").astype(np.float32)
        wavs[("V", n, None)] = (render(torch.from_numpy(mel)[None]), 44100)
    ents = defaultdict(list)
    for m, tag in TAGS.items():
        d = ROOT / "results" / tag
        ck = torch.load(d / "last.pt", map_location="cpu", weights_only=True)
        g = G(ck["mode"]).to(DEV).eval()
        g.load_state_dict(ck["g"])
        nz = np.load(d / "norm.npz")
        mu, sd = nz["mu"], nz["sd"]
        S = {}
        with torch.no_grad():
            for v in tgts + sorted(set(voice_of.values())):
                rmel = np.load(FEAT / f"mel/a0enroll/{v}.npy").astype(np.float32)
                sp = FEAT / (f"sparc/a1src/tref_{v}.npz" if v.startswith("pf") else f"sparc/a0enroll/{v}.npz")
                rp = (playing(sp, rmel.shape[1]) - mu) / sd
                S[v] = g.structure(torch.from_numpy(rmel)[None].to(DEV), torch.from_numpy(rp)[None].to(DEV), torch.ones(1, rmel.shape[1], dtype=torch.bool, device=DEV))
            for n in names:
                n_mel = np.load(FEAT / f"mel/n0r/{n}.npy").shape[1]
                p = playing(FEAT / f"sparc/n0r/{n}.npz", n_mel)
                for cond, v, semi in [("S", voice_of[n], 0)] + [("M", t, SEMI) for t in pairs[n]]:
                    q = p.copy()
                    q[:, 12] += semi * np.log(2) / 12
                    y, att = g(torch.from_numpy((q - mu) / sd)[None].to(DEV), S[v], return_attn=True)
                    if att is not None:
                        ents[(cond, "mixed" if cat_of[n] == "mixed" else "nv")].append(float((-(att.clamp(min=1e-9).log() * att).sum(-1)).mean()))
                    wavs[(f"{cond}_{m}", n, v)] = (render(y), 44100)
        del g
        torch.cuda.empty_cache()
    del voc
    torch.cuda.empty_cache()
    for k, (w, sr) in wavs.items():
        if k[0].startswith("M_") and voice_of[k[1]] in ("pm000", "pm001") and k[2] in pairs[k[1]][:2]:
            sf.write(OUT / f"wav/{k[0]}__{k[1]}__{k[2]}.wav", w, sr)
        if k[0] in ("V", "S_g", "S_x") and voice_of[k[1]] == "pm000":
            sf.write(OUT / f"wav/{k[0]}__{k[1]}.wav", w, sr)
    print("rendered", len(wavs), flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from speechbrain.inference.speaker import EncoderClassifier
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})

    def ecapa(y: np.ndarray, sr: int) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(to16(y, sr)).float()[None].to(DEV)).reshape(-1)
        return torch.nn.functional.normalize(e, dim=0).cpu().numpy()

    X1, X2, yl = [], [], []
    for it in json.loads((N0C / "items.json").read_text()):
        w, sr = load(N0C / f"{it['voice']}_{it['cat']}_{it['t']}_{it['k']}.wav")
        w = trim(w)
        f = cat_feat(to16(w, sr))
        X1.append(f)
        X2.append(np.concatenate([f, ecapa(w, sr)]))
        yl.append(CATS.index(it["cat"]))
    mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
    c1, c2 = mk().fit(np.stack(X1), yl), mk().fit(np.stack(X2), yl)
    T = np.load(FEAT / "ecapa_lda_T.npz")
    Tt = (T["mu"], T["scalings"])
    enr = {}
    for v in tgts + sorted(set(voice_of.values())):
        ys = [load(A0 / f"wav/{v}_{j}.wav") for j in (0, 1, 2)]
        enr[v] = proj(ecapa(np.concatenate([y for y, _ in ys]), ys[0][1]), Tt)
    Mt = np.stack([enr[v] for v in tgts])
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(DEV).eval()
    rec = {}
    for k, (w, sr) in wavs.items():
        wt = trim(np.asarray(w, dtype=np.float32))
        w16 = to16(wt, sr)
        f = cat_feat(w16)
        e = ecapa(wt, sr)
        el = proj(e, Tt)
        with torch.no_grad():
            mos = float(utmos(torch.from_numpy(to16(np.asarray(w, dtype=np.float32), sr)).float()[None].to(DEV), 16000).item())
        r = {"cat": cat_of[k[1]], "voice": voice_of[k[1]], "mos": mos, "hf": hf_share(wt.astype(np.float64), sr)}
        if r["cat"] in CATS:
            r["keep1"] = bool(c1.predict(f[None])[0] == CATS.index(r["cat"]))
            r["keep2"] = bool(c2.predict(np.concatenate([f, e])[None])[0] == CATS.index(r["cat"]))
        if k[0].startswith("M_"):
            s = Mt @ el
            r["top1"] = bool(tgts[int(np.argmax(s))] == k[2])
            r["move"] = float(el @ enr[k[2]] - el @ enr[voice_of[k[1]]])
        rec[k] = r
    del ec, utmos
    torch.cuda.empty_cache()

    res: dict = {"prereg": "results/diag_snv_n1/prereg.yaml", "n_inputs": len(names), "targets_per_input": N_TGT, "semitones": SEMI}
    conds = ["IN", "V", "S_g", "S_x", "M_g", "M_x"]
    for c in conds:
        ks = [k for k in rec if k[0] == c]
        nv = [k for k in ks if rec[k]["cat"] in CATS]
        row = {"n": len(ks), "CAT_keep": round(float(np.mean([rec[k]["keep1"] for k in nv])), 3), "CAT_keep_secondary": round(float(np.mean([rec[k]["keep2"] for k in nv])), 3),
               "per_cat": {cat: round(float(np.mean([rec[k]["keep1"] for k in nv if rec[k]["cat"] == cat])), 3) for cat in CATS},
               "utmos": round(float(np.mean([rec[k]["mos"] for k in ks])), 3), "hf_median": round(float(np.median([rec[k]["hf"] for k in ks])), 2)}
        if c.startswith("M_"):
            for part, sel in (("mixed", lambda k: rec[k]["cat"] == "mixed"), ("nv", lambda k: rec[k]["cat"] != "mixed")):
                kk = [k for k in ks if sel(k)]
                row[f"top1_{part}"] = round(float(np.mean([rec[k]["top1"] for k in kk])), 3)
                row[f"move_pos_{part}"] = round(float(np.mean([rec[k]["move"] > 0 for k in kk])), 3)
                row[f"move_mean_{part}"] = round(float(np.mean([rec[k]["move"] for k in kk])), 3)
        res[c] = row
        print(c, row, flush=True)

    def paired(a: str, b: str, key: str, nv_only: bool) -> dict:
        ks = sorted(k for k in rec if k[0] == a and (not nv_only or rec[k]["cat"] in CATS))
        d = np.array([float(rec[k][key]) - float(rec[(b, k[1], k[2])][key]) for k in ks])
        grp = np.array([voice_of[k[1]] for k in ks])
        return {"mean": round(float(d.mean()), 3), "ci95_voice_cluster": cluster_ci(d, grp), "n": len(ks)}

    res["diff_g_minus_x"] = {
        "M_CAT_keep": paired("M_g", "M_x", "keep1", True),
        "M_CAT_keep_secondary": paired("M_g", "M_x", "keep2", True),
        "S_CAT_keep": paired("S_g", "S_x", "keep1", True),
        "M_utmos": paired("M_g", "M_x", "mos", False),
        "M_top1": paired("M_g", "M_x", "top1", False),
    }
    gx = [k for k in rec if k[0] == "M_g"]
    res["diff_g_minus_x"]["M_utmos_win_g"] = round(float(np.mean([rec[k]["mos"] > rec[("M_x", k[1], k[2])]["mos"] for k in gx])), 3)
    mk_ = [k for k in gx if rec[k]["cat"] == "mixed"]
    dmix = np.array([float(rec[k]["top1"]) - float(rec[("M_x", k[1], k[2])]["top1"]) for k in mk_])
    res["diff_g_minus_x"]["M_top1_mixed"] = {"mean": round(float(dmix.mean()), 3), "ci95_voice_cluster": cluster_ci(dmix, np.array([voice_of[k[1]] for k in mk_])), "n": len(mk_)}
    res["x_attention_entropy"] = {f"{c}_{p}": round(float(np.mean(v)), 3) for (c, p), v in ents.items()}
    print("diff g - x", res["diff_g_minus_x"], flush=True)
    print("attention", res["x_attention_entropy"], flush=True)
    (OUT / "n1_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
