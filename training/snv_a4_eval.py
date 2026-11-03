"""A4(diag)の評価: 連続の回路 c(登録で層ごとの回路の値 Φ を当てはめる)と表 x を、段 1(L1)と段 2(分布の損失)で読む。事前登録 = results/diag_snv_a4/prereg.yaml。
A = 普通の発話の本人らしさ(S・O・M・CER)・B = 自己の reamp の音素の無い発声(A0 の女 24 声の N0r 336 本・構造は普通の発話の登録だけ)の TEX・3〜12 kHz・種類・相対の本人らしさ・
C = N1 の手順(N0r の男 168 本 → 目標 6 声・+7 半音)・D = 本物の男声の音素の無い発声(JNV M1・M2 → 目標 6 声・+7 半音)。C と D の出力は N2b の点のために 16 kHz で書き出す。
    cd training && uv run --no-sync python snv_a4_eval.py <出力の 16 kHz の npz の dir> [smoke]
smoke = 段 1 の重みを段 2 の名前にも当て、各部を数本に絞って評価の経路だけを通す(結果は書かない)。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from snv_a0_measure import DEV, ROOT, cer, kana, load, to16
from snv_a0b_measure import proj
from snv_a2_eval import A1, map_pitch, vocoder
from snv_n0_measure import cat_feat, hf_share, trim
from snv_n0c_gen import CATS
from snv_n2c_tex import tex_dist, tex_stats
from train_snv_a2 import FEAT, G, playing
from train_snv_a4 import FIT_STEPS, C, fit_phi

OUT = ROOT / "results/diag_snv_a4"
A0 = ROOT / "results/snv_a0"
N0R = ROOT / "results/snv_n0r/wav"
N0C = ROOT / "results/snv_n0c/wav"
SYS = {"c1": "diag_snv_a4_c", "x1": "diag_snv_a4_x", "c2": "diag_snv_a4_c_gan", "x2": "diag_snv_a4_x_gan"}
SEMI = 7
N_TGT = 6


def cluster_ci(d: np.ndarray, groups: np.ndarray, n: int = 4000) -> list:
    rng = np.random.default_rng(0)
    gs = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in gs}
    bs = [np.mean(np.concatenate([d[idx[g]] for g in rng.choice(gs, len(gs))])) for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def main() -> int:
    dump = Path(sys.argv[1])
    smoke = len(sys.argv) > 2 and sys.argv[2] == "smoke"
    if smoke:
        SYS.update({"c2": SYS["c1"], "x2": SYS["x1"]})
    dump.mkdir(parents=True, exist_ok=True)
    (OUT / "wav").mkdir(parents=True, exist_ok=True)
    meta = json.loads((A1 / "convert_meta.json").read_text())
    srcs, tgts = meta["sources"], meta["targets"]
    drows = json.loads((A1 / "diag/meta.json").read_text())
    tstats = {v: np.load(FEAT / f"sparc/a1src/tref_{v}.npz")["pitch_stats"].reshape(-1)[:2] for v in tgts}
    n0r = json.loads((N0R / "items.json").read_text())
    nv_self = [f"{it['voice']}_{it['cat']}_{it['k']}" for it in n0r if it["sex"] == "F"]
    nv_male = [f"{it['voice']}_{it['cat']}_{it['k']}" for it in n0r if it["sex"] == "M"]
    cat_of = {f"{it['voice']}_{it['cat']}_{it['k']}": it["cat"] for it in n0r}
    jnvm = sorted(p.stem for p in (FEAT / "sparc/jnvm").glob("*.npz"))
    if smoke:
        nv_self, nv_male, jnvm, drows, srcs = nv_self[:: 42], [n for n in nv_male if n.startswith("pm000")][:7], jnvm[:3], drows[:4], srcs[:2]
    pairs_m = {n: [tgts[(i + 4 * j) % 24] for j in range(N_TGT)] for i, n in enumerate(nv_male)}
    pairs_j = {n: [tgts[(i + 4 * j) % 24] for j in range(N_TGT)] for i, n in enumerate(jnvm)}
    voc = vocoder()

    def render(mel: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return voc(mel.to(DEV)).squeeze().cpu().numpy()

    for r in drows[:3]:
        mel = np.load(FEAT / f"mel/a1diag/D0_{r['target']}_{r['g']}.npy").astype(np.float32)
        w = render(torch.from_numpy(mel)[None])
        orig = librosa.load(str(FEAT / f"wav44/a1diag__D0_{r['target']}_{r['g']}.wav"), sr=44100)[0]
        assert abs(10 * np.log10(np.mean(w ** 2) / np.mean(orig ** 2))) < 3.0, "L-6: 写し合成の音量が元とずれる"
    print("L-6 ok", flush=True)

    truth, vcopy = {}, {}
    for n in nv_self:
        truth[n] = librosa.load(str(FEAT / f"wav44/n0r__{n}.wav"), sr=44100)[0]
        vcopy[n] = render(torch.from_numpy(np.load(FEAT / f"mel/n0r/{n}.npy").astype(np.float32))[None])

    out16: dict = {}
    mel_l1: dict = {}
    hf: dict = {}
    fitlog: dict = {}
    keep44: dict = {}
    for sname, tag in SYS.items():
        d = ROOT / "results" / tag
        ck = torch.load(d / "last.pt", map_location="cpu", weights_only=True)
        net = (C() if ck["mode"] == "c" else G("x")).to(DEV).eval()
        net.load_state_dict(ck["g"])
        nz = np.load(d / "norm.npz")
        mu, sd = nz["mu"], nz["sd"]
        S = {}
        for v in tgts:
            rmel = torch.from_numpy(np.load(FEAT / f"mel/a0enroll/{v}.npy").astype(np.float32))[None].to(DEV)
            rp = torch.from_numpy((playing(FEAT / f"sparc/a1src/tref_{v}.npz", rmel.shape[2]) - mu) / sd)[None].to(DEV)
            mask = torch.ones(1, rmel.shape[2], dtype=torch.bool, device=DEV)
            if ck["mode"] == "c":
                S[v] = fit_phi(net, rp, rmel, mask, steps=FIT_STEPS)
                with torch.no_grad():
                    fitlog[f"{sname}|{v}"] = round(float((net(rp, S[v])[..., 86:] - rmel[..., 86:]).abs().mean()), 4)
            else:
                with torch.no_grad():
                    S[v] = net.structure(rmel, rp, mask)
                    fitlog[f"{sname}|{v}"] = round(float((net(rp, S[v])[..., 86:] - rmel[..., 86:]).abs().mean()), 4)

        @torch.no_grad()
        def run(p: np.ndarray, v: str) -> torch.Tensor:
            return net(torch.from_numpy((p - mu) / sd)[None].to(DEV), S[v])

        for r in drows:
            n_mel = np.load(FEAT / f"mel/a1diag/D0_{r['target']}_{r['g']}.npy").shape[1]
            out16[(f"S_{sname}", r["target"], r["g"])] = to16(render(run(playing(FEAT / f"sparc/a1diag/D0_{r['target']}_{r['g']}.npz", n_mel), r["target"])), 44100)
            o = FEAT / f"sparc/a1diag/src_{r['d2_src']}_{r['g']}.npz"
            out16[(f"O_{sname}", r["target"], r["g"])] = to16(render(run(map_pitch(playing(o, n_mel), np.load(o)["pitch_stats"].reshape(-1)[:2], tstats[r["target"]]), r["target"])), 44100)
        for s in srcs:
            f = FEAT / f"sparc/a1src/{s}.npz"
            n_mel = int(10 * 44100 / 512) + 1
            st = np.load(f)["pitch_stats"].reshape(-1)[:2]
            for v in tgts:
                w = render(run(map_pitch(playing(f, n_mel), st, tstats[v]), v))
                out16[(f"M_{sname}", s, v)] = to16(w, 44100)
                if s in srcs[:2] and v in tgts[:2]:
                    keep44[(f"M_{sname}", s, v)] = w
                p7 = playing(f, n_mel)
                p7[:, 12] += SEMI * np.log(2) / 12
                out16[(f"M7_{sname}", s, v)] = to16(render(run(p7, v)), 44100)
        for n in nv_self:
            mel_t = np.load(FEAT / f"mel/n0r/{n}.npy").astype(np.float32)
            y = run(playing(FEAT / f"sparc/n0r/{n}.npz", mel_t.shape[1]), n.split("_")[0])
            mel_l1[(sname, n)] = float(np.mean(np.abs(y[0].cpu().numpy() - mel_t)))
            w = render(y)
            out16[(f"NS_{sname}", n, None)] = to16(w, 44100)
            hf[(f"NS_{sname}", n)] = hf_share(trim(w.astype(np.float64)), 44100)
            if n.split("_")[0] in ("pf000", "pf001") and n.endswith("_0"):
                keep44[(f"NS_{sname}", n, None)] = w
        for n in nv_male:
            mel_t = np.load(FEAT / f"mel/n0r/{n}.npy")
            for v in pairs_m[n]:
                p = playing(FEAT / f"sparc/n0r/{n}.npz", mel_t.shape[1])
                p[:, 12] += SEMI * np.log(2) / 12
                w = render(run(p, v))
                out16[(f"NM_{sname}", n, v)] = to16(w, 44100)
                hf[(f"NM_{sname}", n, v)] = hf_share(trim(w.astype(np.float64)), 44100)
        for n in jnvm:
            mel_t = np.load(FEAT / f"mel/jnvm/{n}.npy")
            for v in pairs_j[n]:
                p = playing(FEAT / f"sparc/jnvm/{n}.npz", mel_t.shape[1])
                p[:, 12] += SEMI * np.log(2) / 12
                w = render(run(p, v))
                out16[(f"NJ_{sname}", n, v)] = to16(w, 44100)
                hf[(f"NJ_{sname}", n, v)] = hf_share(trim(w.astype(np.float64)), 44100)
                if n in jnvm[:2] and v == pairs_j[n][0]:
                    keep44[(f"NJ_{sname}", n, v)] = w
        del net
        torch.cuda.empty_cache()
        print("rendered", sname, len(out16), flush=True)
    del voc
    torch.cuda.empty_cache()
    for k, w in ({} if smoke else keep44).items():
        sf.write(OUT / f"wav/{k[0]}__{k[1]}__{k[2]}.wav", w, 44100)
    for n in ([] if smoke else nv_self[:14]):
        sf.write(OUT / f"wav/truth__{n}.wav", truth[n], 44100)
    for part in ("NM", "NJ"):
        np.savez(dump / f"{part}.npz", **{f"{k[0]}|{k[1]}|{k[2]}": w for k, w in out16.items() if k[0].startswith(part + "_")})

    scale = np.load(ROOT / "results/snv_n2c/tex_scale.npy")
    t16 = {n: to16(truth[n], 44100) for n in nv_self}
    tst = {n: tex_stats(t16[n]) for n in nv_self}
    tex = {(s, n): tex_dist(tex_stats(out16[(f"NS_{s}", n, None)]), tst[n], scale) for s in SYS for n in nv_self}
    tex_v = {n: tex_dist(tex_stats(to16(vcopy[n], 44100)), tst[n], scale) for n in nv_self}
    hf_t = {n: hf_share(trim(truth[n].astype(np.float64)), 44100) for n in nv_self}
    hf_v = {n: hf_share(trim(vcopy[n].astype(np.float64)), 44100) for n in nv_self}

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from speechbrain.inference.speaker import EncoderClassifier
    X, yl = [], []
    for it in json.loads((N0C / "items.json").read_text()):
        w, sr = load(N0C / f"{it['voice']}_{it['cat']}_{it['t']}_{it['k']}.wav")
        X.append(cat_feat(to16(trim(w), sr)))
        yl.append(CATS.index(it["cat"]))
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(np.stack(X), yl)
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})
    T = np.load(FEAT / "ecapa_lda_T.npz")
    Tt = (T["mu"], T["scalings"])

    def E(y16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(y16).float()[None].to(DEV)).reshape(-1)
        return proj(torch.nn.functional.normalize(e, dim=0).cpu().numpy(), Tt)

    Et = {v: E(np.concatenate([to16(*load(A0 / f"wav/{v}_{j}.wav")) for j in (0, 1, 2)])) for v in tgts}
    Et2 = {v: E(np.concatenate([to16(*load(A0 / f"wav/{v}_{j}.wav")) for j in (6, 7)])) for v in tgts}
    Es = {s: E(sf.read(str(A1 / f"src/{s.rsplit('_', 1)[0]}_enroll.wav"))[0].astype(np.float32)) for s in srcs}
    keys_id = [k for k in out16 if k[0].split("_")[0] in ("S", "O", "M", "M7", "NS") or (k[0].startswith("NM_") and cat_of[k[1]] == "mixed")]
    Ew = {k: E(trim(out16[k])) for k in keys_id}
    del ec
    torch.cuda.empty_cache()
    top = lambda e, ref: max(tgts, key=lambda u: float(e @ ref[u]))
    src_of = {(r["target"], r["g"]): r["d2_src"] for r in drows}
    res: dict = {"prereg": "results/diag_snv_a4/prereg.yaml", "fit_enroll_l1": {s: round(float(np.mean([v for k, v in fitlog.items() if k.startswith(s + "|")])), 4) for s in SYS},
                 "n": {"nv_self": len(nv_self), "nv_male": len(nv_male), "jnvm": len(jnvm)}, "A": {}, "B": {}, "C": {}, "D": {}}
    vec: dict = {}
    for s in SYS:
        row = {}
        for part in ("S", "O", "M", "M7"):
            ks = sorted(k for k in out16 if k[0] == f"{part}_{s}")
            tr = (lambda k: k[2]) if part in ("M", "M7") else (lambda k: k[1])
            hit = np.array([top(Ew[k], Et) == tr(k) for k in ks], float)
            vec[(part, s)] = (hit, np.array([tr(k) for k in ks]))
            row[part] = round(float(hit.mean()), 3)
            row[f"{part}_alt_j67"] = round(float(np.mean([top(Ew[k], Et2) == tr(k) for k in ks])), 3)
            if part == "O":
                row["O_read_as_source"] = round(float(np.mean([top(Ew[k], Et) == src_of[(k[1], k[2])] for k in ks])), 3)
            if part == "M":
                row["M_move_mean"] = round(float(np.mean([Ew[k] @ Et[k[2]] - Ew[k] @ Es[k[1]] for k in ks])), 3)
        res["A"][s] = row
        print("A", s, row, flush=True)
    res["A"]["diff_c_minus_x"] = {f"{part}_{st}": {"mean": round(float((vec[(part, 'c' + st)][0] - vec[(part, 'x' + st)][0]).mean()), 3),
                                                   "ci95_voice_cluster": cluster_ci(vec[(part, "c" + st)][0] - vec[(part, "x" + st)][0], vec[(part, "c" + st)][1])}
                                  for part in ("S", "M") for st in ("1", "2")}
    print("A diff", res["A"]["diff_c_minus_x"], flush=True)

    grp = np.array([n.split("_")[0] for n in nv_self])
    for s in SYS:
        tx = np.array([tex[(s, n)] for n in nv_self])
        keep = np.array([clf.predict(cat_feat(trim(out16[(f"NS_{s}", n, None)]))[None])[0] == CATS.index(cat_of[n]) for n in nv_self if cat_of[n] in CATS], float)
        auc = []
        for n in nv_self:
            sc = {u: float(Ew[(f"NS_{s}", n, None)] @ Et[u]) for u in tgts}
            own = n.split("_")[0]
            auc.append(np.mean([sc[own] > sc[u] for u in tgts if u != own]))
        res["B"][s] = {"tex_mean": round(float(tx.mean()), 4), "tex_by_cat": {c: round(float(np.mean([tex[(s, n)] for n in nv_self if cat_of[n] == c])), 4) for c in CATS + ("mixed",)},
                       "hf_ratio_median": round(float(np.median([hf[(f"NS_{s}", n)] / max(hf_t[n], 1e-6) for n in nv_self])), 3),
                       "cat_keep": round(float(keep.mean()), 3), "rel_id_auc": round(float(np.mean(auc)), 3),
                       "mel_l1": round(float(np.mean([mel_l1[(s, n)] for n in nv_self])), 4)}
        print("B", s, res["B"][s], flush=True)
    res["B"]["V_copy"] = {"tex_mean": round(float(np.mean(list(tex_v.values()))), 4), "hf_ratio_median": round(float(np.median([hf_v[n] / max(hf_t[n], 1e-6) for n in nv_self])), 3)}
    res["B"]["truth_hf_median"] = round(float(np.median(list(hf_t.values()))), 3)
    for a_, b_ in (("c2", "x2"), ("c1", "x1"), ("c2", "c1"), ("x2", "x1")):
        dd = np.array([tex[(a_, n)] - tex[(b_, n)] for n in nv_self])
        dl = np.array([mel_l1[(a_, n)] - mel_l1[(b_, n)] for n in nv_self])
        res["B"][f"tex_{a_}_minus_{b_}"] = {"mean": round(float(dd.mean()), 4), "ci95_voice_cluster": cluster_ci(dd, grp),
                                            "by_cat": {c: round(float(dd[[cat_of[n] == c for n in nv_self]].mean()), 4) for c in CATS + ("mixed",)}}
        res["B"][f"mel_l1_{a_}_minus_{b_}"] = {"mean": round(float(dl.mean()), 4), "ci95_voice_cluster": cluster_ci(dl, grp)}
        print("B diff", a_, b_, res["B"][f"tex_{a_}_minus_{b_}"], res["B"][f"mel_l1_{a_}_minus_{b_}"], flush=True)

    for s in SYS:
        ks = [k for k in out16 if k[0] == f"NM_{s}"]
        nv = [k for k in ks if cat_of[k[1]] in CATS]
        mx = [k for k in ks if cat_of[k[1]] == "mixed"]
        res["C"][s] = {"cat_keep": round(float(np.mean([clf.predict(cat_feat(trim(out16[k]))[None])[0] == CATS.index(cat_of[k[1]]) for k in nv])), 3),
                       "hf_median": round(float(np.median([hf[k] for k in ks])), 3), "top1_mixed": round(float(np.mean([top(Ew[k], Et) == k[2] for k in mx])), 3)}
        res["D"][s] = {"hf_median": round(float(np.median([hf[k] for k in out16 if k[0] == f"NJ_{s}"])), 3), "n": sum(1 for k in out16 if k[0] == f"NJ_{s}")}
        print("C", s, res["C"][s], "D", res["D"][s], flush=True)

    from faster_whisper import WhisperModel
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16")

    def tr(w: np.ndarray) -> str:
        segs, _ = asr.transcribe(w, language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(x.text for x in segs))

    ref = {s: tr(sf.read(str(A1 / f"src/{s}.wav"))[0].astype(np.float32)) for s in srcs}
    for s in SYS:
        ks = [k for k in out16 if k[0] == f"M_{s}" and k[2] in tgts[:8]]
        res["A"][s]["M_cer_median_8tgt"] = round(float(np.median([cer(ref[k[1]], tr(out16[k])) for k in ks])), 3)
        print("CER", s, res["A"][s]["M_cer_median_8tgt"], flush=True)
    if not smoke:
        (OUT / "a4_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
