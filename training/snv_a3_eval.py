"""A3(diag)の評価: 自前の因果の演奏の信号 E + A2 の x の G で A0 の目標 24 声へ描画し、A2 と同じ物差しで読む。事前登録 = results/diag_snv_a3_p/prereg.yaml。
V = BigVGAN の写し合成(天井)・S = 自己の reamp・O = 別の女声の演奏 → 目標・M = 本物の男声 → 目標(高さは A2 と同じ平均・標準偏差の写像)・M12 = 本物の男声 → 目標(運用の固定 +12 半音・本人らしさだけ)。
LEAK = E の 12 次元のフレーム単位の探り(A0 の 36 声・声 / 性別・線形と QDA)と、E から log f0 を当てる線形回帰の R²(SPARC の EMA と比べる)。項目ごとの結果も書き出す。
A3 の n 以外の tag は、A3 の項目(results/diag_snv_a3_p/a3_items.json)の n と M を声で束ねて比べ、A3c 以外の tag は A3c の項目とも比べる。
melQDA = 出力の mel を A0 の登録の mel で当てはめた QDA で読む(フレームの分布の物差し・ECAPA との割れが A3c の診断)。
    cd training && uv run --no-sync python snv_a3_eval.py --tags diag_snv_a3_p diag_snv_a3_n
    cd training && uv run --no-sync python snv_a3_eval.py --tags diag_snv_a3b_a1 diag_snv_a3b_a01
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

from snv_a0_measure import DEV, ROOT, Embedders, cer, kana, load, to16
from snv_a0b_measure import fit_lda, proj, splits
from snv_a1_convert import segments
from snv_a2_eval import map_pitch, vocoder
from train_snv_a2 import FEAT, playing
from train_snv_a3 import HOP, SR, Model, envelope

A0 = ROOT / "results/snv_a0"
A1 = ROOT / "results/snv_a1"


def wav44(p: Path, n: int | None = None) -> np.ndarray:
    y = librosa.load(str(p), sr=SR, mono=True)[0]
    if n is not None:
        assert len(y) // HOP == n, f"{p}: 長さ // 512 = {len(y) // HOP} != mel {n}"
    return y[: (len(y) // HOP) * HOP].astype(np.float32)


def male_segments(srcs: list[str]) -> dict:
    out = {}
    for name, p in (("uirou", ROOT / "外郎売.mp3"), ("namikawa", ROOT / "namikawa.mp3")):
        _, tests, sr = segments(p)
        for i, seg in enumerate(tests):
            y = librosa.resample(seg.astype(np.float32), orig_sr=sr, target_sr=SR)
            out[f"{name}_{i}"] = y[: (len(y) // HOP) * HOP]
    assert set(srcs) <= set(out), "A1 の源の区間が作り直せない"
    return out


def cluster_ci(d: np.ndarray, rng: np.random.Generator) -> list[float]:
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    return [float(x) for x in np.percentile(bs, [2.5, 97.5])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    a = ap.parse_args()
    meta = json.loads((A1 / "convert_meta.json").read_text())
    srcs, tgts = meta["sources"], meta["targets"]
    drows = json.loads((A1 / "diag/meta.json").read_text())
    males = male_segments(srcs)
    voc = vocoder()

    def render(mel: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return voc(mel.to(DEV)).squeeze().cpu().numpy()

    out_root = ROOT / "results" / a.tags[0]
    vdir = out_root / "eval_wav"
    vdir.mkdir(parents=True, exist_ok=True)
    wavs: dict = {}
    for r in drows:
        mel = np.load(FEAT / f"mel/a1diag/D0_{r['target']}_{r['g']}.npy").astype(np.float32)
        w = render(torch.from_numpy(mel)[None])
        orig = librosa.load(str(FEAT / f"wav44/a1diag__D0_{r['target']}_{r['g']}.wav"), sr=SR)[0]
        db = 10 * np.log10(np.mean(w ** 2) / np.mean(orig ** 2))
        assert abs(db) < 3.0, f"L-6: 写し合成の音量が元とずれる {db:.2f} dB"
        wavs[("V", r["target"], r["g"])] = w
    print("L-6 ok: BigVGAN の写し合成は元と ±3 dB 以内", flush=True)

    tstats = {v: np.load(FEAT / f"sparc/a1src/tref_{v}.npz")["pitch_stats"].reshape(-1)[:2] for v in tgts}
    its = json.loads((A0 / "wav/items.json").read_text())
    pv = [it for it in its if it["role"] in ("enroll", "test")]
    a0w = {(it["voice"], it["j"]): librosa.load(str(A0 / f"wav/{it['voice']}_{it['j']}.wav"), sr=SR, mono=True)[0] for it in pv}
    latents: dict = {}
    mels: dict = {}
    for tag in a.tags:
        d = ROOT / "results" / tag
        ck = torch.load(d / "last.pt", map_location="cpu", weights_only=True)
        m = Model().to(DEV).eval()
        m.load_state_dict(ck["model"])
        nz = np.load(d / "norm.npz")
        mu, sd = nz["mu"], nz["sd"]
        ents = []
        with torch.no_grad():
            S = {}
            for v in tgts:
                n = np.load(FEAT / f"mel/a0enroll/{v}.npy").shape[1]
                rmel = np.load(FEAT / f"mel/a0enroll/{v}.npy").astype(np.float32)
                rw = wav44(FEAT / f"wav44/a0enroll__{v}.wav", n)
                rs = (playing(FEAT / f"sparc/a1src/tref_{v}.npz", n)[:, 12:15] - mu) / sd
                S[v] = m.structure(torch.from_numpy(rmel)[None].to(DEV), envelope(torch.from_numpy(rw)[None].to(DEV)),
                                   torch.from_numpy(rs)[None].to(DEV), torch.ones(1, n, dtype=torch.bool, device=DEV))

            def run(y: np.ndarray, p_npz: Path, v: str, mode: str | None, key: tuple) -> np.ndarray:
                n = len(y) // HOP
                p = playing(p_npz, n)
                if mode == "z":
                    p = map_pitch(p, np.load(p_npz)["pitch_stats"].reshape(-1)[:2], tstats[v])
                elif mode == "k12":
                    p = p.copy()
                    p[:, 12] += np.log(2.0)
                src = (p[:, 12:15] - mu) / sd
                mel, att = m(envelope(torch.from_numpy(y)[None].to(DEV)), torch.from_numpy(src)[None].to(DEV), S[v], return_attn=True)
                ents.append(float((-(att.clamp(min=1e-9).log() * att).sum(-1)).mean()))
                mels[key] = mel[0].cpu().numpy()
                return render(mel)

            for r in drows:
                v, gi, o = r["target"], r["g"], r["d2_src"]
                n = np.load(FEAT / f"mel/a1diag/D0_{v}_{gi}.npy").shape[1]
                for c, y_, p_, md in (("S", f"D0_{v}_{gi}", f"D0_{v}_{gi}", None), ("O", f"D0_{o}_{gi}", f"src_{o}_{gi}", "z")):
                    k_ = (f"{c}_{tag}", v, gi)
                    wavs[k_] = run(wav44(FEAT / f"wav44/a1diag__{y_}.wav", n), FEAT / f"sparc/a1diag/{p_}.npz", v, md, k_)
            for s in srcs:
                for v in tgts:
                    for c, md in (("M", "z"), ("M12", "k12")):
                        k_ = (f"{c}_{tag}", s, v)
                        wavs[k_] = run(males[s], FEAT / f"sparc/a1src/{s}.npz", v, md, k_)
            for k, y in a0w.items():
                latents[(tag,) + k] = m.e((envelope(torch.from_numpy(y[: (len(y) // HOP) * HOP])[None].to(DEV)) - m.env_mu) / m.env_sd)[0].cpu().numpy()
        print(tag, "attention entropy mean", float(np.mean(ents)), flush=True)
        for k, w in wavs.items():
            if k[0] in (f"M_{tag}", f"M12_{tag}") and k[2] in tgts[:4] and k[1] in (srcs[0], srcs[-1]):
                sf.write(vdir / f"{k[0]}__{k[1]}__{k[2]}.wav", w, SR)
        del m
        torch.cuda.empty_cache()
    del voc
    torch.cuda.empty_cache()
    print("rendered", len(wavs), flush=True)

    sp = splits()
    emb = Embedders()
    X, yy = [], []
    for i, s in enumerate(sp["lda"]):
        for p in sp["pool"][s]:
            X.append(emb(to16(*load(p)))["ecapa"])
            yy.append(i)
    T = fit_lda(np.stack(X), np.array(yy))
    Et = {v: proj(emb(np.concatenate([to16(*load(A0 / f"wav/{v}_{j}.wav")) for j in (0, 1, 2)]))["ecapa"], T) for v in tgts}
    Es = {s: proj(emb(sf.read(str(A1 / f"src/{s.rsplit('_', 1)[0]}_enroll.wav"))[0].astype(np.float32))["ecapa"], T) for s in srcs}
    E = {k: proj(emb(to16(w.astype(np.float64), SR))["ecapa"], T) for k, w in wavs.items()}
    del emb
    torch.cuda.empty_cache()
    best = {k: max(tgts, key=lambda u: float(E[k] @ Et[u])) for k in E}
    res: dict = {"prereg": f"results/{a.tags[0]}/prereg.yaml",
                 "ref_a2x": {"S": 0.750, "O": 0.681, "M": 0.556, "M_cer_median": 0.227, "M_utmos_mean": 1.385, "V": 0.806},
                 "ref_sparc_ema_leak": {"voice36": 0.194, "sex": 0.667, "voice36_qda": 0.204, "sex_qda": 0.667},
                 "ref_a3": {"M_n": 0.108, "M_p": 0.128, "O_read_as_source_n": 0.597, "voice36_n": 0.852, "voice36_qda_p": 0.880},
                 "ref_a3c": {"M": 0.212, "O_read_as_source": 0.292, "M_cer_median": 0.244, "melQDA_O_to_target": 0.958, "melQDA_M_to_target": 0.806}}
    items: dict = {}
    for k in E:
        if k[0] == "V" or k[0].startswith(("S_", "O_")):
            items["|".join(map(str, k))] = best[k] == k[1]
        else:
            items["|".join(map(str, k))] = best[k] == k[2]
    src_of = {(r["target"], r["g"]): r["d2_src"] for r in drows}
    for cond in ["V"] + [f"{c}_{t}" for t in a.tags for c in ("S", "O", "M", "M12")]:
        keys = [k for k in E if k[0] == cond]
        r = {"top1": float(np.mean([items["|".join(map(str, k))] for k in keys])), "n": len(keys)}
        if cond.startswith(("M_", "M12_")):
            ms = np.array([float(E[k] @ Et[k[2]] - E[k] @ Es[k[1]]) for k in keys])
            r.update({"m_pos": float(np.mean(ms > 0)), "m_mean": float(ms.mean())})
        if cond.startswith("O_"):
            r["read_as_source"] = float(np.mean([best[k] == src_of[(k[1], k[2])] for k in keys]))
        res[cond] = r
        print(cond, r, flush=True)

    rng = np.random.default_rng(0)

    def by_voice(tag: str, c: str) -> np.ndarray:
        g: dict = {}
        for k, b in items.items():
            p = k.split("|")
            if p[0] == f"{c}_{tag}":
                v = p[1] if c in ("S", "O") else p[2]
                g.setdefault(v, []).append(float(b))
        return np.array([np.mean(g[v]) for v in sorted(g)])

    a2b = json.loads((ROOT / "results/diag_snv_a2_xs/a2b_items.json").read_text())
    a2x_M: dict = {}
    for k, b in a2b.items():
        p = k.split("|")
        if p[0] == "diag_snv_a2_x" and p[1] == "M":
            a2x_M.setdefault(p[2], []).append(float(b))
    a2x_M_v = np.array([np.mean(a2x_M[v]) for v in sorted(a2x_M)])
    for t in a.tags:
        d = by_voice(t, "M") - a2x_M_v
        res[f"M_{t}_minus_a2x"] = {"diff": float(d.mean()), "ci95_voice": cluster_ci(d, rng)}
        res[f"M_{t}"]["ci95_voice"] = cluster_ci(by_voice(t, "M"), rng)
    for name, path, base in (("a3n", "results/diag_snv_a3_p/a3_items.json", "diag_snv_a3_n"), ("a3c", "results/diag_snv_a3c_w/a3_items.json", "diag_snv_a3c_w")):
        if not (ROOT / path).exists() or base in a.tags:
            continue
        bm: dict = {}
        for k, b in json.loads((ROOT / path).read_text()).items():
            p = k.split("|")
            if p[0] == f"M_{base}":
                bm.setdefault(p[2], []).append(float(b))
        bm_v = np.array([np.mean(bm[v]) for v in sorted(bm)])
        for t in a.tags:
            d = by_voice(t, "M") - bm_v
            res[f"M_{t}_minus_{name}"] = {"diff": float(d.mean()), "ci95_voice": cluster_ci(d, rng)}
    if len(a.tags) == 2:
        for c in ("S", "O", "M"):
            d = by_voice(a.tags[0], c) - by_voice(a.tags[1], c)
            res[f"{c}_{a.tags[0]}_minus_{a.tags[1]}"] = {"diff": float(d.mean()), "ci95_voice": cluster_ci(d, rng)}
    print({k: v for k, v in res.items() if "minus" in k}, flush=True)

    from sklearn.decomposition import PCA
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
    from sklearn.pipeline import make_pipeline as mkp
    from sklearn.preprocessing import StandardScaler as SS
    a0m = {v: np.load(FEAT / f"mel/a0enroll/{v}.npy").astype(np.float32) for v in tgts}
    mq = mkp(SS(), PCA(40), QDA(reg_param=0.05)).fit(np.concatenate([a0m[v].T for v in tgts]), np.concatenate([np.full(a0m[v].shape[1], i) for i, v in enumerate(tgts)]))
    mread = lambda m: tgts[int(mq.classes_[np.argmax(mq.predict_log_proba(m.T).sum(0))])]
    res["melQDA_real_acc"] = float(np.mean([mread(np.load(FEAT / f"mel/a1diag/D0_{r['target']}_{r['g']}.npy").astype(np.float32)) == r["target"] for r in drows]))
    for t in a.tags:
        ko = [k for k in mels if k[0] == f"O_{t}"]
        km = [k for k in mels if k[0] == f"M_{t}"]
        res[f"melQDA_O_{t}"] = {"to_target": float(np.mean([mread(mels[k]) == k[1] for k in ko])), "read_as_source": float(np.mean([mread(mels[k]) == src_of[(k[1], k[2])] for k in ko]))}
        res[f"melQDA_M_{t}"] = {"to_target": float(np.mean([mread(mels[k]) == k[2] for k in km]))}
    print("melQDA", {k: v for k, v in res.items() if k.startswith("melQDA")}, flush=True)

    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    voices = sorted({it["voice"] for it in pv})
    sex = {it["voice"]: it["sex"] for it in pv}
    tr_its = [it for it in pv if it["role"] == "enroll"]
    te_its = [it for it in pv if it["role"] == "test"]

    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    def probe(tag: str, label, qda: bool = False) -> float:
        Xtr = [latents[(tag, it["voice"], it["j"])] for it in tr_its]
        Xa = np.concatenate(Xtr)
        ya = np.concatenate([np.full(len(f), label(it)) for f, it in zip(Xtr, tr_its)])
        c = QuadraticDiscriminantAnalysis(reg_param=0.05) if qda else LogisticRegression(C=1.0, max_iter=3000)
        clf = make_pipeline(StandardScaler(), c).fit(Xa, ya)
        pred = [clf.classes_[np.argmax(clf.predict_log_proba(latents[(tag, it["voice"], it["j"])]).sum(0))] for it in te_its]
        return float(np.mean(np.array(pred) == np.array([label(it) for it in te_its])))

    def f0_xy(it: dict, x: np.ndarray, rate: float) -> tuple[np.ndarray, np.ndarray]:
        f = np.load(A1 / f"feat/{it['voice']}_{it['j']}.npz")
        ts = np.arange(len(f["pitch"])) / 50.0
        td = np.arange(len(x)) / rate
        lf = np.interp(td, ts, np.log(np.maximum(f["pitch"].reshape(-1), 1.0)))
        pr = np.interp(td, ts, f["periodicity"].reshape(-1))
        return x[pr > 0.5], lf[pr > 0.5]

    def ema_of(it: dict) -> np.ndarray:
        return np.load(A1 / f"feat/{it['voice']}_{it['j']}.npz")["ema"]

    def r2(get, rate: float) -> float:
        tr = [f0_xy(it, get(it), rate) for it in tr_its]
        te = [f0_xy(it, get(it), rate) for it in te_its]
        reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(np.concatenate([a for a, _ in tr]), np.concatenate([b for _, b in tr]))
        X, y = np.concatenate([a for a, _ in te]), np.concatenate([b for _, b in te])
        return float(1 - np.mean((reg.predict(X) - y) ** 2) / np.var(y))

    leak = {"sparc_ema_f0_r2": r2(ema_of, 50.0)}
    for t in a.tags:
        leak[t] = {"voice36": probe(t, lambda it: voices.index(it["voice"])), "sex": probe(t, lambda it: int(sex[it["voice"]] == "F")),
                   "voice36_qda": probe(t, lambda it: voices.index(it["voice"]), True), "sex_qda": probe(t, lambda it: int(sex[it["voice"]] == "F"), True),
                   "f0_r2": r2(lambda it, t=t: latents[(t, it["voice"], it["j"])], SR / HOP)}
    res["LEAK"] = leak
    print("LEAK", leak, flush=True)

    from faster_whisper import WhisperModel
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16")

    def tr_(w: np.ndarray) -> str:
        segs, _ = asr.transcribe(w, language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(s.text for s in segs))

    ref = {s: tr_(sf.read(str(A1 / f"src/{s}.wav"))[0].astype(np.float32)) for s in srcs}
    for t in a.tags:
        keys = [k for k in wavs if k[0] == f"M_{t}"]
        vals = [cer(ref[k[1]], tr_(to16(wavs[k].astype(np.float64), SR))) for k in keys]
        res[f"M_{t}"]["cer_median"] = float(np.median(vals))
        res[f"M_{t}"]["cer_mean"] = float(np.mean(vals))
        print("CER", t, res[f"M_{t}"]["cer_median"], flush=True)
    del asr
    torch.cuda.empty_cache()
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(DEV).eval()
    with torch.no_grad():
        mos = lambda w: float(utmos(torch.from_numpy(w).float()[None].to(DEV), 16000).item())
        sparc = {(s, v): mos(sf.read(str(A1 / f"wav/{s}__{v}__std.wav"))[0].astype(np.float32)) for s in srcs for v in tgts}
        for t in a.tags:
            ours = {(k[1], k[2]): mos(to16(wavs[k].astype(np.float64), SR)) for k in wavs if k[0] == f"M_{t}"}
            res[f"M_{t}"]["utmos_mean"] = float(np.mean(list(ours.values())))
            res[f"M_{t}"]["utmos_win_vs_sparc"] = float(np.mean([ours[k] > sparc[k] for k in ours]))
        res["sparc_M_utmos_mean"] = float(np.mean(list(sparc.values())))
    print("NAT", {k: v for k, v in res.items() if k.startswith("M_") and isinstance(v, dict) and "utmos_mean" in v}, flush=True)
    (out_root / "a3_items.json").write_text(json.dumps(items, indent=0))
    (out_root / "a3_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
