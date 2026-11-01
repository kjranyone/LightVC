"""A2(diag)の評価: 学習した G(g / x)で A0 の目標 24 声へ描画し、A0〜A0c の物差しで読む。事前登録 = results/diag_snv_a2_g/prereg.yaml。
V = BigVGAN の写し合成(天井)・S = 自己の reamp(本人の 10 s の演奏 + 登録の構造)・O = 別の女声の演奏 → 目標・M = 本物の男声 → 目標(高さは SPARC と同じ平均・標準偏差の写像)。
    cd training && uv run --no-sync python snv_a2_eval.py --tags diag_snv_a2_g diag_snv_a2_x
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
from snv_a2_mel import bigvgan_h
from train_snv_a2 import FEAT, G, P_DIM, playing

A1 = ROOT / "results/snv_a1"


def vocoder():
    import bigvgan
    h, d = bigvgan_h()
    m = bigvgan.BigVGAN(h, use_cuda_kernel=False)
    m.load_state_dict(torch.load(d + "bigvgan_generator.pt", map_location="cpu", weights_only=True)["generator"])
    m.remove_weight_norm()
    return m.eval().to(DEV)


def map_pitch(p: np.ndarray, src_stats: np.ndarray, trg_stats: np.ndarray) -> np.ndarray:
    q = p.copy()
    f0 = np.exp(q[:, 12])
    f0 = (f0 - src_stats[0]) / src_stats[1] * trg_stats[1] + trg_stats[0]
    q[:, 12] = np.log(np.maximum(f0, 1.0))
    return q


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    a = ap.parse_args()
    meta = json.loads((A1 / "convert_meta.json").read_text())
    srcs, tgts = meta["sources"], meta["targets"]
    drows = json.loads((A1 / "diag/meta.json").read_text())
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
        orig = librosa.load(str(FEAT / f"wav44/a1diag__D0_{r['target']}_{r['g']}.wav"), sr=44100)[0]
        db = 10 * np.log10(np.mean(w ** 2) / np.mean(orig ** 2))
        assert abs(db) < 3.0, f"L-6: 写し合成の音量が元とずれる {db:.2f} dB"
        wavs[("V", r["target"], r["g"])] = w
    print("L-6 ok: BigVGAN の写し合成は元と ±3 dB 以内", flush=True)

    tstats = {v: np.load(FEAT / f"sparc/a1src/tref_{v}.npz")["pitch_stats"].reshape(-1)[:2] for v in tgts}
    for tag in a.tags:
        d = ROOT / "results" / tag
        ck = torch.load(d / "last.pt", map_location="cpu", weights_only=True)
        g = G(ck["mode"]).to(DEV).eval()
        g.load_state_dict(ck["g"])
        nz = np.load(d / "norm.npz")
        mu, sd = nz["mu"], nz["sd"]
        S = {}
        with torch.no_grad():
            for v in tgts:
                rmel = np.load(FEAT / f"mel/a0enroll/{v}.npy").astype(np.float32)
                rp = (playing(FEAT / f"sparc/a1src/tref_{v}.npz", rmel.shape[1]) - mu) / sd
                S[v] = g.structure(torch.from_numpy(rmel)[None].to(DEV), torch.from_numpy(rp)[None].to(DEV), torch.ones(1, rmel.shape[1], dtype=torch.bool, device=DEV))
            ents = []

            def run(p_npz: Path, n_mel: int, v: str, stats: np.ndarray | None) -> np.ndarray:
                p = playing(p_npz, n_mel)
                if stats is not None:
                    p = map_pitch(p, stats, tstats[v])
                y, att = g(torch.from_numpy((p - mu) / sd)[None].to(DEV), S[v], return_attn=True)
                if att is not None:
                    ents.append(float((-(att.clamp(min=1e-9).log() * att).sum(-1)).mean()))
                return render(y)

            for r in drows:
                n_mel = np.load(FEAT / f"mel/a1diag/D0_{r['target']}_{r['g']}.npy").shape[1]
                wavs[(f"S_{tag}", r["target"], r["g"])] = run(FEAT / f"sparc/a1diag/D0_{r['target']}_{r['g']}.npz", n_mel, r["target"], None)
                o = FEAT / f"sparc/a1diag/src_{r['d2_src']}_{r['g']}.npz"
                wavs[(f"O_{tag}", r["target"], r["g"])] = run(o, n_mel, r["target"], np.load(o)["pitch_stats"].reshape(-1)[:2])
            for s in srcs:
                f = FEAT / f"sparc/a1src/{s}.npz"
                n_mel = int(10 * 44100 / 512) + 1
                for v in tgts:
                    wavs[(f"M_{tag}", s, v)] = run(f, n_mel, v, np.load(f)["pitch_stats"].reshape(-1)[:2])
        if ents:
            print(tag, "attention entropy mean", np.mean(ents), "(一様なら ln(参照のフレーム数) ≈", np.log(1300), ")", flush=True)
        for k, w in wavs.items():
            if k[0].endswith(tag) and k[0].startswith("M_") and k[2] in tgts[:4]:
                sf.write(vdir / f"{k[0]}__{k[1]}__{k[2]}.wav", w, 44100)
        del g
        torch.cuda.empty_cache()
    del voc
    torch.cuda.empty_cache()
    print("rendered", len(wavs), flush=True)

    sp = splits()
    pool = sp["pool"]
    emb = Embedders()
    X, yy = [], []
    for i, s in enumerate(sp["lda"]):
        for p in pool[s]:
            X.append(emb(to16(*load(p)))["ecapa"])
            yy.append(i)
    T = fit_lda(np.stack(X), np.array(yy))
    A0 = ROOT / "results/snv_a0"
    Et = {v: proj(emb(np.concatenate([to16(*load(A0 / f"wav/{v}_{j}.wav")) for j in (0, 1, 2)]))["ecapa"], T) for v in tgts}
    Es = {s: proj(emb(sf.read(str(A1 / f"src/{s.rsplit('_', 1)[0]}_enroll.wav"))[0].astype(np.float32))["ecapa"], T) for s in srcs}
    E = {k: proj(emb(to16(w.astype(np.float64), 44100))["ecapa"], T) for k, w in wavs.items()}
    res: dict = {"prereg": f"results/{a.tags[0]}/prereg.yaml", "ref": {"sparc_D1b": 0.500, "sparc_D2": 0.292, "sparc_M": 0.260, "ceiling_D0": 0.792}}

    def top1(keys: list, truth: callable) -> float:
        return float(np.mean([max(tgts, key=lambda u: float(E[k] @ Et[u])) == truth(k) for k in keys]))

    for cond in ["V"] + [f"{c}_{t}" for t in a.tags for c in ("S", "O", "M")]:
        keys = [k for k in wavs if k[0] == cond]
        if cond.startswith("M_"):
            ms = np.array([float(E[k] @ Et[k[2]] - E[k] @ Es[k[1]]) for k in keys])
            res[cond] = {"top1": top1(keys, lambda k: k[2]), "m_pos": float(np.mean(ms > 0)), "m_mean": float(ms.mean()), "n": len(keys)}
        else:
            res[cond] = {"top1": top1(keys, lambda k: k[1]), "n": len(keys)}
            if cond.startswith("O_"):
                src = {(r["target"], r["g"]): r["d2_src"] for r in drows}
                res[cond]["read_as_source"] = top1(keys, lambda k: src[(k[1], k[2])])
        print(cond, res[cond], flush=True)
    for t in a.tags:
        s1 = np.array([max(tgts, key=lambda u: float(E[k] @ Et[u])) == k[1] for k in sorted(k for k in wavs if k[0] == f"S_{t}")], float)
        res[f"S_{t}_vec"] = s1.tolist()
    if len(a.tags) == 2:
        d = np.array(res[f"S_{a.tags[1]}_vec"]) - np.array(res[f"S_{a.tags[0]}_vec"])
        rng = np.random.default_rng(0)
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        res["S_diff_x_minus_g"] = {"mean": float(d.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
        print("S diff", res["S_diff_x_minus_g"], flush=True)
    del emb
    torch.cuda.empty_cache()

    from faster_whisper import WhisperModel
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16")

    def tr(w: np.ndarray) -> str:
        segs, _ = asr.transcribe(w, language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(s.text for s in segs))

    ref = {s: tr(sf.read(str(A1 / f"src/{s}.wav"))[0].astype(np.float32)) for s in srcs}
    for t in a.tags:
        keys = [k for k in wavs if k[0] == f"M_{t}"]
        vals = [cer(ref[k[1]], tr(to16(wavs[k].astype(np.float64), 44100))) for k in keys]
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
            ours = {(k[1], k[2]): mos(to16(wavs[k].astype(np.float64), 44100)) for k in wavs if k[0] == f"M_{t}"}
            res[f"M_{t}"]["utmos_mean"] = float(np.mean(list(ours.values())))
            res[f"M_{t}"]["utmos_win_vs_sparc"] = float(np.mean([ours[k] > sparc[k] for k in ours]))
        res["sparc_M_utmos_mean"] = float(np.mean(list(sparc.values())))
        res["V_utmos_mean"] = float(np.mean([mos(to16(w.astype(np.float64), 44100)) for k, w in wavs.items() if k[0] == "V"]))
    print("NAT", {k: v for k, v in res.items() if k.startswith("M_")}, res["sparc_M_utmos_mean"], res["V_utmos_mean"], flush=True)
    (out_root / "a2_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
