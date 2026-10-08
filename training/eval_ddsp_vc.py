"""DDSP-VC の評価(事前登録 results/ddsp_vc/prereg.yaml の success 項目)。

  1. held24 の自己再構成 logmel(遅延補正: y[DELAY:] vs x[:-DELAY])・錨 BigVGAN 0.195 と並記
  2. 男→女のゼロショット変換: VCTK 男 8 発話 + JA TTS 男 4 発話 × 除外女声 4 人の参照
     指標 = 目標 ECAPA 類似度(目標の別 3 発話の平均埋め込み)・Whisper CER の増分・出力 f0 の目標レジスタ追従
  3. 学習済み重みでの未来不変性(実音声・入力の t 以降を書き換え)と CPU 1 スレッドの RTF
  4. 耳用の試料(results/ddsp_vc/samples/)

    CUDA_VISIBLE_DEVICES=0 uv run python eval_ddsp_vc.py --ckpt ../results/ddsp_vc/last.pt
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile
import torch

sys.path.insert(0, str(Path(__file__).parent))
import artic_dsp as AD
import ddsp_vc as V
from train_ddsp_vc import index, load48
from train_s1_1 import MEL_SPECS, build_mel, logmel_l1

ROOT = Path(__file__).resolve().parent.parent
VC = ROOT / "data/vctk/VCTK-Corpus/VCTK-Corpus"


def norm_text(s: str) -> str:
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def cer(ref: str, hyp: str) -> float:
    ref, hyp = ref.replace(" ", ""), hyp.replace(" ", "")
    if not ref:
        return 0.0
    dp = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, len(hyp) + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev = cur
    return dp[-1] / len(ref)


def f0_frames(x: np.ndarray) -> np.ndarray:
    f0, _ = AD.causal_yin(x.astype(np.float64), voi_max=0.45)
    return f0.astype(np.float32)


def reg_stats(f0s: list[np.ndarray]) -> tuple[float, float]:
    v = np.log(np.concatenate([f[f > 0] for f in f0s]))
    return float(np.median(v)), float(v.std())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "results/ddsp_vc/last.pt"))
    ap.add_argument("--out", default=str(ROOT / "results/ddsp_vc"))
    a = ap.parse_args()
    dev = "cuda"
    out = Path(a.out)
    (out / "samples").mkdir(parents=True, exist_ok=True)
    st = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model = V.DDSPVC(norm=bool(st.get("norm", False))).to(dev)
    model.load_state_dict(st["ema"])
    model.eval()
    mels = [build_mel(nf, hm, nm).to(dev) for nf, hm, nm in MEL_SPECS]
    spk, tr, ev = index()
    rep = {"ckpt": a.ckpt, "step": int(st["step"])}

    def run(x: np.ndarray, f0: np.ndarray, ref: np.ndarray) -> np.ndarray:
        n = (len(x) // V.HOP) * V.HOP
        xt = torch.from_numpy(x[:n]).to(dev)[None]
        f = np.pad(f0[:n // V.HOP], (0, max(0, n // V.HOP - len(f0))))
        with torch.no_grad():
            mel = model.front(xt)
            s = model.spk(model.front(torch.from_numpy(ref).to(dev)[None]))
            y = model(mel, model.level(mel), torch.from_numpy(f).to(dev)[None], s, n,
                      gen=torch.Generator(device=dev).manual_seed(0))[0]
        return y.cpu().numpy()

    import s0_artic as S
    vals = []
    for it in S.held24():
        k = next((kk for kk in ev if kk.endswith("/" + it["spk"])), None)
        if k is None:
            continue
        x = load48(next(w for z, w, c in spk[k] if z.stem == it["stem"]))[:8 * V.SR]
        z = next(z for z, w, c in spk[k] if z.stem == it["stem"])
        f0 = np.load(z)["f0"].astype(np.float32)
        ref = load48(next(w for zz, w, c in spk[k] if zz.stem != it["stem"]))[:144000]
        y = run(x, f0, ref)
        n = len(y)
        vals.append(float(logmel_l1(torch.from_numpy(y[V.DELAY:]).to(dev)[None, None],
                                    torch.from_numpy(x[:n - V.DELAY]).to(dev)[None, None], mels)))
    rep["held24_logmel"] = round(float(np.mean(vals)), 4)
    rep["held24_n"] = len(vals)
    rep["anchor_bigvgan_logmel"] = 0.195
    print("held24 logmel", rep["held24_logmel"], "n", len(vals), flush=True)

    import whisper
    from a2_dsp_vc import ecapa
    emb = ecapa()
    asr = whisper.load_model("base", device="cpu")
    tg = [kk for kk in ev if kk.startswith("real_female")][6:10]
    targets = []
    for kk in tg:
        its = spk[kk]
        ref = load48(its[0][1])[:144000]
        te = np.mean([emb(load48(w).astype(np.float64)[:8 * V.SR]) for _, w, _ in its[1:4]], 0)
        te /= np.linalg.norm(te)
        mu_t, sd_t = reg_stats([np.load(z)["f0"] for z, _, _ in its[:6]])
        targets.append({"key": kk, "ref": ref, "emb": te, "mu": mu_t, "sd": sd_t})
    info = (VC / "speaker-info.txt").read_text().splitlines()[1:]
    males = sorted("p" + r.split()[0] for r in info if len(r.split()) > 2 and r.split()[2] == "M")
    srcs = []
    for s_ in males[1::6][:8]:
        u = next((u for u in range(40, 90) if (VC / f"wav48/{s_}/{s_}_{u:03d}.wav").is_file() and (VC / f"txt/{s_}/{s_}_{u:03d}.txt").is_file()), None)
        if u is None:
            continue
        enr = [f0_frames(load48(VC / f"wav48/{s_}/{s_}_{v:03d}.wav")) for v in range(u + 1, u + 6) if (VC / f"wav48/{s_}/{s_}_{v:03d}.wav").is_file()]
        oth = [load48(VC / f"wav48/{s_}/{s_}_{v:03d}.wav")[:8 * V.SR] for v in range(u + 10, u + 13) if (VC / f"wav48/{s_}/{s_}_{v:03d}.wav").is_file()]
        srcs.append({"name": f"{s_}_{u:03d}", "x": load48(VC / f"wav48/{s_}/{s_}_{u:03d}.wav")[:8 * V.SR], "lang": "en",
                     "text": norm_text((VC / f"txt/{s_}/{s_}_{u:03d}.txt").read_text()), "enr": enr, "other": oth})
    import glob
    for d in sorted(glob.glob(str(ROOT / "data/artic_feat/tts_male_ja/*")))[:4]:
        fs = sorted(glob.glob(d + "/*.npz"))
        w = ROOT / "data/male_tts_corpus" / Path(d).name / (Path(fs[0]).stem + ".wav")
        oth = [load48(ROOT / "data/male_tts_corpus" / Path(d).name / (Path(f).stem + ".wav"))[:8 * V.SR] for f in fs[6:9]]
        srcs.append({"name": f"ja_{Path(d).name}_{Path(fs[0]).stem}", "x": load48(w)[:8 * V.SR], "lang": "ja", "text": None,
                     "enr": [np.load(f)["f0"] for f in fs[1:6]], "other": oth})
    rows = []
    for sc in srcs:
        x = sc["x"]
        f0x = f0_frames(x)
        mu_s, sd_s = reg_stats(sc["enr"])
        se = np.mean([emb(o.astype(np.float64)) for o in sc["other"]], 0)
        se /= np.linalg.norm(se)
        tmp = out / "samples" / "tmp.wav"
        soundfile.write(tmp, x, V.SR)
        src_txt = sc["text"] if sc["lang"] == "en" else asr.transcribe(str(tmp), language="ja", fp16=False)["text"].replace(" ", "")
        c0 = cer(src_txt, norm_text(asr.transcribe(str(tmp), language="en", fp16=False)["text"])) if sc["lang"] == "en" else 0.0
        for ti, t in enumerate(targets):
            f0m = np.where(f0x > 0, np.exp(t["mu"] + (t["sd"] / max(sd_s, 1e-3)) * (np.log(np.maximum(f0x, 1.0)) - mu_s)), 0.0).astype(np.float32)
            y = run(x, f0m, t["ref"])
            pk = float(np.abs(y).max())
            y = y * (0.95 / pk if pk > 0.95 else 1.0)
            name = f"{sc['name']}__to__{t['key'].split('/')[-1][:8]}"
            soundfile.write(out / "samples" / f"{name}.wav", y.astype(np.float32), V.SR)
            e = emb(y.astype(np.float64))
            hyp = asr.transcribe(str(out / "samples" / f"{name}.wav"), language=sc["lang"], fp16=False)["text"]
            c = cer(sc["text"], norm_text(hyp)) if sc["lang"] == "en" else cer(src_txt, hyp.replace(" ", ""))
            f0y = f0_frames(y)
            k = min(len(f0y), len(f0m))
            both = (f0y[:k] > 0) & (f0m[:k] > 0)
            ferr = float(np.median(np.abs(12 * np.log2(f0y[:k][both] / f0m[:k][both])))) if both.sum() > 10 else None
            rows.append({"src": sc["name"], "tgt": t["key"], "secs_tgt": round(float(e @ t["emb"]), 4),
                         "secs_srcspk": round(float(e @ se), 4),
                         "secs_src_vs_tgt": round(float(emb(x.astype(np.float64)) @ t["emb"]), 4),
                         "cer": round(c, 3), "cer_src": round(c0, 3), "f0_err_st_median": ferr})
            print(json.dumps(rows[-1]), flush=True)
        (out / "samples" / "tmp.wav").unlink(missing_ok=True)
    for t in targets:
        soundfile.write(out / "samples" / f"REF__{t['key'].split('/')[-1][:8]}.wav", t["ref"], V.SR)
    rep["conv"] = {"n": len(rows), "secs_tgt_mean": round(float(np.mean([r["secs_tgt"] for r in rows])), 4),
                   "secs_src_vs_tgt_mean": round(float(np.mean([r["secs_src_vs_tgt"] for r in rows])), 4),
                   "secs_output_vs_source_speaker_mean": round(float(np.mean([r["secs_srcspk"] for r in rows])), 4),
                   "cer_increase_median_en": round(float(np.median([r["cer"] - r["cer_src"] for r in rows if not r["src"].startswith("ja_")])), 3),
                   "cer_median_ja_vs_src_transcript": round(float(np.median([r["cer"] for r in rows if r["src"].startswith("ja_")])), 3),
                   "f0_err_st_median": round(float(np.median([r["f0_err_st_median"] for r in rows if r["f0_err_st_median"] is not None])), 3)}
    rep["conv_rows"] = rows
    print(json.dumps(rep["conv"], ensure_ascii=False), flush=True)
    xs = S.held24()[0]["x"].astype(np.float32)[:4 * V.SR]
    f0 = f0_frames(xs)
    ref = targets[0]["ref"]
    y0 = run(xs, f0, ref)
    worst, sens = -10 ** 9, False
    rng = np.random.default_rng(0)
    for c in (int(len(xs) * r) for r in (0.4, 0.6, 0.8)):
        x2 = xs.copy()
        x2[c:] = rng.standard_normal(len(xs) - c).astype(np.float32) * xs.std()
        y1 = run(x2, f0_frames(x2), ref)
        d = np.nonzero(np.abs(y1 - y0) > 1e-6)[0]
        if len(d):
            sens = True
            worst = max(worst, c - int(d[0]))
    rep["future_invariance_trained"] = {"lookahead_samples": int(max(worst, 0)) if sens else None, "inconclusive": not sens,
                                        "note": "学習済み重み・実音声。出力の変化が編集点より前なら先読みあり"}
    cpu = V.DDSPVC(norm=bool(st.get("norm", False)))
    cpu.load_state_dict({k: v.cpu() for k, v in st["ema"].items()})
    cpu.eval()
    torch.set_num_threads(1)
    xt = torch.from_numpy(xs[:2 * V.SR])[None]
    with torch.no_grad():
        mel = cpu.front(xt)
        s = cpu.spk(cpu.front(torch.from_numpy(ref)[None]))
        f = torch.from_numpy(f0[:mel.shape[-1]])[None]
        t0 = time.time()
        cpu(mel, cpu.level(mel), f, s, xt.shape[-1])
        dt = time.time() - t0
    rep["rtf_cpu_1thread_python_offline"] = round(dt / 2.0, 3)
    print("future invariance", rep["future_invariance_trained"], "RTF", rep["rtf_cpu_1thread_python_offline"], flush=True)
    (out / "eval.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
