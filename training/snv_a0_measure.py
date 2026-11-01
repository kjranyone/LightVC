"""A0(synth_nam_vc §5・学習なし): 物差しの較正。事前登録 = results/snv_a0/prereg.yaml。
ID(ECAPA・WavLM-SV)・CER(faster-whisper large-v3 の仮名 CER)・NAT(UTMOS22 strong)・LEAK(発話の統計 + ロジスティック回帰)を既知の対照で読む。
    cd training && uv run --no-sync python snv_a0_measure.py
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pyworld as pw
import soundfile as sf
import torch
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/snv_a0"
WAV = OUT / "wav"
VCTK = ROOT / "data/vctk/VCTK-Corpus/VCTK-Corpus/wav48"
VCTK_M = ("p226", "p227", "p232", "p237", "p241", "p243")
SR = 16000
ALPHAS = (1.0, 1.05, 1.10, 1.20, 1 / 1.05, 1 / 1.10, 1 / 1.20)
DEV = "cuda"


def load(p: Path) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(str(p), sr=None, mono=True)
    return y.astype(np.float64), sr


def to16(y: np.ndarray, sr: int) -> np.ndarray:
    return librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SR) if sr != SR else y.astype(np.float32)


def match_rms(y: np.ndarray, ref: np.ndarray) -> np.ndarray:
    g = np.sqrt(np.mean(ref ** 2) / max(np.mean(y ** 2), 1e-12))
    y = y * g
    pk = np.max(np.abs(y))
    return y / pk * 0.99 if pk > 0.99 else y


def world_job(args: tuple) -> dict:
    y, sr, alphas, sts = args
    f0, t = pw.harvest(y, sr, f0_floor=50.0, f0_ceil=1000.0, frame_period=5.0)
    sp = pw.cheaptrick(y, f0, t, sr)
    ap = pw.d4c(y, f0, t, sr)
    k = np.arange(sp.shape[1], dtype=np.float64)
    out = {}
    for a in alphas:
        spw = np.stack([np.interp(k / a, k, row, right=row[-1]) for row in sp]) if a != 1.0 else sp
        out[("a", a)] = match_rms(pw.synthesize(f0, spw, ap, sr, 5.0), y)
    for s in sts:
        out[("st", s)] = match_rms(pw.synthesize(f0 * 2 ** (s / 12), sp, ap, sr, 5.0), y)
    return out


def world_grid_job(args: tuple) -> dict:
    y, sr, ks, alphas = args
    f0, t = pw.harvest(y, sr, f0_floor=50.0, f0_ceil=1000.0, frame_period=5.0)
    sp = pw.cheaptrick(y, f0, t, sr)
    ap = pw.d4c(y, f0, t, sr)
    kk = np.arange(sp.shape[1], dtype=np.float64)
    out = {}
    for a in alphas:
        spw = np.stack([np.interp(kk / a, kk, row, right=row[-1]) for row in sp]) if a != 1.0 else sp
        for s in ks:
            out[(s, a)] = match_rms(pw.synthesize(f0 * 2 ** (s / 12), spw, ap, sr, 5.0), y)
    return out


def noise(y: np.ndarray, snr: float, seed: int) -> np.ndarray:
    n = np.random.default_rng(seed).standard_normal(len(y))
    n *= np.sqrt(np.mean(y ** 2) / 10 ** (snr / 10) / np.mean(n ** 2))
    return match_rms(y + n, y)


def lowpass(y: np.ndarray, sr: int, fc: float) -> np.ndarray:
    return match_rms(sosfiltfilt(butter(8, fc, fs=sr, output="sos"), y), y)


def griffinlim(y: np.ndarray) -> np.ndarray:
    S = np.abs(librosa.stft(y.astype(np.float32), n_fft=1024, hop_length=256))
    return match_rms(librosa.griffinlim(S, n_iter=32, hop_length=256, n_fft=1024, random_state=0, length=len(y)).astype(np.float64), y)


def buzz(y: np.ndarray, sr: int) -> np.ndarray:
    t = np.arange(len(y)) / sr
    return match_rms(y * (1 + 0.3 * np.sin(2 * np.pi * 200 * t)), y)


def segments(p: Path) -> tuple[np.ndarray, list[np.ndarray], int]:
    y, sr = load(p)
    w = 10 * sr
    enroll = y[: 3 * w]
    tests = []
    s = 6 * w
    while len(tests) < 6 and s + w <= len(y):
        seg = y[s: s + w]
        if np.sqrt(np.mean(seg ** 2)) > 0.01:
            tests.append(seg)
        s += w
    return enroll, tests, sr


def vctk(spk: str) -> tuple[np.ndarray, list[np.ndarray], int]:
    fs = sorted(p for p in (VCTK / spk).glob("*.wav") if int(p.stem.split("_")[1]) >= 10)[:6]
    ys = [load(p) for p in fs]
    sr = ys[0][1]
    return np.concatenate([y for y, _ in ys[:3]]), [y for y, _ in ys[3:]], sr


KANA = re.compile(r"[^ァ-ヺー]")


def kana(text: str) -> str:
    import pyopenjtalk
    return KANA.sub("", pyopenjtalk.g2p(text, kana=True))


def cer(ref: str, hyp: str) -> float:
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[-1] / max(len(ref), 1)


def boot_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = [np.mean(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def eer(pos: np.ndarray, neg: np.ndarray) -> float:
    th = np.sort(np.concatenate([pos, neg]))
    frr = np.array([np.mean(pos < t) for t in th])
    far = np.array([np.mean(neg >= t) for t in th])
    i = int(np.argmin(np.abs(frr - far)))
    return float((frr[i] + far[i]) / 2)


class Embedders:
    def __init__(self) -> None:
        from speechbrain.inference.speaker import EncoderClassifier
        from transformers import AutoFeatureExtractor, WavLMForXVector
        self.ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})
        self.fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
        self.wl = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(DEV).eval()

    @torch.no_grad()
    def __call__(self, y16: np.ndarray) -> dict:
        e = self.ec.encode_batch(torch.from_numpy(y16).float()[None].to(DEV)).reshape(-1)
        inp = self.fe(y16, sampling_rate=SR, return_tensors="pt").to(DEV)
        w = self.wl(**inp).embeddings.reshape(-1)
        return {"ecapa": torch.nn.functional.normalize(e, dim=0).cpu().numpy(), "wavlm_sv": torch.nn.functional.normalize(w, dim=0).cpu().numpy()}


def main() -> int:
    its = json.loads((WAV / "items.json").read_text())
    for it in its:
        it["y"], it["sr"] = load(WAV / f"{it['voice']}_{it['j']}.wav")
    voices = sorted({it["voice"] for it in its})
    sex = {it["voice"]: it["sex"] for it in its}
    enroll = {v: np.concatenate([it["y"] for it in its if it["voice"] == v and it["role"] == "enroll"]) for v in voices}
    sr_i = its[0]["sr"]
    tests = [it for it in its if it["role"] != "enroll"]
    ftest = [it for it in tests if it["sex"] == "F" and it["role"] == "test"]
    print("irodori", len(its), "voices", len(voices), "female plain tests", len(ftest), flush=True)

    real = {}
    for name, p in (("uirou", ROOT / "外郎売.mp3"), ("namikawa", ROOT / "namikawa.mp3")):
        real[name] = segments(p)
    for spk in VCTK_M:
        real[f"vctk_{spk}"] = vctk(spk)
    print("real", {k: len(v[1]) for k, v in real.items()}, flush=True)

    with ProcessPoolExecutor(10) as ex:
        wf = list(ex.map(world_job, [(it["y"], it["sr"], ALPHAS, (-4.0, 4.0)) for it in ftest]))
        rtests = [(k, i, seg, sr) for k, (_, segs, sr) in real.items() for i, seg in enumerate(segs)]
        wr = list(ex.map(world_grid_job, [(seg, sr, (0.0, 5.0, 12.0), (1.0, 1.10, 1.20)) for _, _, seg, sr in rtests]))
    print("world done", flush=True)

    res: dict = {"prereg": "results/snv_a0/prereg.yaml"}
    emb = Embedders()
    E = {v: emb(to16(enroll[v], sr_i)) for v in voices}
    ER = {k: emb(to16(e, sr)) for k, (e, _, sr) in real.items()}

    def id_eval(items: list, ys: list) -> dict:
        out = {}
        for m in ("ecapa", "wavlm_sv"):
            top, pos, neg, own = [], [], [], []
            for it, y in zip(items, ys):
                e = emb(to16(y, it["sr"]))[m]
                cands = [v for v in voices if sex[v] == it["sex"]]
                sc = {v: float(e @ E[v][m]) for v in cands}
                top.append(max(sc, key=sc.get) == it["voice"])
                own.append(sc[it["voice"]])
                pos.append(sc[it["voice"]])
                neg += [s for v, s in sc.items() if v != it["voice"]]
            out[m] = {"top1": float(np.mean(top)), "eer": eer(np.array(pos), np.array(neg)), "cos_own": own}
        return out

    id1 = {}
    for s in ("F", "M"):
        for role in ("test", "whisper", "tender"):
            sel = [it for it in tests if it["sex"] == s and it["role"] == role]
            r = id_eval(sel, [it["y"] for it in sel])
            id1[f"{s}_{role}"] = {m: {"top1": r[m]["top1"], "eer": r[m]["eer"], "n": len(sel)} for m in r}
    res["ID1_ID2"] = id1
    print(json.dumps(id1, indent=1), flush=True)

    id3 = {}
    cos3 = {}
    for a in ALPHAS:
        r = id_eval(ftest, [w[("a", a)] for w in wf])
        cos3[a] = {m: np.array(r[m]["cos_own"]) for m in r}
        id3[f"{a:.4f}"] = {m: {"top1": r[m]["top1"], "cos_own_mean": float(np.mean(r[m]["cos_own"]))} for m in r}
    for m in ("ecapa", "wavlm_sv"):
        for a in ALPHAS[1:]:
            d = cos3[a][m] - cos3[1.0][m]
            id3[f"{a:.4f}"][m]["diff_vs_1_ci"] = boot_ci(d)
    res["ID3"] = id3
    id4 = {}
    for s in (-4.0, 4.0):
        r = id_eval(ftest, [w[("st", s)] for w in wf])
        id4[f"{s:+.0f}st"] = {m: {"top1": r[m]["top1"], "cos_own_mean": float(np.mean(r[m]["cos_own"]))} for m in r}
    res["ID4"] = id4
    print(json.dumps({"ID3": id3, "ID4": id4}, indent=1), flush=True)

    id5 = {}
    for (k, i, seg, sr), w in zip(rtests, wr):
        for (s, a), y in w.items():
            e = emb(to16(y, sr))
            key = f"k{s:+.0f}_a{a:.2f}"
            for m in ("ecapa", "wavlm_sv"):
                id5.setdefault(key, {}).setdefault(m, []).append(float(e[m] @ ER[k][m]))
    res["ID5"] = {key: {m: float(np.mean(v)) for m, v in d.items()} for key, d in id5.items()}
    print(json.dumps(res["ID5"], indent=1), flush=True)
    del emb
    torch.cuda.empty_cache()

    from faster_whisper import WhisperModel
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16")

    def transcribe(y16: np.ndarray) -> str:
        segs, _ = asr.transcribe(y16, language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return "".join(s.text for s in segs)

    conds = {"clean": lambda it, w: it["y"], "snr20": lambda it, w: noise(it["y"], 20, 1), "snr10": lambda it, w: noise(it["y"], 10, 1),
             "snr0": lambda it, w: noise(it["y"], 0, 1), "lp4000": lambda it, w: lowpass(it["y"], it["sr"], 4000),
             "lp2000": lambda it, w: lowpass(it["y"], it["sr"], 2000), "world_a1.00": lambda it, w: w[("a", 1.0)],
             "world_a1.10": lambda it, w: w[("a", 1.10)], "world_a1.20": lambda it, w: w[("a", 1.20)]}
    cr = {}
    examples = []
    for c, f in conds.items():
        vals = []
        for it, w in zip(ftest, wf):
            hyp = transcribe(to16(f(it, w), it["sr"]))
            ref = it["text"]
            vals.append(cer(kana(ref), kana(hyp)))
            if c == "clean" and len(examples) < 6:
                examples.append({"ref": ref, "hyp": hyp, "cer": vals[-1]})
        cr[c] = {"median": float(np.median(vals)), "mean": float(np.mean(vals)), "p90": float(np.percentile(vals, 90))}
        print("CER", c, cr[c], flush=True)
    res["CER"] = cr
    res["CER_examples"] = examples
    del asr
    torch.cuda.empty_cache()

    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(DEV).eval()

    @torch.no_grad()
    def mos(y16: np.ndarray) -> float:
        return float(utmos(torch.from_numpy(y16).float()[None].to(DEV), SR).item())

    ncond = {"snr20": lambda it, w: noise(it["y"], 20, 1), "snr10": lambda it, w: noise(it["y"], 10, 1), "lp4000": lambda it, w: lowpass(it["y"], it["sr"], 4000),
             "world": lambda it, w: w[("a", 1.0)], "griffinlim": lambda it, w: griffinlim(it["y"]), "buzz200": lambda it, w: buzz(it["y"], it["sr"])}
    base = np.array([mos(to16(it["y"], it["sr"])) for it in ftest])
    nr = {"clean_mean": float(base.mean())}
    for c, f in ncond.items():
        v = np.array([mos(to16(f(it, w), it["sr"])) for it, w in zip(ftest, wf)])
        nr[c] = {"mean": float(v.mean()), "win_rate": float(np.mean(base > v))}
        print("NAT", c, nr[c], flush=True)
    lv = {}
    for s in ("F", "M"):
        for role in ("test", "whisper", "tender"):
            sel = [it for it in tests if it["sex"] == s and it["role"] == role]
            lv[f"irodori_{s}_{role}"] = float(np.mean([mos(to16(it["y"], it["sr"])) for it in sel]))
    for k, (_, segs, sr) in real.items():
        lv[k] = float(np.mean([mos(to16(s, sr)) for s in segs]))
    nr["levels"] = lv
    res["NAT"] = nr
    print("NAT levels", lv, flush=True)

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def stats(y: np.ndarray, sr: int) -> np.ndarray:
        m = np.log(librosa.feature.melspectrogram(y=to16(y, sr), sr=SR, n_fft=1024, hop_length=160, n_mels=80) + 1e-6)
        return np.concatenate([m.mean(1), m.std(1)])

    tr = [it for it in its if it["role"] == "enroll"]
    te = [it for it in its if it["role"] == "test"]
    Xtr = np.stack([stats(it["y"], it["sr"]) for it in tr])
    Xte = np.stack([stats(it["y"], it["sr"]) for it in te])
    ytr = np.array([voices.index(it["voice"]) for it in tr])
    yte = np.array([voices.index(it["voice"]) for it in te])

    def probe(a: np.ndarray, b: np.ndarray, ya: np.ndarray) -> float:
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
        clf.fit(a, ya)
        return float(np.mean(clf.predict(b) == yte))

    rng = np.random.default_rng(0)
    from scipy.stats import binom
    res["LEAK"] = {"logmel": probe(Xtr, Xte, ytr), "permuted": probe(Xtr, Xte, rng.permutation(ytr)),
                   "random": probe(rng.standard_normal(Xtr.shape), rng.standard_normal(Xte.shape), ytr),
                   "chance": 1 / len(voices), "chance_upper95": float(binom.ppf(0.95, len(yte), 1 / len(voices)) / len(yte)), "n_test": int(len(yte))}
    print("LEAK", res["LEAK"], flush=True)

    (OUT / "a0_measure.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
