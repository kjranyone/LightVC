"""C3(current/converter.md §3c)の目標側の前計算: 学習話者(女声の実音声と TTS・評価話者の実音声を除く)ごとに、参照 ≥ 10s(最大 ~25s)から
 T [24, K](C1 の最大の単位ごとの CheapTrick 包絡 c1..c24 の平均・[.25,.5,.25] 平滑・参照に無い単位はコードブック空間で最も近い単位の値)・
 mu(有声フレームの log f0 の中央値・f0 は f0hi の教師)・per(周期性 4 帯の有声フレームの平均)・ECAPA と WavLM-SV の埋め込み(参照全体)。
出力: data/c3/targets.npz(spk・src・T float16・mu・per・e_ecapa・e_wavlm・secs)
    OMP_NUM_THREADS=1 uv run python prep_c3_targets.py --procs 10 [--limit 20]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from math import gcd
from multiprocessing import Pool
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).parent))
ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data/rvoc_f0hi/manifest.json"
MIN_S, TGT_S, MAX_S = 10.0, 20.0, 25.0
AUG = False
VC = ROOT / "data/vctk/VCTK-Corpus/VCTK-Corpus"


def smooth3(E: np.ndarray, w: float = 0.25) -> np.ndarray:
    e = np.pad(E, ((0, 0), (1, 1)), mode="edge")
    return w * e[:, :-2] + (1 - 2 * w) * e[:, 1:-1] + w * e[:, 2:]


def augment_ref(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """参照の音声の加工(チャンネルの多様性): 対数周波数の 3 点 EQ(±5dB)・帯域制限(8/10/12/16kHz か無し)・雑音 SNR 25〜45dB。"""
    from scipy.signal import butter, sosfilt
    X = np.fft.rfft(x)
    f = np.linspace(0, 1, len(X))
    a = rng.uniform(-5, 5, 3)
    db = a[0] * (f - 0.5) * 2 + a[1] * np.sin(2 * np.pi * 2 * f) + a[2] * np.cos(2 * np.pi * 3 * f)
    x = np.fft.irfft(X * 10 ** (db / 20), n=len(x)).astype(np.float32)
    cut = rng.choice([8000, 10000, 12000, 16000, 0])
    if cut:
        x = sosfilt(butter(6, cut, "lowpass", fs=48000, output="sos"), x).astype(np.float32)
    snr = rng.uniform(25, 45)
    return (x + rng.standard_normal(len(x)).astype(np.float32) * float(np.sqrt((x ** 2).mean())) * 10 ** (-snr / 20)).astype(np.float32)


def work(job):
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    import f0hi as H
    import pae as PA
    key, rows = job
    try:
        xs, fs = [], []
        for r in rows:
            x, sr = sf.read(r["wav"], dtype="float32", always_2d=True)
            x = x.mean(1)
            if sr != 48000:
                g = gcd(sr, 48000)
                x = resample_poly(x, 48000 // g, sr // g).astype(np.float32)
            if r.get("f0"):
                f0 = np.load(ROOT / r["f0"]).astype(np.float32)
            else:
                f0, _ = H.teacher_f0(x[:len(x) // 240 * 240])
            n = min(len(f0), len(x) // 240)
            xs.append(x[:n * 240])
            fs.append(f0[:n])
        x = np.concatenate(xs)
        f0 = np.concatenate(fs)
        if AUG:
            x = augment_ref(x, np.random.default_rng(abs(hash(key)) % (2 ** 31)))
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        env = smooth3(PA.envelope(xa, f0))
        per = PA.periodicity(torch.from_numpy(xa)[None], torch.from_numpy(f0)[None])[0].numpy()
        return {"key": key, "x": x, "f0": f0, "env": env.astype(np.float32), "per": per.astype(np.float32)}
    except Exception as e:
        return {"key": key, "err": f"{type(e).__name__}: {e}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--c1", default=str(ROOT / "results/c1_1"))
    ap.add_argument("--out", default=str(ROOT / "data/c3/targets.npz"))
    ap.add_argument("--ref_sec", type=float, default=20.0, help="参照の目標の長さ(最小 10s)")
    ap.add_argument("--aug", action="store_true", help="参照の音声に EQ・帯域制限・雑音をかける(表・埋め込みとも加工後から)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vctk", action="store_true", help="VCTK の評価に使わない女声を目標に足す")
    a = ap.parse_args()
    global TGT_S, MAX_S, AUG
    TGT_S, MAX_S, AUG = a.ref_sec, a.ref_sec + 5.0, a.aug
    import torch
    import c1_content as CC
    import conv_c0 as C0
    import idloss as ID
    import nvoc as N
    import train_c1 as T1
    import train_rvoc as TR
    ex, exs = TR.hi_eval_paths(), TR.held_speakers()
    rows = [r for r in json.loads(MANIFEST.read_text())["rows"] if r.get("keep") and not TR.eval_excluded(r, ex, exs)]
    groups: dict = {}
    for r in rows:
        groups.setdefault((("real" if r["src"].startswith("real") else "tts"), r["spk"]), []).append(r)
    rng = random.Random(a.seed)
    jobs = []
    for key, rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: r["wav"])
        rng.shuffle(rs)
        pick, tot = [], 0.0
        for r in rs:
            if tot >= TGT_S:
                break
            if tot + r["dur"] > MAX_S and pick:
                continue
            pick.append(r)
            tot += r["dur"]
        if tot >= MIN_S:
            jobs.append((key, pick))
    if a.vctk:
        import soundfile as sf
        info = (VC / "speaker-info.txt").read_text().splitlines()
        J = json.loads(Path("/tmp/claude-1000/-home-kojirotanaka-kjranyone-LightVC/2e1a59c3-e906-4c48-bba4-03855ea95b2e/scratchpad/r4_spk/jobs.json").read_text())
        evs = set(J["fems"]) | set(J["males"])
        fem = ["p" + l.split()[0] for l in info[1:] if l.strip() and l.split()[2] == "F"]
        for sp in sorted(fem):
            if sp in evs or not (VC / "wav48" / sp).is_dir():
                continue
            ws = sorted((VC / "wav48" / sp).glob("*.wav"))
            rng.shuffle(ws)
            pick, tot = [], 0.0
            for w in ws:
                if tot >= a.ref_sec:
                    break
                d = sf.info(str(w)).duration
                pick.append({"wav": str(w), "f0": None, "dur": d})
                tot += d
            if tot >= MIN_S:
                jobs.append((("vctk", sp), pick))
        print("VCTK の評価に使わない女声", sum(1 for k, _ in jobs if k[0] == "vctk"), flush=True)
    if a.limit:
        jobs = jobs[::max(1, len(jobs) // a.limit)][:a.limit]
    print("speakers", len(jobs), "real", sum(1 for k, _ in jobs if k[0] == "real"), "tts", sum(1 for k, _ in jobs if k[0] == "tts"), flush=True)
    dev = "cuda"
    c1d = Path(a.c1)
    st = torch.load(c1d / "last.pt", map_location="cpu", weights_only=False)
    net = CC.C1(st["cfg"]["k"], st["cfg"]["ch"], tuple(st["cfg"]["dils"])).to(dev).eval()
    net.load_state_dict(st["net"])
    mfront = CC.MelFront().to(dev)
    Cb = torch.load(c1d / "codebook.pt").to(dev)
    K = Cb.shape[0]
    ecapa, wsv = ID.Ecapa(dev), ID.WavlmSV(dev)
    out = {"spk": [], "src": [], "T": [], "mu": [], "per": [], "e_ecapa": [], "e_wavlm": [], "secs": []}
    with Pool(a.procs) as pool, torch.no_grad():
        for i, r in enumerate(pool.imap(work, jobs, chunksize=1)):
            if "err" in r:
                print("ERR", r["key"], r["err"], flush=True)
                continue
            x, f0, env, per = r["x"], r["f0"], r["env"], r["per"]
            n = len(f0)
            xp = torch.from_numpy(T1.prime(x))[None].to(dev)
            p = net(mfront(xp))[..., -(len(x) // N.HOP):].softmax(1)[0].T.cpu().numpy()
            p = np.pad(p, ((0, max(0, n - len(p))), (0, 0)), mode="edge")[:n]
            T = C0.table(env[1:25], p.argmax(1), K, Cb.cpu().numpy())
            vo = f0 > 0
            xt = torch.from_numpy(x)[None].to(dev)
            out["spk"].append(r["key"][1]); out["src"].append(r["key"][0])
            out["T"].append(T.astype(np.float16)); out["mu"].append(float(np.median(np.log(f0[vo]))) if vo.any() else 5.3)
            out["per"].append(per[:, vo].mean(1) if vo.any() else per.mean(1))
            out["e_ecapa"].append(ecapa(xt)[0].cpu().numpy()); out["e_wavlm"].append(wsv(xt)[0].cpu().numpy()); out["secs"].append(len(x) / 48000)
            if (i + 1) % 100 == 0:
                print(i + 1, "/", len(jobs), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, spk=np.array(out["spk"]), src=np.array(out["src"]), T=np.stack(out["T"]), mu=np.array(out["mu"], np.float32),
             per=np.stack(out["per"]).astype(np.float32), e_ecapa=np.stack(out["e_ecapa"]), e_wavlm=np.stack(out["e_wavlm"]), secs=np.array(out["secs"], np.float32))
    print("saved", a.out, "speakers", len(out["spk"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
