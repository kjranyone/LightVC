"""話者らしさは「音素単位ごとの目標の形」(静的)か「単位の中の動き」(動的)か(変換器の方式を決める測定・学習なし)。

材料 = artic_g2 と同じ差し替えラダー(VCTK 男 → 女 45 組・共通文 3〜6・R0 = 男声を RRPS で音域移動・T0 = 目標の同じ文)。
単位 = ContentVec(話者に依らない内容表現)の k-means。R0 の STFT に「目標の包絡の系列 − R0 の DCT24 包絡」を掛け(全フレーム・±6 で切る = H1_24 と同じ)、
女声 45 人の中での目標の順位(ECAPA・WavLM-SV・重心 = 各話者の発話 60〜の 3 つ)で採点する。
  ORACLE        目標の DCT24 軌道を DTW で整列(= H1_24)
  ORACLE_SM     同じ軌道を 25ms の移動平均で平滑化(単位平均の系列と同じ平滑化の対照)
  U_T0          T0 自身の単位平均を T0 の単位の並びで(同じ文・単位内の動きを消す)
  U_REF10/30    目標の別の文(発話 41〜59・10s / 30s)の単位平均を T0 の単位の並びで(参照に無い単位は参照の平均)
  U_REF*_SRCU   同じ表を R0 自身の単位の並び(男声の内容 = 変換器が実際に見る側)で置く(男女の単位の不一致の代価)
  MAP_REF30     R0 自身の単位で、R0 の単位平均 → 目標の参照 30s の単位平均 の差だけを掛ける(元話者の単位内の動きを残す)
単位の並びは T0 側(女声どうしの割り当て)。MAP は R0 側の割り当て(男女をまたぐ)。健全性: 同じ文の DTW 対応での単位の一致率。

    uv run python artic_g2_unit.py --ladder <scratchpad>/r4_spk --out ../results/artic_inv/g2_unit.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
VC = ROOT / "data/vctk/VCTK-Corpus/VCTK-Corpus"
CV_HOP = 0.02
CV_C = 0.0125


def utt(s: str, u: int) -> Path:
    return VC / f"wav48/{s}/{s}_{u:03d}.wav"


class CV:
    def __init__(self, dev: str):
        from transformers import HubertModel
        self.m = HubertModel.from_pretrained("lengyue233/content-vec-best").to(dev).eval()
        self.dev = dev

    @torch.no_grad()
    def __call__(self, x48: np.ndarray) -> np.ndarray:
        import librosa
        y = librosa.resample(x48.astype(np.float64), orig_sr=48000, target_sr=16000).astype(np.float32)
        h = self.m(torch.from_numpy(y)[None].to(self.dev)).last_hidden_state[0]
        return torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy()


class E2:
    """既存の因果な ContentVec の生徒(results/diag_e2・44.1kHz の因果 mel80・172fps・先読み 0)。フレーム j の窓は (j+1)·256/44100 s に閉じる。"""
    hop_s = 256 / 44100
    c_s = 0.0

    def __init__(self, dev: str):
        import ship_front as SF
        from train_vc_e import E1
        st = torch.load(ROOT / "results/diag_e2/diag_e2_best.pt", map_location="cpu", weights_only=False)
        self.net = E1(dim=st["args"]["dim"], layers=st["args"]["L"], look=st["args"]["look"]).to(dev).eval()
        self.net.load_state_dict(st["net"])
        self.SF, self.dev = SF, dev

    @torch.no_grad()
    def __call__(self, x48: np.ndarray) -> np.ndarray:
        import librosa
        from causal_mel import causal_mel
        w = torch.from_numpy(librosa.resample(x48.astype(np.float64), orig_sr=48000, target_sr=44100).astype(np.float32)) * 32768.0
        mel = causal_mel(w, n_fft=self.SF.NFFT_A, hop=self.SF.HOP_A, num_mels=self.SF.N_MEL, sr=44100)[0]
        h = self.net(mel[None].to(self.dev))[0].T
        return torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy()


def kmeans(X: np.ndarray, k: int, dev: str, iters: int = 40, seed: int = 0) -> np.ndarray:
    g = torch.Generator(device="cpu").manual_seed(seed)
    Xt = torch.from_numpy(X).to(dev)
    C = Xt[torch.randperm(len(Xt), generator=g)[:k].to(dev)].clone()
    for _ in range(iters):
        a = (Xt @ C.T).argmax(1)
        S = torch.zeros_like(C).index_add_(0, a, Xt)
        cnt = torch.bincount(a, minlength=k)
        empty = cnt == 0
        C = torch.nn.functional.normalize(S, dim=-1)
        if empty.any():
            C[empty] = Xt[torch.randint(len(Xt), (int(empty.sum()),), generator=g).to(dev)]
    return C.cpu().numpy()


FEAT = {"hop": CV_HOP, "c": CV_C, "floor": False}


def frame_units(cv_feat: np.ndarray, C: np.ndarray, n_frames: int) -> np.ndarray:
    import nvoc as N
    u = (cv_feat @ C.T).argmax(1)
    t = np.arange(n_frames) * N.HOP / N.SR
    if FEAT["floor"]:
        j = np.clip(np.floor(t / FEAT["hop"]).astype(int), 0, len(u) - 1)
    else:
        j = np.clip(np.round((t - FEAT["c"]) / FEAT["hop"]).astype(int), 0, len(u) - 1)
    return u[j]


def unit_means(E: np.ndarray, u: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    M = np.zeros((E.shape[0], k))
    has = np.zeros(k, bool)
    for j in range(k):
        m = u == j
        if m.any():
            M[:, j] = E[:, m].mean(1)
            has[j] = True
    return M, has


def smooth(E: np.ndarray, w: int = 5) -> np.ndarray:
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(E, size=w, axis=1, mode="nearest")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, nargs="+", default=[50, 200])
    ap.add_argument("--n", type=int, default=45)
    ap.add_argument("--feat", default="cv", choices=("cv", "e2"), help="単位の特徴: ContentVec(非因果・教師)/ E2(因果な生徒・製品の経路)")
    a = ap.parse_args()
    lad = Path(a.ladder)
    sys.path.insert(0, str(lad))
    import gen as G
    from train_dec2 import load48
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    J = json.loads((lad / "jobs.json").read_text())
    mf = [j for j in J["jobs"] if j[0] == "MF"][: a.n]
    fems, males = J["fems"], J["males"]
    cv = CV(dev) if a.feat == "cv" else E2(dev)
    if a.feat == "e2":
        FEAT.update(hop=E2.hop_s, c=0.0, floor=True)
    outdir = ROOT / ("results/artic_inv/g2_unit_wav" if a.feat == "cv" else "results/artic_inv/g2_unit_wav_e2")
    outdir.mkdir(parents=True, exist_ok=True)

    pool = []
    for s in fems + males:
        for u in range(41, 52):
            if utt(s, u).exists():
                pool.append(cv(G.trim(load48(str(utt(s, u))).astype(np.float64))))
    P = np.concatenate(pool)
    print("codebook frames", len(P), flush=True)
    books = {k: kmeans(P, k, dev) for k in a.k}

    def ref_set(s: str, secs: float) -> list:
        xs, tot = [], 0.0
        for u in range(41, 60):
            if tot >= secs:
                break
            if utt(s, u).exists():
                x = G.trim(load48(str(utt(s, u))).astype(np.float64))
                xs.append(x)
                tot += len(x) / 48000
        return xs

    stats: dict = {"agree": {k: [] for k in a.k}, "ref_secs": {10: [], 30: []}, "cover": {f"{k}_{s}": [] for k in a.k for s in (10, 30)}}
    for tag, src, tgt, ratio in mf:
        xr, _ = sf.read(lad / "wav" / f"MF__{src}__{tgt}__R0.wav")
        xt, _ = sf.read(lad / "wav" / f"MF__{src}__{tgt}__T0.wav")
        xr, xt = xr.astype(np.float64), xt.astype(np.float64)
        Xb, Xo = G.stft(xr), G.stft(xt)
        Eb, Eo = G.env(G.logmel(Xb), 24), G.env(G.logmel(Xo), 24)
        pos = G.align_map(xr, xt, Xb.shape[1], Eo.shape[1])
        jo = np.clip(np.round(pos).astype(int), 0, Eo.shape[1] - 1)
        fr, ft = cv(xr), cv(xt)
        refs = {sec: ref_set(tgt, sec) for sec in (10, 30)}
        for sec in (10, 30):
            stats["ref_secs"][sec].append(sum(len(x) for x in refs[sec]) / 48000)
        conds = {"ORACLE": G.interp_frames(Eo, pos), "ORACLE_SM": smooth(G.interp_frames(Eo, pos))}
        for k, C in books.items():
            ut = frame_units(ft, C, Eo.shape[1])
            ub = frame_units(fr, C, Eb.shape[1])
            stats["agree"][k].append(float((ub == ut[jo]).mean()))
            Mt0, _ = unit_means(Eo, ut, k)
            conds[f"U_T0_k{k}"] = smooth(Mt0[:, ut[jo]])
            for sec in (10, 30):
                Er = [G.env(G.logmel(G.stft(x)), 24) for x in refs[sec]]
                ur = [frame_units(cv(x), C, e.shape[1]) for x, e in zip(refs[sec], Er)]
                Ecat, ucat = np.concatenate(Er, 1), np.concatenate(ur)
                Mr, has = unit_means(Ecat, ucat, k)
                Mr[:, ~has] = Ecat.mean(1, keepdims=True)
                stats["cover"][f"{k}_{sec}"].append(float(has[ut[jo]].mean()))
                conds[f"U_REF{sec}_k{k}"] = smooth(Mr[:, ut[jo]])
                conds[f"U_REF{sec}_SRCU_k{k}"] = smooth(Mr[:, ub])
                if sec == 30:
                    Mb, hb = unit_means(Eb, ub, k)
                    off = Mr[:, ub] - Mb[:, ub]
                    conds[f"MAP_REF30_k{k}"] = Eb + smooth(off)
        for name, seq in conds.items():
            if (outdir / f"MF__{src}__{tgt}__{name}.wav").exists():
                continue
            g = np.clip(seq - Eb, -6, 6)
            y = G.istft(Xb * np.exp(G.to_lin(g)), len(xr))
            sf.write(outdir / f"MF__{src}__{tgt}__{name}.wav", G.norm(y).astype(np.float32), 48000)
        print(src, tgt, "agree", {k: round(v[-1], 3) for k, v in stats["agree"].items()}, "ref s", [round(stats["ref_secs"][s][-1], 1) for s in (10, 30)], flush=True)

    from a2_dsp_vc import ecapa
    from transformers import AutoFeatureExtractor, WavLMForXVector
    import librosa
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    wm = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval().float()

    def emb_w(x48):
        y = librosa.resample(x48.astype(np.float64), orig_sr=48000, target_sr=16000).astype(np.float32)
        with torch.no_grad():
            v = wm(**fe(y, sampling_rate=16000, return_tensors="pt")).embeddings[0].double().numpy()
        return v / np.linalg.norm(v)
    emb_e = ecapa()
    rep: dict = {"n": len(mf), "unit_agree_R0_vs_T0": {k: round(float(np.mean(v)), 3) for k, v in stats["agree"].items()},
                 "ref_secs_mean": {s: round(float(np.mean(v)), 1) for s, v in stats["ref_secs"].items()},
                 "ref_unit_coverage_of_T0_frames": {k: round(float(np.mean(v)), 3) for k, v in stats["cover"].items()}}
    rep["feat"] = a.feat
    names = sorted({p.stem.split("__")[-1] for p in outdir.glob("MF__*.wav")})
    for name, f_ in (("ecapa", emb_e), ("wavlm", emb_w)):
        Cn = {}
        for s in fems:
            c = np.mean([f_(load48(str(utt(s, u))).astype(np.float64)[:8 * 48000]) for u in [u for u in range(60, 200) if utt(s, u).exists()][:3]], 0)
            Cn[s] = c / np.linalg.norm(c)
        res = {}
        for cond in names + ["R0"]:
            d = lad / "wav" if cond == "R0" else outdir
            rk = []
            for _, src, tgt, _ in mf:
                f = d / f"MF__{src}__{tgt}__{cond}.wav"
                if not f.exists():
                    continue
                x, _ = sf.read(f)
                v = f_(x.astype(np.float64)[: 8 * 48000])
                sims = {gg: float(v @ Cn[gg]) for gg in fems}
                rk.append(1 + sum(1 for gg in fems if gg != tgt and sims[gg] > sims[tgt]))
            rk = np.array(rk)
            res[cond] = {"n": len(rk), "top1": round(float((rk == 1).mean()), 3), "top5": round(float((rk <= 5).mean()), 3), "median_rank": float(np.median(rk))}
            print(name, cond, res[cond], flush=True)
        rep[name] = res
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
