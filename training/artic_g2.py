"""構音の界面 G2(話者を運ぶか・current/artic_inv.md §5): 物理モデル(vtract + 声門)の包絡の時間軌道で話者が動くか。学習なし。

材料 = a2vc 設計レビュー 4 巡目の差し替えラダー(VCTK 男 → 女 45 組・同文 3〜6・R0 = 男声を RRPS で音域移動・T0 = 目標の同じ文)。
条件 PHYS: R0 と T0 に artic_fit で物理モデルを当てはめ、両側とも有声のフレームだけ 物理モデルの包絡の差(T0 を DTW 整列 − R0)を R0 の STFT に掛ける
  (それ以外のフレームは変えない)。対照 D24V: 同じフレームだけ DCT24 の差を掛ける = 同じフレームでの比較。単位は倍音の振幅(dB)どうしの差なので相殺する。
採点: 女声 45 人の中での目標の順位(ECAPA・WavLM-SV・重心 = 各話者の別の 3 発話)。同じ採点で H1_24(DCT24)・R0 も測る。

    uv run python artic_g2.py --ladder <scratchpad>/r4_spk --out ../results/artic_inv/g2.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).parent))
import artic_fit as AF

ROOT = Path(__file__).resolve().parent.parent
VC = ROOT / "data/vctk/VCTK-Corpus/VCTK-Corpus"


def phys_env_mel(x: np.ndarray, cb: tuple, dev: str, G) -> tuple[np.ndarray, np.ndarray, float]:
    """x(48k)→ E_dct [128, T](ln・DCT24 の log-mel 包絡・200fps)と P [128, T](物理モデルの包絡・ln 単位・mel の中心・無声は NaN)。"""
    import nvoc as N
    X = G.stft(x)
    E = G.env(G.logmel(X), 24)
    f0, _ = AF.harvest(x)
    fr, am, mk = AF.harmonic_obs(x, f0)
    r = AF.fit(fr, am, mk, f0, iters=1000 if AF.K == 6 else 2500, dev=dev, cb=cb)
    vt = r["voiced"]
    B = AF.basis().to(dev)
    mc = torch.from_numpy(G.mc.astype(np.float64)).to(dev)
    with torch.no_grad():
        q = torch.from_numpy(r["q"][vt]).to(dev)
        src = torch.from_numpy(r["src"]).to(dev)
        logL = torch.tensor(math.log(r["L"]), device=dev)
        f0v = torch.from_numpy(f0[vt].astype(np.float64)).to(dev)
        z = torch.zeros(len(vt), device=dev)
        env_db = AF.model_db(q, logL, src, z, mc[None].expand(len(vt), -1), B, f0v)
        obs = torch.from_numpy(am[vt]).to(dev)
        mkt = torch.from_numpy(mk[vt]).to(dev).double()
        top = torch.where(mkt > 0, obs, torch.full_like(obs, -1e9)).amax(1, keepdim=True)
        w = mkt * torch.sigmoid((obs - (top - 25.0)) / 3.0)
        mh = AF.model_db(q, logL, src, z, torch.from_numpy(fr[vt]).to(dev), B, f0v)
        gain = ((obs - mh) * w).sum(1) / w.sum(1).clamp(min=1e-9)
        env_ln = ((env_db + gain[:, None]) * math.log(10) / 20).cpu().numpy()
    full = np.full((len(f0), 128), np.nan)
    full[vt] = env_ln
    Tm = E.shape[1]
    j = np.clip(np.round(np.arange(Tm) * N.HOP / AF.HOP).astype(int), 0, len(f0) - 1)
    P = full[j].T
    return E, P, r["err_db"]


INV = {}


def inv_env_mel(x: np.ndarray, dev: str, G) -> tuple[np.ndarray, np.ndarray, float]:
    """学習した推定器(train_artic_inv)の出力 q・src・L で物理の包絡を作る(利得は倍音の観測へ閉じた式)。返り値は phys_env_mel と同じ形。"""
    import nvoc as N
    import artic_dsp as D
    import train_artic_inv as TI
    from artic_prep import causal_logmel
    net = INV["net"]
    X = G.stft(x)
    E = G.env(G.logmel(X), 24)
    f0, _ = AF.harvest(x)
    fr, am, mk = AF.harmonic_obs(x, f0)
    n = len(x) // AF.HOP * AF.HOP
    T = n // AF.HOP
    fb = N.mel_fb().numpy().astype(np.float64)
    pad = np.concatenate([np.zeros(1024 - AF.HOP), x[:n]])
    frm = np.lib.stride_tricks.sliding_window_view(pad, 1024)[::AF.HOP][:T] * np.hanning(1024)
    mel = np.log(np.maximum(np.abs(np.fft.rfft(frm, n=2048, axis=1)) @ fb.T, 1e-5)).T
    y, _ = D.causal_yin(x[:n], voi_max=0.45)
    y = np.pad(y[0::2][:T], (0, max(0, T - len(y[0::2][:T]))))
    with torch.no_grad():
        q, src, lL = net(torch.from_numpy(mel).float()[None].to(dev), torch.from_numpy(y).double()[None].to(dev))
    q, src, lL = q[0], src[0], lL[0]
    Tn = min(T, len(f0))
    vt = np.nonzero(mk[:Tn].any(1))[0]
    B = INV["B"]
    mc = torch.from_numpy(G.mc.astype(np.float64)).to(dev)
    with torch.no_grad():
        qv, sv, lv = q[vt], src[vt], lL[vt]
        f0v = torch.from_numpy(f0[vt].astype(np.float64)).to(dev)
        env_db = TI.model_db_frames(qv, lv, sv, mc[None].expand(len(vt), -1), f0v, B)
        mh = TI.model_db_frames(qv, lv, sv, torch.from_numpy(fr[vt]).to(dev), f0v, B)
        obs = torch.from_numpy(am[vt]).to(dev)
        mkt = torch.from_numpy(mk[vt]).to(dev).double()
        top = torch.where(mkt > 0, obs, torch.full_like(obs, -1e9)).amax(1, keepdim=True)
        w = mkt * torch.sigmoid((obs - (top - 25.0)) / 3.0)
        gain = ((obs - mh) * w).sum(1) / w.sum(1).clamp(min=1e-9)
        err = float(torch.sqrt((((mh + gain[:, None] - obs) ** 2) * w).sum() / w.sum()))
        env_ln = ((env_db + gain[:, None]) * math.log(10) / 20).cpu().numpy()
    full = np.full((len(f0), 128), np.nan)
    full[vt] = env_ln
    Tm = E.shape[1]
    j = np.clip(np.round(np.arange(Tm) * N.HOP / AF.HOP).astype(int), 0, len(f0) - 1)
    INV.setdefault("traj", []).append(q.cpu().numpy())
    c = (G.Dm @ G.logmel(X))[1:25].T[::2]
    INV.setdefault("dct", []).append(c)
    return E, full[j].T, err


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ladder", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=45)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--ls", action="store_true", help="壁の損失(帯域幅)をフレームごとに可変")
    ap.add_argument("--inv", default=None, help="学習した推定器の重み(results/<tag>/last.pt)。与えると最適化の代わりにこれで包絡を作る(条件名 INV)")
    a = ap.parse_args()
    AF.K = a.k
    AF.LOSS_SCALE = a.ls
    tagc = "PHYS" if (a.k == 6 and not a.ls) else f"PHYS_K{a.k}{'LS' if a.ls else ''}"
    if a.inv:
        import train_artic_inv as TI
        st0 = torch.load(a.inv, map_location="cpu", weights_only=False)
        net = TI.Inv(st0["k"]).to("cuda" if torch.cuda.is_available() else "cpu")
        net.load_state_dict(st0["net"])
        net.eval()
        INV["net"], INV["B"] = net, AF.basis(AF.N_SEC, st0["k"]).to(next(net.parameters()).device)
        tagc = f"INV_{Path(a.inv).parent.name}_{st0['step'] // 1000}k"
    lad = Path(a.ladder)
    sys.path.insert(0, str(lad))
    import gen as G
    torch.set_default_dtype(torch.float64)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    AF.K = 6
    cb = AF.codebook(dev=dev) if not a.inv else None
    AF.K = a.k
    J = json.loads((lad / "jobs.json").read_text())
    mf = [j for j in J["jobs"] if j[0] == "MF"][: a.n]
    outdir = ROOT / "results/artic_inv/g2_wav"
    outdir.mkdir(parents=True, exist_ok=True)
    errs = []
    for tag, src, tgt, ratio in mf:
        if (outdir / f"MF__{src}__{tgt}__{tagc}.wav").exists():
            continue
        xr, _ = sf.read(lad / "wav" / f"MF__{src}__{tgt}__R0.wav")
        xt, _ = sf.read(lad / "wav" / f"MF__{src}__{tgt}__T0.wav")
        xr, xt = xr.astype(np.float64), xt.astype(np.float64)
        envf = inv_env_mel if a.inv else (lambda xx, d_, g_: phys_env_mel(xx, cb, d_, g_))
        Eb, Pb, eb = envf(xr, dev, G)
        Eo, Po, eo = envf(xt, dev, G)
        Xb = G.stft(xr)
        pos = G.align_map(xr, xt, Xb.shape[1], Eo.shape[1])
        jo = np.clip(np.round(pos).astype(int), 0, Po.shape[1] - 1)
        Pa = Po[:, jo]
        both = ~np.isnan(Pb[0]) & ~np.isnan(Pa[0])
        gd = G.interp_frames(Eo, pos) - Eb
        gp = np.zeros_like(gd)
        gp[:, both] = Pa[:, both] - Pb[:, both]
        gv = np.zeros_like(gd)
        gv[:, both] = gd[:, both]
        for cond, g in ((tagc, gp), ("D24V", gv)):
            y = G.istft(Xb * np.exp(G.to_lin(np.clip(g, -6, 6))), len(xr))
            sf.write(outdir / f"MF__{src}__{tgt}__{cond}.wav", G.norm(y).astype(np.float32), 48000)
        errs.append((eb, eo))
        print(src, tgt, "fit err R0/T0", round(eb, 2), round(eo, 2), "both-voiced frames", round(float(both.mean()), 2), flush=True)
    from a2_dsp_vc import ecapa
    from transformers import AutoFeatureExtractor, WavLMForXVector
    import librosa
    from train_dec2 import load48
    fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    wm = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval().float()

    def emb_w(x48):
        y = librosa.resample(x48.astype(np.float64), orig_sr=48000, target_sr=16000).astype(np.float32)
        with torch.no_grad():
            v = wm(**fe(y, sampling_rate=16000, return_tensors="pt")).embeddings[0].double().numpy()
        return v / np.linalg.norm(v)
    emb_e = ecapa()
    fems = J["fems"]

    def utt(s, u):
        return VC / f"wav48/{s}/{s}_{u:03d}.wav"
    rep: dict = {"fit_err_db_mean_this_run": round(float(np.mean(errs)), 2) if errs else None, "n": len(mf)}
    for name, f_ in (("ecapa", emb_e), ("wavlm", emb_w)):
        C = {}
        for s in fems:
            c = np.mean([f_(load48(str(utt(s, u))).astype(np.float64)[:8 * 48000]) for u in [u for u in range(60, 200) if utt(s, u).exists()][:3]], 0)
            C[s] = c / np.linalg.norm(c)
        res = {}
        conds = [(tagc, outdir)] + [(c, outdir) for c in ("PHYS", "PHYS_K16LS", "D24V_LO") if c != tagc] + [("D24V", outdir), ("H1_24", lad / "wav"), ("R0", lad / "wav")]
        for cond, d in conds:
            rk = []
            for _, src, tgt, _ in mf:
                if not (d / f"MF__{src}__{tgt}__{cond}.wav").exists():
                    continue
                x, _ = sf.read(d / f"MF__{src}__{tgt}__{cond}.wav")
                v = f_(x.astype(np.float64)[: 8 * 48000])
                sims = {gg: float(v @ C[gg]) for gg in fems}
                rk.append(1 + sum(1 for gg in fems if gg != tgt and sims[gg] > sims[tgt]))
            rk = np.array(rk)
            if len(rk) == 0:
                continue
            res[cond] = {"n": len(rk), "top1": round(float((rk == 1).mean()), 3), "top5": round(float((rk <= 5).mean()), 3), "median_rank": float(np.median(rk))}
            print(name, cond, res[cond], flush=True)
        rep[name] = res
    if a.inv and INV.get("traj"):
        from scipy.signal import butter, sosfiltfilt
        sos = butter(2, [4 / 50, 15 / 50], btype="band", output="sos")
        sos2 = butter(2, [15 / 50, 49 / 50], btype="band", output="sos")
        def bands(trajs):
            r4, r15 = [], []
            for q in trajs:
                if len(q) < 60:
                    continue
                nat = q.std(0) + 1e-9
                r4.append(np.sqrt((sosfiltfilt(sos, q, axis=0) ** 2).mean(0)) / nat)
                r15.append(np.sqrt((sosfiltfilt(sos2, q, axis=0) ** 2).mean(0)) / nat)
            return round(float(np.median(r4)), 3), round(float(np.median(r15)), 3)
        qa, qb = bands(INV["traj"])
        da, db = bands(INV["dct"])
        rep["g3_traj_band_over_nat"] = {"q_4_15Hz": qa, "q_15_50Hz": qb, "dct24_4_15Hz": da, "dct24_15_50Hz": db,
                                        "note": "推定器の q の帯域成分 / q 自身の標準偏差(自然な発話の構音の速い動きを含む = 上限。実音声の DCT24 の同じ比と並べる)"}
        print("G3", rep["g3_traj_band_over_nat"], flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
