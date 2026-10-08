"""D1 G0/G1判定(prereg: results/d1_g0/prereg.yaml)・v2(bug7修正 2026-09-23)。

bug7: 旧版はGT参照を**正規化潜在のまま**decodeしていた(生成側は逆正規化してdecode)。
参照音声はRMS約2倍・hi_mid 1/2〜1/9に歪み、全hi_mid比が発話依存の係数で水増しされた
(証拠 results/d1_g0/reeval_bug7.json)。v2は参照を生スケールでdecodeし、起動時に
元wavとの一致(RMS±3dB・hi_mid比0.4〜1.5)をassertする(codecは高域を減衰: 正常0.50〜1.01・bug7型0.11〜0.30)。

各腕(D1AR・CFMYS[s7/s11]とも同一物差し)について、N話者×1発話×3seed:
- hi_mid比(高域ノイズ代理・コーラス盲目と実測済み results/earbattery/chorus_proxy_validation.json)
- env_l1(線形周波数logパワーのL1 vs 生参照。melではなく倍音・ピッチ誤差も含む総合スペクトル差=包絡代理ではない)
- コーラス補助代理 d_ncc / d_comb(生成−参照・chorus_proxy.py・弱検証)
- f0_st(decode音f0中央値の参照比[半音])・d_voiced(有声率差) — 代理の良化が退化(ピッチ崩落)でないことの確認
- D1ARのみ: 衝突掃引・プライム切替ホライズン(同一窓の生参照で正規化・潜在誤差rmse/lag1自己相関)
出力は腕ごと results/<tag>/g0_report_v2_<source>_<ckpt>_K<K>.json(上書き衝突なし)。

    CUDA_VISIBLE_DEVICES=0 uv run python eval_d1_g0.py --arms d1_g0,d1_g0par --utt-source train
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from d1_model import D1AR, Z_DIM, shift_right, sample_frame_ar, COND_DIM, COND_DIM_M80
from train_d1 import build_index, cond_of
from train_cfmys import LAT, F0FIX, CFMYS, sample_k
from diag_cfm_audit import decode_f0
from eval_d4b_gates import band_metrics
import chorus_proxy as cp

ROOT = Path(__file__).resolve().parent.parent
LF0_ROW = 768
SR = 48000


def logmel_l1(y: np.ndarray, yg: np.ndarray) -> float:
    def m(w):
        w44 = librosa.resample(w.astype(np.float64), orig_sr=SR, target_sr=44100)
        return np.log(np.clip(np.abs(librosa.stft(w44, n_fft=1024, hop_length=256)) ** 2,
                              1e-8, None))
    a, b = m(y), m(yg)
    T = min(a.shape[1], b.shape[1])
    return float(np.abs(a[:, :T] - b[:, :T]).mean())


def assert_ref_matches_source(y_ref: np.ndarray, wav_path: str, n: int) -> dict:
    src, _ = librosa.load(wav_path, sr=SR, mono=True)
    src = src[:n].astype(np.float64)
    ref = y_ref[:len(src)].astype(np.float64)
    db = 20 * np.log10(np.sqrt((ref ** 2).mean() + 1e-12) / np.sqrt((src ** 2).mean() + 1e-12))
    hr = band_metrics(ref)["hi_mid"] / max(band_metrics(src)["hi_mid"], 1e-4)
    ok = abs(db) <= 3.0 and 0.4 <= hr <= 1.5
    assert ok, (f"参照decodeが元wavと不一致(bug7型スケール取り違え): RMS {db:+.2f}dB "
                f"hi_mid比 {hr:.2f} ({wav_path})")
    return {"ref_vs_src_db": round(float(db), 2), "ref_vs_src_hi_mid": round(float(hr), 2)}


def load_arm(tag: str, dev: str, which: str):
    ck = torch.load(ROOT / "results" / tag / f"{tag}_{which}.pt", map_location=dev,
                    weights_only=False)
    if ck.get("args", {}).get("arch") == "cfmys":
        cli = ck["cli"]
        net = CFMYS(dim=cli.get("dim", 384), cin=850 if cli.get("mel_in") else 770,
                    spk_in=bool(cli.get("spk_in"))).to(dev).eval()
        net.load_state_dict(ck["net"])
        return "cfm", net, ck
    mel80 = bool(ck["cli"].get("mel80"))
    net = D1AR(dim=ck["cli"].get("dim", 256), layers=ck["cli"].get("layers", 12),
               cond_dim=COND_DIM_M80 if mel80 else COND_DIM,
               no_history=bool(ck["cli"].get("no_history"))).to(dev).eval()
    net.load_state_dict(ck["net"])
    return "d1", net, ck


def sample_hist_gt(net, cond, z_gt_n, K=8, seed=0, prime_until=None, z0_rho: float = 0.0):
    """GT履歴teacher-forced条件付きサンプル。prime_until=PでP以降は自己履歴に切替
    (切替時点の履歴窓はGT=GTプライム)。z0はsample_frame_arと同じAR(1)規約。
    注: GT履歴モードの各フレームは互いの標本でなくGTに条件付く=フレーム独立な偏差を
    構造的に持つ(自由走行の上限ではない)。"""
    B, C, T = cond.shape
    dev = cond.device
    RF = net.rf
    torch.manual_seed(seed)
    e = torch.randn(B, Z_DIM, T, device=dev)
    z0 = e.clone()
    for j in range(1, T):
        z0[:, :, j] = z0_rho * z0[:, :, j - 1] + math.sqrt(1 - z0_rho ** 2) * e[:, :, j]
    hist_seq = shift_right(z_gt_n)
    out = torch.zeros(B, Z_DIM, T, device=dev)
    zbuf = torch.zeros(B, Z_DIM, RF + 1, device=dev)
    with torch.no_grad():
        for i in range(T):
            if prime_until is None or i <= prime_until:
                lo = max(0, i - RF)
                win_z = hist_seq[:, :, lo:i + 1]
                if win_z.shape[2] < RF + 1:
                    win_z = F.pad(win_z, (RF + 1 - win_z.shape[2], 0))
                if prime_until is not None and i == prime_until:
                    zbuf = win_z.clone()
            else:
                win_z = zbuf
            cw = cond[:, :, max(0, i - RF):i + 1]
            cwin = F.pad(cw, (RF + 1 - cw.shape[2], 0))
            ft = net.feat(win_z, cwin)[:, :, -1]
            z = z0[:, :, i]
            for k in range(K):
                tk = torch.full((B,), k / K, device=dev)
                z = z + net.vel(z, ft, tk) / K
            out[:, :, i] = z
            zbuf = torch.cat([zbuf[:, :, 1:], z[:, :, None]], -1)
    return out


def pick_utts(pairs, lats, held_spk, source: str, n_spk: int, min_T: int = 400):
    hset = set(held_spk)
    spk_all = sorted({f.parent.name for f in pairs})
    spks = held_spk if source == "held" else [s for s in spk_all if s not in hset][:10]
    out = []
    for s in spks:
        for f in sorted(f for f in pairs if f.parent.name == s):
            if torch.load(lats[f.stem], map_location="cpu", weights_only=False)["z"].shape[0] >= min_T:
                out.append(f)
                break
        if len(out) >= n_spk:
            break
    return out


def ac1(e: torch.Tensor) -> float:
    e = e - e.mean(1, keepdim=True)
    return float((e[:, 1:] * e[:, :-1]).sum() / (e.pow(2).sum() + 1e-9))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="d1_g0,d1_g0m80,d1_g0par")
    ap.add_argument("--n-spk", type=int, default=6)
    ap.add_argument("--utt-source", choices=["train", "held"], default="train",
                    help="train=seen話者(先頭10話者=overfit集合内)/held=末尾24話者")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--ckpt", choices=["best", "last"], default="last")
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--z0-rho", type=float, default=None,
                    help="D1のz0相関(既定=学習時cli値)・CFMYSは学習時rho固定")
    ap.add_argument("--no-proxy", action="store_true")
    ap.add_argument("--save-wav", action="store_true")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [int(s) for s in a.seeds.split(",")]

    from causal_codec import CausalCodec
    ckc = torch.load(ROOT / "results/s1_3_c32/s1_3_c32_last.pt", map_location=dev,
                     weights_only=False)
    codec = CausalCodec(latent_dim=32, channels=(32, 64, 128, 256, 512)).to(dev)
    codec.load_state_dict(ckc.get("ema") or ckc["net"])
    codec.eval()
    abi = torch.load(LAT / "abi.pt", map_location=dev, weights_only=False)
    MU, SD = abi["mu"].to(dev), abi["sd"].to(dev)
    spk_emb = torch.load(ROOT / "data/ecapa_spk_mean_full.pt", map_location="cpu",
                         weights_only=False)

    def dec(z_raw: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return codec.decode(z_raw[None].to(dev))[0, 0].cpu().numpy()

    pairs, lats, held_spk = build_index(0)
    utts = pick_utts(pairs, lats, held_spk, a.utt_source, a.n_spk)

    mel_cache: dict = {}

    def mel_of(f, d):
        if f not in mel_cache:
            from causal_mel import causal_mel
            wv, _ = librosa.load(d["path"], sr=44100, mono=True)
            mel_cache[f] = causal_mel(torch.from_numpy(wv) * 32768.0, n_fft=1024, hop=256,
                                      num_mels=80, sr=44100)[0].half()
        return mel_cache[f]

    def load_item(f):
        d = torch.load(f, map_location="cpu", weights_only=False)
        d = {**d, "f0": torch.load(F0FIX / f.parent.name / f.name, map_location="cpu",
                                   weights_only=False)["f0"]}
        z = torch.load(lats[f.stem], map_location="cpu", weights_only=False)["z"].float().T
        return d, z

    refs = {}
    for f in utts:
        d, z = load_item(f)
        T = min(z.shape[1], 600)
        y_ref = dec(z[:, :T].to(dev))
        chk = assert_ref_matches_source(y_ref, d["path"], len(y_ref))
        m_ref = None if a.no_proxy else cp.measure(y_ref.astype(np.float64))
        chk["f0_ref"] = decode_f0(np.clip(y_ref, -1, 1))
        refs[f] = (d, z, T, y_ref, chk, m_ref)

    for tag in a.arms.split(","):
        kind, net, ck = load_arm(tag, dev, a.ckpt)
        cli = ck["cli"]
        out_dir = ROOT / "results" / tag
        res = {"eval": "v2(bug7修正)", "ckpt": a.ckpt, "step": int(ck["step"]), "K": a.K,
               "utt_source": a.utt_source, "utts": {}}
        agg = {"hi_mid": [], "env": [], "d_ncc": [], "d_comb": [], "f0_st": [], "d_voiced": []}
        for f in utts:
            d, z, T, y_ref, chk, m_ref = refs[f]
            if kind == "d1":
                m = mel_of(f, d) if cli.get("mel80") else None
            else:
                m = mel_of(f, d) if cli.get("mel_in") else None
            cond = cond_of(d, T, m)[None].to(dev)
            per = {"ref_check": chk, "hi_mid": [], "env": [], "d_ncc": [], "d_comb": [],
                   "f0_st": [], "d_voiced": []}
            for s in seeds:
                with torch.no_grad():
                    if kind == "d1":
                        rho = a.z0_rho if a.z0_rho is not None else float(cli.get("z0_rho", 0))
                        zh = sample_frame_ar(net, cond, K=a.K, seed=s, z0_rho=rho)
                        zr = (zh[0] * SD[:, None] + MU[:, None]).clamp(-8, 8)
                    else:
                        s_ = None if cli.get("no_spk") else spk_emb.get(d.get("speaker"))
                        s_ = s_[None].to(dev) if s_ is not None else None
                        g = torch.Generator(device=dev).manual_seed(s)
                        mu, sd = ck["abi"]["mu"].to(dev), ck["abi"]["sd"].to(dev)
                        zh = sample_k(net, T, cli.get("rho", 0.9), g, dev, cond, s_, a.K)
                        zr = (zh[0] * sd[:, None] + mu[:, None]).clamp(-8, 8)
                y = dec(zr)
                if a.save_wav:
                    soundfile.write(out_dir / f"v2_{f.stem}_s{s}{a.suffix}.wav",
                                    np.clip(y, -1, 1), SR)
                per["hi_mid"].append(round(band_metrics(y)["hi_mid"]
                                           / max(band_metrics(y_ref)["hi_mid"], 1e-4), 3))
                per["env"].append(round(logmel_l1(y, y_ref), 3))
                fg, fr = decode_f0(np.clip(y, -1, 1)), chk["f0_ref"]
                per["f0_st"].append(round(12 * np.log2(fg["f0_median"] / fr["f0_median"]), 2)
                                    if fg["f0_median"] > 0 and fr["f0_median"] > 0 else float("nan"))
                per["d_voiced"].append(round(fg["voiced_ratio"] - fr["voiced_ratio"], 3))
                if m_ref is not None:
                    mg = cp.measure(y.astype(np.float64))
                    per["d_ncc"].append(round(mg["period_ncc"] - m_ref["period_ncc"], 4))
                    per["d_comb"].append(round(mg["comb_db"] - m_ref["comb_db"], 3))
            for k in agg:
                agg[k] += per[k]
            res["utts"][f.stem] = per
        summ = {}
        for k, v in agg.items():
            if not v:
                continue
            arr = np.array(v, dtype=np.float64)
            if k == "d_voiced":
                summ[k] = {"median": round(float(np.nanmedian(arr)), 4),
                           "min": round(float(np.nanmin(arr)), 4),
                           "max": round(float(np.nanmax(arr)), 4)}
                continue
            if k == "f0_st":
                summ[k] = {"median": round(float(np.nanmedian(arr)), 2),
                           "median_abs": round(float(np.nanmedian(np.abs(arr))), 2),
                           "worst_abs": round(float(np.nanmax(np.abs(arr))), 2)}
                continue
            worst = arr.max() if k in ("hi_mid", "env") else arr.min()
            summ[k] = {"median": round(float(np.nanmedian(arr)), 4),
                       "worst": round(float(worst), 4)}
        res["summary"] = summ

        if kind == "d1":
            rho_d1 = a.z0_rho if a.z0_rho is not None else float(cli.get("z0_rho", 0))
            f = utts[0]
            d, z = load_item(f)
            T = min(z.shape[1], 400)
            zn = ((z[:, :T].to(dev) - MU[:, None]) / SD[:, None]).clamp(-8, 8)
            sweep, f0base = {}, None
            for st in (0.0, 7.0):
                cond = cond_of(d, T, mel_of(f, d) if cli.get("mel80") else None)[None].to(dev)
                if st:
                    hz = 200.0 * torch.exp(cond[0, LF0_ROW])
                    cond = cond.clone()
                    cond[0, LF0_ROW] = torch.log(
                        torch.where(hz > 50.5, hz * 2 ** (st / 12), hz) / 200.0)
                zh = sample_hist_gt(net, cond, zn[None], K=a.K, seed=0, z0_rho=rho_d1)
                y = dec((zh[0] * SD[:, None] + MU[:, None]).clamp(-8, 8))
                mm = decode_f0(np.clip(y, -1, 1))
                sweep[f"st{int(st)}"] = round(mm["f0_median"], 1)
                if st == 0.0:
                    f0base = mm["f0_median"]
            if f0base and sweep.get("st7"):
                sweep["follow_st"] = round(12 * np.log2(sweep["st7"] / max(f0base, 1e-3)), 2)
            res["conflict_sweep"] = sweep

            T = min(z.shape[1], 800)
            zr = z[:, :T].to(dev)
            zn2 = ((zr - MU[:, None]) / SD[:, None]).clamp(-8, 8)
            condp = cond_of(d, T, mel_of(f, d) if cli.get("mel80") else None)[None].to(dev)
            P = min(500, T - 250)
            zh = sample_hist_gt(net, condp, zn2[None], K=a.K, seed=0, prime_until=P,
                                z0_rho=rho_d1)
            yp = dec((zh[0] * SD[:, None] + MU[:, None]).clamp(-8, 8))
            ygt = dec(zr)
            err = zh[0] - zn2
            e_rms = err.pow(2).mean(0).sqrt()
            hor = {}
            for nm, (s0, e0) in {"prime": (P - 200, P), "+0-50": (P, P + 50),
                                 "+50-150": (P + 50, P + 150),
                                 "+150-250": (P + 150, min(P + 250, T))}.items():
                a2, b2 = s0 * 480, e0 * 480
                hor[nm] = {"hi_mid_vs_samewin_ref": round(
                               band_metrics(yp[a2:b2])["hi_mid"]
                               / max(band_metrics(ygt[a2:b2])["hi_mid"], 1e-4), 3),
                           "ref_win_over_global": round(
                               band_metrics(ygt[a2:b2])["hi_mid"]
                               / max(band_metrics(ygt)["hi_mid"], 1e-4), 2),
                           "latent_rmse": round(float(e_rms[s0:e0].mean()), 3),
                           "err_lag1": round(ac1(err[:, s0:e0]), 3)}
            hor["loop_gain"] = round(float(e_rms[P + 40:P + 50].mean()
                                           / e_rms[P:P + 10].mean().clamp(min=1e-6)), 2)
            res["prime_horizon"] = hor
        p = out_dir / f"g0_report_v2_{a.utt_source}_{a.ckpt}_K{a.K}{a.suffix}.json"
        p.write_text(json.dumps(res, indent=1, ensure_ascii=False))
        print(f"  {tag}[{a.ckpt}@{res['step']}] {json.dumps(summ, ensure_ascii=False)} -> {p}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
