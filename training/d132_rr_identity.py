"""D132(学習なし・R の同一性): train_rr.py の重みで、D106 と同じ組(評価用の男声 m000〜m023 → 女声 TTS の評価外 24 目標 × 5 文)を
操作者の時間のまま(DTW なし = 推論と同じ)で変換し、出力部(rvtab_f3 380k・D106 と同じ)で鳴らす:
  TABR = c0 = 男声の F2・包絡 = Σp_M T_A(A の別の 9 文の CheapTrick の表)・RR = R の出力(包絡と c0)・RRSWAP = R の記憶だけ別の目標の参照に(表は本物の A)・
  RRE = R の包絡 + 男声の c0・RRC = 表 + R の c0(D133: どちらが同一性を下げたか)。--sfx で条件名に接尾辞(重みごと)。
f0 = 男声の F1(確信度 0.5)を A の中央値へ移す・周期性 = 男声の F4 − 男声の平均 + A の平均(D106 と同じ式)。入力特徴は data/rr_feat(学習と同じ)。
出力 = <d106>/m{i}_{A}_t{k}_{cond}.wav → d118_identity_ceiling.py --dir <d106> --tag d132 --conds CONV TABR RR RRSWAP。
    uv run python d132_rr_identity.py --dir <scratchpad>/d106 --tag diag_rr1
--flow(diag_flow3c): --tag が F の重みのとき、過去なしの標本で FN<温度><sfx> だけを書く。D160: --scales 1.5,2 --std_out <json> = R の包絡の残差を s 倍した条件だけ + 残差の標準偏差(R vs 目標本人の実音声・有声)。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import train_rr as RRm

ROOT = Path(__file__).resolve().parent.parent
FEM = ROOT / "data/female_tts_corpus"
MPAR = ROOT / "data/male_tts_v4l_par"
STY = ["neutral", "breathy", "low_tension", "soft", "warm"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rvoc_ckpt", default=str(ROOT / "results/rvtab_f3/snap/ema_380k.pt"))
    ap.add_argument("--scales", default="", help="D160: R の包絡の残差(R − 表)を s 倍した条件 RRS<s><sfx> だけを書き、R と目標本人の残差の標準偏差の比を --std_out に")
    ap.add_argument("--std_out", default="")
    ap.add_argument("--flow", action="store_true", help="--tag が F(train_dynflow)の重み: 過去なしの標本(6 段)で FN<温度×10><sfx> を書く")
    ap.add_argument("--temps", default="1.0,0.7")
    ap.add_argument("--sfx", default="", help="条件名の接尾辞(例 2 → RR2・RRE2・RRC2・RRSWAP2)")
    a = ap.parse_args()
    import conv_c0 as C0
    import nvoc as N
    import rvoc as R
    import train_rvoc as TR
    dev = "cuda"
    od = Path(a.dir)
    st = RRm.Store()
    ck = torch.load(ROOT / "results" / a.tag / "last.pt", map_location="cpu", weights_only=False)
    if a.flow:
        import train_dynflow as FL
        fnet = FL.Flow(**{k: ck["cfg"][k] for k in ("nq", "ch", "heads", "dk", "K", "hid")}).to(dev).eval()
        fnet.load_state_dict(ck["net"])
        ck = {"cfg": fnet.rr.cfg, "net": fnet.rr.state_dict()}
    net = RRm.RR(**{k: ck["cfg"][k] for k in ("nq", "ch", "heads", "dk")}, **{k_: tuple(ck["cfg"][k_]) for k_ in ("qd", "kd") if k_ in ck["cfg"]}).to(dev).eval()
    net.load_state_dict(ck["net"])
    sr_ = torch.load(a.rvoc_ckpt, map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=sr_["cfg"]["ch"], kernels=tuple(sr_["cfg"]["kernels"]), dils=tuple(sr_["cfg"]["dils"]), d_cond=sr_["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(sr_["ema"]); gen.eval().requires_grad_(False)
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()

    @torch.no_grad()
    def render(c0n, env24, f0, per_t) -> np.ndarray:
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)[None]
        v = (f0 > 0).astype(np.float64)[None]
        cond = torch.from_numpy(np.concatenate([c0n[None], env24 / 10, lf, v, np.clip(per_t, 0, 1.5) * v], 0).astype(np.float32))[None].to(dev)
        y = gen(cond, TR.excitation(torch.from_numpy(f0.astype(np.float32))[None].to(dev), cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)

    def refmem(A: str, tk: int) -> tuple:
        refs = [st.f[("fem_neutral", A, f"t{t:02d}_neutral")] for t in range(10) if t != tk]
        rp = np.concatenate([RRm.dense(r["pk"], r["pv"]) for r in refs], 1)[:, :5000]
        renv = np.concatenate([r["env"] for r in refs], 1)[:, :5000]
        ru = np.concatenate([r["u"] for r in refs])[:5000]
        T = C0.table(renv[1:25], ru, RRm.K, Cb).astype(np.float32)
        rc0 = (renv[0] - RRm.C0_SIL) / 10
        rv = np.concatenate([renv[1:25] / 10, (rc0 - rc0.mean())[None]], 0).astype(np.float32)
        fts = np.concatenate([r["ft"] for r in refs])
        return rp, rv, T, float(np.median(np.log(fts[fts > 0])))

    males = sorted(d.name for d in MPAR.iterdir() if d.is_dir() and len(list(d.glob("f*.wav"))) == 10)[:24]
    fems = sorted(d.name for d in FEM.iterdir() if d.is_dir() and all((d / f"t{i:02d}_{s}.wav").exists() for i in range(10) for s in STY))
    random.Random(80).shuffle(fems)
    p4 = {}

    def p4mean(corpus: str, spk: str, stems: list) -> np.ndarray:
        if (corpus, spk) not in p4:
            ds = [torch.load(RRm.FEAT / corpus / spk / f"{s}.pt", weights_only=False) for s in stems]
            P = np.concatenate([d["p4"].float().numpy()[:, (d["f1"] > 0).numpy()] for d in ds], 1)
            p4[(corpus, spk)] = P.mean(1)
        return p4[(corpus, spk)]

    sacc: list = []
    with torch.no_grad():
        for mi, M in enumerate(males):
            A = fems[mi]
            Asw = fems[(mi + 5) % 24]
            sM = st.stats[M]
            m_stems = sorted(s for (c, sp, s) in st.f if c == "male_ref" and sp == M)
            pO = p4mean("male_ref", M, m_stems)
            for k in range(5):
                pre = f"{M}_{A}_t{k:02d}"
                scs = [float(z) for z in a.scales.split(",")] if a.scales else []
                if (od / (f"{pre}_FN{int(round(float(a.temps.split(',')[-1]) * 10)):02d}{a.sfx}.wav" if a.flow else f"{pre}_RRS{scs[-1]:g}{a.sfx}.wav" if scs else f"{pre}_RRC{a.sfx}.wav")).exists():
                    continue
                b = st.f[("male_par", M, f"f{k:02d}")]
                mfile = torch.load(RRm.FEAT / "male_par" / M / f"f{k:02d}.pt", weights_only=False)
                pM = RRm.dense(b["pk"], b["pv"])
                vM = b["f1"] > 0
                q = np.concatenate([pM, (b["e3"] - sM["e3"][:, None]) / 10, (b["c0f"] - sM["c0"])[None],
                                    np.where(vM, np.log(np.maximum(b["f1"], 1.0)) - sM["lf"], 0.0)[None], vM[None].astype(np.float32)], 0).astype(np.float32)
                rp, rv, T, muA = refmem(A, k)
                rp2, rv2, _, _ = refmem(Asw, k)
                pA = p4mean("fem_neutral", A, [f"t{t:02d}_neutral" for t in range(10) if t != k])
                tab = (T @ pM).astype(np.float32)
                f0o = np.where(vM, b["f1"] * np.exp(muA - sM["lf"]), 0.0)
                per_c = mfile["p4"].float().numpy() - pO[:, None] + pA[:, None]
                if ck["cfg"].get("no_op_env"):
                    q[RRm.K:RRm.K + 24] = 0
                qt = torch.from_numpy(q)[None].to(dev); tt = torch.from_numpy(tab)[None].to(dev)
                outs = {}
                for nm, (rpx, rvx) in {"RR" + a.sfx: (rp, rv), "RRSWAP" + a.sfx: (rp2, rv2)}.items():
                    mem = net.memory(torch.from_numpy(rpx)[None].to(dev), torch.from_numpy(rvx)[None].to(dev))
                    rm = torch.ones(1, rpx.shape[1], dtype=torch.bool, device=dev)
                    env, dc0, _ = net(qt, tt, mem, rm)
                    outs[nm] = (b["c0f"] + dc0[0].cpu().numpy(), env[0].cpu().numpy())
                conds = {"TABR": (b["c0f"], tab), **outs, "RRE" + a.sfx: (b["c0f"], outs["RR" + a.sfx][1]), "RRC" + a.sfx: (outs["RR" + a.sfx][0], tab)}
                if (od / f"{pre}_TABR.wav").exists():
                    conds.pop("TABR")
                if a.flow:
                    mem = fnet.rr.memory(torch.from_numpy(rp)[None].to(dev), torch.from_numpy(rv)[None].to(dev))
                    h_, _ = fnet.context(qt, mem, torch.ones(1, rp.shape[1], dtype=torch.bool, device=dev))
                    conds = {}
                    for tp in [float(z) for z in a.temps.split(",")]:
                        g_ = torch.Generator(device=dev).manual_seed(mi * 10 + k)
                        x_ = fnet.sample_nopast(h_, tt, steps=6, gen=g_, temp=tp)[0].cpu().numpy().astype(np.float64)
                        conds[f"FN{int(round(tp * 10)):02d}{a.sfx}"] = (b["c0f"] + x_[24], tab + 10 * x_[:24])
                if scs:
                    eR = outs["RR" + a.sfx][1]
                    conds = {f"RRS{sc:g}{a.sfx}": (outs["RR" + a.sfx][0], tab + sc * (eR - tab)) for sc in scs}
                    ra = st.f[("fem_neutral", A, f"t{k:02d}_neutral")]
                    vA = ra["ft"] > 0
                    sacc.append({"pair": pre, "sd_R": (eR - tab)[:, vM].std(1).tolist(), "sd_A": (ra["env"][1:25] - T[:, ra["u"]])[:, vA].std(1).tolist(),
                                 "sd_c0R": float(outs["RR" + a.sfx][0][vM].std()), "sd_c0A": float(((ra["env"][0] - RRm.C0_SIL) / 10)[vA].std()), "sd_c0M": float(b["c0f"][vM].std())})
                for c, (c0n, env) in conds.items():
                    sf.write(od / f"{pre}_{c}.wav", render(np.asarray(c0n, np.float64), np.asarray(env, np.float64), f0o.astype(np.float64), per_c.astype(np.float64)), N.SR)
                print(pre, flush=True)
    if a.std_out:
        Path(a.std_out).write_text(json.dumps(sacc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
