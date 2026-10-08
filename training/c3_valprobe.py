"""C3 の G の検証の広げ直し(学習なし): 学習に使わない目標(話者 ID で分けた検証の 137 件)に対して、多数のバッチで G あり / Δ = 0 の ECAPA・WavLM-SV の cos・top-1・平均順位・KL を測る。
1k step ごとの監視(48 件)は標本が小さいので、独立の検証器(WavLM)の順位まで含めて 480 件以上で見る。
    uv run python c3_valprobe.py --g ../results/c3_1/last.pt --batches 60 --out ../results/c3_1/valprobe.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import c1_content as CC
import f0est as FE
import idloss as ID
import lvl as LV
import nvoc as N
import rvoc as R
import train_c3 as C3
import train_rvoc as TR

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g", required=True)
    ap.add_argument("--batches", type=int, default=60)
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", default=str(ROOT / "data/c3/targets.npz"))
    a = ap.parse_args()
    dev = "cuda"
    sr_ = torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=sr_["cfg"]["ch"], kernels=tuple(sr_["cfg"]["kernels"]), dils=tuple(sr_["cfg"]["dils"]), d_cond=sr_["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(sr_["ema"]); gen.eval().requires_grad_(False)
    sf1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev); f1.load_state_dict(sf1["net"]); f1.eval().requires_grad_(False)
    sf2 = torch.load(ROOT / "results/f2_1/last.pt", map_location="cpu", weights_only=False)
    f2 = LV.Lvl(sf2["cfg"]["ch"], tuple(sf2["cfg"]["dils"])).to(dev); f2.load_state_dict(sf2["net"]); f2.eval().requires_grad_(False)
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev); c1.load_state_dict(st1["net"]); c1.eval().requires_grad_(False)
    mfront = CC.MelFront().to(dev)
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").to(dev)
    tau = st1.get("tau", 0.05)
    ecapa, wsv, cvec = ID.Ecapa(dev), ID.WavlmSV(dev), ID.ContentVec(dev)
    Z = np.load(a.targets)
    Tt = torch.from_numpy(Z["T"].astype(np.float32)).to(dev) / 10.0
    mu = torch.from_numpy(Z["mu"]).to(dev); per_t = torch.from_numpy(Z["per"]).to(dev)
    e_ec = torch.from_numpy(Z["e_ecapa"]).to(dev); e_wv = torch.from_numpy(Z["e_wavlm"]).to(dev)
    uniq = sorted(set(Z["spk"].tolist()))
    rs = np.random.default_rng(0).permutation(len(uniq))
    val_spk = {uniq[i] for i in rs[:max(20, len(uniq) // 20)]}
    isval = np.array([sp in val_spk for sp in Z["spk"].tolist()])
    val_idx = torch.from_numpy(np.where(isval)[0]).to(dev)
    summ = torch.cat([Tt.mean(2), mu[:, None], per_t], 1)
    sg = torch.load(a.g, map_location="cpu", weights_only=False)
    G = C3.ConvG(sg["cfg"]["K"], sg["cfg"]["ch"], tuple(sg["cfg"]["dils"]), sg["cfg"]["d_film"], sg["cfg"].get("m_basis", 0)).to(dev)
    G.load_state_dict(sg["G"], strict=False); G.eval()
    chain = C3.Chain(c1, mfront, Cb, f1, f1front := FE.Front().to(dev), f2, G, Tt, mu, per_t, summ, dev)
    rows = C3.male_rows_spk(); meds = C3.speaker_med(rows)
    loader = iter(torch.utils.data.DataLoader(C3.SrcDS(rows, meds, 77), batch_size=8, num_workers=6, persistent_workers=True))
    isreal = torch.from_numpy(np.array([s == "real" for s in Z["src"].tolist()])).to(dev)
    res = {k: [] for k in ("ec", "wv", "ec0", "wv0", "rk", "rk0", "wrk", "wrk0", "t1", "t10", "wt1", "wt10", "kl", "kl0", "real_rk", "real_rk0", "tts_rk", "tts_rk0")}
    with torch.no_grad():
        for it in range(a.batches):
            x, med = (t.to(dev) for t in next(loader))
            j = torch.randint(len(val_idx), (x.shape[0],), device=dev)
            idx = val_idx[j]
            for zero in (False, True):
                cond, f0o, dc, dp, p = chain(x, idx, med, zero)
                exc = TR.excitation(f0o, x.shape[1], torch.Generator().manual_seed(it))
                y = gen(cond, exc)
                o = C3.NCTX * C3.HOP
                ys = y[:, o + N.DELAY:o + C3.NSEG * C3.HOP]
                xs = x[:, o:o + C3.NSEG * C3.HOP - N.DELAY]
                ee, ww = ecapa(ys), wsv(ys)
                sim, simw = ee @ e_ec[val_idx].T, ww @ e_wv[val_idx].T
                s = "0" if zero else ""
                res["ec" + s].append((ee * e_ec[idx]).sum(-1).cpu().numpy())
                res["wv" + s].append((ww * e_wv[idx]).sum(-1).cpu().numpy())
                rk = ((sim > sim.gather(1, j[:, None])).sum(1) + 1).float().cpu().numpy()
                wrk = ((simw > simw.gather(1, j[:, None])).sum(1) + 1).float().cpu().numpy()
                res["rk" + s].append(rk); res["wrk" + s].append(wrk)
                res["t1" + ("0" if zero else "")].append((rk == 1).astype(float)); res["wt1" + ("0" if zero else "")].append((wrk == 1).astype(float))
                qy = (cvec(ys) @ Cb.T / tau).log_softmax(-1)
                res["kl" + s].append(np.array([F.kl_div(qy, C3.anchor_from_c1(p, qy.shape[1]), reduction="none").sum(-1).mean().item()]))
                ir = isreal[idx].cpu().numpy()
                res["real_rk" + s].append(rk[ir]); res["tts_rk" + s].append(rk[~ir])
    out = {k: round(float(np.concatenate(v).mean()), 4) if len(np.concatenate(v)) else None for k, v in res.items()}
    out["n_samples"] = int(len(np.concatenate(res["rk"])))
    out["n_val_targets"] = int(len(val_idx))
    print(json.dumps(out, ensure_ascii=False, indent=1))
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
