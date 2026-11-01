"""D22: 登録の表(参照 20s の CheapTrick 包絡の単位平均)を、教師の f0(f0hi.teacher_f0 = Harvest + DIO48 の組み合わせ)の代わりに
因果の F1(f0est3・Rust に移植済み)の f0 で作っても同一性が落ちないか(学習なし・自己変換・包絡だけ替える・c0/f0/周期性は実の分析)。
TAB = 教師 f0 の表(製品の現行)・TAB_F1 = F1 の f0 の表。表どうしの差(rms)と、学習外の LDA の同一性を比べる(wav を d4b の形式で書く)。
    uv run python d22_enroll_f0.py --out_dir <scratchpad>/d22 --out ../results/conv_p0/d22.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import diag_pool as P
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import conv_c0 as C0
    import eval_nvoc as E
    import f0est as FE
    import nvoc as N
    import pae as PA
    import rvoc as R
    import train_c3 as C3
    import train_f0est as TF
    import train_rvoc as TR
    dev = "cuda"
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    pool = P.load_pool(150, 3)
    refs = P.load_refs(150, 3)
    sr_ = torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=sr_["cfg"]["ch"], kernels=tuple(sr_["cfg"]["kernels"]), dils=tuple(sr_["cfg"]["dils"]), d_cond=sr_["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(sr_["ema"]); gen.eval().requires_grad_(False)
    sm_w = TR.ckpt_env_smooth(sr_, None)
    sf1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev); f1.load_state_dict(sf1["net"]); f1.eval()
    f1front = FE.Front().to(dev)
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()
    K = Cb.shape[0]

    def sm3(Ev):
        e = np.pad(Ev, ((0, 0), (1, 1)), mode="edge")
        return sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:]

    @torch.no_grad()
    def render(c0, env24, f0, per):
        cc = np.concatenate([c0[None] - PA.C0_SIL, env24], 0)
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)[None]
        v = (f0 > 0).astype(np.float32)[None]
        cond = torch.from_numpy(np.concatenate([cc / 10, lf, v, per * v], 0).astype(np.float32))[None].to(dev)
        fo = torch.from_numpy(f0.astype(np.float32))[None].to(dev)
        y = gen(cond, TR.excitation(fo, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)

    ca = lambda z: C3.causal_avg(torch.from_numpy(z.astype(np.float32))[None], 3)[0].numpy()
    diffs, vdis, am = [], [], {"TAB": [], "TAB_F1": []}
    pi = 0
    for ti, (t, r) in enumerate(zip(pool, refs)):
        ws = sorted((ROOT / "female-dataset" / t["spk"]).glob("*.wav"))
        ref, tot = [], 0.0
        for w in ws:
            if tot >= 20.0:
                break
            ref.append(w); tot += sf.info(str(w)).duration
        xr = np.concatenate([load48(w) for w in ref]); xr = xr[: len(xr) // N.HOP * N.HOP]
        n = len(xr) // N.HOP
        f0f1 = TF.infer(f1front, f1, xr, dev)
        f0f1 = np.pad(f0f1, (0, max(0, n - len(f0f1))))[:n].astype(np.float64)
        f0t = r["f0"].astype(np.float64)[:n]
        vdis.append(float(((f0f1 > 0) != (f0t > 0)).mean()))
        xa = np.concatenate([np.zeros(PA.A, np.float32), xr, np.zeros(PA.A, np.float32)])
        envF = sm3(PA.envelope(xa, f0f1))
        u = r["u"].astype(int)[:n]
        TF1 = C0.table(envF[1:25, : len(u)], u, K, Cb)
        T0 = t["T"]
        diffs.append(float(np.sqrt(((TF1 - T0) ** 2).mean())))
        sf.write(od / f"cen_{ti:03d}.wav", t["cen"], 48000)
        for ut in t["utts"]:
            uu = ut["u"].astype(int)
            for c, TT in (("TAB", T0), ("TAB_F1", TF1)):
                y = render(ut["c0"], ca(TT[:, uu]), ut["f0"], ut["per"])
                sf.write(od / f"p{pi:04d}_t{ti:03d}_{c}.wav", y, 48000)
                am[c].append(E.amline(y))
            sf.write(od / f"p{pi:04d}_t{ti:03d}_SRC.wav", ut["x"], 48000)
            pi += 1
        if ti % 25 == 0:
            print(ti, "table rms diff", round(float(np.mean(diffs)), 3), "voicing disagreement", round(float(np.mean(vdis)), 3), flush=True)
    rep = {"n_targets": len(pool), "table_rms_diff_teacher_vs_F1": round(float(np.mean(diffs)), 4), "table_rms_between_speakers_ref": None,
           "voicing_disagreement_F1_vs_teacher": round(float(np.mean(vdis)), 4), "am_line_db": {k: round(float(np.mean(v)), 3) for k, v in am.items()}}
    Ts = np.stack([t["T"] for t in pool])
    rep["table_rms_between_speakers_ref"] = round(float(np.sqrt(((Ts - Ts.mean(0, keepdims=True)) ** 2).mean())), 4)
    print(json.dumps(rep, indent=1))
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
