"""D578 / D582(推論だけ): 製品 P57S0 の条件の包絡の行に、既存の学習した読み R(diag_rr19L・S 形)の話者の分を足して鳴らす。規則は OVERNIGHT の D578 / D582。
--ablate: D582 の部品外し = OURS<k>NC(文脈の網 ctx の補正 0)・OURS<k>NV(連続の内容の特徴 cvf を 0)・OURS<k>RNC(R の話者の分 + ctx 0)。
R の差分は d571_oracle.py --only RRDUMP が <out>/rr_delta/<pNNNN_tTTT>.npz(dS = 包絡 c1〜c24 の差分・dC = c0 の差分・D571 の 5 ms の格子)に書く。
条件と登録は D573(d573_product.py)の書き出しを使う。前置き 12 s の後ろのフレームにだけ dS / 10 を遅い揺らぎの口(g.make の z)から足す。
版: OURS<k>R = 包絡の話者の分 / OURS<k>RC = 包絡 + c0 の話者の分(c0 は記録)。
    uv run python d578_product_rr.py --out <dev>/d571par --k 7
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import nvoc as N
import train_ppgvoc as P
import train_rvoc as TR
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "results/diag_ppgvoc57/product_002500_m0.pt"
PRE = 12.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--ckpt", default=str(CKPT))
    ap.add_argument("--wsfx", default="", help="別の重みの版: OURS<k><wsfx>(R なし)と OURS<k>R<wsfx> を描く")
    a = ap.parse_args()
    od = Path(a.out)
    en_dir = od.parent / "d85_work"
    work, ed, ko = od / "d573_work", od / "d573_dump", od / f"k{a.k}"
    items = json.loads((od / "items.json").read_text())
    n0 = int(PRE * N.SR)
    f0 = n0 // N.HOP
    dev = "cuda"
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    st["slow_sigma"], st["slow_causal"], st["opslow_tau"], st["opslow_gain"] = 1.5, True, 4.0, 0.0
    P.cvin_setup(st["cvin"] if Path(st["cvin"]).is_absolute() else str(Path(__file__).parent / st["cvin"]), dev)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    P.slow_from_ckpt(st)
    assert P.OPSLOW["on"] and P.SLOW["on"] is not None
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    import train_c1 as TC
    sfx = "" if a.gain == 1.0 else f"g{int(round(a.gain * 100))}"
    done = 0
    for it in items:
        k = f"p{it['item']:04d}_t{it['ti']:03d}"
        outs = {f"OURS{a.k}R{sfx}": ko / f"{k}_OURS{a.k}R{sfx}.wav", f"OURS{a.k}RC{sfx}": ko / f"{k}_OURS{a.k}RC{sfx}.wav"}
        if a.wsfx:
            outs = {f"OURS{a.k}{a.wsfx}": ko / f"{k}_OURS{a.k}{a.wsfx}.wav", f"OURS{a.k}R{a.wsfx}": ko / f"{k}_OURS{a.k}R{a.wsfx}.wav"}
        if a.ablate:
            outs = {f"OURS{a.k}{v}": ko / f"{k}_OURS{a.k}{v}.wav" for v in ("NC", "NV", "RNC")}
        if all(o.exists() for o in outs.values()):
            continue
        rr = np.load(od / "rr_delta" / f"{k}.npz")
        c, lg = P.load_dump(ed, k)
        pre = c[:2400, 0]
        mu = float(pre[pre > 0.5].mean()) if (pre > 0.5).sum() > 50 else 4.40
        c = c.copy()
        c[:, 0] = np.where(c[:, 0] > 0.05, c[:, 0] + (4.40 - mu), c[:, 0])
        c[:, 25] = np.where(c[:, 26] > 0.5, c[:, 25] + a.k * np.log(2) / 12, c[:, 25])
        T = len(c)
        nr = rr["dS"].shape[1]
        assert abs((T - f0) - nr) <= 3, (k, T - f0, nr)
        m = min(nr, T - f0)
        z = np.zeros((24, T), np.float32)
        z[:, f0:f0 + m] = a.gain * rr["dS"][:, :m] / 10
        xin = load48(work / f"{k}_in.wav").astype(np.float32)
        cvf = P.cvin_frames(torch.from_numpy(TC.prime(xin))[None].to(dev), len(lg))
        en = P.load_enroll(en_dir / f"en_{it['ti']:03d}")
        nx = len(xin) - n0
        for nm, o in outs.items():
            if o.exists():
                continue
            cc = P.c0low_np(c.copy()) if P.C0LOW["on"] else c.copy()
            zz = np.zeros_like(z) if (nm.endswith(("NC", "NV")) and not nm.endswith("RNC")) or (a.wsfx and nm == f"OURS{a.k}{a.wsfx}") else z
            ctx_off = nm.endswith("NC")
            cv_use = None if nm.endswith("NV") else cvf
            if ctx_off:
                ctx_mod = g.ctx
                del g.ctx
            if nm.startswith(f"OURS{a.k}RC"):
                cc[f0:f0 + m, 0] = np.where(cc[f0:f0 + m, 0] > 0.05, cc[f0:f0 + m, 0] + a.gain * rr["dC"][:m], cc[f0:f0 + m, 0])
            with torch.no_grad():
                p = P.ppg_form(torch.from_numpy(lg.T.copy()).float().to(dev)[None], P.PPG)
                ct = torch.from_numpy(cc.T.copy()).float().to(dev)[None]
                stt = P.spk_tensor(en, dev)
                if P.PERDYN["on"] and isinstance(stt, tuple):
                    stt = (stt[0], ct[:, 27:31])
                cond = g.make(p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], stt, torch.from_numpy(zz)[None].to(dev), cvf=cv_use)
                fz = torch.where(ct[:, 26] > 0.5, 200.0 * torch.exp(ct[:, 25]), torch.zeros_like(ct[:, 25]))
                exc = TR.excitation(fz, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0))
                y = g(cond, exc, stt)[0].cpu().numpy()[N.DELAY:]
            if ctx_off:
                g.ctx = ctx_mod
            y = y[n0:n0 + nx]
            sf.write(o, (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
        if done == 0:
            with torch.no_grad():
                p = P.ppg_form(torch.from_numpy(lg.T.copy()).float().to(dev)[None], P.PPG)
                c2 = P.c0low_np(c.copy()) if P.C0LOW["on"] else c.copy()
                ct = torch.from_numpy(c2.T.copy()).float().to(dev)[None]
                stt = P.spk_tensor(en, dev)
                if P.PERDYN["on"] and isinstance(stt, tuple):
                    stt = (stt[0], ct[:, 27:31])
                cond = g.make(p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], stt, torch.zeros(1, 24, T, device=dev), cvf=cvf)
                fz = torch.where(ct[:, 26] > 0.5, 200.0 * torch.exp(ct[:, 25]), torch.zeros_like(ct[:, 25]))
                y0 = g(cond, TR.excitation(fz, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)), stt)[0].cpu().numpy()[N.DELAY:]
            y1 = P.render(g, c, lg, en, dev, T_op=P.load_enroll(work / f"op_{it['spk']}")[0], cvf=cvf)
            y1b = P.render(g, c, lg, en, dev, T_op=P.load_enroll(work / f"op_{it['spk']}")[0], cvf=cvf)
            y0 = (y0 / (np.abs(y0).max() + 1e-9) * 0.9).astype(np.float32)
            print("self-check max|z=0 − P.render|", float(np.abs(y0 - y1).max()), "P.render twice", float(np.abs(y1 - y1b).max()), flush=True)
            assert float(np.abs(y0 - y1).max()) < 1e-3, float(np.abs(y0 - y1).max())
        done += 1
    print("rendered", done, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
