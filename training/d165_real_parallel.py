"""D165(学習なし・診断専用): 実女声での一般化の検査。D157(目標本人の c0 + 動きを C1 で整列すると明瞭さと同一性が両立)・D162(c0 の同一性は低域が主)は TTS 女声だけ。
実女声には同じ文を読んだ話者の組が無い(41,230 本・共有 0)ので、開発用の組(D143)の 30 目標の、参照(先頭 20s)に入らないクリップのうち普通の台詞(1.2〜12 秒・記号を除いて 6〜60 字・同じ字の 3 連続なし・小書きが 2 割以下)を 1 人 4 本、
同じ目標に割り当てた男声 TTS(Irodori・錨 t10 + caption)に読ませる(generate_male_for_real.py)= 男声 TTS → 実女声のパラレルの組。
  --stage select: <out>/items.json を書く(目標・男声・文・実女声のクリップ)
  --stage convert: D106 / D157 / D162 と同じ Python の製品経路の式・C1 の整列器で CONV・TAB0・XDYN・XC0・XCD・XC0S・XC0F・ASELF(実女声の写し合成)を
                   <out>/p{item:04d}_t{ti:03d}_{cond}.wav に・cen_TTT.wav = 参照と評価のクリップを除いた残りから 25s(この検査専用の中心) → d4b_lda_verifier / d146 / d23 をそのまま使える。
    uv run python d165_real_parallel.py --stage select --dev ~/lightvc_scratch_20261013/dev --out <s>/d165
    uv run python d165_real_parallel.py --stage convert --dev ~/lightvc_scratch_20261013/dev --out <s>/d165 --rvoc_ckpt ../results/rvtab_f3/snap/ema_380k.pt
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
MALE = ROOT / "data/male_tts_v4l"


def select(dev: Path, od: Path, per: int = 4) -> None:
    from d6_unit_cov import load48
    ds = json.loads((dev / "devset.json").read_text())
    items = []
    for ti, t in enumerate(ds["targets"]):
        ws = sorted((ROOT / "female-dataset" / t).glob("*.wav"))
        ref, tot = [], 0.0
        for w in ws:
            if tot >= 20.0:
                break
            ref.append(w); tot += sf.info(str(w)).duration
        rest = [w for w in ws if w not in ref]
        spk = [p for p in ds["pairs"] if p["ti"] == ti][0]["spk"]
        picks = []
        for w in rest:
            if len(picks) >= per:
                break
            lab = w.with_suffix(".lab")
            if not lab.exists() or not (1.2 <= sf.info(str(w)).duration <= 12.0):
                continue
            text = re.sub(r"[「」『』（）()\s]", "", unicodedata.normalize("NFKC", lab.read_text(errors="ignore")).strip())
            core = re.sub(r"[…ー~!?・、。,.―-]", "", text)
            if not (6 <= len(core) <= 60) or re.search(r"(.)\1\1", core) or sum(ch in "ぁぃぅぇぉっッ" for ch in core) > 0.2 * len(core):
                continue
            picks.append((w, text))
        cen, tc = [], 0.0
        for w in rest:
            if tc >= 25.0:
                break
            if w not in [q for q, _ in picks]:
                cen.append(w); tc += sf.info(str(w)).duration
        if not picks or tc < 8.0:
            continue
        od.mkdir(parents=True, exist_ok=True)
        sf.write(od / f"cen_{ti:03d}.wav", np.concatenate([load48(w) for w in cen])[: 25 * 48000].astype(np.float32), 48000)
        for w, text in picks:
            items.append({"item": len(items), "ti": ti, "target": t, "spk": spk, "clip": str(w), "text": text, "ref": [str(r) for r in ref], "cen_sec": round(tc, 1)})
    od.mkdir(parents=True, exist_ok=True)
    (od / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=1))
    print("items", len(items), "targets", len({i["ti"] for i in items}))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["select", "convert"])
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rvoc_ckpt", default=str(ROOT / "results/rvtab_f3/snap/ema_380k.pt"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--only", default="", help="D169 / D174 / D182: この条件だけ(XCDT = XCD の f0 を操作者の教師 f0 に・XCDL / XCDM / XCDH = c0 は目標本人で動きの 1〜4 / 5〜12 / 13〜24 次元だけ目標本人・XF0 = f0 だけ目標本人の教師 f0(C1 で整列・有声も目標)・XCDF = c0 + 動き + f0・XCDFP = + 周期性)")
    a = ap.parse_args()
    dev_dir, od = Path(a.dev), Path(a.out)
    if a.stage == "select":
        select(dev_dir, od)
        return 0
    import librosa
    import scipy.signal as ss
    import c1_content as CC
    import conv_c0 as C0
    import f0est as FE
    import f0hi as H
    import lvl as LV
    import nvoc as N
    import pae as PA
    import rvoc as R
    import train_c1 as T1
    import train_c3 as C3
    import train_rvoc as TR
    from d6_unit_cov import load48
    from d88_perturb_identity import msqrt, wmat
    dev = a.device
    torch.set_num_threads(4)
    sr_ = torch.load(a.rvoc_ckpt, map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=sr_["cfg"]["ch"], kernels=tuple(sr_["cfg"]["kernels"]), dils=tuple(sr_["cfg"]["dils"]), d_cond=sr_["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(sr_["ema"]); gen.eval().requires_grad_(False)
    sm_w = TR.ckpt_env_smooth(torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False), None)
    ld = lambda p: torch.load(ROOT / p, map_location="cpu", weights_only=False)
    s1 = ld("results/f0est3/last.pt"); f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev); f1.load_state_dict(s1["net"]); f1.eval()
    f1front = FE.Front().to(dev)
    s2 = ld("results/f2_2/last.pt"); f2 = LV.Lvl(s2["cfg"]["ch"], tuple(s2["cfg"]["dils"])).to(dev); f2.load_state_dict(s2["net"]); f2.eval()
    s3 = ld("results/f3_1/last.pt"); f3 = LV.EnvEst(s3["cfg"]["ch"], tuple(s3["cfg"]["dils"])).to(dev); f3.load_state_dict(s3["net"]); f3.eval()
    s4 = ld("results/f4_1/last.pt"); f4 = LV.PerEst(s4["cfg"]["ch"], tuple(s4["cfg"]["dils"])).to(dev); f4.load_state_dict(s4["net"]); f4.eval()
    sc = ld("results/c1_1/last.pt"); c1 = CC.C1(sc["cfg"]["k"], sc["cfg"]["ch"], tuple(sc["cfg"]["dils"])).to(dev); c1.load_state_dict(sc["net"]); c1.eval()
    mfront = CC.MelFront().to(dev)
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt", map_location="cpu").cpu().numpy()
    K = Cb.shape[0]
    W = wmat(1.3)

    @torch.no_grad()
    def front(x: np.ndarray) -> dict:
        n = len(x) // N.HOP
        x = x[: n * N.HOP].astype(np.float32)
        xt = torch.from_numpy(x)[None].to(dev)
        v_, p_ = f1(f1front(torch.cat([torch.zeros(1, FE.CTX48, device=dev), xt], 1), n))
        f0f = FE.decode(v_, p_, 0.5)[0].cpu().numpy()[:n].astype(np.float64)
        m0 = mfront(torch.cat([torch.zeros(1, N.WIN - N.HOP, device=dev), xt], 1))
        c0f = C3.causal_avg(f2(m0)[:, None], 3)[0, 0].cpu().numpy()[:n].astype(np.float64)
        e3 = 10.0 * f3(m0)[0].cpu().numpy()[1:25, :n].astype(np.float64)
        p4 = f4(m0)[0].cpu().numpy()[:, :n].astype(np.float64)
        p = c1(mfront(torch.from_numpy(T1.prime(x))[None].to(dev)))[..., -n:].softmax(1)[0].cpu().numpy().astype(np.float64)
        pad1 = lambda z: np.pad(z, (0, n - len(z)), mode="edge")
        pad = lambda z: np.pad(z, ((0, 0), (0, n - z.shape[1])), mode="edge")
        ft, _ = H.teacher_f0(x, n)
        return {"x": x, "n": n, "f0": pad1(f0f), "c0": pad1(c0f), "e3": pad(e3), "p4": pad(p4), "p": pad(p), "u": pad(p).argmax(0), "ft": pad1(ft[:n].astype(np.float64))}

    def cheap(fr: dict) -> np.ndarray:
        xa = np.concatenate([np.zeros(PA.A, np.float32), fr["x"], np.zeros(PA.A, np.float32)])
        e = np.pad(PA.envelope(xa, fr["ft"]), ((0, 0), (1, 1)), mode="edge")
        return (sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:])[:, :fr["n"]][1:25].astype(np.float64)

    def enroll(frs: list, use_f3: bool) -> dict:
        E = np.concatenate([fr["e3"] if use_f3 else cheap(fr) for fr in frs], 1)
        U = np.concatenate([fr["u"] for fr in frs])
        T = C0.table(E, U, K, Cb)
        F0 = np.concatenate([fr["f0"] if use_f3 else fr["ft"] for fr in frs])
        v = F0 > 0
        r = E - T[:, U]
        rv = r[:, v] - r[:, v].mean(1, keepdims=True)
        P4 = np.concatenate([fr["p4"] for fr in frs], 1)
        return {"T": T, "mu": float(np.median(np.log(F0[v]))), "cov": rv @ rv.T / rv.shape[1], "p4mean": P4[:, v].mean(1)}

    @torch.no_grad()
    def render(c0n, env24, f0, per_t) -> np.ndarray:
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)[None]
        v = (f0 > 0).astype(np.float64)[None]
        cond = torch.from_numpy(np.concatenate([c0n[None], env24 / 10, lf, v, np.clip(per_t, 0, 1.5) * v], 0).astype(np.float32))[None].to(dev)
        y = gen(cond, TR.excitation(torch.from_numpy(f0.astype(np.float32))[None].to(dev), cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)

    def dtwj(X_: np.ndarray, Y_: np.ndarray) -> np.ndarray:
        _, wq = librosa.sequence.dtw(X=X_, Y=Y_, metric="euclidean")
        jq = np.zeros(X_.shape[1], int)
        for i_, j_ in wq[::-1]:
            jq[i_] = j_
        return jq

    bl, al = ss.butter(2, 3.0 / 100.0)
    lo = lambda z: ss.filtfilt(bl, al, z)
    items = json.loads((od / "items.json").read_text())
    si, sk = (int(z) for z in a.shard.split("/"))
    ops: dict = {}
    tgs: dict = {}
    for it in items:
        if it["item"] % sk != si:
            continue
        pre = od / f"p{it['item']:04d}_t{it['ti']:03d}"
        if Path(f"{pre}_{a.only.split(',')[-1] if a.only else 'ASELF'}.wav").exists():
            continue
        if it["spk"] not in ops:
            mf, tot = [], 0.0
            for f in sorted(MALE.glob(f"{it['spk']}/t*.wav")):
                if tot >= 22.0:
                    break
                x = load48(f).astype(np.float32); mf.append(front(x)); tot += len(x) / 48000
            ops[it["spk"]] = enroll(mf, True)
            ops[it["spk"]]["_frs"] = mf
        if it["ti"] not in tgs:
            rf = [front(load48(r).astype(np.float32)) for r in it["ref"]]
            tgs[it["ti"]] = (enroll(rf, False), enroll(rf, True))
            tgs[it["ti"]][1]["_frs"] = rf
        O_, (T_, TA3) = ops[it["spk"]], tgs[it["ti"]]
        fm = front(load48(f"{pre}_SRC.wav").astype(np.float32))
        fa = front(load48(it["clip"])[: 8 * 48000].astype(np.float32))
        j = dtwj(np.sqrt(np.maximum(fm["p"], 0)), np.sqrt(np.maximum(fa["p"], 0)))
        f0o = np.where(fm["f0"] > 0, fm["f0"] * np.exp(T_["mu"] - O_["mu"]), 0.0)
        per_c = fm["p4"] - O_["p4mean"][:, None] + T_["p4mean"][:, None]
        M_ = msqrt(T_["cov"], 0.5) @ msqrt(W @ O_["cov"] @ W.T, -0.5)
        tab = T_["T"] @ fm["p"]
        dyn_c = 0.7 * M_ @ W @ (fm["e3"] - O_["T"] @ fm["p"])
        fa_dyn = 0.7 * (fa["e3"][:, j] - TA3["T"] @ fa["p"][:, j])
        cm, ca = fm["c0"], fa["c0"][j]
        conds = {"CONV": (cm, tab + dyn_c, f0o, per_c), "TAB0": (cm, tab, f0o, per_c), "XDYN": (cm, tab + fa_dyn, f0o, per_c),
                 "XC0": (ca, tab + dyn_c, f0o, per_c), "XCD": (ca, tab + fa_dyn, f0o, per_c),
                 "XC0S": (lo(ca) + cm - lo(cm), tab + dyn_c, f0o, per_c), "XC0F": (lo(cm) + ca - lo(ca), tab + dyn_c, f0o, per_c),
                 "ASELF": (fa["c0"], fa["e3"], np.where(fa["ft"] > 0, fa["ft"], 0.0), fa["p4"])}
        if a.only:
            ftm = np.where((fm["ft"] > 0) & (fm["f0"] > 0), fm["ft"] * np.exp(T_["mu"] - O_["mu"]), 0.0)
            conds["XCDT"] = (ca, tab + fa_dyn, ftm, per_c)
            fa_f0 = np.where(fa["ft"][j] > 0, fa["ft"][j], 0.0)
            conds["XF0"] = (cm, tab + dyn_c, fa_f0, per_c)
            c0m = lambda frs: float(np.mean(np.concatenate([fr["c0"][fr["f0"] > 0] for fr in frs])))
            loc = lambda z: ss.lfilter(bl, al, z - z[0]) + z[0]
            ssd = lambda frs: float(np.mean([np.std((loc(fr["c0"]) - loc(fr["c0"]).mean())[fr["f0"] > 0]) for fr in frs if (fr["f0"] > 0).sum() > 50]))
            mO, mT = c0m(O_["_frs"]), c0m(TA3["_frs"])
            conds["C0MEAN"] = (cm + (mT - mO), tab + dyn_c, f0o, per_c)
            zf = Path(f"{pre}_TTSZ.wav")
            if zf.exists() and "XCDZ" in a.only:
                fz = front(load48(zf)[: 12 * 48000].astype(np.float32))
                jz = dtwj(np.sqrt(np.maximum(fm["p"], 0)), np.sqrt(np.maximum(fz["p"], 0)))
                conds["XCDZ"] = (fz["c0"][jz], tab + 0.7 * (fz["e3"][:, jz] - TA3["T"] @ fz["p"][:, jz]), f0o, per_c)
                conds["XC0Z"] = (fz["c0"][jz], tab + dyn_c, f0o, per_c)
            sl = loc(cm)
            conds["C0MS"] = (mT + ssd(TA3["_frs"]) / ssd(O_["_frs"]) * (sl - mO) + (cm - sl), tab + dyn_c, f0o, per_c)
            conds["XCDF"] = (ca, tab + fa_dyn, fa_f0, per_c)
            conds["XCDFP"] = (ca, tab + fa_dyn, fa_f0, fa["p4"][:, j])
            for nm_, (lo_, hi_) in (("XCDL", (0, 4)), ("XCDM", (4, 12)), ("XCDH", (12, 24))):
                dd_ = dyn_c.copy(); dd_[lo_:hi_] = fa_dyn[lo_:hi_]
                conds[nm_] = (ca, tab + dd_, f0o, per_c)
            conds = {k_: v_ for k_, v_ in conds.items() if k_ in a.only.split(",")}
        for c, (cc, ee, ff, pp) in conds.items():
            sf.write(f"{pre}_{c}.wav", render(cc, ee, ff, pp), N.SR)
        print(pre.name, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
