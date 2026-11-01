"""D249(学習なし・results/conv_p0/d249_prereg.yaml): c0・包絡の推定の害が遅れか均しか推定の誤差かを目標関数 v1 で切り分ける(出力部 rvoc3hi・rvtab_x)。
C0LAG / ENVLAG = 教師の値を 1 フレーム遅らせる・C0SMC = 教師の c0 の中心 3 平均・C0SM3 = 因果 3 平均・F2 / F3 = 製品の推定・F2LA / F3LA = 製品の推定を 1 フレーム先へ(5 ms の先読み)。
    uv run python d249_lag.py --s <dev の親> --out ../results/conv_p0/d249.json
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

ROOT = Path(__file__).resolve().parent.parent
CK = {"R3": "results/rvoc3hi/snap/ema_330k.pt", "TX": "results/rvtab_x/snap/ema_400k.pt"}
RUNGS = ("ORA", "C0LAG", "C0SMC", "C0SM3", "F2", "F2LA", "ENVLAG", "F3", "F3LA")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import c1_content as CC
    import f0est as FE
    import f0hi as H
    import lvl as LV
    import nvoc as N
    import pae as PA
    import rvoc as R
    import train_c3 as C3
    import train_rvoc as TR
    S = Path(a.s)
    F = S / "devff"
    od = F / "d249"
    od.mkdir(exist_ok=True)
    dev = "cuda"
    gens, smw = {}, {}
    for k, p in CK.items():
        s_ = torch.load(ROOT / p, map_location="cpu", weights_only=False)
        g = R.RVoc(ch=s_["cfg"]["ch"], kernels=tuple(s_["cfg"]["kernels"]), dils=tuple(s_["cfg"]["dils"]), d_cond=s_["cfg"]["d_cond"]).to(dev)
        g.load_state_dict(s_["ema"]); g.eval().requires_grad_(False)
        gens[k], smw[k] = g, TR.ckpt_env_smooth(s_, None)
    sf1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev); f1.load_state_dict(sf1["net"]); f1.eval()
    f1front = FE.Front().to(dev)
    sf3 = torch.load(ROOT / "results/f3_1/last.pt", map_location="cpu", weights_only=False)
    f3 = LV.EnvEst(sf3["cfg"]["ch"], tuple(sf3["cfg"]["dils"])).to(dev); f3.load_state_dict(sf3["net"]); f3.eval()
    sf4 = torch.load(ROOT / "results/f4_1/last.pt", map_location="cpu", weights_only=False)
    f4 = LV.PerEst(sf4["cfg"]["ch"], tuple(sf4["cfg"]["dils"])).to(dev); f4.load_state_dict(sf4["net"]); f4.eval()
    sf2 = torch.load(ROOT / "results/f2_2/last.pt", map_location="cpu", weights_only=False)
    f2 = LV.Lvl(sf2["cfg"]["ch"], tuple(sf2["cfg"]["dils"])).to(dev); f2.load_state_dict(sf2["net"]); f2.eval()
    mfront = CC.MelFront().to(dev)

    @torch.no_grad()
    def render(k, c0n, env24n, f0, per_t):
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)[None]
        v = (f0 > 0).astype(np.float64)[None]
        cond = torch.from_numpy(np.concatenate([c0n[None], env24n, lf, v, np.clip(per_t, 0, 1.5) * v], 0).astype(np.float32))[None].to(dev)
        y = gens[k](cond, TR.excitation(torch.from_numpy(f0.astype(np.float32))[None].to(dev), cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)

    srcs = sorted((F / "d56").glob("p0*_SRC.wav"))
    vf: dict = {}
    for f in srcs:
        st = f.stem[:-4]
        if all((od / f"{st}_{c}{k}.wav").exists() for k in CK for c in RUNGS):
            continue
        x, sr = sf.read(f, dtype="float32")
        n = len(x) // N.HOP
        x = x[: n * N.HOP]
        f0, _ = H.teacher_f0(x, n)
        f0 = f0[:n].astype(np.float64)
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        e = np.pad(PA.envelope(xa, f0), ((0, 0), (1, 1)), mode="edge")
        per = PA.periodicity(torch.from_numpy(xa)[None], torch.from_numpy(f0.astype(np.float32))[None])[0].numpy()[:, :n].astype(np.float64)
        with torch.no_grad():
            m = mfront(torch.cat([torch.zeros(1, N.WIN - N.HOP, device=dev), torch.from_numpy(x)[None].to(dev)], 1))
            avg3 = C3.causal_avg(f2(m)[:, None], 3)[0, 0].cpu().numpy()[:n].astype(np.float64)
            e3 = f3(m)[0].cpu().numpy()[1:25, :n].astype(np.float64)
            p4 = f4(m)[0].cpu().numpy()[:, :n].astype(np.float64)
            v_, p_ = f1(f1front(torch.cat([torch.zeros(1, FE.CTX48, device=dev), torch.from_numpy(x)[None].to(dev)], 1), n))
            f0c0 = FE.decode(v_, p_, 0.5)[0].cpu().numpy()[:n].astype(np.float64)
            pr_ = p_.softmax(1)
            kk_ = pr_.argmax(1, keepdim=True)
            idx_ = (kk_ + torch.arange(-4, 5, device=dev)[None, :, None]).clamp(0, pr_.shape[1] - 1)
            conf = torch.gather(pr_, 1, idx_).sum(1)[0].cpu().numpy()[:n]
            f0c5 = np.where(conf[: len(f0c0)] >= 0.5, f0c0, 0.0)
        pad1 = lambda z: np.pad(z, (0, n - len(z)), mode="edge")
        pad2 = lambda z: np.pad(z, ((0, 0), (0, n - z.shape[1])), mode="edge")
        f0c0, f0c5 = pad1(f0c0), pad1(f0c5)
        vf[st] = {"ORA": float((f0 > 0).mean()), "F1C0": float((f0c0 > 0).mean()), "F1C5": float((f0c5 > 0).mean())}
        for k in CK:
            env = (smw[k] * e[:, :-2] + (1 - 2 * smw[k]) * e[:, 1:-1] + smw[k] * e[:, 2:])[:, :n].astype(np.float64)
            oc0, oe = (env[0] - PA.C0_SIL) / 10, env[1:25] / 10
            lag = lambda z: np.concatenate([z[..., :1], z[..., :-1]], -1)
            adv = lambda z: np.concatenate([z[..., 1:], z[..., -1:]], -1)
            smc = np.convolve(np.pad(oc0, 1, mode="edge"), np.ones(3) / 3, "valid")
            sm3 = np.convolve(np.pad(oc0, (2, 0), mode="edge"), np.ones(3) / 3, "valid")
            a3, ee3 = pad1(avg3), pad2(e3)
            conds = {"ORA": (oc0, oe, f0, per), "C0LAG": (lag(oc0), oe, f0, per), "C0SMC": (smc, oe, f0, per), "C0SM3": (sm3, oe, f0, per),
                     "F2": (a3, oe, f0, per), "F2LA": (adv(a3), oe, f0, per), "ENVLAG": (oc0, lag(oe), f0, per), "F3": (oc0, ee3, f0, per), "F3LA": (oc0, adv(ee3), f0, per)}
            for c, (cc, ee, ff, pp) in conds.items():
                sf.write(od / f"{st}_{c}{k}.wav", render(k, cc, ee, ff, pp), N.SR)
        print(st, flush=True)
    from scipy.signal import resample_poly
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from transformers import WavLMModel
    del gens
    torch.cuda.empty_cache()
    net = WavLMModel.from_pretrained("microsoft/wavlm-large").to(dev).eval().float()

    @torch.no_grad()
    def emb(p: Path) -> np.ndarray:
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        x = resample_poly(x.mean(1)[: int(8.0 * sr)], 16000, sr).astype(np.float32)
        x = (x - x.mean()) / (x.std() + 1e-7)
        hs = net(torch.from_numpy(x)[None].to(dev), output_hidden_states=True).hidden_states
        return torch.cat([t for l in (4, 6, 8) for t in (hs[l][0].mean(0), hs[l][0].std(0))]).cpu().numpy()

    base = ROOT / "female-dataset"
    Rr = []
    for r in json.loads((ROOT / "results/conv_p0/d237.json").read_text())["rows"]:
        if r["grp"] == "REAL_POOL":
            ws = sorted((base / r["id"]).glob("*.wav"))
            Rr.append(emb(ws[len(ws) // 2]))
    Rr = np.array(Rr)
    rf = np.arange(len(Rr)) % 5
    rep: dict = {}
    for part, pat in (("FF", "p00"), ("AA", "p01")):
        for c in RUNGS:
            for k in CK:
                files = sorted(od.glob(f"{pat}*_{c}{k}.wav"))
                X = np.array([emb(f) for f in files])
                tg = np.array([int(f.name.split("_t")[1][:3]) for f in files])
                Xa = np.concatenate([Rr, X])
                ya = np.r_[np.zeros(len(Rr)), np.ones(len(X))]
                fa = np.r_[rf, tg % 5]
                pr = np.zeros(len(ya))
                for kk in range(5):
                    te = fa == kk
                    sc = StandardScaler().fit(Xa[~te])
                    clf = LogisticRegression(C=0.01, max_iter=2000).fit(sc.transform(Xa[~te]), ya[~te])
                    pr[te] = clf.predict_proba(sc.transform(Xa[te]))[:, 1]
                key = f"{part}_{c}_{k}"
                rep[key] = {"auc": round(float(roc_auc_score(ya, pr)), 4), "p_fake_median": round(float(np.median(pr[len(Rr):])), 4), "n": len(X)}
                print(key, rep[key], flush=True)
    up = {c: round(rep[f"FF_{c}_R3"]["p_fake_median"] - rep["FF_ORA_R3"]["p_fake_median"], 4) for c in RUNGS if c != "ORA"}
    rep["rise_R3_FF"] = up
    def kind(lagk, smk, prod, la):
        if up[lagk] >= 0.6 * up[prod] and up[la] <= 0.5 * up[prod]:
            return "LAG"
        if smk and up[smk] >= 0.6 * up[prod]:
            return "SMOOTH"
        return "ESTIMATE"
    rep["verdict"] = {"c0": kind("C0LAG", "C0SMC", "F2", "F2LA"), "env": kind("ENVLAG", None, "F3", "F3LA")}
    print("verdict", rep["verdict"], up, flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
