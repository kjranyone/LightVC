"""診断(D16〜)の評価プールの準備を保存して使い回す。評価外の日本語女声のうち、d4b の LDA 学習話者(RandomState(0) の先頭 1,200 人)以外から
RandomState(5) の順で n_tg 人。目標ごとに: 参照 20s の表 T(c1..c24)と c0 の表 Tc0(参照の平均からの相対)・log f0 の中央値と標準偏差・c0 の平均・周期性の平均、
別の発話 n_utt 本(8s)の実の分析(包絡・c0・教師 f0・周期性・C1 の単位・同じ発話の単位平均 INS)と音声、中心の埋め込み用の残りの発話(25s)。
保存先: data/diag_pool/n{n_tg}_u{n_utt}/t{ti:03d}.npz(初回だけ計算)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from d6_unit_cov import TTS_LEAK, load48

ROOT = Path(__file__).resolve().parent.parent


def load_pool(n_tg: int = 150, n_utt: int = 3, dev: str = "cuda") -> list:
    import c1_content as CC
    import conv_c0 as C0
    import f0hi as H
    import nvoc as N
    import pae as PA
    import train_c1 as T1
    import train_rvoc as TR
    cdir = ROOT / "data/diag_pool" / f"n{n_tg}_u{n_utt}"
    if (cdir / "done").exists():
        out = []
        for f in sorted(cdir.glob("t*.npz")):
            z = np.load(f, allow_pickle=True)
            t = {k: z[k] for k in ("T", "Tc0", "per_mean", "cen")}
            t.update({k: float(z[k]) for k in ("mu", "sd", "c0m")})
            t["spk"] = str(z["spk"])
            t["utts"] = [{k: z[f"u{j}_{k}"] for k in ("x", "er", "c0", "f0", "per", "u", "INS")} for j in range(int(z["n_utt"]))]
            out.append(t)
        return out
    cdir.mkdir(parents=True, exist_ok=True)
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev); c1.load_state_dict(st1["net"]); c1.eval()
    mfront = CC.MelFront().to(dev)
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()
    K = Cb.shape[0]
    sr_ = torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False)
    sm_w = TR.ckpt_env_smooth(sr_, None)

    def sm3(Ev):
        e = np.pad(Ev, ((0, 0), (1, 1)), mode="edge")
        return sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:]

    def analyse(x, f0):
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        return sm3(PA.envelope(xa, f0)), PA.periodicity(torch.from_numpy(xa)[None], torch.from_numpy(f0)[None])[0].numpy()

    @torch.no_grad()
    def units(x):
        n = len(x) // N.HOP
        u = c1(mfront(torch.from_numpy(T1.prime(x))[None].to(dev)))[..., -n:].argmax(1)[0].cpu().numpy()
        return np.pad(u, (0, max(0, n - len(u))), mode="edge")[:n]

    hs = TR.held_speakers()
    allspk = sorted(p.name for p in (ROOT / "female-dataset").iterdir() if p.is_dir() and p.name not in (hs | TTS_LEAK))
    lda_tr = {allspk[i] for i in np.random.RandomState(0).permutation(len(allspk))[:1200]}
    cand = [s_ for s_ in allspk if s_ not in lda_tr]
    pool = [cand[i] for i in np.random.RandomState(5).permutation(len(cand))[: n_tg * 2]]
    out = []
    for s in pool:
        ws = sorted((ROOT / "female-dataset" / s).glob("*.wav"))
        ref, tot = [], 0.0
        for w in ws:
            if tot >= 20.0:
                break
            ref.append(w); tot += sf.info(str(w)).duration
        rest = [w for w in ws if w not in ref]
        if len(rest) < n_utt + 2:
            continue
        xr = np.concatenate([load48(w) for w in ref]); xr = xr[: len(xr) // N.HOP * N.HOP]
        f0r, _ = H.teacher_f0(xr, len(xr) // N.HOP)
        envr, perr = analyse(xr, f0r)
        ur = units(xr)
        T = C0.table(envr[1:25, : len(ur)], ur, K, Cb)
        Tc0 = C0.table(envr[0:1, : len(ur)] - envr[0, : len(ur)].mean(), ur, K, Cb)[0]
        vo = f0r > 0
        act_r = (envr[0] - PA.C0_SIL) / 10 > 1.0
        utts = []
        for w in rest[:n_utt]:
            x = load48(w)[: 8 * 48000]
            n = len(x) // N.HOP
            x = x[: n * N.HOP]
            if n < 200:
                continue
            f0t, _ = H.teacher_f0(x, n)
            env, per = analyse(x, f0t)
            u = units(x)
            er = env[1:25, :n]
            utts.append({"x": x.astype(np.float32), "er": er.astype(np.float32), "c0": env[0, :n].astype(np.float32), "f0": f0t[:n].astype(np.float32),
                         "per": per[:, :n].astype(np.float32), "u": u.astype(np.int16), "INS": C0.table(er, u, K, Cb).astype(np.float32)})
        if not utts:
            continue
        cen = np.concatenate([load48(w) for w in rest[n_utt:]])[: 25 * 48000].astype(np.float32)
        t = {"spk": s, "T": T.astype(np.float32), "Tc0": Tc0.astype(np.float32), "mu": float(np.median(np.log(f0r[vo]))), "sd": float(np.log(f0r[vo]).std()),
             "c0m": float(envr[0][act_r].mean()), "per_mean": perr[:, vo].mean(1).astype(np.float32), "utts": utts, "cen": cen}
        np.savez(cdir / f"t{len(out):03d}.npz", spk=s, T=t["T"], Tc0=t["Tc0"], mu=t["mu"], sd=t["sd"], c0m=t["c0m"], per_mean=t["per_mean"], cen=cen,
                 n_utt=len(utts), **{f"u{j}_{k}": v for j, d in enumerate(utts) for k, v in d.items()})
        out.append(t)
        print("pool", len(out), s, flush=True)
        if len(out) >= n_tg:
            break
    (cdir / "done").write_text(str(len(out)))
    return out


def load_refs(n_tg: int = 150, n_utt: int = 3, dev: str = "cuda") -> list:
    """load_pool と同じ目標の参照 20s の軌道(c1..c24・C1 の単位)を保存して使い回す: data/diag_pool/n{n_tg}_u{n_utt}/r{ti:03d}.npz。"""
    import c1_content as CC
    import f0hi as H
    import nvoc as N
    import pae as PA
    import train_c1 as T1
    import train_rvoc as TR
    pool = load_pool(n_tg, n_utt, dev)
    cdir = ROOT / "data/diag_pool" / f"n{n_tg}_u{n_utt}"
    if (cdir / "refs_done").exists():
        return [dict(np.load(cdir / f"r{ti:03d}.npz")) for ti in range(len(pool))]
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev); c1.load_state_dict(st1["net"]); c1.eval()
    mfront = CC.MelFront().to(dev)
    sr_ = torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False)
    sm_w = TR.ckpt_env_smooth(sr_, None)
    out = []
    for ti, t in enumerate(pool):
        ws = sorted((ROOT / "female-dataset" / t["spk"]).glob("*.wav"))
        ref, tot = [], 0.0
        for w in ws:
            if tot >= 20.0:
                break
            ref.append(w); tot += sf.info(str(w)).duration
        xr = np.concatenate([load48(w) for w in ref]); xr = xr[: len(xr) // N.HOP * N.HOP]
        n = len(xr) // N.HOP
        f0r, _ = H.teacher_f0(xr, n)
        xa = np.concatenate([np.zeros(PA.A, np.float32), xr, np.zeros(PA.A, np.float32)])
        E = PA.envelope(xa, f0r)
        e = np.pad(E, ((0, 0), (1, 1)), mode="edge")
        E = sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:]
        with torch.no_grad():
            u = c1(mfront(torch.from_numpy(T1.prime(xr))[None].to(dev)))[..., -n:].argmax(1)[0].cpu().numpy()
        u = np.pad(u, (0, max(0, n - len(u))), mode="edge")[:n]
        r = {"env": E[1:25, :n].astype(np.float32), "c0": E[0, :n].astype(np.float32), "u": u.astype(np.int16), "f0": f0r[:n].astype(np.float32)}
        np.savez(cdir / f"r{ti:03d}.npz", **r)
        out.append(r)
        if ti % 25 == 0:
            print("refs", ti, flush=True)
    (cdir / "refs_done").write_text(str(len(out)))
    return out
