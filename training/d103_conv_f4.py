"""D103 = D91 の派生(変換の周期性を F4 に): 周期性 = F4(元の音声・因果)の有声の平均を、操作者の F4 の平均 → 目標の周期性の平均へ移す。
条件 = TAB / C05 / C07 / C10(周期性 = F4 を移したもの)・C07P(周期性 = 定数の 75 パーセンタイル = 従来)。出力部 = --rvoc_ckpt。
以下は D91 の説明。D91 = D89 の派生(γ の掃引と操作者の表の作り方の切り分け): 条件 = TAB・C03/C05/C07/C10(γ・操作者の表 = F3)・C10CT(γ1・操作者の表 = CheapTrick = Rust の現行)。
以下は D89 の説明。D89(学習なし): 単位・遷移ごとの軌道の型の入れ替え(話者を内容で索引・オーナー 10-08)。今の製品は単位の平均だけを男声 → 目標に入れ替える。
これを (前の単位, 今の単位, 単位に入ってからの経過の段 φ) ごとの包絡の型 TPH に広げる(観測が --min 未満は (単位, φ) → 単位の平均へ戻す)。φ は硬い単位の列から因果に決まる。
  目標の型 = 登録の参照(<s>/d37/_work/ref_TTT.wav)の CheapTrick 平滑包絡(製品の表と同じ)・操作者の型 = 操作者の参照(<s>/d56/_work/op_PPPP.wav)の F3 の包絡(実行時の元の包絡と同じ推定器)。
条件(全て同じ Python の製品経路: F1 の f0 を操作者の中央値 → 目標の μ・F2 の c0(因果 3 平均)・周期性 = 目標の 75 パーセンタイル・出力部 = --rvoc_ckpt):
  TAB = Σp T_目標・C10 = TAB + M·W(F3 − Σp T_操作者)(Rust の γ1 + 色づけと同じ式)・
  TPH = TPH_目標[文脈]・TPHC = TPH_目標[文脈] + M·W(F3 − TPH_操作者[文脈])(型を入れ替え、男声の型からの残差 = 実現だけを写す)。
出力 = d4b 形式(pNNNN_tNNN_<cond>.wav・cen・SRC)。
    uv run python d89_tph_swap.py --s <scratchpad> --out_dir <scratchpad>/d89 --rvoc_ckpt ../results/rvtab_full/snap/ema_360k.pt
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
PHB = [0, 1, 2, 4, 8, 16, 10 ** 6]


def ctx(u: np.ndarray) -> tuple:
    prev = np.full(len(u), -1)
    ph = np.zeros(len(u), int)
    cp, st = -1, 0
    for t in range(len(u)):
        if t > 0 and u[t] != u[t - 1]:
            cp, st = u[t - 1], t
        prev[t] = cp
        ph[t] = np.searchsorted(PHB, t - st, side="right") - 1
    return prev, ph


def templates(E: np.ndarray, U: np.ndarray, T: np.ndarray, mn: int) -> dict:
    prev, ph = ctx(U)
    acc: dict = {"UP": {}, "TP": {}}
    for t in range(len(U)):
        for key, d in (((U[t], ph[t]), acc["UP"]), ((prev[t], U[t], ph[t]), acc["TP"])):
            s = d.setdefault(key, [np.zeros(E.shape[0]), 0])
            s[0] += E[:, t]; s[1] += 1
    return {lv: {k: v[0] / v[1] for k, v in d.items() if v[1] >= mn} for lv, d in acc.items()} | {"T": T}


def lookup(tp: dict, u: np.ndarray, p: np.ndarray) -> np.ndarray:
    prev, ph = ctx(u)
    base = tp["T"] @ p
    out = base.copy()
    for t in range(len(u)):
        v = tp["TP"].get((prev[t], u[t], ph[t]))
        if v is None:
            v = tp["UP"].get((u[t], ph[t]))
        if v is not None:
            out[:, t] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rvoc_ckpt", required=True)
    ap.add_argument("--min", type=int, default=3)
    ap.add_argument("--max", type=int, default=60)
    ap.add_argument("--dump_cond", default="", help="診断: 条件 [31, T] を <dir>/<key>_<cond>.npy に保存")
    a = ap.parse_args()
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
    S = Path(a.s)
    dev = "cuda"
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    sr_ = torch.load(a.rvoc_ckpt, map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=sr_["cfg"]["ch"], kernels=tuple(sr_["cfg"]["kernels"]), dils=tuple(sr_["cfg"]["dils"]), d_cond=sr_["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(sr_["ema"]); gen.eval().requires_grad_(False)
    sm_w = TR.ckpt_env_smooth(torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False), None)
    sf1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev); f1.load_state_dict(sf1["net"]); f1.eval()
    f1front = FE.Front().to(dev)
    sf4 = torch.load(ROOT / "results/f4_1/last.pt", map_location="cpu", weights_only=False)
    f4 = LV.PerEst(sf4["cfg"]["ch"], tuple(sf4["cfg"]["dils"])).to(dev); f4.load_state_dict(sf4["net"]); f4.eval()
    sf2 = torch.load(ROOT / "results/f2_2/last.pt", map_location="cpu", weights_only=False)
    f2 = LV.Lvl(sf2["cfg"]["ch"], tuple(sf2["cfg"]["dils"])).to(dev); f2.load_state_dict(sf2["net"]); f2.eval()
    sf3 = torch.load(ROOT / "results/f3_1/last.pt", map_location="cpu", weights_only=False)
    f3 = LV.EnvEst(sf3["cfg"]["ch"], tuple(sf3["cfg"]["dils"])).to(dev); f3.load_state_dict(sf3["net"]); f3.eval()
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev); c1.load_state_dict(st1["net"]); c1.eval()
    mfront = CC.MelFront().to(dev)
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()
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
        pad = lambda z: np.pad(z, ((0, 0), (0, max(0, n - z.shape[1]))), mode="edge")[:, :n]
        return {"x": x, "n": n, "f0": np.pad(f0f, (0, n - len(f0f))), "c0": np.pad(c0f, (0, n - len(c0f))), "e3": pad(e3), "p": pad(p), "u": pad(p).argmax(0), "p4": pad(p4)}

    def cheaptrick(x: np.ndarray, n: int) -> tuple:
        f0, _ = H.teacher_f0(x, n)
        f0 = f0[:n].astype(np.float64)
        xa = np.concatenate([np.zeros(PA.A, np.float32), x[: n * N.HOP], np.zeros(PA.A, np.float32)])
        e = np.pad(PA.envelope(xa, f0), ((0, 0), (1, 1)), mode="edge")
        env = (sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:])[:, :n].astype(np.float64)
        per = PA.periodicity(torch.from_numpy(xa)[None], torch.from_numpy(f0.astype(np.float32))[None])[0].numpy()[:, :n]
        return env[1:25], f0, per

    def enroll_target(x: np.ndarray) -> dict:
        fr = front(x)
        env, f0, per = cheaptrick(fr["x"], fr["n"])
        T = C0.table(env, fr["u"], K, Cb)
        v = f0 > 0
        r = env - T[:, fr["u"]]
        rv = r[:, v] - r[:, v].mean(1, keepdims=True)
        return {"tp": templates(env, fr["u"], T, a.min), "mu": float(np.median(np.log(f0[v]))), "p75": np.percentile(per[:, v], 75, axis=1), "cov": rv @ rv.T / rv.shape[1],
                "pmean": per[:, v].mean(1)}

    def enroll_operator(x: np.ndarray) -> dict:
        fr = front(x)
        T = C0.table(fr["e3"], fr["u"], K, Cb)
        v = fr["f0"] > 0
        r = fr["e3"] - T[:, fr["u"]]
        rv = r[:, v] - r[:, v].mean(1, keepdims=True)
        envc, f0c, _ = cheaptrick(fr["x"], fr["n"])
        Tc = C0.table(envc, fr["u"], K, Cb)
        rc = envc - Tc[:, fr["u"]]
        vc = f0c > 0
        rcv = rc[:, vc] - rc[:, vc].mean(1, keepdims=True)
        return {"tp": templates(fr["e3"], fr["u"], T, a.min), "mu": float(np.median(np.log(fr["f0"][v]))), "cov": rv @ rv.T / rv.shape[1], "p4mean": fr["p4"][:, v].mean(1),
                "Tc": Tc, "covc": rcv @ rcv.T / rcv.shape[1]}

    @torch.no_grad()
    def render(c0n, env24, f0, per_t) -> np.ndarray:
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)[None]
        v = (f0 > 0).astype(np.float64)[None]
        cond = torch.from_numpy(np.concatenate([c0n[None], env24 / 10, lf, v, np.clip(per_t, 0, 1.5) * v], 0).astype(np.float32))[None].to(dev)
        G_["last"] = cond[0].cpu().numpy()
        y = gen(cond, TR.excitation(torch.from_numpy(f0.astype(np.float32))[None].to(dev), cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)

    G_: dict = {}
    tg, op = {}, {}
    stats = []
    for f in sorted((S / "d56").glob("p*_SRC.wav"))[: a.max]:
        k = f.name[: -len("_SRC.wav")]
        m = re.match(r"p(\d+)_t(\d+)", k)
        pi, ti = int(m.group(1)), int(m.group(2))
        if ti not in tg:
            tg[ti] = enroll_target(load48(S / "d37/_work" / f"ref_{ti:03d}.wav").astype(np.float32))
        if pi not in op:
            op[pi] = enroll_operator(load48(S / "d56/_work" / f"op_{pi:04d}.wav").astype(np.float32))
        T_, O_ = tg[ti], op[pi]
        x, _ = sf.read(f, dtype="float32")
        fr = front(x)
        n = fr["n"]
        f0o = np.where(fr["f0"] > 0, fr["f0"] * np.exp(T_["mu"] - O_["mu"]), 0.0)
        per_t = np.repeat(T_["p75"][:, None], n, 1)
        M = msqrt(T_["cov"], 0.5) @ msqrt(W @ O_["cov"] @ W.T, -0.5)
        Mc = msqrt(T_["cov"], 0.5) @ msqrt(W @ O_["covc"] @ W.T, -0.5)
        tab = T_["tp"]["T"] @ fr["p"]
        dlt = M @ W @ (fr["e3"] - O_["tp"]["T"] @ fr["p"])
        conds = {"TAB": tab, "C05": tab + 0.5 * dlt, "C07": tab + 0.7 * dlt, "C10": tab + dlt, "C07P": tab + 0.7 * dlt}
        per_f4 = fr["p4"] - O_["p4mean"][:, None] + T_["pmean"][:, None]
        for c, ev in conds.items():
            sf.write(od / f"{k}_{c}.wav", render(fr["c0"], ev, f0o, per_t if c == "C07P" else per_f4), N.SR)
            if a.dump_cond:
                Path(a.dump_cond).mkdir(parents=True, exist_ok=True)
                np.save(Path(a.dump_cond) / f"{k}_{c}.npy", G_["last"])
        shutil.copy(f, od / f.name)
        stats.append({"k": k, "tph_cover_TP": float(np.mean([1.0 for _ in [0]]))})
        print(k, flush=True)
    for c in sorted((S / "d56").glob("cen_*.wav")):
        shutil.copy(c, od / c.name)
    cov = {ti: {"TP": len(v["tp"]["TP"]), "UP": len(v["tp"]["UP"])} for ti, v in tg.items()}
    (od / "d89_templates.json").write_text(json.dumps(cov, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
