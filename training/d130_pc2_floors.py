"""D130 = ref_reader.md §4 PC2(学習なし): 参照を読む生成器 R の学習の正解(目標 A の CheapTrick 包絡 c1..c24 と c0・伸縮しない)に対する誤差の床と基準。
交差ペア = 操作者 B と目標 A が同じ文 tK を読む(B の入力は DTW で A の時間へ = 入力側だけ)。女声 → 女声(女声 TTS・--n_ff 組)と男声 → 女声(male_tts_v4l_par・--n_mf 組)。
基準(有声フレームの L1・c1..c24 の次元平均 / c0 の L1・相関):
  TAB = Σp_B T_A(B の事後で A の表を引く)・CONV = TAB + 0.7·M W(F3_B − Σp_B T_B)(現行の変換・男声のときだけ W_1.3)・IDB = B の CheapTrick 包絡そのもの(写しの解)・
  READ = 素朴な鍵の柔らかい注意で A の登録から読む(温度 0.05)・FLOOR = A の同じ文の別の話し方(soft)の CheapTrick(DTW)= 決まらない揺らぎの床・MEAN = A の登録の有声平均の定数。
c0 は A の発話の平均を引いた相対値で比べる(水準は操作者の演技)。出力 = results/conv_p0/d130.json。
    uv run python d130_pc2_floors.py --n_ff 48 --n_mf 24
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent
FEM = ROOT / "data/female_tts_corpus"
MPAR = ROOT / "data/male_tts_v4l_par"
MALE = ROOT / "data/male_tts_v4l"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_ff", type=int, default=48)
    ap.add_argument("--n_mf", type=int, default=24)
    a = ap.parse_args()
    import librosa
    import c1_content as CC
    import conv_c0 as C0
    import f0hi as H
    import lvl as LV
    import nvoc as N
    import pae as PA
    import train_c1 as T1
    import train_rvoc as TR
    from d88_perturb_identity import msqrt, wmat
    dev = "cuda"
    ld = lambda p: torch.load(p, map_location="cpu", weights_only=False)
    sm_w = TR.ckpt_env_smooth(ld(ROOT / "results/rvoc3hi/snap/ema_330k.pt"), None)
    s3 = ld(ROOT / "results/f3_1/last.pt"); f3 = LV.EnvEst(s3["cfg"]["ch"], tuple(s3["cfg"]["dils"])).to(dev); f3.load_state_dict(s3["net"]); f3.eval()
    sc = ld(ROOT / "results/c1_1/last.pt"); c1 = CC.C1(sc["cfg"]["k"], sc["cfg"]["ch"], tuple(sc["cfg"]["dils"])).to(dev); c1.load_state_dict(sc["net"]); c1.eval()
    mfront = CC.MelFront().to(dev)
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()
    K = Cb.shape[0]
    W13 = wmat(1.3)
    cache: dict = {}

    def ctxkey(p: np.ndarray) -> np.ndarray:
        sp = np.sqrt(np.maximum(p, 0))
        cs = np.cumsum(np.pad(sp, ((0, 0), (5, 0))), 1)
        k_ = np.concatenate([sp, (cs[:, 4:-1] - cs[:, :-5]) / 4], 0)
        return k_ / (np.linalg.norm(k_, axis=0, keepdims=True) + 1e-9)

    @torch.no_grad()
    def analyse(p_: Path) -> dict:
        if p_ in cache:
            return cache[p_]
        x = load48(p_).astype(np.float32)
        n = len(x) // N.HOP
        x = x[: n * N.HOP]
        f0, _ = H.teacher_f0(x, n)
        f0 = f0[:n].astype(np.float64)
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        e = np.pad(PA.envelope(xa, f0), ((0, 0), (1, 1)), mode="edge")
        env = (sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:])[:, :n].astype(np.float64)
        m0 = mfront(torch.cat([torch.zeros(1, N.WIN - N.HOP, device=dev), torch.from_numpy(x)[None].to(dev)], 1))
        e3 = 10.0 * f3(m0)[0].cpu().numpy()[1:25, :n].astype(np.float64)
        p = c1(mfront(torch.from_numpy(T1.prime(x))[None].to(dev)))[..., -n:].softmax(1)[0].cpu().numpy().astype(np.float64)
        pad = lambda z: np.pad(z, ((0, 0), (0, n - z.shape[1])), mode="edge")
        r = {"n": n, "f0": f0, "env": env[1:25], "c0": (env[0] - PA.C0_SIL) / 10, "e3": pad(e3), "p": pad(p), "lm": pad(m0[0].cpu().numpy()[:, :n].astype(np.float64))}
        r["u"] = r["p"].argmax(0)
        cache[p_] = r
        return r

    def enroll(paths: list) -> dict:
        rs = [analyse(q) for q in paths]
        E = np.concatenate([r["env"] for r in rs], 1); E3 = np.concatenate([r["e3"] for r in rs], 1)
        U = np.concatenate([r["u"] for r in rs]); P = np.concatenate([r["p"] for r in rs], 1)
        V = np.concatenate([r["f0"] > 0 for r in rs]); C0v = np.concatenate([r["c0"] for r in rs])
        T = C0.table(E, U, K, Cb); T3 = C0.table(E3, U, K, Cb)
        R3 = E3 - T3 @ P
        R3 = R3[:, V] - R3[:, V].mean(1, keepdims=True)
        Rt = (E - T @ P)[:, V]; Rt = Rt - Rt.mean(1, keepdims=True)
        return {"T": T, "T3": T3, "C3": R3 @ R3.T / R3.shape[1], "Ct": Rt @ Rt.T / Rt.shape[1], "Kc": np.concatenate([ctxkey(r["p"]) for r in rs], 1),
                "E": E, "C0r": C0v - C0v[V].mean(), "mean": E[:, V].mean(1)}

    def align(dst: dict, src: dict) -> np.ndarray:
        an = dst["lm"] / (np.linalg.norm(dst["lm"], axis=0, keepdims=True) + 1e-9)
        bn = src["lm"] / (np.linalg.norm(src["lm"], axis=0, keepdims=True) + 1e-9)
        _, wp = librosa.sequence.dtw(C=1 - an.T @ bn)
        j = np.zeros(dst["n"], int)
        for i_, j_ in wp[::-1]:
            j[i_] = j_
        return j

    def read(q: np.ndarray, EA: dict) -> tuple:
        w = torch.softmax(torch.from_numpy(q.T).float().to(dev) @ torch.from_numpy(EA["Kc"]).float().to(dev) / 0.05, 1)
        v = torch.from_numpy(np.concatenate([EA["E"], EA["C0r"][None]], 0).T).float().to(dev)
        o = (w @ v).cpu().numpy().T.astype(np.float64)
        return o[:24], o[24]

    def score(pred: np.ndarray, tgt: np.ndarray, v: np.ndarray) -> dict:
        return {"l1": float(np.abs(pred[:, v] - tgt[:, v]).mean()), "corr": float(np.mean([np.corrcoef(pred[d, v], tgt[d, v])[0, 1] for d in range(24)]))}

    fems = sorted(d.name for d in FEM.iterdir() if d.is_dir() and all((d / f"t{i:02d}_{s}.wav").exists() for i in range(10) for s in ("neutral", "soft")))
    random.Random(130).shuffle(fems)
    males = sorted(d.name for d in MPAR.iterdir() if d.is_dir() and len(list(d.glob("f*.wav"))) == 10)
    jobs = [("ff", fems[2 * k], fems[2 * k + 1], k % 10) for k in range(a.n_ff)] + [("mf", males[k % len(males)], fems[200 + k], k % 10) for k in range(a.n_mf)]
    rows = []
    for kind, sb, sa, tk in jobs:
        EA = enroll([FEM / sa / f"t{i:02d}_neutral.wav" for i in range(10) if i != tk])
        if kind == "ff":
            Bp = FEM / sb / f"t{tk:02d}_neutral.wav"
            EB = enroll([FEM / sb / f"t{i:02d}_neutral.wav" for i in range(10) if i != tk])
            Wm = np.eye(24)
        else:
            Bp = MPAR / sb / f"f{tk:02d}.wav"
            EB = enroll(sorted((MALE / sb).glob("t*.wav"))[:8])
            Wm = W13
        A_ = analyse(FEM / sa / f"t{tk:02d}_neutral.wav"); B_ = analyse(Bp); As = analyse(FEM / sa / f"t{tk:02d}_soft.wav")
        j = align(A_, B_)
        v = A_["f0"] > 0
        pB = B_["p"][:, j]
        tab = EA["T"] @ pB
        Mc = msqrt(EA["C3"], 0.5) @ msqrt(Wm @ EB["C3"] @ Wm.T, -0.5)
        conv = tab + 0.7 * Mc @ Wm @ (B_["e3"] - EB["T3"] @ B_["p"])[:, j]
        rd, rc0 = read(ctxkey(B_["p"])[:, j], EA)
        js = align(A_, As)
        preds = {"TAB": tab, "CONV": conv, "IDB": B_["env"][:, j], "READ": rd, "FLOOR": As["env"][:, js], "MEAN": np.repeat(EA["mean"][:, None], A_["n"], 1)}
        row = {"kind": kind, "A": sa, "B": sb, "text": tk}
        for c, pr in preds.items():
            for k_, val in score(pr, A_["env"], v).items():
                row[f"{c}_{k_}"] = round(val, 4)
        c0t = A_["c0"] - A_["c0"][v].mean()
        c0s = {"C0_B": B_["c0"][j] - B_["c0"][j][v].mean(), "C0_READ": rc0 - rc0[v].mean(), "C0_FLOOR": As["c0"][js] - As["c0"][js][v].mean()}
        for c, z in c0s.items():
            row[f"{c}_l1"] = round(float(np.abs(z[v] - c0t[v]).mean()), 4)
            row[f"{c}_corr"] = round(float(np.corrcoef(z[v], c0t[v])[0, 1]), 4)
        rows.append(row)
        print(row, flush=True)
    keys = [k_ for k_ in rows[0] if k_ not in ("kind", "A", "B", "text")]
    summ = {kd: {k_: round(float(np.median([r[k_] for r in rows if r["kind"] == kd])), 4) for k_ in keys} for kd in ("ff", "mf")}
    print("median", json.dumps(summ), flush=True)
    (ROOT / "results/conv_p0/d130.json").write_text(json.dumps({"median": summ, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
