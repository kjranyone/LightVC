"""D70(診断・学習なし): 単位の中の動きを別の話者から写せるか(元の話者の動きを写す変換の前提)。女声 TTS コーパス(全話者が同じ 10 文)を使う。
話者 A の文 tK_neutral を写し合成(f0 = 教師・c0 / 周期性 = A の本物・出力部 = rvoc3hi 330k)し、包絡だけ替える:
  TAB = T_A·p_A(A の表 = 同じ文 tK を除く A のファイル 20s 以上から)・SELF = TAB + res_A(A 自身の動き)・XFER = TAB + res_B→A(同じ文を読んだ別の話者 B の動きを DTW で A の時間へ写す)。
res_X = 本物の包絡 − 同じ発話の単位平均の事後混合(D68 と同じ定義)。DTW は対数メルのコサイン距離(診断専用・非因果)。出力 = d4b の形式(cen = A の本物・SRC = A の本物)。
    uv run python d70_xdyn.py --out_dir <scratchpad>/d70 --n 12
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data/female_tts_corpus"


def dtw_path(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    an = a / (np.linalg.norm(a, axis=0, keepdims=True) + 1e-9)
    bn = b / (np.linalg.norm(b, axis=0, keepdims=True) + 1e-9)
    D = 1 - an.T @ bn
    n, m = D.shape
    C = np.full((n + 1, m + 1), np.inf)
    C[0, 0] = 0
    for i in range(1, n + 1):
        ci, cp = C[i], C[i - 1]
        d = D[i - 1]
        for j in range(1, m + 1):
            ci[j] = d[j - 1] + min(cp[j - 1], cp[j], ci[j - 1])
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = np.argmin([C[i - 1, j - 1], C[i - 1, j], C[i, j - 1]])
        i, j = (i - 1, j - 1) if k == 0 else (i - 1, j) if k == 1 else (i, j - 1)
    return np.array(path[::-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    import c1_content as CC
    import conv_c0 as C0
    import f0hi as H
    import nvoc as N
    import pae as PA
    import rvoc as R
    import train_c1 as T1
    import train_rvoc as TR
    dev = "cuda"
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    sr_ = torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=sr_["cfg"]["ch"], kernels=tuple(sr_["cfg"]["kernels"]), dils=tuple(sr_["cfg"]["dils"]), d_cond=sr_["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(sr_["ema"]); gen.eval().requires_grad_(False)
    sm_w = TR.ckpt_env_smooth(sr_, None)
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev); c1.load_state_dict(st1["net"]); c1.eval()
    mfront = CC.MelFront().to(dev)
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()
    K = Cb.shape[0]

    def analyse(x: np.ndarray) -> dict:
        n = len(x) // N.HOP
        x = x[: n * N.HOP]
        f0, _ = H.teacher_f0(x, n)
        f0 = f0[:n].astype(np.float64)
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        e = np.pad(PA.envelope(xa, f0), ((0, 0), (1, 1)), mode="edge")
        env = (sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:])[:, :n].astype(np.float64)
        per = PA.periodicity(torch.from_numpy(xa)[None], torch.from_numpy(f0.astype(np.float32))[None])[0].numpy()[:, :n].astype(np.float64)
        with torch.no_grad():
            mf = mfront(torch.from_numpy(T1.prime(x))[None].to(dev))
            p = c1(mf)[..., -n:].softmax(1)[0].cpu().numpy().astype(np.float64)
            lm = mf[0, :, -n:].cpu().numpy().astype(np.float64)
        p = np.pad(p, ((0, 0), (0, max(0, n - p.shape[1]))), mode="edge")[:, :n]
        lm = np.pad(lm, ((0, 0), (0, max(0, n - lm.shape[1]))), mode="edge")[:, :n]
        return {"x": x, "n": n, "f0": f0, "env": env, "per": per, "p": p, "lm": lm}

    def table(spk: str, skip: str) -> np.ndarray:
        files = sorted(f for f in (CORPUS / spk).glob("*.wav") if not f.name.startswith(skip + "_"))
        envs, us, tot = [], [], 0.0
        for f in files:
            if tot >= 20.0:
                break
            r = analyse(load48(f).astype(np.float32))
            envs.append(r["env"][1:25]); us.append(r["p"].argmax(0)); tot += r["n"] * N.HOP / N.SR
        return C0.table(np.concatenate(envs, 1), np.concatenate(us), K, Cb)

    @torch.no_grad()
    def render(c0n, env24n, f0, per_t):
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)[None]
        v = (f0 > 0).astype(np.float64)[None]
        cond = torch.from_numpy(np.concatenate([c0n[None], env24n, lf, v, np.clip(per_t, 0, 1.5) * v], 0).astype(np.float32))[None].to(dev)
        fo = torch.from_numpy(f0.astype(np.float32))[None].to(dev)
        y = gen(cond, TR.excitation(fo, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)

    def res(r: dict) -> np.ndarray:
        er = r["env"][1:25]
        return er - C0.table(er, r["p"].argmax(0), K, Cb) @ r["p"]

    spks = sorted(d.name for d in CORPUS.iterdir() if d.is_dir() and all((d / f"t{i:02d}_neutral.wav").exists() for i in range(10)))
    rng = random.Random(70)
    rng.shuffle(spks)
    log = []
    for k in range(a.n):
        sa, sb = spks[2 * k], spks[2 * k + 1]
        tk = f"t{k % 10:02d}"
        A = analyse(load48(CORPUS / sa / f"{tk}_neutral.wav").astype(np.float32))
        B = analyse(load48(CORPUS / sb / f"{tk}_neutral.wav").astype(np.float32))
        TA = table(sa, tk)
        rA, rB = res(A), res(B)
        path = dtw_path(A["lm"], B["lm"])
        rBA = np.zeros_like(rA)
        cnt = np.zeros(A["n"])
        for i, j in path:
            rBA[:, i] += rB[:, j]; cnt[i] += 1
        rBA /= np.maximum(cnt, 1)[None]
        tab = TA @ A["p"]
        c0n = (A["env"][0] - PA.C0_SIL) / 10
        key = f"p{k:04d}_t{k:03d}"
        for c, ev in {"TAB": tab, "SELF": tab + rA, "XFER": tab + rBA}.items():
            sf.write(od / f"{key}_{c}.wav", render(c0n, ev / 10, A["f0"], A["per"]), N.SR)
        sf.write(od / f"{key}_SRC.wav", A["x"], N.SR)
        sf.write(od / f"cen_{k:03d}.wav", A["x"], N.SR)
        cc = float(np.mean([np.corrcoef(rA[d], rBA[d])[0, 1] for d in range(24)]))
        log.append({"k": k, "A": sa, "B": sb, "text": tk, "res_corr_A_vs_BtoA": round(cc, 3), "rms_A": round(float(rA.std()), 3), "rms_BtoA": round(float(rBA.std()), 3)})
        print(log[-1], flush=True)
    (od / "d70.json").write_text(json.dumps(log, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
