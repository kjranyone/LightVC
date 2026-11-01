"""D65(診断・学習なし): 耳 env_ab1(オーナー 2026-10-09: 女声の写し合成で包絡だけ本人の表の柔らかい引きにすると 4 試行とも『失格』)の帰属。
目標の実音声 30 本(D37 の cen)・f0 = 教師・c0 / 周期性 = 本物(D51 と同じ)・包絡だけ替える:
  TAB = Σp T(本人の Rust 登録の表・柔らかい引き = env_ab1 で失格)・TABPF = TAB の c2..c24 を 1.3 倍(周波数方向に鋭く)・
  DYN = 本物の包絡 + Σp(T − INS)(表の単位平均の誤差だけを足す = 単位の中の動きは本物)・INS = Σp INS(同じ発話から作った単位平均 = 平均は正確・動きなし)。
INS = その発話の実の包絡を C1 の硬い単位で平均した表(conv_c0.table・未観測の単位は近いコードブックから借用)。出力 = <out>/tNNN_<cond>.wav。
    uv run python d65_table_env_fail.py --s <scratchpad> --out_dir <scratchpad>/d65
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    import c1_content as CC
    import conv_c0 as C0
    import f0hi as H
    import nvoc as N
    import pae as PA
    import rvoc as R
    import train_c1 as T1
    import train_rvoc as TR
    S = Path(a.s)
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

    @torch.no_grad()
    def render(c0n, env24n, f0, per_t):
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)[None]
        v = (f0 > 0).astype(np.float64)[None]
        cond = torch.from_numpy(np.concatenate([c0n[None], env24n, lf, v, np.clip(per_t, 0, 1.5) * v], 0).astype(np.float32))[None].to(dev)
        fo = torch.from_numpy(f0.astype(np.float32))[None].to(dev)
        y = gen(cond, TR.excitation(fo, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)

    for ti in range(30):
        T = np.fromfile(S / "d53_en" / f"en_{ti:03d}" / "enroll.bin", dtype="<f4").reshape(24, -1).astype(np.float64)
        x, _ = sf.read(S / "d37" / f"cen_{ti:03d}.wav", dtype="float32")
        n = len(x) // N.HOP
        x = x[: n * N.HOP]
        f0, _ = H.teacher_f0(x, n)
        f0 = f0[:n].astype(np.float64)
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        e = np.pad(PA.envelope(xa, f0), ((0, 0), (1, 1)), mode="edge")
        env = (sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:])[:, :n].astype(np.float64)
        per = PA.periodicity(torch.from_numpy(xa)[None], torch.from_numpy(f0.astype(np.float32))[None])[0].numpy()[:, :n].astype(np.float64)
        with torch.no_grad():
            p = c1(mfront(torch.from_numpy(T1.prime(x))[None].to(dev)))[..., -n:].softmax(1)[0].cpu().numpy().astype(np.float64)
        p = np.pad(p, ((0, 0), (0, max(0, n - p.shape[1]))), mode="edge")[:, :n]
        u = p.argmax(0)
        er = env[1:25]
        INS = C0.table(er, u, K, Cb)
        c0n = (env[0] - PA.C0_SIL) / 10
        tab = T @ p
        pf = tab.copy(); pf[1:] *= 1.3
        conds = {"TAB": tab, "TABPF": pf, "DYN": er + (T - INS) @ p, "INS": INS @ p}
        for c, ev in conds.items():
            sf.write(od / f"t{ti:03d}_{c}.wav", render(c0n, ev / 10, f0, per), N.SR)
        print("target", ti, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
