"""D98 = D97 の派生(製品経路の写し合成の部品ごとの代価): ORA(本物の分析)から 1 つずつ製品の推定に替える = LVL(F2 + 因果 3 平均)・ENV(F3 の包絡)・PER(周期性 = 登録の 75 パーセンタイルの定数 × 有声)・F1(f0 = F1)・ALL(4 つとも)。
登録 = <s>/d85_work/en_TTT(Rust・res_cov つき)。D95 の --extra で区別の AUC を測る。
以下は D97 の説明。D97(学習なし): 音量 c0 の作り方の比べ。D96: 製品の F2 + 因果 3 平均は 1 フレーム遅れ・RMSE 0.314(生の F2 は遅れなし 0.147 だが 32〜100Hz の揺れ +3.6dB)。
評価の 30 目標の女声の写し合成(包絡・f0・周期性は本物の分析 = D64 と同じ・出力部 = --rvoc_ckpt)で c0 だけ替える:
  ORA = 本物(CheapTrick 平滑)・AVG3 = F2 + 因果 3 平均(製品)・RAW = F2 生・AVG2 = 因果 2 平均・EMA{a} = 指数平滑 y_t = a·r_t + (1 − a)·y_{t−1}。
出力 = <out>/pNNNN_tNNN_<cond>.wav(D95 の --extra で区別の AUC を測る)と c0 の誤差・遅れ。
    uv run python d97_c0_variants.py --s <scratchpad> --out_dir <scratchpad>/d97
"""
from __future__ import annotations

import argparse
import json
import shutil
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
    ap.add_argument("--rvoc_ckpt", default=str(ROOT / "results/rvoc3hi/snap/ema_330k.pt"))
    a = ap.parse_args()
    import c1_content as CC
    import f0hi as H
    import lvl as LV
    import nvoc as N
    import pae as PA
    import rvoc as R
    import train_c3 as C3
    import train_rvoc as TR
    S = Path(a.s)
    dev = "cuda"
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    sr_ = torch.load(a.rvoc_ckpt, map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=sr_["cfg"]["ch"], kernels=tuple(sr_["cfg"]["kernels"]), dils=tuple(sr_["cfg"]["dils"]), d_cond=sr_["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(sr_["ema"]); gen.eval().requires_grad_(False)
    sm_w = TR.ckpt_env_smooth(torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False), None)
    import f0est as FE
    sf1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev); f1.load_state_dict(sf1["net"]); f1.eval()
    f1front = FE.Front().to(dev)
    sf3 = torch.load(ROOT / "results/f3_1/last.pt", map_location="cpu", weights_only=False)
    f3 = LV.EnvEst(sf3["cfg"]["ch"], tuple(sf3["cfg"]["dils"])).to(dev); f3.load_state_dict(sf3["net"]); f3.eval()
    sf2 = torch.load(ROOT / "results/f2_2/last.pt", map_location="cpu", weights_only=False)
    f2 = LV.Lvl(sf2["cfg"]["ch"], tuple(sf2["cfg"]["dils"])).to(dev); f2.load_state_dict(sf2["net"]); f2.eval()
    mfront = CC.MelFront().to(dev)

    @torch.no_grad()
    def render(c0n, env24n, f0, per_t):
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)[None]
        v = (f0 > 0).astype(np.float64)[None]
        cond = torch.from_numpy(np.concatenate([c0n[None], env24n, lf, v, np.clip(per_t, 0, 1.5) * v], 0).astype(np.float32))[None].to(dev)
        y = gen(cond, TR.excitation(torch.from_numpy(f0.astype(np.float32))[None].to(dev), cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)

    def ema(r: np.ndarray, al: float) -> np.ndarray:
        y = np.zeros_like(r)
        acc = r[0]
        for i, v in enumerate(r):
            acc = al * v + (1 - al) * acc
            y[i] = acc
        return y

    stats: dict = {}
    for t in range(30):
        x, _ = sf.read(S / "d37" / f"cen_{t:03d}.wav", dtype="float32")
        n = len(x) // N.HOP
        x = x[: n * N.HOP]
        f0, _ = H.teacher_f0(x, n)
        f0 = f0[:n].astype(np.float64)
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        e = np.pad(PA.envelope(xa, f0), ((0, 0), (1, 1)), mode="edge")
        env = (sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:])[:, :n].astype(np.float64)
        per = PA.periodicity(torch.from_numpy(xa)[None], torch.from_numpy(f0.astype(np.float32))[None])[0].numpy()[:, :n].astype(np.float64)
        c0 = (env[0] - PA.C0_SIL) / 10
        with torch.no_grad():
            m = mfront(torch.cat([torch.zeros(1, N.WIN - N.HOP, device=dev), torch.from_numpy(x)[None].to(dev)], 1))
            r = f2(m)[:, None]
            raw = r[0, 0].cpu().numpy()[:n].astype(np.float64)
            avg3 = C3.causal_avg(r, 3)[0, 0].cpu().numpy()[:n].astype(np.float64)
            e3 = f3(m)[0].cpu().numpy()[1:25, :n].astype(np.float64)
            xt = torch.from_numpy(x)[None].to(dev)
            v_, p_ = f1(f1front(torch.cat([torch.zeros(1, FE.CTX48, device=dev), xt], 1), n))
            f0f = FE.decode(v_, p_, 0.5)[0].cpu().numpy()[:n].astype(np.float64)
        pad = lambda z: np.pad(z, (0, n - len(z)), mode="edge")
        e3 = np.pad(e3, ((0, 0), (0, n - e3.shape[1])), mode="edge")
        f0f = pad(f0f)
        p75 = np.asarray(json.loads((S / "d85_work" / f"en_{t:03d}" / "enroll.json").read_text())["per_p75"], np.float64)
        pconst = np.repeat(p75[:, None], n, 1)
        key = f"p{t:04d}_t{t:03d}"
        conds = {"ORA": (c0, env[1:25] / 10, f0, per), "LVL": (pad(avg3), env[1:25] / 10, f0, per), "ENV": (c0, e3, f0, per),
                 "PER": (c0, env[1:25] / 10, f0, pconst), "F1": (c0, env[1:25] / 10, f0f, per), "ALL": (pad(avg3), e3, f0f, pconst)}
        for k, (cc, ee, ff, pp) in conds.items():
            sf.write(od / f"{key}_{k}.wav", render(cc, ee, ff, pp), N.SR)
            stats.setdefault(k, []).append(float(np.sqrt(((cc - c0) ** 2).mean())))
        shutil.copy(S / "d37" / f"cen_{t:03d}.wav", od / f"cen_{t:03d}.wav")
        print(key, flush=True)
    rep = {k: round(float(np.mean(v)), 4) for k, v in stats.items()}
    print("c0 rmse vs real", json.dumps(rep), flush=True)
    (od / "d97_c0_rmse.json").write_text(json.dumps(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
