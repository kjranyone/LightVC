"""D51(診断・学習なし): 変換の荒れ(倍音間コントラスト・出力の周期性の中域の低下)のうち、表から作る包絡の分。目標の実音声 30 本(D37 の cen)を、本人の表(D37 の Rust 登録)で作り直す。
周期性・f0(教師)・c0 は本物のまま: ORA = 本物の包絡・TABENV = 柔らかい引き Σp T(本人の表・β 1)・ORA75 / TABENV75 = 周期性を有声の 75 パーセンタイルの定数に。
出力の倍音間コントラストと出力の周期性(4 帯)を測って JSON に書く。
    uv run python d51_env_roughness.py --s <scratchpad> --out_dir <scratchpad>/d51 --out ../results/conv_p0/d51_env_roughness.json
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import c1_content as CC
    import f0hi as H
    import nvoc as N
    import pae as PA
    import rvoc as R
    import train_c1 as T1
    import train_rvoc as TR
    from eval_zsvc import contrast
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

    def analyse(x, f0):
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        e = np.pad(PA.envelope(xa, f0), ((0, 0), (1, 1)), mode="edge")
        env = sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:]
        per = PA.periodicity(torch.from_numpy(xa)[None], torch.from_numpy(f0.astype(np.float32))[None])[0].numpy()
        return env[:, : len(f0)].astype(np.float64), per[:, : len(f0)].astype(np.float64)

    @torch.no_grad()
    def render(c0n, env24n, f0, per_t):
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)[None]
        v = (f0 > 0).astype(np.float64)[None]
        cond = torch.from_numpy(np.concatenate([c0n[None], env24n, lf, v, np.clip(per_t, 0, 1.5) * v], 0).astype(np.float32))[None].to(dev)
        fo = torch.from_numpy(f0.astype(np.float32))[None].to(dev)
        y = gen(cond, TR.excitation(fo, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)

    def measure(y):
        m = len(y) // N.HOP
        y = y[: m * N.HOP].astype(np.float32)
        f, _ = H.teacher_f0(y, m)
        f = f[:m]
        _, p = analyse(y, f.astype(np.float64))
        v = f > 0
        return float(contrast(y.astype(np.float64), f.astype(np.float64))), (p[:, v].mean(1) if v.sum() > 10 else np.full(4, np.nan)).tolist()

    res: dict = {}
    for ti in range(30):
        T = np.fromfile(S / "d37/_work" / f"enroll_{ti:03d}" / "enroll.bin", dtype="<f4").reshape(24, -1).astype(np.float64)
        x, _ = sf.read(S / "d37" / f"cen_{ti:03d}.wav", dtype="float32")
        n = len(x) // N.HOP
        x = x[: n * N.HOP]
        f0, _ = H.teacher_f0(x, n)
        f0 = f0[:n].astype(np.float64)
        env, per = analyse(x, f0)
        with torch.no_grad():
            p = c1(mfront(torch.from_numpy(T1.prime(x))[None].to(dev)))[..., -n:].softmax(1)[0].cpu().numpy().astype(np.float64)
        p = np.pad(p, ((0, 0), (0, max(0, n - p.shape[1]))), mode="edge")[:, :n]
        c0n = (env[0] - PA.C0_SIL) / 10
        p75 = np.repeat(np.percentile(per[:, f0 > 0], 75, axis=1)[:, None], n, 1)
        tab = (T @ p) / 10.0
        for c, (e, pt) in {"ORA": (env[1:25] / 10, per), "TABENV": (tab, per), "ORA75": (env[1:25] / 10, p75), "TABENV75": (tab, p75)}.items():
            y = render(c0n, e, f0, pt)
            sf.write(od / f"t{ti:03d}_{c}.wav", y, N.SR)
            res.setdefault(c, []).append(measure(y))
        res.setdefault("REAL", []).append(measure(x))
        print("target", ti, flush=True)
    summ = {c: {"contrast": round(float(np.nanmedian([r[0] for r in v])), 3), "per": [round(float(z), 3) for z in np.nanmedian(np.array([r[1] for r in v]), 0)]} for c, v in res.items()}
    print(json.dumps(summ, ensure_ascii=False), flush=True)
    Path(a.out).write_text(json.dumps({"summary": summ, "rows": res}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
