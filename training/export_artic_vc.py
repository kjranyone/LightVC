"""製品経路(表の引き・G なし)を Rust(crates/lightvc-core/src/artic_vc.rs)で組むための登録ファイルと、条件の列の一致検査の固定の入出力。
登録 <out>/enroll.bin・enroll.json: 表 T [24][K](CheapTrick c1..c24・参照 20s の単位平均・生の単位)・mu(参照の有声 log f0 の中央値)・per_mean [4]。
fixture/: x.f32(元の声・48kHz)・med.f32(元の声の有声 log f0 の中央値・固定値で比較)・cond.f32 [T][31]・f0o.f32 [T](Python の Chain = 製品の処方: F2 + 因果 3 平均・表の引きの因果 3 平均・ρ = 1)。
各部品の重みは export_rvoc / export_lvl / export_f0est / export_c1 の出力を使う。
    CUDA_VISIBLE_DEVICES= uv run python export_artic_vc.py --out ../results/artic_vc_export
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import c1_content as CC
import diag_pool as P
import f0est as FE
import lvl as LV
import nvoc as N
import train_c3 as C3

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "fixture").mkdir(parents=True, exist_ok=True)
    dev = "cpu"
    pool = P.load_pool(150, 3, dev)
    t = pool[a.target]
    T = t["T"].astype(np.float32)
    enroll = {"spk": t["spk"], "mu": float(t["mu"]), "per_mean": [float(v) for v in t["per_mean"]], "K": int(T.shape[1]), "D": int(T.shape[0])}
    T.astype("<f4").tofile(out / "enroll.bin")
    (out / "enroll.json").write_text(json.dumps(enroll, indent=1))
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).eval(); c1.load_state_dict(st1["net"])
    mfront = CC.MelFront().eval()
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").float()
    sf1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).eval(); f1.load_state_dict(sf1["net"])
    f1front = FE.Front().eval()
    sf2 = torch.load(ROOT / "results/f2_2/last.pt", map_location="cpu", weights_only=False)
    f2 = LV.Lvl(sf2["cfg"]["ch"], tuple(sf2["cfg"]["dils"])).eval(); f2.load_state_dict(sf2["net"])
    G = C3.ConvG(Cb.shape[0]).eval()
    Tt = torch.from_numpy(T)[None] / 10.0
    mu = torch.tensor([enroll["mu"]])
    per_t = torch.tensor([enroll["per_mean"]], dtype=torch.float32)
    summ = torch.cat([Tt.mean(2), mu[:, None], per_t], 1)
    chain = C3.Chain(c1, mfront, Cb, f1, f1front, f2, G, Tt, mu, per_t, summ, dev).eval()
    assert chain.w_sm == 3 and chain.c0_avg == 3 and chain.rho == 1.0
    mrows = [r for r in json.loads((ROOT / "data/f0est_hi/manifest.json").read_text())["rows"] if r["ok"] and r["src"] == "tts_m"]
    from d6_unit_cov import load48
    x = load48(sorted(mrows, key=lambda r: r["wav"])[0]["wav"])[: 6 * N.SR].astype(np.float32)
    n = len(x) // N.HOP
    x = x[: n * N.HOP]
    from train_f0est import infer
    f0s = infer(f1front, f1, x, dev)
    med = float(np.log(np.median(f0s[f0s > 0])))
    with torch.no_grad():
        cond, f0o, dc, dp, p = chain(torch.from_numpy(x)[None], torch.tensor([0]), torch.tensor([med]), True)
    x.astype("<f4").tofile(out / "fixture" / "x.f32")
    np.array([med], np.float32).astype("<f4").tofile(out / "fixture" / "med.f32")
    cond[0].T.contiguous().numpy().astype("<f4").tofile(out / "fixture" / "cond.f32")
    f0o[0].numpy().astype("<f4").tofile(out / "fixture" / "f0o.f32")
    print("enroll", enroll["spk"], "mu", enroll["mu"], "| source frames", n, "| med", med, "| cond", tuple(cond.shape), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
