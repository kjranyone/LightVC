"""登録(参照 → 構音目標表)を Rust(crates/lightvc-core/src/enroll.rs)で行うための定数と一致検査の固定の入出力。
出力 <out>/: weights.bin・manifest.json(mel.fb [128][1025]・dct [25][128]・const)・fixture/(x.f32 = 参照 48kHz・f0.f32 = F1 の f0・env.f32 [T][25] = pae.envelope の
CheapTrick の DCT 包絡(3 点平滑の前)・per.f32 [T][4] = pae.periodicity)。
    CUDA_VISIBLE_DEVICES= uv run python export_enroll.py --out ../results/enroll_export
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import f0est as FE
import nvoc as N
import pae as PA

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "fixture").mkdir(parents=True, exist_ok=True)
    fb, dct = PA._np_consts()
    ten = {"mel.fb": torch.from_numpy(fb).float(), "dct": torch.from_numpy(dct[:25]).float()}
    man = {"tensors": {}, "const": {"NFFT": PA.NFFT, "A": PA.A, "HOP": N.HOP, "SR": N.SR, "C0_SIL": PA.C0_SIL, "HB": [list(b) for b in PA.HB], "SM_W": 0.25}}
    off = 0
    with open(out / "weights.bin", "wb") as f:
        for k in sorted(ten):
            v = ten[k].numpy()
            f.write(v.astype("<f4").tobytes())
            man["tensors"][k] = [off, list(v.shape)]
            off += v.size
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    import s0_artic as S
    from train_f0est import infer
    x = S.held24()[3]["x"].astype(np.float32)[: 4 * N.SR]
    T = len(x) // N.HOP
    x = x[: T * N.HOP]
    sf1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).eval(); f1.load_state_dict(sf1["net"])
    f0 = infer(FE.Front().eval(), f1, x, "cpu")
    f0 = np.pad(f0, (0, max(0, T - len(f0))))[:T].astype(np.float32)
    xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
    env = PA.envelope(xa, f0.astype(np.float64))
    per = PA.periodicity(torch.from_numpy(xa)[None], torch.from_numpy(f0)[None])[0].numpy()
    x.astype("<f4").tofile(out / "fixture" / "x.f32")
    f0.astype("<f4").tofile(out / "fixture" / "f0.f32")
    env.T.copy().astype("<f4").tofile(out / "fixture" / "env.f32")
    per.T.copy().astype("<f4").tofile(out / "fixture" / "per.f32")
    print("exported ->", out, "| T", T, "| voiced", float((f0 > 0).mean()), flush=True)
    return 0


if __name__ == "__main__" and not (len(sys.argv) > 1 and sys.argv[1] == "--full"):
    raise SystemExit(main())


def export_full(out: Path, target: int = 0) -> None:
    """登録の全工程の参照: 評価プールの目標の参照 20s → F1 の f0・C1 の単位(prime)・CheapTrick → 3 点平滑 → conv_c0.table・mu・per_mean。"""
    import c1_content as CC
    import conv_c0 as C0
    import diag_pool as P
    import soundfile as sf
    import train_c1 as T1
    from d6_unit_cov import load48
    from train_f0est import infer
    t = P.load_pool(150, 3, "cpu")[target]
    ws = sorted((ROOT / "female-dataset" / t["spk"]).glob("*.wav"))
    ref, tot = [], 0.0
    for w in ws:
        if tot >= 20.0:
            break
        ref.append(w); tot += sf.info(str(w)).duration
    xr = np.concatenate([load48(w) for w in ref]).astype(np.float32)
    T = len(xr) // N.HOP
    xr = xr[: T * N.HOP]
    sf1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).eval(); f1.load_state_dict(sf1["net"])
    f0 = infer(FE.Front().eval(), f1, xr, "cpu")
    f0 = np.pad(f0, (0, max(0, T - len(f0))))[:T].astype(np.float64)
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).eval(); c1.load_state_dict(st1["net"])
    with torch.no_grad():
        u = c1(CC.MelFront()(torch.from_numpy(T1.prime(xr))[None]))[..., -T:].argmax(1)[0].numpy()
    xa = np.concatenate([np.zeros(PA.A, np.float32), xr, np.zeros(PA.A, np.float32)])
    E = PA.envelope(xa, f0)
    e = np.pad(E, ((0, 0), (1, 1)), mode="edge")
    E = 0.25 * e[:, :-2] + 0.5 * e[:, 1:-1] + 0.25 * e[:, 2:]
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()
    tab = C0.table(E[1:25], u, Cb.shape[0], Cb)
    vo = f0 > 0
    per = PA.periodicity(torch.from_numpy(xa)[None], torch.from_numpy(f0.astype(np.float32))[None])[0].numpy()
    fx = out / "fixture_full"
    fx.mkdir(parents=True, exist_ok=True)
    xr.astype("<f4").tofile(fx / "x.f32")
    tab.astype("<f4").tofile(fx / "table.f32")
    np.array([np.median(np.log(f0[vo]))] + list(per[:, vo].mean(1)), np.float32).astype("<f4").tofile(fx / "mu_per.f32")
    u.astype("<f4").tofile(fx / "unit.f32")
    print("full enroll fixture", t["spk"], "T", T, "| unseen units", int((np.bincount(u, minlength=Cb.shape[0]) == 0).sum()), flush=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--full":
    export_full(Path(sys.argv[2]))
