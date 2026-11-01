"""D44(診断・学習なし): 製品経路の写し合成の明瞭さの損失(D43 の男声の写し 0.13)を部品に分ける。元 = D37 の男声 TTS の 60 文(高さは男声のまま)。出力部 = rvoc3hi(Python)。
  ORA = CheapTrick の包絡 + 実の c0 + 教師 f0 + 実の周期性(時変)= 出力部だけの損失・PC = ORA の周期性を有声フレームの平均の定数に(製品と同じ扱い)・
  F3 = 包絡だけ F3(因果)・F1 = f0 だけ F1(因果)・F2 = c0 だけ F2(因果・3 平均)・PROD = F3 + F2 + F1 + 定数の周期性(= 製品の写し合成)。
CER は d23_cer.py(基準 = SRC の文字起こし)。
    uv run python d44_copy_decomp.py --src_dir <scratchpad>/d37 --out_dir <scratchpad>/d44
"""
from __future__ import annotations

import argparse
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
    ap.add_argument("--src_dir", required=True)
    ap.add_argument("--out_dir", required=True)
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
    dev = "cuda"
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    sr_ = torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=sr_["cfg"]["ch"], kernels=tuple(sr_["cfg"]["kernels"]), dils=tuple(sr_["cfg"]["dils"]), d_cond=sr_["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(sr_["ema"]); gen.eval().requires_grad_(False)
    sm_w = TR.ckpt_env_smooth(sr_, None)
    sf1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev); f1.load_state_dict(sf1["net"]); f1.eval()
    f1front = FE.Front().to(dev)
    sf2 = torch.load(ROOT / "results/f2_2/last.pt", map_location="cpu", weights_only=False)
    f2 = LV.Lvl(sf2["cfg"]["ch"], tuple(sf2["cfg"]["dils"])).to(dev); f2.load_state_dict(sf2["net"]); f2.eval()
    sf3 = torch.load(ROOT / "results/f3_1/last.pt", map_location="cpu", weights_only=False)
    f3 = LV.EnvEst(sf3["cfg"]["ch"], tuple(sf3["cfg"]["dils"])).to(dev); f3.load_state_dict(sf3["net"]); f3.eval()
    mfront = CC.MelFront().to(dev)

    @torch.no_grad()
    def render(c0, env24, f0, per):
        cc = np.concatenate([c0[None] - PA.C0_SIL, env24], 0)
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)[None]
        v = (f0 > 0).astype(np.float32)[None]
        cond = torch.from_numpy(np.concatenate([cc / 10, lf, v, per * v], 0).astype(np.float32))[None].to(dev)
        fo = torch.from_numpy(f0.astype(np.float32))[None].to(dev)
        y = gen(cond, TR.excitation(fo, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)

    for f in sorted(Path(a.src_dir).glob("p*_SRC.wav")):
        base = f.name[: -len("_SRC.wav")]
        x, sr = sf.read(f, dtype="float32")
        assert sr == N.SR
        n = len(x) // N.HOP
        x = x[: n * N.HOP]
        xt = torch.from_numpy(x)[None].to(dev)
        f0t, _ = H.teacher_f0(x, n)
        f0t = f0t[:n]
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        e = np.pad(PA.envelope(xa, f0t), ((0, 0), (1, 1)), mode="edge")
        ct = (sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:])[:, :n].astype(np.float64)
        per = PA.periodicity(torch.from_numpy(xa)[None], torch.from_numpy(f0t)[None])[0].numpy()[:, :n]
        vo = f0t > 0
        pc = np.repeat(per[:, vo].mean(1, keepdims=True) if vo.any() else np.zeros((4, 1)), n, 1)
        with torch.no_grad():
            v_, p_ = f1(f1front(torch.cat([torch.zeros(1, FE.CTX48, device=dev), xt], 1), n))
            f0f = FE.decode(v_, p_, 0.5)[0].cpu().numpy()[:n].astype(np.float64)
            c0n = C3.causal_avg(f2(mfront(torch.cat([torch.zeros(1, N.WIN - N.HOP, device=dev), xt], 1)))[:, None], 3)[0, 0].cpu().numpy()[:n]
            e3 = f3(mfront(torch.cat([torch.zeros(1, N.WIN - N.HOP, device=dev), xt], 1)))[0].cpu().numpy()
        c0f = c0n.astype(np.float64) * 10 + PA.C0_SIL
        env3 = (e3[1:25] * 10.0).astype(np.float64)[:, :n]
        conds = {"ORA": (ct[0], ct[1:25], f0t, per), "PC": (ct[0], ct[1:25], f0t, pc), "F3": (ct[0], env3, f0t, per),
                 "F1": (ct[0], ct[1:25], f0f, per), "F2": (c0f, ct[1:25], f0t, per), "PROD": (c0f, env3, f0f, pc)}
        for c, args in conds.items():
            sf.write(od / f"{base}_{c}.wav", render(*args), N.SR)
        shutil.copy(f, od / f.name)
        print(base, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
