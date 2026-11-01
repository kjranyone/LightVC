"""D292: 男声の操作者のずれ(包絡 − Σp T_操作者)を周波数軸で α 倍に伸ばしてから目標の表に足す(男声の声道は長い = 動きが低い周波数で起きる・D283 の線形の色づけは周波数の位置を直さない)。
ずれの DCT c1..c24 → log-mel 128 帯(c0 = 0)→ 帯の中心周波数(Hz・mel のフィルタの重心)で f/α の値を線形補間 → DCT に戻す。α 1 は恒等(assert)。
PV = dev の 60 組(操作者の表 = d85_work/op_PPPP)・学習なし。
    uv run python d292_warpres.py --ckpt <last_ema.pt> --out <dir> --variants g070a100,g070a115,g070a130,g035a130
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import pae as PA
import train_ppgvoc as P


def warp_mat(alpha: float) -> np.ndarray:
    """24 次の DCT のずれ → 伸ばしたずれ の線形写像 [24, 24](右から掛ける: r' = r @ M)。"""
    fb, dct = PA._np_consts()
    D = dct[1:25]
    fr = np.fft.rfftfreq((fb.shape[1] - 1) * 2, 1 / N.SR)
    cen = (fb * fr[None]).sum(1) / (fb.sum(1) + 1e-12)
    W = np.zeros((len(cen), len(cen)))
    for k, f in enumerate(cen):
        src = np.clip(f / alpha, cen[0], cen[-1])
        j = int(np.clip(np.searchsorted(cen, src) - 1, 0, len(cen) - 2))
        w = (src - cen[j]) / (cen[j + 1] - cen[j])
        W[k, j], W[k, j + 1] = 1 - w, w
    return (D @ W.T @ D.T).astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="g070a100,g070a115,g070a130,g035a130")
    a = ap.parse_args()
    M1 = warp_mat(1.0)
    assert np.abs(M1 - np.eye(24)).max() < 1e-3, ("α 1 の恒等", np.abs(M1 - np.eye(24)).max())
    dev = "cuda"
    O = Path(a.out)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    print("ckpt step", st["step"], "dyn_g", st.get("dyn_g"), flush=True)
    _, pv_items = P.eval_items(P.e2e_held())
    op_tab = {}
    groups = []
    for v in a.variants.split(","):
        gm, al = int(v[1:4]) / 100, int(v[5:]) / 100
        Mw = warp_mat(al)
        od = O / v
        od.mkdir(parents=True, exist_ok=True)
        for nm, c, lg, spk in pv_items:
            pi = int(nm[1:5])
            if pi not in op_tab:
                op_tab[pi] = P.load_enroll(P.H / "dev" / "d85_work" / f"op_{pi:04d}")[0]
            with torch.no_grad():
                cond, exc, stt = P.cond_exc(g, c, lg, spk, dev)
                p = P.ppg_form(torch.from_numpy(lg.T.copy()).float()[None], P.PPG)[0].numpy().T
                res = (c[:, 1:25] - p @ op_tab[pi].T / 10) @ Mw
                cond[0, 1:25] += torch.from_numpy((gm * res).T.astype(np.float32)).to(dev)
                y = g(cond, exc, stt)[0].cpu().numpy()[N.DELAY:]
            sf.write(od / f"{nm}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
        groups.append(f"{v}={od}/p*_PV.wav")
        print("rendered", v, flush=True)
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "eval_v1perp.py"), "--groups", *groups, "--out", str(O / "v1perp.json")], capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-1500:] if r.returncode else "", flush=True)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
