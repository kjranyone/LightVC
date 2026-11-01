"""D451b(利得と時定数の掃引・D451 と同じ置き場に足す): 遅い音色の揺らぎの源を乱数から操作者自身の遅い揺らぎ(操作者の包絡 − 操作者の表の引き の 0.5 s 平均 − 平均 → 基底へ射影)に替えると、
長い窓(8 s)の自然さが局所(0.5 s)を損ねずに写し合成へ近づくか(診断・中心の窓の平均 = 非因果の oracle・推論に使うなら因果な版を別に設計)。
女声 → 女声(devff の部分集合 0・30 本)と 男声 → 女声(dev の目標ごとに 1 本・30 本)で、OFF / 乱数(非因果 σ 1.0)/ 操作者 γ 0.5・1.0 を描画 → d286_window。
    uv run python d450_opslow.py --ckpt ../results/diag_ppgvoc43/snap/ema_002500.pt --out ../results/conv_p0/d450
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
import train_ppgvoc as P

VARS = ("OPC4G20", "OPC4G25", "OPC8G20")


def slow_causal(r: torch.Tensor, tau_s: float) -> torch.Tensor:
    """因果な版: 過去 0.5 s(100 フレーム)の平均 − 時定数 tau_s の指数移動平均(長い平均 = 区間の平均の代わり・発話の統計なし)→ 基底へ射影。r [1, 24, T]。"""
    import torch.nn.functional as F
    s = F.avg_pool1d(F.pad(r, (99, 0), mode="replicate"), 100, 1)
    a = float(np.exp(-1.0 / (tau_s * 200.0)))
    m = torch.empty_like(s)
    acc = s[..., 0]
    for t in range(s.shape[-1]):
        acc = a * acc + (1 - a) * s[..., t]
        m[..., t] = acc
    B = P.SLOW["B"].to(r.device)
    return torch.einsum("dk,nkt->ndt", B, torch.einsum("dk,ndt->nkt", B, s - m))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="+", default=["devff", "dev"])
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    st["slow_sigma"], st["slow_causal"] = 1.0, False
    P.slow_from_ckpt(st)
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    for s in a.sets:
        seen, items = set(), []
        for f in sorted((P.H / "e2e" / f"eval_{s}").glob("p00*.cond")):
            ti = int(f.stem.split("_t")[1][:3])
            if ti not in seen:
                seen.add(ti)
                items.append((f.stem, int(f.stem[1:5]), ti))
        for v in VARS:
            (O / s / v).mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for k, pi, ti in items:
                c, lg = P.load_dump(P.H / "e2e" / f"eval_{s}", k)
                spk = P.load_enroll(P.H / s / "d85_work" / f"en_{ti:03d}")
                T_op = P.load_enroll(P.H / s / "d85_work" / f"op_{pi:04d}")[0]
                p = P.ppg_form(torch.from_numpy(lg.T.copy()).float().to(dev)[None], P.PPG)
                ct = torch.from_numpy(c.T.copy()).float().to(dev)[None]
                stt = P.spk_tensor(spk, dev)
                base_op = torch.einsum("dk,bkt->bdt", torch.from_numpy(T_op).float().to(dev), p) / 10
                zo = P.slow_real(ct[:, 1:25], base_op)
                zc = {tau: slow_causal(ct[:, 1:25] - base_op, tau) for tau in (4.0, 8.0)}
                f0 = torch.where(ct[:, 26] > 0.5, 200.0 * torch.exp(ct[:, 25]), torch.zeros_like(ct[:, 25]))
                exc = P.TR.excitation(f0, ct.shape[-1] * N.HOP, torch.Generator().manual_seed(0))
                for v in VARS:
                    if v == "OFF":
                        P.SLOW["on"] = False
                        cond = g.make(p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], stt)
                        P.SLOW["on"] = True
                    elif v == "RND10":
                        cond = g.make(p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], stt)
                    elif v == "OP10":
                        cond = g.make(p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], stt, zo)
                    else:
                        z_ = zc[8.0 if v.startswith("OPC8") else 4.0] * float(v[-2:]) / 10
                        cond = g.make(p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], stt, z_)
                    y = g(cond, exc, stt)[0].cpu().numpy()[N.DELAY:]
                    sf.write(O / s / v / f"{k}_{v}.wav", y / (np.abs(y).max() + 1e-9) * 0.9, N.SR)
        print("rendered", s, len(items), flush=True)
        r = subprocess.run([sys.executable, str(Path(__file__).parent / "d286_window.py"), "--eq_dir", str(O / s), "--out", str(O / f"{s}_window.json")] + (["--perp"] if s.startswith("dev") and s != "devff" else []), capture_output=True, text=True)
        print(s, r.stdout[-1500:], r.stderr[-800:] if r.returncode else "", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
