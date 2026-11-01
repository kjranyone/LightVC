"""D539(推論だけ・D538 の判定の描画): 書き出し(eval_<set>_pre)の有声と log f0 だけを 音量で閾値を変える F1 の判定(c0n_rel ≥ τ で p > 0.5・未満で p > θq・p = Python の F1 = Rust とフレーム一致)に
置き換えて候補を描画する(無声にしたフレームは log f0 = 0)。c0n_rel の基準 = 前置き 12 s(操作者の参照の末尾)の有声の c0n の中央値(製品では登録の定数)。包絡・c0n は元の書き出しのまま(Rust では因果 CT の窓が変わる分だけ違う)。
    uv run python d539_level_render.py --ckpt ../results/diag_ppgvoc43/product_002500_m0.pt --rules L10_80=-1.0,0.8 --out ../results/conv_p0/d539
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
import f0est as FE
import nvoc as N
import train_ppgvoc as P
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent
PRE = 12.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rules", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="devff5")
    a = ap.parse_args()
    dev = "cuda"
    s1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    front = FE.Front().to(dev)
    net = FE.F0Est(ch=s1["cfg"]["ch"], dils=tuple(s1["cfg"]["dils"])).to(dev)
    net.load_state_dict(s1["net"])
    net.eval()
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch") or 192), adapt=bool(st.get("adapt", False)))
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    st["slow_sigma"], st["slow_causal"], st["opslow_tau"], st["opslow_gain"] = 1.0, True, 4.0, 1.5
    P.slow_from_ckpt(st)
    S = P.H / a.set
    ed = P.H / "e2e" / f"eval_{a.set}_pre"
    n0 = int(PRE * N.SR)
    rules = {r.split("=")[0]: tuple(float(v) for v in r.split("=")[1].split(",")) for r in a.rules}
    info = {nm: [] for nm in rules}
    for r in json.loads((S / "devset.json").read_text())["pairs"]:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        if not (ed / f"{k}.cond").exists():
            continue
        c, lg = P.load_dump(ed, k)
        x = load48(S / "d467" / f"{k}.wav").astype(np.float32)
        n = len(x) // N.HOP
        xx = torch.from_numpy(np.concatenate([np.zeros(FE.CTX48, np.float32), x[:n * N.HOP]]))[None].to(dev)
        with torch.no_grad():
            v, _ = net(front(xx, n))
        pv = torch.sigmoid(v[0]).cpu().numpy()[: len(c)]
        assert np.mean((pv > 0.5) == (c[:, 26] > 0.5)) > 0.999, k
        pre = c[:2400]
        ref = float(np.median(pre[pre[:, 26] > 0.5, 0])) if (pre[:, 26] > 0.5).sum() > 50 else float(np.median(c[c[:, 26] > 0.5, 0]))
        spk = P.load_enroll(S / "d85_work" / f"en_{r['ti']:03d}")
        for nm, (tau, tq) in rules.items():
            vv = np.where(c[:, 0] - ref >= tau, pv > 0.5, pv > tq)
            c2 = c.copy()
            c2[:, 26] = vv.astype(np.float32)
            c2[:, 25] = np.where(vv, c[:, 25], 0.0)
            info[nm].append(float(np.mean(vv != (c[:, 26] > 0.5))))
            y = P.render(g, c2, lg, spk, dev, T_op=P.op_tab(S, k))[n0:n0 + 8 * N.SR]
            od = Path(a.out) / nm
            od.mkdir(parents=True, exist_ok=True)
            sf.write(od / f"{k}_{nm}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
    for nm, v in info.items():
        print("rendered", nm, len(v), "voicing changed (mean)", round(float(np.mean(v)), 4), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
