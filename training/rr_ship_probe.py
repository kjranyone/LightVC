"""R の出荷ゲート 1(未来不変性)を学習済みの重みと実音声の特徴で測る: 開発用の組の男声 TTS の特徴の t 以降を同じ発話の時間反転で書き換え、
t より前の包絡・c0 の出力が変わらないこと(max|差| = 0)と t 以降が動くこと(感度)を、記憶あり R_A・記憶なし R_∅(--null)・差 R_A − R_∅ で確かめる。
    uv run python rr_ship_probe.py --s <scratchpad>/dev --tag diag_rr15c --out ../results/diag_rr16c/ship_probe.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import train_rr as RRm

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--key", default="p0000_t000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rand_layers", type=int, default=0, help="学習前の台帳: 重みを読まず build(layers = N) の乱数の初期値(出力の最後の層も乱数にして感度を持たせる)で測る")
    a = ap.parse_args()
    import conv_c0 as C0
    S = Path(a.s)
    FR = S / "rr_real"
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()

    def ld(corpus: str, stem: str) -> dict:
        man = {it["stem"]: it for it in json.loads((FR / corpus / "manifest.json").read_text())["items"]}
        d = torch.load(FR / man[stem]["pt"], weights_only=False)
        return {"p": RRm.dense(d["pk"], d["pv"]), "u": d["pk"][:, 0].long().numpy(), "e3": d["e3"].float().numpy(), "c0f": d["c0f"].numpy(), "f1": d["f1"].numpy(),
                "env": d["env"].float().numpy(), "ft": d["ft"].numpy(), "p4": d["p4"].float().numpy()}

    if a.rand_layers:
        torch.manual_seed(0)
        net = RRm.build({"nq": RRm.K + 24 + 3, "ch": 192, "heads": 4, "dk": 32, "qd": [1, 2, 4, 8, 16, 32], "kd": [1, 2, 4], "layers": a.rand_layers}).eval()
        torch.nn.init.normal_(net.dout.c.weight, std=0.05)
    else:
        ck = torch.load(ROOT / "results" / a.tag / "last.pt", map_location="cpu", weights_only=False)
        net = RRm.build(ck["cfg"]).eval()
        net.load_state_dict(ck["net"])
    pi, ti = int(a.key[1:5]), int(a.key[7:10])
    r = ld("real_ref", f"ref_{ti:03d}")
    T = C0.table(r["env"][1:25], r["u"], RRm.K, Cb).astype(np.float32)
    n = min(r["env"].shape[1], 5000)
    rc0 = (r["env"][0, :n] - RRm.C0_SIL) / 10
    rp = r["p"][:, :n].astype(np.float32)
    rv = np.concatenate([r["env"][1:25, :n] / 10, (rc0 - rc0.mean())[None]], 0).astype(np.float32)
    b = ld("d56_src", f"{a.key}_SRC"); o = ld("d56_op", f"op_{pi:04d}")
    vo = o["f1"] > 0
    sM = {"e3": o["e3"][:, vo].mean(1), "c0": float(o["c0f"][vo].mean()), "lf": float(np.median(np.log(o["f1"][vo])))}
    vM = b["f1"] > 0
    q = np.concatenate([b["p"], (b["e3"] - sM["e3"][:, None]) / 10, (b["c0f"] - sM["c0"])[None],
                        np.where(vM, np.log(np.maximum(b["f1"], 1.0)) - sM["lf"], 0.0)[None], vM[None].astype(np.float32)], 0).astype(np.float32)
    tab = (T @ b["p"]).astype(np.float32)
    L = q.shape[1]
    with torch.no_grad():
        mem = net.memory(torch.from_numpy(rp)[None], torch.from_numpy(rv)[None])
        rm = torch.ones(1, rp.shape[1], dtype=torch.bool)

        def outs(qq: np.ndarray, tt: np.ndarray) -> dict:
            qt, tb = torch.from_numpy(qq)[None], torch.from_numpy(tt)[None]
            eA, dA, _ = net(qt, tb, mem, rm)
            eN, dN, _ = net(qt, tb, mem, rm, torch.ones(1, dtype=torch.bool))
            return {"A": (eA[0].numpy(), dA[0].numpy()), "N": (eN[0].numpy(), dN[0].numpy()), "H": ((eA - eN)[0].numpy(), (dA - dN)[0].numpy())}

        base = outs(q, tab)
        rep: dict = {"tag": a.tag, "key": a.key, "frames": L, "probe": "t 以降を同じ発話の時間反転で書き換え(問い・表とも)", "points": {}}
        ok = True
        for t in [int(L * f) for f in (0.15, 0.35, 0.55, 0.8)]:
            q2, tab2 = q.copy(), tab.copy()
            q2[:, t:] = q[:, ::-1][:, : L - t]; tab2[:, t:] = tab[:, ::-1][:, : L - t]
            o2 = outs(q2, tab2)
            pt = {}
            for k in ("A", "N", "H"):
                past = max(float(np.abs(base[k][0][..., :t] - o2[k][0][..., :t]).max()), float(np.abs(base[k][1][..., :t] - o2[k][1][..., :t]).max()))
                fut = float(np.abs(base[k][0][..., t:] - o2[k][0][..., t:]).max())
                pt[k] = {"past_max": past, "future_max": round(fut, 4)}
                ok = ok and past == 0.0 and fut > 1e-3
            rep["points"][str(t)] = pt
    rep["verdict"] = "PASS" if ok else "FAIL_OR_INCONCLUSIVE"
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
