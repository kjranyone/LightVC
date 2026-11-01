"""D367: 単位のトークンごとの実現のばらつき(同じ単位でも発話のたびに違う包絡のずれ)を、内容にそろった乱数で与える(学習なし)。
統計: e2e/fem の実女声(保留を除く・表のある話者): 残差 r = 包絡 − Σp T / 10 の速い成分(r − 0.5 s 平均)を argmax の単位の連続(3 フレーム以上)ごとに平均 = トークンのずれ → 主成分 8 本と成分の標準偏差。
推論: argmax が切り替わったら新しいトークン = 成分ごとに N(0, 標準偏差²) を引いて保持・一極の低域通過(τ = 15 ms)で滑らかに・基底へ戻して σ_t 倍を包絡の行へ(遅い揺らぎの後)。乱数 = 共通の乱数(splitmix64・種 = 遅い揺らぎの種 + 長さ + 31337)。因果・先読み 0・発話の統計なし。
    uv run python d367_tokennoise.py --ckpt ../results/diag_ppgvoc24/product_s15_007500.pt --out ../results/conv_p0/d367
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P


def token_stats(exclude: set, k: int = 8) -> tuple:
    E2 = P.H / "e2e"
    TD = P.ROOT / "data/rvoc_tables_real"
    V = []
    for m in json.loads((E2 / "meta.json").read_text())["fem"]:
        f = E2 / "fem" / f"{m['key']}.cond"
        if m["spk"] in exclude or not f.exists() or not (TD / f"{m['spk']}.npz").exists():
            continue
        c, lg = P.load_dump(E2 / "fem", m["key"])
        p = torch.from_numpy(lg.T.copy()).float().softmax(0)
        r = torch.from_numpy(c[:, 1:25].T.copy()).float() - torch.from_numpy(np.load(TD / f"{m['spk']}.npz")["T"]).float() @ p / 10
        fast = (r - F.avg_pool1d(F.pad(r[None], (50, 49), mode="replicate"), 100, 1)[0]).numpy()
        a = p.argmax(0).numpy()
        b = np.r_[0, np.flatnonzero(np.diff(a)) + 1, len(a)]
        for s, e in zip(b[:-1], b[1:]):
            if e - s >= 3:
                V.append(fast[:, s:e].mean(1))
    V = np.array(V)
    U, S, _ = np.linalg.svd(V.T @ V / len(V))
    B = U[:, :k]
    return B, np.sqrt((V @ B).var(0)), {"tokens": len(V), "explained": float(S[:k].sum() / S.sum()), "rms": float(np.sqrt((V ** 2).mean()))}


def token_noise(argmax: np.ndarray, B: np.ndarray, sd: np.ndarray, seed: int, sigma: float) -> np.ndarray:
    T, k = len(argmax), B.shape[1]
    nrm = P.splitmix_normals(seed, T * k + (k % 2)).reshape(-1)[: T * k].reshape(T, k)
    u, cur, out = np.zeros(k), np.zeros(k), np.zeros((T, k))
    al = 1.0 - np.exp(-1.0 / 3.0)
    for t in range(T):
        if t == 0 or argmax[t] != argmax[t - 1]:
            u = nrm[t] * sd
        cur = cur + al * (u - cur)
        out[t] = cur
    return (sigma * (B @ out.T)).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.5, 1.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--sets", nargs="+", default=["sr", "devff", "devff2"])
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    held = P.e2e_held()
    B, sd, info = token_stats({h[0] for h in held})
    print("token stats", json.dumps(info), flush=True)
    np.savez(O / "token_stats.npz", B=B, sd=sd)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    P.slow_from_ckpt(st)
    g = P.init_gen(dev, "table")
    g.load_state_dict(st["ema"])
    g.eval()
    its = {s_: (P.eval_items(held, "table", dev)[0] if s_ == "sr" else P.ff_items([s_])) for s_ in a.sets}
    groups: dict = {s_: [] for s_ in a.sets}
    with torch.no_grad():
        for sg in a.sigmas:
            for sdd in a.seeds:
                P.SLOW["seed"] = sdd
                tag = f"K{sg}_r{sdd}"
                for s_ in a.sets:
                    od = O / s_ / tag
                    od.mkdir(parents=True, exist_ok=True)
                    for nm, c, lg, spk in its[s_]:
                        cond, exc, s2 = P.cond_exc(g, c, lg, spk, dev)
                        T = cond.shape[-1]
                        tz = token_noise(lg.argmax(1)[:T], B, sd, sdd + T + 31337, sg)
                        cond = torch.cat([cond[:, :1], cond[:, 1:25] + torch.from_numpy(tz).to(dev)[None], cond[:, 25:]], 1)
                        y = g(cond, exc, s2)[0].cpu().numpy()[N.DELAY:]
                        sf.write(od / f"{nm}.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
                    groups[s_].append(f"{tag}={od}/*.wav")
                print("rendered", tag, flush=True)
    rows = {}
    for s_ in a.sets:
        r = subprocess.run([sys.executable, str(Path(__file__).parent / "eval_v1perp.py"), "--groups", *groups[s_], "--out", str(O / f"{s_}_v1perp.json")], capture_output=True, text=True)
        if r.returncode:
            print(r.stderr[-1500:], flush=True)
        j = json.loads((O / f"{s_}_v1perp.json").read_text())
        rows[s_] = {k_: v["v1perp"]["logit"] for k_, v in j.items()}
    (O / "summary.json").write_text(json.dumps(rows, indent=1))
    for sg in a.sigmas:
        m = [np.mean([v for k_, v in rows[s_].items() if k_.startswith(f"K{sg}_")]) for s_ in a.sets]
        print(f"token σ {sg}:", " ".join(f"{s_} {x:.3f}" for s_, x in zip(a.sets, m)), f"| 3-set {np.mean(m):.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
