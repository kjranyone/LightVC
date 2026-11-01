"""D337: 表(単位ごとの平均)からの包絡の残差のうち、因果な文脈(事後・c0n・log f0・有声・表の包絡)から予測できる分はどれだけか(検査 I・D334 の続き)。
因果な文脈の符号器(CtxEnc・96 ch・受容野 約 1.27 秒)を本物の残差に L2 で合わせる(最適 = 文脈の下の条件付き平均 = 予測できる遅い揺れ)。
学習 = e2e/fem の実女声(保留の 30 人を除く・表 data/rvoc_tables_real のある話者)・評価 = 保留の SR クリップ(Rust の登録の表)で
遅い残差(0.5 s の移動平均)の説明率と、予測を ppgvoc18 の包絡の行に β 倍で足した SR の v1⊥。診断(生成器は学習し直さない)。
    uv run python d337_resprobe.py --ckpt ../results/diag_ppgvoc18/last_ema.pt --out ../results/conv_p0/d337
"""
from __future__ import annotations

import argparse
import json
import random
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


def slow(x: torch.Tensor) -> torch.Tensor:
    return F.avg_pool1d(F.pad(x, (50, 49), mode="replicate"), 100, 1)


def feats(c: np.ndarray, lg: np.ndarray, T: np.ndarray, dev: str) -> tuple:
    p = torch.from_numpy(lg.T.copy()).float().to(dev)[None].softmax(1)
    ct = torch.from_numpy(c.T.copy()).float().to(dev)[None]
    base = torch.einsum("dk,bkt->bdt", torch.from_numpy(T).float().to(dev), p) / 10
    x = torch.cat([p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], base], 1)
    return x, ct[:, 1:25] - base


def r2(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    return float(1 - ((tgt - pred) ** 2).sum() / ((tgt - tgt.mean(-1, keepdim=True)) ** 2).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--ch", type=int, default=96)
    ap.add_argument("--band", default="raw", choices=["raw", "mid"], help="予測の対象: raw = 残差そのもの(D337)・mid = 中間の帯(0.15 s 平均 − 0.5 s 平均・D366)")
    a = ap.parse_args()
    if a.band == "mid":
        def tgt_of(y):
            return F.avg_pool1d(F.pad(y, (15, 14), mode="replicate"), 30, 1) - F.avg_pool1d(F.pad(y, (50, 49), mode="replicate"), 100, 1)
    else:
        def tgt_of(y):
            return y
    dev = "cuda"
    torch.manual_seed(337); random.seed(337)
    O = Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    E2 = P.H / "e2e"
    held = P.e2e_held()
    hs = {h[0] for h in held}
    TD = P.ROOT / "data/rvoc_tables_real"
    tr = []
    for m in json.loads((E2 / "meta.json").read_text())["fem"]:
        f = E2 / "fem" / f"{m['key']}.cond"
        if m["spk"] in hs or not f.exists() or not (TD / f"{m['spk']}.npz").exists():
            continue
        c, lg = P.load_dump(E2 / "fem", m["key"])
        if len(c) >= 900:
            tr.append((c, lg, np.load(TD / f"{m['spk']}.npz")["T"], m["spk"]))
    print("train clips", len(tr), "speakers", len({t[3] for t in tr}), flush=True)
    enc = P.CtxEnc(a.ch).to(dev)
    torch.nn.init.normal_(enc.out.weight, std=1e-3)
    opt = torch.optim.AdamW(enc.parameters(), lr=3e-4)
    for step in range(1, a.steps + 1):
        xs, ys = [], []
        for _ in range(16):
            c, lg, T, _ = random.choice(tr)
            o = random.randrange(0, len(c) - 800)
            x, y = feats(c[o:o + 800], lg[o:o + 800], T, dev)
            xs.append(x); ys.append(tgt_of(y))
        x, y = torch.cat(xs), torch.cat(ys)
        d = enc(x)
        loss = F.mse_loss(d[..., 255:], y[..., 255:])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 500 == 0:
            print("step", step, "mse", round(float(loss), 4), "var", round(float(y[..., 255:].var()), 4), flush=True)
    enc.eval()
    sr, _ = P.eval_items(held, "table", dev)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX["on"], P.CTX["ch"], P.CTX["adapt"] = bool(st.get("ctx", False)), int(st.get("ctx_ch", 192)), bool(st.get("adapt", False))
    P.slow_from_ckpt(st)
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    rs, rr, rg = [], [], []
    betas = (0.5, 1.0, 2.0)
    groups = []
    with torch.no_grad():
        for nm, c, lg, spk in sr:
            x, y = feats(c, lg, spk[0], dev)
            d = enc(x)
            rs.append(r2(d, slow(y)) if a.band == "raw" else float(1 - ((tgt_of(y) - d) ** 2).sum() / (tgt_of(y) ** 2).sum()))
            rr.append(r2(d, y))
            cond0, _, _ = P.cond_exc(g, c, lg, spk, dev)
            rg.append(r2(cond0[:, 1:25] - x[:, -24:], slow(y)))
        for b in betas:
            od = O / f"B{b}"
            od.mkdir(exist_ok=True)
            for nm, c, lg, spk in sr:
                x, _ = feats(c, lg, spk[0], dev)
                cond, exc, s_ = P.cond_exc(g, c, lg, spk, dev)
                cond = torch.cat([cond[:, :1], cond[:, 1:25] + b * enc(x), cond[:, 25:]], 1)
                yy = g(cond, exc, s_)[0].cpu().numpy()[N.DELAY:]
                sf.write(od / f"{nm}.wav", (yy / (np.abs(yy).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
            groups.append(f"B{b}={od}/h*_SR.wav")
    res = {"r2_slow": float(np.mean(rs)), "r2_raw": float(np.mean(rr)), "r2_slow_ppgvoc18_ctx": float(np.mean(rg))}
    print("held SR:", json.dumps(res), flush=True)
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "eval_v1perp.py"), "--groups", *groups, "--out", str(O / "v1perp.json")], capture_output=True, text=True)
    print(r.stdout[-2500:], r.stderr[-1500:] if r.returncode else "", flush=True)
    torch.save(enc.state_dict(), O / "ctx_probe.pt")
    (O / "r2.json").write_text(json.dumps(res, indent=1))
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
