"""D477(本人らしさの誤差の分析の続き・診断): D476 で本人らしさの最大の損失は操作者が別人であること(自己変換 0.18 → 別人の女声から 0.11・ECAPA)。どの通り道で操作者の声が漏れるかを、
学習の外の女声 → 女声(devff2・30 組・目標 = dev2 の 30 人・中心 = dev2/cen_reg)で操作者の条件を 1 つずつ断って測る(推論の診断・発話の統計を使う変種あり)。
版: NONE(P43OP)・PHARD(事後を argmax の one-hot)・PSHARP(事後の温度 0.5)・C0FLAT(c0n を有声の平均の定数に・無音は 0)・F0FLAT(有声の log f0 を平均の定数に)・NOSLOW(遅い揺らぎ 0)。
指標 = 識別の余白(WavLM-SV・ECAPA)の版 − NONE の対の差と、同じ発話の v1⊥ の対の差(自然さの代償)。
    uv run python d477_leak_identity.py --ckpt ../results/diag_ppgvoc43/product_002500_op15.pt --out ../results/conv_p0/d477
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
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P

VARS = ("NONE", "PHARD", "PSHARP", "C0FLAT", "F0FLAT", "NOSLOW")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", default="devff2")
    ap.add_argument("--cen", default="dev2")
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    P.slow_from_ckpt(st)
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    S = P.H / a.set
    ed = P.H / "e2e" / f"eval_{a.set}"
    pairs = json.loads((S / "devset.json").read_text())["pairs"]
    keys = []
    for r in pairs:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        c, lg = P.load_dump(ed, k)
        keys.append(k)
        v = c[:, 26] > 0.5
        for var in VARS:
            cc, ll = c.copy(), lg.copy()
            if var == "PHARD":
                ll = np.where(np.arange(lg.shape[1])[None] == lg.argmax(1)[:, None], 0.0, -30.0).astype(np.float32)
            elif var == "PSHARP":
                ll = lg / 0.5
            elif var == "C0FLAT":
                cc[:, 0] = np.where(c[:, 0] > 1.0, c[v, 0].mean() if v.any() else c[:, 0], c[:, 0])
            elif var == "F0FLAT":
                cc[:, 25] = np.where(v, c[v, 25].mean() if v.any() else 0.0, c[:, 25])
            if var == "NOSLOW":
                on = P.OPSLOW["on"], P.SLOW["on"]
                P.OPSLOW["on"], P.SLOW["on"] = False, False
                y = P.render(g, cc, ll, P.load_enroll(S / "d85_work" / f"en_{r['ti']:03d}"), dev)
                P.OPSLOW["on"], P.SLOW["on"] = on
            else:
                y = P.render(g, cc, ll, P.load_enroll(S / "d85_work" / f"en_{r['ti']:03d}"), dev, T_op=P.op_tab(S, k) if P.OPSLOW["on"] else None)
            (O / var).mkdir(parents=True, exist_ok=True)
            sf.write(O / var / f"{k}_{var}.wav", y, N.SR)
    print("rendered", len(keys), flush=True)
    from transformers import WavLMForXVector
    sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()
    import idloss as ID
    ec = ID.Ecapa(dev)

    @torch.no_grad()
    def emb(p: Path) -> dict:
        x, sr = sf.read(str(p), dtype="float32", always_2d=True)
        x = x.mean(1)[: 8 * sr]
        w = torch.nn.functional.normalize(sv(input_values=torch.from_numpy(resample_poly(x, 16000, sr).astype(np.float32))[None].to(dev)).embeddings, dim=-1)[0].cpu().numpy()
        e = ec(torch.from_numpy(x)[None].to(dev))[0].cpu().numpy()
        return {"wsv": w, "ecapa": e / np.linalg.norm(e)}

    meta = json.loads((P.H / a.cen / "cen_reg/meta.json").read_text())
    C = {m["ti"]: emb(P.H / a.cen / "cen_reg" / f"cen_{m['ti']:03d}.wav") for m in meta if m["ok"]}
    mg = {}
    for var in VARS:
        for k in keys:
            ti = int(k[7:10])
            if ti not in C:
                continue
            e = emb(O / var / f"{k}_{var}.wav")
            mg.setdefault(var, {})[k] = {m: float(e[m] @ C[ti][m]) - float(np.mean([e[m] @ C[t][m] for t in C if t != ti])) for m in ("wsv", "ecapa")}
    rng = np.random.default_rng(477)
    rep = {"margin_mean": {v: {m: round(float(np.mean([x[m] for x in d.values()])), 4) for m in ("wsv", "ecapa")} for v, d in mg.items()}, "diff_vs_NONE": {}}
    ks = sorted(mg["NONE"])
    tg = np.array([int(k[7:10]) for k in ks])
    us = sorted(set(tg.tolist()))
    idx = {u: np.flatnonzero(tg == u) for u in us}
    for var in VARS[1:]:
        rep["diff_vs_NONE"][var] = {}
        for m in ("wsv", "ecapa"):
            d = np.array([mg[var][k][m] - mg["NONE"][k][m] for k in ks])
            bs = [d[np.concatenate([idx[u] for u in rng.choice(us, len(us))])].mean() for _ in range(2000)]
            rep["diff_vs_NONE"][var][m] = [round(float(d.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
    print(json.dumps(rep), flush=True)
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "d467_paired_v1.py"), "--dir", str(O), "--ref", "NONE", "--out", str(O / "paired.json")], capture_output=True, text=True)
    print(r.stdout[-1500:], flush=True)
    (O / "identity.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
