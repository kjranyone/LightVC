"""D443: 女声 → 女声(devff・実女声の入力)の本人らしさ(D394 と同じ識別の余白)。devff の目標は dev と同じ 30 人 = 中心は dev/cen_reg(声域をそろえた独立の中心)。
群: SRC(入力そのもの = 別人の女声・下限の目安)・REF(目標本人の登録の参照の先頭 8 s = 上限の目安・中心とは別の録音)・KNN・PROD・学習の評価の FF の書き出し(--ff NAME=<ev の dir>)。
同じ組(pPPPP_tTTT)どうしの対の差を目標でまとめたブートストラップ。
    uv run python d443_ffid.py --ff B=../results/diag_ppgvoc27/ev_002500 D2=../results/diag_ppgvoc41/ev_002500 --vs KNN PROD --out ../results/conv_p0/d443.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))
H = Path("/home/kojirotanaka/lightvc_scratch_20261013")


def load16(p, sec: float = 0.0) -> np.ndarray:
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    x = x.mean(1)
    if sec:
        x = x[: int(sec * sr)]
    return resample_poly(x, 16000, sr).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ff", nargs="*", default=[])
    ap.add_argument("--vs", nargs="*", default=["KNN", "PROD"])
    ap.add_argument("--d444", nargs="*", default=[], help="d444_ffgate の書き出し(<devff>/d444/NAME/p*_NAME.wav・devff の 90 組の全部)")
    a = ap.parse_args()
    dev = "cuda"
    from transformers import WavLMForXVector
    sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()
    import idloss as ID
    ec = ID.Ecapa(dev)

    @torch.no_grad()
    def emb(p, sec: float = 0.0) -> dict:
        x16 = load16(p, sec)
        w = torch.nn.functional.normalize(sv(input_values=torch.from_numpy(x16)[None].to(dev)).embeddings, dim=-1)[0].cpu().numpy()
        x48, sr = sf.read(str(p), dtype="float32", always_2d=True)
        x48 = x48.mean(1)[: int(sec * sr)] if sec else x48.mean(1)
        e = ec(torch.from_numpy(x48)[None].to(dev))[0].cpu().numpy()
        return {"wsv": w, "ecapa": e / np.linalg.norm(e)}

    meta = json.loads((H / "dev/cen_reg/meta.json").read_text())
    ok = sorted(m["ti"] for m in meta if m["ok"])
    C = {ti: emb(H / "dev/cen_reg" / f"cen_{ti:03d}.wav") for ti in ok}
    files: dict = {"SRC": {Path(f).name[:10]: f for f in glob.glob(str(H / "devff/d56/p*_SRC.wav"))},
                   "KNN": {Path(f).name[:10]: f for f in glob.glob(str(H / "devff/d135_real/p*_KNN.wav"))},
                   "PROD": {Path(f).name[:10]: f for f in glob.glob(str(H / "devff/d114/p*_G03S10ZH.wav"))}}
    for e in a.ff:
        nm, d = e.split("=", 1)
        files[nm] = {f"p{Path(f).name[2:6]}_{Path(f).name[7:11]}": f for f in glob.glob(f"{d}/f*_FF.wav")}
    for nm in a.d444:
        files[nm] = {Path(f).name[:10]: f for f in glob.glob(str(H / "devff/d444" / nm / f"p*_{nm}.wav"))}
    own = [e.split("=", 1)[0] for e in a.ff] + a.d444
    keys = sorted(set.intersection(*[set(files[k]) for k in own])) if own else sorted(files["SRC"])
    per: dict = {}

    def margin(e: dict, ti: int) -> dict:
        out = {}
        for m in ("wsv", "ecapa"):
            cs = {t: float(e[m] @ C[t][m]) for t in C}
            out[m] = cs[ti] - np.mean([v for t, v in cs.items() if t != ti])
        return out

    for k in keys:
        ti = int(k[7:10])
        if ti not in C:
            continue
        for g, d in files.items():
            if k in d:
                per.setdefault(g, {})[k] = margin(emb(d[k]), ti)
        per.setdefault("REF", {})[k] = margin(emb(H / "dev/d37/_work" / f"ref_{ti:03d}.wav", 8.0), ti)
    rep: dict = {"n_keys": len(keys), "margin": {g: {m: round(float(np.mean([v[m] for v in d.values()])), 4) for m in ("wsv", "ecapa")} | {"n": len(d)} for g, d in per.items()}, "paired": {}}
    rng = np.random.default_rng(0)
    for g in own:
        for b in a.vs + [x for x in own if x != g]:
            ks = sorted(set(per[g]) & set(per[b]))
            tg = np.array([k[7:10] for k in ks])
            out = {"n": len(ks)}
            for m in ("wsv", "ecapa"):
                dd = np.array([per[g][k][m] - per[b][k][m] for k in ks])
                us = sorted(set(tg))
                idx = {u: np.flatnonzero(tg == u) for u in us}
                bs = [dd[np.concatenate([idx[u] for u in rng.choice(us, len(us))])].mean() for _ in range(2000)]
                out[m] = {"diff": round(float(dd.mean()), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}
            rep["paired"][f"{g}-{b}"] = out
    print(json.dumps(rep["margin"]), flush=True)
    for k, v in rep["paired"].items():
        print(k, v, flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
