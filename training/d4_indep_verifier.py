"""D4: 学習に使っていない物差し(凍結 SSL の層平均埋め込み)で、日本語 21 目標の変換音声の同一性を測り直す(推論のみ)。
物差し = microsoft/wavlm-base-plus と wavlm-large の層ごとのフレーム平均(話者検証の学習はしていない)・候補平均を引いてコサイン。
較正: 目標の実音声(cen)の前半を中心・後半を問いにして、実音声の本人 top-1(物差しが話者を識別できるか)。
    uv run python d4_indep_verifier.py --wav_dir <save_dir> --out ../results/conv_p0/d4.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from scipy.stats import wilcoxon

os.environ.setdefault("HF_HUB_OFFLINE", "1")


def load16(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(1)
    return resample_poly(x, 16000, sr).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", nargs="*", default=["microsoft/wavlm-base-plus", "microsoft/wavlm-large"])
    a = ap.parse_args()
    from transformers import WavLMModel
    d = Path(a.wav_dir)
    cens = sorted(d.glob("cen_*.wav"))
    S = len(cens)
    pw = sorted(d.glob("p*_t*_*.wav"))
    meta = []
    for f in pw:
        m = re.match(r"p(\d+)_t(\d+)_(.+)\.wav", f.name)
        meta.append((int(m.group(1)), int(m.group(2)), m.group(3), f))
    names = sorted({m[2] for m in meta})
    dev = "cuda"
    rep: dict = {"n_targets": S, "conditions": names}
    for mname in a.models:
        net = WavLMModel.from_pretrained(mname).to(dev).eval().float()

        @torch.no_grad()
        def emb(x: np.ndarray) -> np.ndarray:
            x = (x - x.mean()) / (x.std() + 1e-7)
            hs = net(torch.from_numpy(x)[None].to(dev), output_hidden_states=True).hidden_states
            return torch.stack([h[0].mean(0) for h in hs]).cpu().numpy()

        cen_x = [load16(f) for f in cens]
        half = [(x[: len(x) // 2], x[len(x) // 2:]) for x in cen_x]
        Ea = np.stack([emb(h[0]) for h in half])
        Eb = np.stack([emb(h[1]) for h in half])
        Ec = np.stack([emb(x) for x in cen_x])
        Eo = {f.name: emb(load16(f)) for _, _, _, f in meta}
        out: dict = {}
        nl = Ec.shape[1]
        for layer in range(1, nl):
            def prep(E, ref):
                mu = ref[:, layer].mean(0, keepdims=True)
                v = E[..., layer, :] - mu
                return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
            ca, cb = prep(Ea, Ea), prep(Eb, Ea)
            cal = float(((cb @ ca.T).argmax(1) == np.arange(S)).mean())
            C = prep(Ec, Ec)
            res = {}
            for nm in names:
                rk, tg = [], []
                for pi, ti, n, f in meta:
                    if n != nm:
                        continue
                    v = prep(Eo[f.name][None], Ec)[0]
                    se = C @ v
                    rk.append(int((se > se[ti]).sum()) + 1)
                    tg.append(ti)
                res[nm] = (np.array(rk), np.array(tg))
            out[layer] = {"calib_real_top1": cal, "res": res}
        best = [l for l in out if out[l]["calib_real_top1"] >= 0.9]
        use = best if best else sorted(out, key=lambda l: -out[l]["calib_real_top1"])[:3]
        rep[mname] = {"calib_real_top1_by_layer": {str(l): round(out[l]["calib_real_top1"], 3) for l in out}, "layers_used": [int(l) for l in use]}
        for l in use:
            r = out[l]["res"]
            blk = {}
            for nm in names:
                rk, tg = r[nm]
                blk[nm] = {"top1": round(float((rk == 1).mean()), 3), "mean_rank": round(float(rk.mean()), 2)}
                if nm not in ("SRC", "TAB") and "SHUF" not in nm:
                    rb, _ = r["TAB"]
                    ma = np.array([rk[tg == t].mean() for t in range(S)])
                    mb = np.array([rb[tg == t].mean() for t in range(S)])
                    blk[nm]["cluster_p_improve_vs_TAB"] = round(float(wilcoxon(ma, mb, alternative="less").pvalue), 4)
                    blk[nm]["targets_better/worse"] = [int((ma < mb).sum()), int((ma > mb).sum())]
            rep[mname][f"layer{l}"] = blk
        del net
        torch.cuda.empty_cache()
        print(mname, json.dumps(rep[mname], ensure_ascii=False)[:2000], flush=True)
    Path(a.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
