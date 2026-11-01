"""話者の守り(E-C1r・D703 の事前登録の守りの量): 男声の行の覆いで出力が『本物らしいが男声』に寄っていないかを、v1(本物 / 合成)と別に測る。
WavLM-SV(microsoft/wavlm-base-plus-sv)の x-vector で、目標の余白 = cos(出力, 自分の目標の参照 d37/_work/ref_TTT.wav)− 同じ組の他の目標の参照との cos の平均、
操作者への近さ = cos(出力, 入力 d56/<key>_SRC.wav)。腕と基準の対の差の中央値(条件ごと)。参照は表の登録と同じ録音 = 絶対値は水増し・対の差だけを読む。
    uv run python spk_guard.py --base N67 --arm N71 --out ../results/diag_ppgvoc71/spk_guard.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
COND = [("uiro_k12", "d487_uiro", None, "RUST12"), ("uiro_k5", "d487_uiro", None, "RUST5Q3"), ("vctkME_k12", "devvctkm", "ME", "RUST12"), ("vctkME_k5", "devvctkm", "ME", "RUST5Q3"),
        ("devff2_k0", "devff2", None, "RUST0")]


def load16(p: Path) -> np.ndarray:
    x, sr = sf.read(str(p), dtype="float32", always_2d=True)
    return resample_poly(x.mean(1), 16000, sr).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from transformers import WavLMForXVector
    sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to("cuda").eval()

    @torch.no_grad()
    def emb(p: Path) -> np.ndarray:
        x = torch.from_numpy(load16(p))[None].to("cuda")
        return torch.nn.functional.normalize(sv(input_values=x).embeddings, dim=-1)[0].cpu().numpy()

    rep: dict = {"embedding": "wavlm-base-plus-sv", "base": a.base, "arm": a.arm}
    for nm, s, kd, d in COND:
        P = [r for r in json.loads((H / s / "devset.json").read_text())["pairs"] if kd is None or r["kind"] == kd]
        tis = sorted({r["ti"] for r in P})
        R = {ti: emb(H / s / "d37" / "_work" / f"ref_{ti:03d}.wav") for ti in tis}
        out: dict = {}
        for sx in (a.base, a.arm, "Y58"):
            mg, op = [], []
            for r in P:
                k = f"p{r['pi']:04d}_t{r['ti']:03d}"
                e = emb(H / s / "gate" / f"{d}{sx}" / f"{k}_{d}{sx}.wav")
                own = float(e @ R[r["ti"]])
                oth = [float(e @ R[t]) for t in tis if t != r["ti"]]
                mg.append(own - (float(np.mean(oth)) if oth else 0.0))
                op.append(float(e @ emb(H / s / "d56" / f"{k}_SRC.wav")))
            out[sx] = (np.array(mg), np.array(op))
        b, x = out[a.base], out[a.arm]
        rep[nm] = {"n": len(P), "margin_median": {sx: round(float(np.median(v[0])), 4) for sx, v in out.items()},
                   "op_sim_median": {sx: round(float(np.median(v[1])), 4) for sx, v in out.items()},
                   "margin_paired_diff_median": round(float(np.median(x[0] - b[0])), 4), "op_sim_paired_diff_median": round(float(np.median(x[1] - b[1])), 4)}
        print(nm, rep[nm], flush=True)
    male = [n for n, *_ in COND[:4]]
    rep["guard_ok"] = all(rep[n]["margin_paired_diff_median"] >= -0.01 and rep[n]["op_sim_paired_diff_median"] <= 0.02 for n in male)
    print(json.dumps({"guard_ok": rep["guard_ok"]}), flush=True)
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
