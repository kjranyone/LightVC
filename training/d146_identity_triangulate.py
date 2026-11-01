"""D146(学習なし): 同一性の物差しの三角測量。D140〜D145: 学習外 LDA(WavLM 層 4/6/8 の平均 + 標準偏差 → LDA)では『動きを足すと同一性が下がる』(表だけ 0.33 > 動きあり 0.12〜0.22)。
これがこの物差し固有か、同一性一般かを、独立の話者照合器で同じ組を測り直して確かめる(耳では較正しない = オーナー 10-13『解析で続ける』)。
照合器: ECAPA(speechbrain spkrec-ecapa-voxceleb)・WavLM-SV(microsoft/wavlm-base-plus-sv の x-vector)。目標の中心 = <dir>/cen_TTT.wav(d4b と同じ mkcen = 登録の参照の中央 8s)・
変換 = <dir>/pNNNN_tNNN_<cond>.wav を 30 目標の中心との cos で順位づけ(本人の順位)。top-1 / top-5 / 平均順位。本物の較正 = 中心の前半と後半(d4b と同じ)。
注意: diag_rr3 / rr4 は ECAPA を学習の批評家に使った = R の条件(RR*)の ECAPA の値は独立でない。出力 = results/conv_p0/<tag>.json。
    uv run python d146_identity_triangulate.py --dir <dev>/lda145 --tag d146dev
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import idloss as ID
    from d6_unit_cov import load48
    dev = a.device
    d = Path(a.dir)
    vers = {"ECAPA": ID.Ecapa(dev), "WAVLM_SV": ID.WavlmSV(dev)}
    cens = sorted(d.glob("cen_*.wav"), key=lambda f: int(re.match(r"cen_(\d+)", f.name).group(1)))
    meta = []
    for f in sorted(d.glob("p*_t*_*.wav")):
        m = re.match(r"p(\d+)_t(\d+)_(.+)\.wav", f.name)
        meta.append((int(m.group(2)), m.group(3), f))
    rep = {}
    with torch.no_grad():
        for vn, ver in vers.items():
            emb = lambda x: ver(torch.from_numpy(x.astype(np.float32))[None].to(dev))[0].cpu().numpy()
            cx = [load48(f).astype(np.float32) for f in cens]
            C = np.stack([emb(x) for x in cx])
            Ca = np.stack([emb(x[: len(x) // 2]) for x in cx]); Cb = np.stack([emb(x[len(x) // 2:]) for x in cx])
            nz = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
            cal = float(((nz(Cb) @ nz(Ca).T).argmax(1) == np.arange(len(cx))).mean())
            Cn = nz(C)
            R: dict = {}
            It: dict = {}
            for ti, c, f in meta:
                s = Cn @ nz(emb(load48(f).astype(np.float32)))
                R.setdefault(c, []).append(int((s > s[ti]).sum()) + 1)
                It.setdefault(c, []).append(f.name[: -len(c) - 5])
            rep[vn] = {"calib_real_top1": round(cal, 3)} | {c: {"top1": round(float(np.mean(np.array(r) == 1)), 3), "top5": round(float(np.mean(np.array(r) <= 5)), 3),
                                                              "mean_rank": round(float(np.mean(r)), 2), "n": len(r)} for c, r in sorted(R.items())}
            print(vn, json.dumps(rep[vn]), flush=True)
            for c in R:
                rep[vn][c] |= {"items": It[c], "ranks": R[c]}
    (ROOT / f"results/conv_p0/{a.tag}.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
