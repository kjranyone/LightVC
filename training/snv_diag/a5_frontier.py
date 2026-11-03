"""A5 の途中信号の停止の後の記録(学習なし): A4 の c1 の登録の当てはめの回数を替えて(Φ を小さく)開発の組 E1V の音素の無い発声の TEX と普通の発話の top-1 を測り、
A5 の a5_1(学習と推論で同じ 8 歩)と x1 を同じ平面に置く。問い = A5 の質感の利得は「Φ を小さくした」だけで c1 にも出るか(本人らしさとの引き合いの上にあるか)。
出力 = results/diag_snv_a5/frontier.json。dev1.json の a5_1・c1(400 回)・x1 と同じ項目・同じ物差し。
    cd training && uv run --no-sync python snv_diag/a5_frontier.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snv_a0_measure import DEV, ROOT, load, to16
from snv_a2_eval import vocoder
from snv_a4b_fit import ENROLL_T, HELD_T, N_E1
from snv_n0_measure import trim
from snv_n2c_tex import tex_stats
from train_snv_a2 import FEAT, playing
from train_snv_a4 import C, fit_phi

OUT = ROOT / "results/diag_snv_a5"
STEPS = (10, 25, 50, 100)


def main() -> int:
    scale = np.load(ROOT / "results/snv_n2c/tex_scale.npy")
    tag = "diag_snv_a4_c"
    ck = torch.load(ROOT / "results" / tag / "last.pt", map_location="cpu", weights_only=True)
    net = C().to(DEV).eval()
    net.load_state_dict(ck["g"])
    for p in net.parameters():
        p.requires_grad_(False)
    nz = np.load(ROOT / "results" / tag / "norm.npz")
    mu, sd = nz["mu"], nz["sd"]
    names_tr = json.loads((ROOT / "results" / tag / "train_log.json").read_text())["names_train"]
    man = sorted(v["voice"] for v in json.loads((ROOT / "data/snv_e1/manifest.json").read_text())["voices"])
    e1v = [v for v in man if v not in names_tr][-N_E1:]
    nvman = [it for it in json.loads((ROOT / "data/snv_e1nv/manifest.json").read_text())["items"] if it["k"] == 0 and it["voice"] in e1v]
    voc = vocoder()
    from speechbrain.inference.speaker import EncoderClassifier
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})

    def emb(y16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(y16).float()[None].to(DEV)).reshape(-1)
        return torch.nn.functional.normalize(e, dim=0).cpu().numpy()

    enroll, items = {}, []
    for v in e1v:
        ms = [np.load(FEAT / f"mel/e1/{v}/{t}.npy").astype(np.float32) for t in ENROLL_T]
        ps = [playing(FEAT / f"sparc/e1/{v}/{t}.npz", m.shape[1]) for t, m in zip(ENROLL_T, ms)]
        enroll[v] = (torch.from_numpy((np.concatenate(ps, 0) - mu) / sd)[None].to(DEV), torch.from_numpy(np.concatenate(ms, 1))[None].to(DEV),
                     emb(np.concatenate([to16(*load(ROOT / f"data/snv_e1/{v}/{t}.wav")) for t in ENROLL_T])))
        items += [("normal", v, FEAT / f"mel/e1/{v}/{t}.npy", FEAT / f"sparc/e1/{v}/{t}.npz", None) for t in HELD_T]
    for it in nvman:
        n = f"{it['cat']}_0"
        items.append(("nv", it["voice"], FEAT / f"mel/e1nv/{it['voice']}/{n}.npy", FEAT / f"sparc/e1nv/{it['voice']}/{n}.npz",
                      tex_stats(to16(*load(ROOT / f"data/snv_e1nv/{it['voice']}/{n}.wav")))))
    dev1 = json.loads((OUT / "dev1.json").read_text())["sys"]
    res: dict = {"note": "記録だけ・A5 の途中信号の停止の後・A4 の c1 の当てはめの回数だけを替える", "ref_dev1": {s: {k: dev1[s][k] for k in ("nv_tex", "normal_top1", "phi_rms")} for s in dev1},
                 "c1_by_fit_steps": {}}
    for st in STEPS:
        phis, rms = {}, []
        for v, (rp, rmel, _) in enroll.items():
            phis[v] = fit_phi(net, rp, rmel, torch.ones(1, rmel.shape[2], dtype=torch.bool, device=DEV), steps=st)
            rms.append(float(phis[v].pow(2).mean().sqrt()))
        top, txs = [], []
        for kind, v, mp, spp, tst in items:
            p = torch.from_numpy((playing(spp, np.load(mp).shape[1]) - mu) / sd)[None].to(DEV)
            with torch.no_grad():
                w = voc(net(p, phis[v])).squeeze().cpu().numpy()
            if kind == "nv":
                txs.append(float(np.mean(np.abs(tex_stats(to16(w, 44100)) - tst) / scale)))
            else:
                e = emb(trim(to16(w, 44100)))
                top.append(max(e1v, key=lambda u: float(e @ enroll[u][2])) == v)
        res["c1_by_fit_steps"][st] = {"nv_tex": round(float(np.mean(txs)), 4), "normal_top1": round(float(np.mean(top)), 3), "phi_rms": round(float(np.median(rms)), 4)}
        print(st, res["c1_by_fit_steps"][st], flush=True)
    (OUT / "frontier.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
