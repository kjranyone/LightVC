"""A4 の記録(事前登録の record・判定なし): 評価の出力の NM(N0r の男 → 目標 6 声)・NJ(JNV の M → 目標 6 声)に N2b の点(FAIL の物差し・記録だけ)を付ける。
近傍の基準 = N2 の iro_F の清音(目標の声と同じ A0 の女の N0r)。出力 = results/diag_snv_a4/n2b_record.json。
    cd .. && ../aes_env/bin/python training/snv_diag/a4_n2b_record.py <dump> aes
    cd training && uv run --no-sync python snv_diag/a4_n2b_record.py <dump>
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
BS = 16
PARTS = ("NM", "NJ")


def aes(dump: Path) -> None:
    from audiobox_aesthetics.infer import initialize_predictor
    pred = initialize_predictor()
    res = {}
    for p in PARTS:
        d = np.load(dump / f"{p}.npz")
        ks = d.files
        for i in range(0, len(ks), BS):
            rows = pred.forward([{"path": torch.from_numpy(d[k]).float()[None], "sample_rate": 16000} for k in ks[i: i + BS]])
            for k, r in zip(ks[i: i + BS], rows):
                res[k] = r
        print(p, len(ks), flush=True)
    (dump / "aes.json").write_text(json.dumps(res))


def main() -> int:
    dump = Path(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] == "aes":
        aes(dump)
        return 0
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from transformers import WavLMModel
    from snv_a0_measure import DEV, SR
    from snv_a4_eval import cluster_ci
    ruler = json.loads((ROOT / "results/snv_n2b/ruler.json").read_text())
    its = json.loads((ROOT / "results/snv_n2/items.json").read_text())
    bank_all = np.load(ROOT / "results/snv_n2b/knn_bank_clean.npz")
    bank = torch.from_numpy(np.stack([bank_all[f"clean|{it['name']}"] for it in its if it["pop"] == "iro_F"])).to(DEV)
    av = json.loads((dump / "aes.json").read_text())
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(DEV).eval()
    wl = WavLMModel.from_pretrained(ruler["knn"]["model"]).to(DEV).eval()
    K = ruler["knn"]["k"]
    m, s, w = ruler["standardize"]["mean"], ruler["standardize"]["std"], ruler["w"]
    score: dict = {}
    for p in PARTS:
        d = np.load(dump / f"{p}.npz")
        for k in d.files:
            x = torch.from_numpy(d[k]).float()[None].to(DEV)
            with torch.no_grad():
                mos = float(utmos(x, SR).item())
                xn = (x - x.mean()) / (x.std() + 1e-7)
                h = wl(xn, output_hidden_states=True).hidden_states[ruler["knn"]["layer"]][0]
                e = torch.nn.functional.normalize(torch.cat([h.mean(0), h.std(0)]), dim=0)
                kn = float(torch.sort(bank @ e).values[-K:].mean())
            f = {"utmos": mos, "aes_pq": av[k]["PQ"], "aes_ce": av[k]["CE"], "aes_cu": av[k]["CU"], "aes_pc": av[k]["PC"], "knn": kn}
            score[k] = {"n2b": float(sum(w[n] * (f[n] - m[n]) / s[n] for n in w))} | f
        print(p, len(d.files), flush=True)
    by = defaultdict(dict)
    for k, v in score.items():
        sysp, src, tgt = k.split("|")
        part, sname = sysp.split("_")
        by[(part, sname)][f"{src}|{tgt}"] = v
    res: dict = {"note": "記録だけ(N2b は FAIL の物差し・判定に使わない)", "bank": "results/snv_n2b/knn_bank_clean.npz の iro_F 336 本", "agg": {}, "diff": {}}
    for (part, sname), d in sorted(by.items()):
        res["agg"][f"{part}|{sname}"] = {f: round(float(np.mean([v[f] for v in d.values()])), 4) for f in ("n2b", "utmos", "aes_pq", "aes_ce", "knn")} | {"n": len(d)}
    for part in PARTS:
        for a_, b_ in (("c2", "x2"), ("c1", "x1"), ("c2", "c1"), ("x2", "x1")):
            ks = sorted(set(by[(part, a_)]) & set(by[(part, b_)]))
            dd = np.array([by[(part, a_)][k]["n2b"] - by[(part, b_)][k]["n2b"] for k in ks])
            res["diff"][f"{part}|{a_}_minus_{b_}"] = {"mean": round(float(dd.mean()), 3), "ci95_target_cluster": cluster_ci(dd, np.array([k.split("|")[1] for k in ks]), 2000)}
    (ROOT / "results/diag_snv_a4/n2b_record.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
