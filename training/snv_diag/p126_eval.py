from __future__ import annotations

import os

os.environ.setdefault("SNV_WIN_CAP_MS", "7.5")
os.environ.setdefault("SNV_W_YIN", "600")
os.environ.setdefault("SNV_TAU_UV", "500")
os.environ.setdefault("SNV_ENVC", "1")
os.environ.setdefault("SNV_ESM", "1")
os.environ.setdefault("SNV_WF16", "1")
os.environ.setdefault("SNV_KV16", "1")

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import p125_eval as E5

ROOT = E5.ROOT
MEM_P125 = ("snv_p125a_c", "snv_p125b_c", "snv_p125e_c")
SYS_P126 = "ens:" + "|".join(f"{t}+mask:snv_p126_gate_i80+fb:anch" for t in MEM_P125)


def main() -> None:
    out_dir = ROOT / "results/snv_p126"
    out_dir.mkdir(parents=True, exist_ok=True)
    rck = "results/snv_p123_s/snap/ema_20k.pt"
    res: dict = {"p120": {}, "probe4": {}}
    prev = ROOT / "results/snv_p125/eval_probes.json"
    if prev.exists():
        pr = json.loads(prev.read_text())
        for k in ("B0_oldA", "P125_anch"):
            if k in pr.get("p120", {}):
                res["p120"][k] = pr["p120"][k]
            if k in pr.get("probe4", {}):
                res["probe4"][k] = pr["probe4"][k]
    r = E5.run_p120_arm("P126_inst_anch", rck, SYS_P126)
    res["p120"]["P126_inst_anch"] = r
    print("P120 P126_inst_anch PESQ", {k: r["pesq"][k]["mean"] for k in r["pesq"]}, "d2_ratio_max", r["d2_sm_ratio_max"], flush=True)
    p = E5.run_probe4_arm("P126_inst_anch", rck, SYS_P126, out_dir / "probe")
    res["probe4"]["P126_inst_anch"] = p
    print("PROBE4 P126_inst_anch", json.dumps(p, ensure_ascii=False), flush=True)
    (out_dir / "eval_probes.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
