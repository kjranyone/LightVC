"""A4 の判定: results/diag_snv_a4/a4_eval.json に事前登録(results/diag_snv_a4/prereg.yaml の success・fork)の規則をそのまま当てる。
出力 = results/diag_snv_a4/judge.json(verdict.yaml はこれを写して書く)。
    cd training && uv run --no-sync python snv_a4_judge.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/diag_snv_a4"
ID_TOL = 0.05
CER_TOL = 0.05


def side(d: dict) -> str:
    lo, hi = d["ci95_voice_cluster"]
    if d["mean"] < 0 and hi < 0:
        return "below"
    if d["mean"] > 0 and lo > 0:
        return "above"
    return "within"


def main() -> int:
    r = json.loads((OUT / "a4_eval.json").read_text())
    A, B = r["A"], r["B"]
    guard = {s: round(A[s + "2"]["M_cer_median_8tgt"] - A[s + "1"]["M_cer_median_8tgt"], 3) for s in "cx"}
    withheld = [s for s in "cx" if guard[s] > CER_TOL]
    j: dict = {"source": "results/diag_snv_a4/a4_eval.json", "prereg": "results/diag_snv_a4/prereg.yaml",
               "guard_cer_stage2_minus_stage1": guard, "stage2_withheld": withheld}
    for st in ("1", "2"):
        j[f"identity_stage{st}"] = {p: {"c": A["c" + st][p], "x": A["x" + st][p], "ok": A["c" + st][p] >= A["x" + st][p] - ID_TOL} for p in ("S", "M")}
    j["H_id_ok"] = None if withheld else all(v["ok"] for v in j["identity_stage2"].values())
    ext = side(B["tex_c2_minus_x2"])
    j["extrapolation"] = {"tex_c2_minus_x2": B["tex_c2_minus_x2"], "tex_c1_minus_x1_record": B["tex_c1_minus_x1"], "side": ext}
    j["H_ext"] = None if withheld else {"below": "H_circuit_ext", "above": "H_table_ext", "within": "H_equal_ext"}[ext]
    g = {s: side(B[f"tex_{s}2_minus_{s}1"]) for s in "cx"}
    j["texture"] = {s: {"tex": B[f"tex_{s}2_minus_{s}1"], "side": g[s], "hf_ratio_stage1": B[s + "1"]["hf_ratio_median"],
                        "hf_ratio_stage2": B[s + "2"]["hf_ratio_median"]} for s in "cx"}
    j["H_gan_ok"] = None if withheld else all(v == "below" for v in g.values())
    if withheld:
        j["fork"] = "guard: 段 2 の判定を出さない系がある → 段 1 の記録と CER の悪化の型を報告"
    elif not j["H_id_ok"]:
        j["fork"] = "not id_ok"
    elif j["H_ext"] == "H_table_ext":
        j["fork"] = "id_ok + H_table_ext"
    elif j["H_gan_ok"]:
        j["fork"] = "id_ok + (H_circuit_ext か H_equal_ext) + H_gan_ok"
    else:
        j["fork"] = "id_ok + (H_circuit_ext か H_equal_ext) + not H_gan_ok"
    (OUT / "judge.json").write_text(json.dumps(j, ensure_ascii=False, indent=1))
    print(json.dumps(j, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
