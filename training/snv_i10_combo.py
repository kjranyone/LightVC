"""段 I10(paper_snv rev21)の組み合わせ(学習なし): G・軌跡・型 = 段 I8(snv_i8_m13)・C = snv_i10_c_pr(M15)・推論の E = snv_i10_e_rec(M16)を 1 つの保存に組む。
M13 の役割の分け方では E と C は互いの出力で学ばない(C は軌跡で・E は軌跡への回帰で)ので、別々に学び直した重みをそのまま組める。凍結した部品が 3 つの保存で同じことを確かめる。
出力 = results/snv_i10_combo/{last.pt, split.json}。
    cd training && uv run --no-sync python snv_i10_combo.py [e=results/snv_i10_e_rec/last.pt] [c=results/snv_i10_c_pr/last.pt] [out=results/snv_i10_combo](段 I12 = e=results/snv_i12_e_vr2/last.pt out=results/snv_i12_combo)
E の保存の e_lift と lift_mean(paper_snv §5 M19)・e_play と cpc と pca(§5 M28)も組み合わせに写す。
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent


def same(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    pe, pc = ROOT / opt.get("e", "results/snv_i10_e_rec/last.pt"), ROOT / opt.get("c", "results/snv_i10_c_pr/last.pt")
    base = torch.load(ROOT / "results/snv_i8_m13/last.pt", map_location="cpu", weights_only=False)
    e, c = torch.load(pe, map_location="cpu", weights_only=False), torch.load(pc, map_location="cpu", weights_only=False)
    assert e["e_only"] and e["e_rec"] and c["c_only"], "腕の種類が違う"
    assert same(e["Gs"], base["Gs"]) and same(c["Gs"], base["Gs"]), "G が段 I8 と違う"
    assert same(e["C"], base["C"]), "e_rec の C が段 I8 と違う"
    assert same(c["Es"], base["Es"]), "c_pr の推論の E が段 I8 と違う"
    assert torch.equal(e["tpl"], base["tpl"]) and torch.equal(c["tpl"], base["tpl"]), "型が段 I8 と違う"
    ck = dict(c)
    ck["Es"] = e["Es"]
    ck["e_lift"], ck["lift_mean"] = e.get("e_lift", 0), e.get("lift_mean")
    ck["e_play"], ck["cpc"], ck["pca"] = e.get("e_play", ""), e.get("cpc"), e.get("pca")
    ck["combo_e"], ck["combo_c"], ck["step"] = str(pe.relative_to(ROOT)), str(pc.relative_to(ROOT)), min(e["step"], c["step"])
    out = ROOT / opt.get("out", "results/snv_i10_combo")
    out.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out / "last.pt")
    shutil.copy(pc.parent / "split.json", out / "split.json")
    print("combo", ck["combo_e"], ck["combo_c"], "step", ck["step"], "e_lift", ck["e_lift"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
