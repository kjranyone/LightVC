"""配布物の前段の門と重みの網(results/snv_p45_gate_t80・train_snv_sep / train_snv_mask の族)の学習の入力側に混ぜた、本物の非音声の音源の帰属の一覧を作る(段 P94・NOTICE から参照)。
台帳 = data/snv_p_feat_rec2/sources.json(MUSAN の音楽と雑音)・data/snv_sep/fma_sources.json(FMA)・data/snv_sep/operator_sources.json(MUSAN の手や物の音)・data/snv_sep/fsd50k_dev_sources.json(FSD50K の開発の組)。
学習の族ごとの使い分けより広く(台帳の全ての音源を)挙げる。CC BY の音源は 1 本ずつ、CC0 / パブリックドメインは数だけ。
    cd training && uv run --no-sync python make_attribution.py [out=../ATTRIBUTION_SOURCES.md]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    out = Path(opt.get("out", ROOT / "ATTRIBUTION_SOURCES.md"))
    J = lambda p: json.loads((ROOT / p).read_text())
    mu, fm, op, fs = J("data/snv_p_feat_rec2/sources.json"), J("data/snv_sep/fma_sources.json"), J("data/snv_sep/operator_sources.json"), J("data/snv_sep/fsd50k_dev_sources.json")
    one = lambda s: " ".join(str(s).split()).replace("|", "/")
    L = ["# Attribution of non-speech sources used in training", "",
         "The gate and mask networks of the front end (models/front_gate) were trained on mixtures of generated speech with real non-speech audio",
         "(music, noise, hand and object sounds). Only sources under CC0, Public Domain or CC BY (without NC / ND / SA) were used, and clips with",
         "detected speech were removed (AudioSet classifier MIT/ast-finetuned-audioset-10-10-0.4593, threshold 0.02). No source audio is distributed.",
         "This list covers every source in the selection ledgers (a superset of what each training run drew from).", ""]
    mby = sorted((k, v) for k, v in mu["files"].items() if v["lic"] == "by")
    L += ["## MUSAN (D. Snyder, G. Chen, D. Povey, 2015; openslr.org/17)", "",
          f"Music and noise: {len(mu['files'])} files ({sum(v['lic'] == 'pd' for v in mu['files'].values())} public domain / CC0, {len(mby)} CC BY).",
          "Full credits for each file are in the LICENSE files of the MUSAN corpus. CC BY files used:", "", "| file | license / credit (as recorded in MUSAN) |", "|---|---|"]
    L += [f"| {k} | {one(v['text'])} |" for k, v in mby]
    oby = sorted((k, v) for k, v in op.get("files", {}).items() if v.get("lic") == "by")
    L += ["", f"Hand and object sounds from MUSAN noise: {len(op.get('files', {}))} files ({len(oby)} CC BY; credits in MUSAN's LICENSE files).", ""]
    L += [f"- {k}" for k, _ in oby]
    fby = sorted((k, v) for k, v in fm["files"].items() if v["lic"] == "by")
    L += ["", "## Free Music Archive (fma_small; M. Defferrard et al., 2017)", "",
          f"Instrumental tracks: {len(fm['files'])} ({sum(v['lic'] == 'pd' for v in fm['files'].values())} public domain / CC0, {len(fby)} CC BY).", "",
          "| track id | title | artist | album | license |", "|---|---|---|---|---|"]
    L += [f"| {k} | {one(v['title'])} | {one(v['artist'])} | {one(v.get('album', ''))} | {one(v['text'])} |" for k, v in fby]
    sby = sorted((f for f in fs["files"] if f["lic"] == "by"), key=lambda f: int(f["fname"]))
    L += ["", "## FSD50K (development split; E. Fonseca et al., 2022; Zenodo 4060432)", "",
          f"Hand and object sounds: {len(fs['files'])} clips ({sum(f['lic'] == 'cc0' for f in fs['files'])} CC0, {len(sby)} CC BY). Each clip is a Freesound sound: https://freesound.org/s/<id>/", "",
          "| freesound id | title | uploader | license |", "|---|---|---|---|"]
    L += [f"| {f['fname']} | {one(f['title'])} | {one(f['uploader'])} | CC BY |" for f in sby]
    out.write_text("\n".join(L) + "\n")
    print("written", out, "lines", len(L), "CC BY", len(mby) + len(oby) + len(fby) + len(sby), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
