from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

BAD_TAGS = ["metallic", "muffled", "rough", "sibilant", "source_leak", "weak_vc",
            "breath_dead", "whisper_broken", "tiring", "latency_feel", "uncanny"]

DEFAULT_SELECTORS = {
    "persona": ["", "imouto", "onee", "kanojo", "asmr_narrator"],
    "scene": ["", "whisper_close", "normal", "emotional", "live"],
    "relation": ["", "public", "friendly", "intimate"],
    "preset": ["", "intimate_soft", "bright_energetic", "neutral"],
}


def _abspath(p) -> str:
    return str(Path(p).resolve())


def build_gate0(export_dir, a_key: str, b_key: str, ref_key: str) -> list:
    d = Path(export_dir)
    groups = defaultdict(dict)
    for w in sorted(d.glob("*.wav")):
        stem = w.stem
        key = stem.rsplit("_", 1)[-1]
        group = stem[: -(len(key) + 1)]
        groups[group][key] = w
    pairs = []
    for group, files in groups.items():
        if a_key not in files or b_key not in files:
            continue
        pair = {
            "pair_id": group,
            "reference": _abspath(files[ref_key]) if ref_key in files else None,
            "cand_a": {"path": _abspath(files[a_key]), "label": a_key,
                       "checkpoint": _ckpt_for(a_key), "controls": {}},
            "cand_b": {"path": _abspath(files[b_key]), "label": b_key,
                       "checkpoint": _ckpt_for(b_key), "controls": {}},
        }
        pairs.append(pair)
    return pairs


def _ckpt_for(key: str) -> str:
    return {"base": "dac_44khz", "finetuned": "dac_44khz_finetuned",
            "ceiling": "dac_44khz(no-rvq)", "orig": "reference"}.get(key, key)


def build_dirs(a_dir, b_dir, ref_dir, a_label: str, b_label: str) -> list:
    a_dir, b_dir = Path(a_dir), Path(b_dir)
    ref_dir = Path(ref_dir) if ref_dir else None
    pairs = []
    for wa in sorted(a_dir.glob("*.wav")):
        wb = b_dir / wa.name
        if not wb.exists():
            continue
        ref = None
        if ref_dir and (ref_dir / wa.name).exists():
            ref = _abspath(ref_dir / wa.name)
        pairs.append({
            "pair_id": wa.stem,
            "reference": ref,
            "cand_a": {"path": _abspath(wa), "label": a_label, "checkpoint": a_label, "controls": {}},
            "cand_b": {"path": _abspath(wb), "label": b_label, "checkpoint": b_label, "controls": {}},
        })
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    pg = sub.add_parser("gate0")
    pg.add_argument("--dir", required=True)
    pg.add_argument("--a", default="base")
    pg.add_argument("--b", default="finetuned")
    pg.add_argument("--ref", default="orig")

    pd = sub.add_parser("dirs")
    pd.add_argument("--a-dir", required=True)
    pd.add_argument("--b-dir", required=True)
    pd.add_argument("--ref-dir", default=None)
    pd.add_argument("--a-label", default="A")
    pd.add_argument("--b-label", default="B")

    for p in (pg, pd):
        p.add_argument("--out", required=True)
        p.add_argument("--session", default="ab_session")
        p.add_argument("--prompt", default="どちらが良いか（滑らか・近い・疲れない・source leakなし）")

    args = ap.parse_args()

    if args.mode == "gate0":
        pairs = build_gate0(args.dir, args.a, args.b, args.ref)
    else:
        pairs = build_dirs(args.a_dir, args.b_dir, args.ref_dir, args.a_label, args.b_label)

    session = {
        "session": args.session,
        "task_prompt": args.prompt,
        "bad_tags": BAD_TAGS,
        "selectors": DEFAULT_SELECTORS,
        "pairs": pairs,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(session, indent=2, ensure_ascii=False))
    print(f"session '{args.session}': {len(pairs)} pairs -> {out}")
    if not pairs:
        print("WARNING: 0 pairs — check keys/dirs")


if __name__ == "__main__":
    main()
