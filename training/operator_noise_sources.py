"""paper_snv rev29 段 I15 の操作者の音(声と時間で重なる手・物の音: 手をたたく・叩く・擦る・触る)の音源の一覧(data/snv_sep/operator_sources.json)。
候補 = 段 I12 の MUSAN の雑音のうち許諾 CC0 / CC BY / パブリックドメインで人の声の選別(AST 閾 0.02)を通ったもの(data/snv_p_feat_rec2/sources.json の noise)。
各ファイルを AudioSet の分類器(MIT/ast-finetuned-audioset-10-10-0.4593・選別だけ)に 2 s の窓(1 s おき)で通し、操作者の音の類(手・打つ・擦る・紙や布・小物)の確率の窓の最大を取る。
操作者の音 = その最大が thr 以上。学習と開発の分け = train_snv_sep.bg_split と同じファイルの分け(crc32)。
    cd training && HF_HUB_DISABLE_IMPLICIT_TOKEN=1 uv run --no-sync python operator_noise_sources.py [thr=0.3] [dev=cuda]
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/snv_p_feat_rec2/sources.json"
OUT = ROOT / "data/snv_sep/operator_sources.json"
MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SR = 16000
WIN, HOP = 2 * SR, SR
OPERATOR = (
    "Clapping", "Hands", "Finger snapping", "Applause", "Slap, smack", "Whack, thwack", "Knock", "Tap", "Thump, thud", "Smash, crash",
    "Rustle", "Scratch", "Scrape", "Rub", "Crumpling, crinkling", "Tearing", "Zipper (clothing)", "Keys jangling", "Coin (dropping)",
    "Typing", "Computer keyboard", "Writing", "Drawer open or close", "Cupboard open or close", "Door", "Sliding door", "Cutlery, silverware",
    "Dishes, pots, and pans", "Chopping (food)", "Wood", "Glass", "Clink", "Shuffle", "Walk, footsteps", "Squeak", "Creak", "Mouse click",
    "Clicking", "Tick", "Ratchet, pawl", "Velcro, hook and loop fastener", "Paper rustling",
)


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    thr, dev = float(opt.get("thr", 0.3)), opt.get("dev", "cuda")
    import soundfile as sf
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    src = json.loads(SRC.read_text())
    fe = ASTFeatureExtractor.from_pretrained(MODEL)
    m = ASTForAudioClassification.from_pretrained(MODEL).to(dev).eval()
    lab = m.config.id2label
    idx = [i for i in sorted(lab) if lab[i] in OPERATOR]
    print("operator classes", len(idx), "missing", sorted(set(OPERATOR) - {lab[i] for i in idx}), flush=True)
    rows = {}
    for n, p in enumerate(src["pools"]["noise"]):
        x = sf.read(p, dtype="float32")[0]
        x = x.mean(1) if x.ndim > 1 else x
        st = list(range(0, max(1, len(x) - WIN + 1), HOP))
        pr = []
        for b in range(0, len(st), 64):
            f = fe([x[s:s + WIN] for s in st[b:b + 64]], sampling_rate=SR, return_tensors="pt")["input_values"].to(dev)
            with torch.no_grad():
                pr.append(torch.sigmoid(m(f).logits).float().cpu().numpy())
        pr = np.concatenate(pr)[:, idx]
        top = pr.max(0)
        stem = Path(p).stem
        rows[stem] = {"path": p, "max": round(float(top.max()), 4), "cls": lab[idx[int(top.argmax())]],
                      "top3": [[lab[idx[j]], round(float(top[j]), 4)] for j in np.argsort(-top)[:3]],
                      "frac_win": round(float((pr.max(1) >= thr).mean()), 3), "lic": src["files"][stem]["lic"], "voice": src["files"][stem]["voice"],
                      "split": "dev" if zlib.crc32(f"musan|{stem}".encode()) % 5 == 0 else "train"}
        if (n + 1) % 100 == 0:
            print("scored", n + 1, flush=True)
    keep = {k: r for k, r in rows.items() if r["max"] >= thr}
    by: dict = {}
    for r in keep.values():
        by[r["cls"]] = by.get(r["cls"], 0) + 1
    out = {"thr": thr, "model": MODEL, "classes": OPERATOR, "n_scored": len(rows), "n_keep": len(keep),
           "by_class": dict(sorted(by.items(), key=lambda kv: -kv[1])), "pools": {s: sorted(r["path"] for r in keep.values() if r["split"] == s) for s in ("train", "dev")}, "files": rows}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: out[k] for k in ("n_scored", "n_keep", "by_class")}, ensure_ascii=False), {s: len(v) for s, v in out["pools"].items()}, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
