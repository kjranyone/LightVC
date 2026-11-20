"""段 D5(paper_snv rev41 §6・§7): 開発の組の本物の日本語の男声。Common Voice 22.0 ja(CC0・写し = fsicoli/common_voice_22_0)の train と dev の分割から、性の自己申告が男性で 12 本以上ある話者を採る。
話者ごとに先頭 40 本(path の順)から、評価の発話 4 本(長い順)と登録 8 本(残りの先頭から・2 s 以上)を選ぶ。評価の入力だけに使い、学習・目標にしない。
話者の名前 = c + client_id の sha1 の先頭 7 桁。組 = devB(確認)は sha1 の先頭 8 桁の値 mod 3 == 0・残りが devA(台帳とてこの設計)。
出力 = data/cv22_ja/dev_manifest.json(台帳)・data/cv22_ja/clips/*.mp3・data/snv_p_feat/cv/{話者}_{e|s}{番号}.wav(48 kHz)と .npz(前段 v3)。
    cd training && uv run --no-sync python snv_dev_set.py [--plan]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tarfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

import snv_p0_front as P

ROOT = Path(__file__).resolve().parent.parent
CV = ROOT / "data/cv22_ja"
OUT = ROOT / "data/snv_p_feat/cv"
MIN_CLIPS = 12
N_HEAD = 40
N_SRC = 4
N_ENR = 8
MIN_ENR_MS = 2000


def read_tsv(p: Path) -> list[dict]:
    csv.field_size_limit(10**9)
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def plan() -> dict:
    dur = {r["clip"]: int(r["duration[ms]"]) for r in read_tsv(CV / "transcript/ja/clip_durations.tsv")}
    spk: dict = defaultdict(list)
    for part in ("train", "dev"):
        for r in read_tsv(CV / f"transcript/ja/{part}.tsv"):
            if r["gender"] == "male_masculine":
                spk[r["client_id"]].append(r | {"part": part})
    out = []
    for cid, rows in sorted(spk.items()):
        if len(rows) < MIN_CLIPS:
            continue
        h = hashlib.sha1(cid.encode()).hexdigest()
        head = sorted(rows, key=lambda r: r["path"])[:N_HEAD]
        src = sorted(sorted(head, key=lambda r: -dur[r["path"]])[:N_SRC], key=lambda r: r["path"])
        rest = [r for r in head if r not in src and dur[r["path"]] >= MIN_ENR_MS][:N_ENR]
        if len(rest) < N_ENR:
            continue
        item = lambda r, tag: {"name": tag, "clip": r["path"], "ms": dur[r["path"]], "sentence": r["sentence"]}
        out.append({"speaker": "c" + h[:7], "split": "devB" if int(h[:8], 16) % 3 == 0 else "devA", "part": rows[0]["part"], "age": rows[0]["age"], "n_clips": len(rows),
                    "src": [item(r, f"s{i}") for i, r in enumerate(src)], "enroll": [item(r, f"e{i}") for i, r in enumerate(rest)]})
    return {"source": "Common Voice 22.0 ja (CC0-1.0) via fsicoli/common_voice_22_0", "use": "evaluation input only", "rule": {"gender": "male_masculine", "min_clips": MIN_CLIPS, "head": N_HEAD, "src": N_SRC, "enroll": N_ENR},
            "n_speakers_male": len(spk), "speakers": out}


def extract(man: dict) -> None:
    need = {c["clip"]: s["part"] for s in man["speakers"] for c in s["src"] + s["enroll"]}
    (CV / "clips").mkdir(exist_ok=True)
    for part in ("train", "dev"):
        want = {c for c, p in need.items() if p == part and not (CV / "clips" / c).exists()}
        if not want:
            continue
        with tarfile.open(CV / f"audio/ja/{part}/ja_{part}_0.tar") as tf:
            for m in tf:
                nm = Path(m.name).name
                if nm in want:
                    (CV / "clips" / nm).write_bytes(tf.extractfile(m).read())
                    want.discard(nm)
        assert not want, f"{part}: {len(want)} 本が tar に無い"


def work(job: tuple) -> tuple:
    key, clip = job
    torch.set_num_threads(1)
    dst = OUT / f"{key}.npz"
    sr0 = sf.info(str(CV / "clips" / clip)).samplerate
    if dst.exists():
        return key, sr0
    x = librosa.load(str(CV / "clips" / clip), sr=P.SR, mono=True)[0].astype(np.float32)
    sf.write(OUT / f"{key}.wav", x, P.SR)
    T = P.n_frames(len(x))
    f = P.analyze(x[: T * P.H])
    np.savez(dst, **{k: v.astype(np.float32) for k, v in f.items()})
    return key, sr0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args()
    mp = CV / "dev_manifest.json"
    man = plan()
    n = {s: sum(1 for x in man["speakers"] if x["split"] == s) for s in ("devA", "devB")}
    print("male speakers", man["n_speakers_male"], "selected", len(man["speakers"]), n, flush=True)
    if a.plan:
        return 0
    extract(man)
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(f"{s['speaker']}_{c['name']}", c["clip"]) for s in man["speakers"] for c in s["src"] + s["enroll"]]
    srs: dict = {}
    with ProcessPoolExecutor(11) as ex:
        for i, (key, sr0) in enumerate(ex.map(work, jobs, chunksize=4)):
            srs[key] = sr0
            if (i + 1) % 500 == 0:
                print(i + 1, "/", len(jobs), flush=True)
    for s in man["speakers"]:
        s["mp3_sr"] = sorted({srs[f"{s['speaker']}_{c['name']}"] for c in s["src"] + s["enroll"]})
    mp.write_text(json.dumps(man, ensure_ascii=False, indent=1))
    print("done", len(jobs), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
