"""paper_snv rev29 段 I15 の開発の組の操作者の音(本物の手拍子・手・叩く・擦る)の一覧(data/snv_sep/fsd50k_eval_sources.json)。FSD50K(Fonseca ら・Zenodo 4060432)の評価の組から、学習に使わず評価だけに使う。
許諾 = クリップごとの license が CC0 か CC BY(NC・Sampling+ は除く)。ラベル = 操作者の類(Clapping・Hands・Finger_snapping・Knock・Tap・Slap_and_smack・Crumpling_and_crinkling・Tearing・Zipper_(clothing)・Typing・Scratching_(performance_technique)・Rustle)を持ち、人の声の類のラベルを持たない。
人の声 = musan_sources と同じ AST の人の声の類の 10 秒窓の最大 < 0.02。分割の zip(FSD50K.eval_audio.z01 + .zip)の中央の目録を読み、候補のファイルだけを取り出す(extract)。帰属 = uploader と freesound の id を残す。
split=dev(paper_snv rev33 段 I18)= 開発の組(FSD50K.dev_audio.z01〜z05 + .zip)から同じ選び方で学習の手の音を作る(data/snv_sep/fsd50k_dev_sources.json・data/fsd50k_dev)。評価の組は評価だけ。
    cd training && HF_HUB_DISABLE_IMPLICIT_TOKEN=1 uv run --no-sync python fsd50k_sources.py [split=eval|dev] [dev=cuda]
"""
from __future__ import annotations

import csv
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DL = ROOT / "data/fsd50k_dl"
META = ROOT / "data/fsd50k_meta"
PARTS = {"eval": ["FSD50K.eval_audio.z01", "FSD50K.eval_audio.zip"], "dev": [f"FSD50K.dev_audio.z0{i}" for i in range(1, 6)] + ["FSD50K.dev_audio.zip"]}
CLAP = {"Clapping", "Hands"}
OPER = CLAP | {"Finger_snapping", "Knock", "Tap", "Slap_and_smack", "Crumpling_and_crinkling", "Tearing", "Zipper_(clothing)", "Typing", "Scratching_(performance_technique)", "Rustle"}
VOICE_LABELS = {"Human_voice", "Speech", "Male_speech_and_man_speaking", "Female_speech_and_woman_speaking", "Child_speech_and_kid_speaking", "Conversation", "Laughter",
                "Giggle", "Chuckle_and_chortle", "Cheering", "Crowd", "Shout", "Yell", "Screaming", "Singing", "Whispering", "Breathing", "Sigh", "Cough", "Sneeze",
                "Chatter", "Applause", "Human_group_actions", "Gasp", "Burping_and_eructation", "Run", "Walk_and_footsteps"}


class Split:
    def __init__(self, parts: list[Path]) -> None:
        self.parts = parts
        self.base = np.cumsum([0] + [q.stat().st_size for q in parts]).tolist()
        self.fh = [q.open("rb") for q in parts]

    def read(self, off: int, n: int) -> bytes:
        out = b""
        while n > 0:
            k = max(i for i in range(len(self.parts)) if self.base[i] <= off)
            self.fh[k].seek(off - self.base[k])
            b = self.fh[k].read(min(n, self.base[k + 1] - off))
            out, off, n = out + b, off + len(b), n - len(b)
        return out


def zip64_extra(extra: bytes, vals: dict) -> dict:
    i = 0
    while i + 4 <= len(extra):
        hid, ln = struct.unpack("<HH", extra[i:i + 4])
        if hid == 1:
            d, j = extra[i + 4:i + 4 + ln], 0
            for k in ("usize", "csize", "off", "disk"):
                if vals[k] == (0xFFFF if k == "disk" else 0xFFFFFFFF):
                    vals[k] = struct.unpack("<I" if k == "disk" else "<Q", d[j:j + (4 if k == "disk" else 8)])[0]
                    j += 4 if k == "disk" else 8
        i += 4 + ln
    return vals


def extract(names: set[str], dst: Path, split: str = "eval") -> int:
    z = Split([DL / q for q in PARTS[split]])
    last = DL / PARTS[split][-1]
    with last.open("rb") as fh:
        fh.seek(max(0, last.stat().st_size - (1 << 16) - 22))
        tail = fh.read()
    e = tail.rfind(b"PK\x05\x06")
    disk_cd, cd_size, cd_off = struct.unpack("<H", tail[e + 6:e + 8])[0], *struct.unpack("<II", tail[e + 12:e + 20])
    loc = tail.rfind(b"PK\x06\x07")
    if loc >= 0 and (cd_off == 0xFFFFFFFF or cd_size == 0xFFFFFFFF or disk_cd == 0xFFFF):
        z64_disk, z64_off = struct.unpack("<IQ", tail[loc + 4:loc + 16])
        rec = z.read(z.base[z64_disk] + z64_off, 56)
        disk_cd, cd_size, cd_off = struct.unpack("<I", rec[20:24])[0], *struct.unpack("<QQ", rec[40:56])
    cd = z.read(z.base[disk_cd] + cd_off, cd_size)
    i, n = 0, 0
    dst.mkdir(parents=True, exist_ok=True)
    while i + 46 <= len(cd) and cd[i:i + 4] == b"PK\x01\x02":
        comp, = struct.unpack("<H", cd[i + 10:i + 12])
        csize, usize = struct.unpack("<II", cd[i + 20:i + 28])
        nl, el, cl, disk = struct.unpack("<HHHH", cd[i + 28:i + 36])
        off, = struct.unpack("<I", cd[i + 42:i + 46])
        name = cd[i + 46:i + 46 + nl].decode("utf-8", "replace")
        v = zip64_extra(cd[i + 46 + nl:i + 46 + nl + el], {"usize": usize, "csize": csize, "off": off, "disk": disk})
        i += 46 + nl + el + cl
        stem = Path(name).stem
        if not name.endswith(".wav") or stem not in names or (dst / f"{stem}.wav").exists():
            continue
        a = z.base[v["disk"]] + v["off"]
        lh = z.read(a, 30)
        lnl, lel = struct.unpack("<HH", lh[26:30])
        data = z.read(a + 30 + lnl + lel, v["csize"])
        raw = data if comp == 0 else zlib.decompress(data, -15)
        assert len(raw) == v["usize"], name
        (dst / f"{stem}.wav").write_bytes(raw)
        n += 1
    return n


def lic_ok(url: str) -> str | None:
    if "publicdomain/zero" in url:
        return "cc0"
    if "/licenses/by/" in url:
        return "by"
    return None


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    split = opt.get("split", "eval")
    aud, out = ROOT / f"data/fsd50k_{split}", ROOT / f"data/snv_sep/fsd50k_{split}_sources.json"
    info = json.loads((META / f"FSD50K.metadata/{split}_clips_info_FSD50K.json").read_text())
    rows = list(csv.DictReader((META / f"FSD50K.ground_truth/{split}.csv").open()))
    cand = []
    for r in rows:
        labs = set(r["labels"].split(","))
        lic = lic_ok(info[r["fname"]]["license"])
        if lic and labs & OPER and not labs & VOICE_LABELS:
            cand.append({"fname": r["fname"], "labels": sorted(labs & OPER), "clap": bool(labs & CLAP), "lic": lic, "uploader": info[r["fname"]].get("uploader"), "title": info[r["fname"]].get("title")})
    print("extracted", extract({c["fname"] for c in cand}, aud / f"FSD50K.{split}_audio", split), flush=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import musan_sources as MS
    rel = [{"rel": f"fsd50k_{split}/{c['fname']}", "path": str(aud / f"FSD50K.{split}_audio" / f"{c['fname']}.wav")} for c in cand]
    for r in rel:
        st = Path(r["path"]).stat()
        r["size"], r["mtime"] = st.st_size, int(st.st_mtime)
    MS.OUT = ROOT / "data/snv_sep"
    vs = MS.voice_scores(rel, opt.get("dev", "cuda"))
    keep = []
    for c, r in zip(cand, rel):
        v = vs[r["rel"]]["max"]
        if v < 0.02:
            keep.append(c | {"path": r["path"], "voice": round(v, 4)})
    res = {"source": f"FSD50K {split} (Zenodo 4060432)" + ("・評価だけ" if split == "eval" else "・学習の手の音(評価の組とは別)"), "n_candidates": len(cand), "n_keep": len(keep),
           "pools": {"clap": [k["path"] for k in keep if k["clap"]], "op": [k["path"] for k in keep]}, "files": keep}
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("candidates", len(cand), "keep", len(keep), "clap", len(res["pools"]["clap"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
