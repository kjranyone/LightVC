"""paper_snv rev24 §5 M18 の本物の音源の一覧(段 I12・data/snv_p_feat_rec2/sources.json): MUSAN(openslr 17)の音楽と雑音から、許諾が CC0 / CC BY / パブリックドメインで、歌と人の声の無い 16 kHz の wav だけ。
許諾 = 各ディレクトリの LICENSE を区切り(= の行)の塊ごとに読む(塊の中の名前にその塊の記述・License の行があればそれだけ・名前の無い塊が 1 つならディレクトリの既定)。SA・NC・ND・記述なしは除く。
歌 = 音楽の ANNOTATIONS の vocals N だけ(載っていないものは除く)。
人の声 = AudioSet の分類器(MIT/ast-finetuned-audioset-10-10-0.4593・BSD-3)の人の声・息・人の集まりの声の類の確率の、10 秒窓(5 秒おき)の最大が thr 未満(音楽と雑音の両方)。
較正 = 音楽の vocals Y を thr で拾う率(scan で分布を出す)。分類器の値 = data/snv_p_feat_rec2/voice_scores.json(鍵 = 相対の道・大きさ・時刻)。
    cd training && HF_HUB_DISABLE_IMPLICIT_TOKEN=1 uv run --no-sync python musan_sources.py scan [root=data/musan/musan] [dev=cuda]
    cd training && HF_HUB_DISABLE_IMPLICIT_TOKEN=1 uv run --no-sync python musan_sources.py thr=0.1 [root=data/musan/musan] [dev=cuda]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/snv_p_feat_rec2"
MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SR = 16000
WIN, HOP = 10 * SR, 5 * SR
ID = re.compile(r"\b((?:music|noise)-[a-z-]+-\d{4})\b")
BAD = re.compile(r"share[\s-]?alike|by-sa|by-nc|by-nd|non-?commercial|no-?derivs?|noderiv|\b(?:sa|nc|nd)\b", re.I)
RANK = {"pd": 0, "by": 1, "unknown": 2, "restricted": 3}
VOICE = (
    "Speech", "Male speech, man speaking", "Female speech, woman speaking", "Child speech, kid speaking", "Conversation",
    "Narration, monologue", "Babbling", "Speech synthesizer", "Shout", "Bellow", "Whoop", "Yell", "Battle cry",
    "Children shouting", "Screaming", "Whispering", "Laughter", "Baby laughter", "Giggle", "Snicker", "Belly laugh",
    "Chuckle, chortle", "Crying, sobbing", "Baby cry, infant cry", "Whimper", "Wail, moan", "Sigh", "Singing", "Choir",
    "Yodeling", "Chant", "Mantra", "Male singing", "Female singing", "Child singing", "Synthetic singing", "Rapping",
    "Humming", "Groan", "Grunt", "Whistling", "Breathing", "Wheeze", "Snoring", "Gasp", "Pant", "Snort", "Cough",
    "Throat clearing", "Sneeze", "Sniff", "Burping, eructation", "Hiccup", "Gargling", "Cheering", "Booing", "Chatter",
    "Crowd", "Hubbub, speech noise, speech babble", "Children playing", "Vocal music", "A capella", "Beatboxing",
)


def lic_class(text: str) -> str:
    if BAD.search(text):
        return "restricted"
    if re.search(r"attribution|\bcc[\s-]by\b", text, re.I):
        return "by"
    if re.search(r"public domain|\bcc0\b", text, re.I):
        return "pd"
    return "unknown"


def blocks(path: Path) -> list[tuple[list[str], str]]:
    out, cur = [], []
    for line in path.read_text(errors="replace").splitlines() + ["===="]:
        if not line.startswith("===="):
            cur.append(line)
            continue
        if any(x.strip() for x in cur):
            ids = sorted({i for x in cur for i in ID.findall(x)})
            lic = [x.strip() for x in cur if x.strip().lower().startswith("license")]
            out.append((ids, " ".join(lic) if lic else " ".join(x.strip() for x in cur if x.strip() and not ID.fullmatch(x.strip()))))
        cur = []
    return out


def licenses(d: Path) -> dict[str, tuple[str, str]]:
    bl = blocks(d / "LICENSE")
    head = [t for ids, t in bl if not ids and lic_class(t) != "unknown"]
    default = head[0] if len(head) == 1 else ""
    out: dict[str, tuple[str, str]] = {}
    for ids, t in bl:
        text = t if lic_class(t) != "unknown" else default
        for i in ids:
            c = (lic_class(text), text[:200])
            if i not in out or RANK[c[0]] > RANK[out[i][0]]:
                out[i] = c
    for w in d.glob("*.wav"):
        out.setdefault(w.stem, (lic_class(default), default[:200]) if default else ("unknown", ""))
    return out


def vocals(d: Path) -> dict[str, str]:
    p = d / "ANNOTATIONS"
    return {x.split()[0]: x.split()[2] for x in p.read_text(errors="replace").splitlines() if len(x.split()) >= 3} if p.exists() else {}


def catalog(root: Path) -> list[dict]:
    import soundfile as sf
    rows = []
    for kind in ("music", "noise"):
        for d in sorted(x for x in (root / kind).iterdir() if x.is_dir()):
            lic, voc = licenses(d), vocals(d) if kind == "music" else {}
            for w in sorted(d.glob("*.wav")):
                info = sf.info(str(w))
                rows.append({"kind": kind, "dir": d.name, "id": w.stem, "path": str(w.resolve()), "rel": str(w.relative_to(ROOT)),
                             "sr": info.samplerate, "dur": info.frames / info.samplerate, "lic": lic[w.stem][0], "lic_text": lic[w.stem][1],
                             "vocals": voc.get(w.stem, "?") if kind == "music" else "-", "size": w.stat().st_size, "mtime": w.stat().st_mtime_ns})
    return rows


def voice_scores(rows: list[dict], dev: str) -> dict[str, dict]:
    p = OUT / "voice_scores.json"
    cache = json.loads(p.read_text()) if p.exists() else {}
    todo = [r for r in rows if cache.get(r["rel"], {}).get("key") != [r["size"], r["mtime"]]]
    if todo:
        import soundfile as sf
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        fe = ASTFeatureExtractor.from_pretrained(MODEL)
        m = ASTForAudioClassification.from_pretrained(MODEL).to(dev).eval()
        lab = m.config.id2label
        idx = [i for i in sorted(lab) if lab[i] in VOICE]
        miss = sorted(set(VOICE) - {lab[i] for i in idx})
        print("voice classes", len(idx), "missing", miss, flush=True)
        for n, r in enumerate(todo):
            x = sf.read(r["path"], dtype="float32")[0]
            x = x.mean(1) if x.ndim > 1 else x
            st = list(range(0, max(1, len(x) - WIN + 1), HOP))
            if len(x) > WIN and st[-1] + WIN < len(x):
                st.append(len(x) - WIN)
            pr = []
            for b in range(0, len(st), 32):
                feats = fe([x[s:s + WIN] for s in st[b:b + 32]], sampling_rate=SR, return_tensors="pt")["input_values"].to(dev)
                with torch.no_grad():
                    pr.append(torch.sigmoid(m(feats).logits).float().cpu().numpy())
            pr = np.concatenate(pr)
            v = pr[:, idx]
            top = v.max(0)
            cache[r["rel"]] = {"key": [r["size"], r["mtime"]], "max": float(v.max()), "cls": lab[idx[int(top.argmax())]],
                               "top3": [[lab[idx[j]], round(float(top[j]), 4)] for j in np.argsort(-top)[:3]], "windows": len(st)}
            if (n + 1) % 100 == 0:
                print("scored", n + 1, "/", len(todo), flush=True)
                OUT.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(cache))
        OUT.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cache))
    return cache


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    root = ROOT / opt.get("root", "data/musan/musan")
    rows = catalog(root)
    sc = voice_scores(rows, opt.get("dev", "cuda"))
    for r in rows:
        r["voice"], r["voice_cls"] = sc[r["rel"]]["max"], sc[r["rel"]]["cls"]
    for kind in ("music", "noise"):
        for d in sorted({r["dir"] for r in rows if r["kind"] == kind}):
            rs = [r for r in rows if r["dir"] == d]
            print(kind, d, len(rs), {c: sum(r["lic"] == c for r in rs) for c in RANK}, {v: sum(r["vocals"] == v for r in rs) for v in sorted({r["vocals"] for r in rs})},
                  "sr", sorted({r["sr"] for r in rs}), "h", round(sum(r["dur"] for r in rs) / 3600, 2), flush=True)
    q = lambda xs: [round(float(np.quantile(xs, a)), 3) for a in (0.05, 0.25, 0.5, 0.75, 0.95)] if xs else []
    mY, mN = [r["voice"] for r in rows if r["vocals"] == "Y"], [r["voice"] for r in rows if r["vocals"] == "N"]
    nz = [r["voice"] for r in rows if r["kind"] == "noise"]
    print("voice score quantiles(5/25/50/75/95%) music Y", q(mY), "music N", q(mN), "noise", q(nz), flush=True)
    for t in (0.05, 0.1, 0.2, 0.3, 0.5):
        print(f"thr {t}: music Y caught {np.mean(np.array(mY) >= t):.3f} · music N cut {np.mean(np.array(mN) >= t):.3f} · noise cut {np.mean(np.array(nz) >= t):.3f}", flush=True)
    if "scan" in sys.argv[1:]:
        for r in sorted((r for r in rows if r["kind"] == "noise" or r["vocals"] == "N"), key=lambda r: -r["voice"])[:60]:
            print(f"  {r['voice']:.3f} {r['voice_cls']:<38} {r['id']} {r['lic']} {r['lic_text'][:60] if r['dir'] == 'sound-bible' else ''}", flush=True)
        return 0
    thr = float(opt["thr"])
    why = lambda r: ("sr" if r["sr"] != SR or r["dur"] < 0.5 else f"license_{r['lic']}" if r["lic"] not in ("pd", "by")
                     else f"vocals_{r['vocals']}" if r["kind"] == "music" and r["vocals"] != "N" else "voice_screen" if r["voice"] >= thr else "")
    keep = [r for r in rows if not why(r)]
    excluded: dict[str, dict[str, int]] = {}
    for r in rows:
        if why(r):
            excluded.setdefault(r["kind"], {}).setdefault(why(r), 0)
            excluded[r["kind"]][why(r)] += 1
    out = {"pools": {k: [r["path"] for r in keep if r["kind"] == k] for k in ("music", "noise")},
           "licenses": {k: {c: sum(r["lic"] == c for r in keep if r["kind"] == k) for c in ("pd", "by")} for k in ("music", "noise")},
           "hours": {k: round(sum(r["dur"] for r in keep if r["kind"] == k) / 3600, 2) for k in ("music", "noise")},
           "excluded": excluded,
           "screen": {"model": MODEL, "thr": thr, "window_s": WIN / SR, "hop_s": HOP / SR, "classes": list(VOICE),
                      "music_Y_caught": round(float(np.mean(np.array(mY) >= thr)), 4), "music_N_cut": round(float(np.mean(np.array(mN) >= thr)), 4), "noise_cut": round(float(np.mean(np.array(nz) >= thr)), 4)},
           "files": {r["id"]: {"lic": r["lic"], "text": r["lic_text"], "voice": round(r["voice"], 4)} for r in keep},
           "source": "MUSAN (Snyder, Chen, Povey 2015, arXiv:1510.08484, openslr 17)"}
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "sources.json.tmp"
    tmp.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    tmp.replace(OUT / "sources.json")
    print("sources", {k: len(v) for k, v in out["pools"].items()}, out["licenses"], out["hours"], "excluded", excluded, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
