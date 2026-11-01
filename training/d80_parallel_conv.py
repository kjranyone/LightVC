"""D80(学習なし): 男声 → 女声の変換を、目標の女性が同じ文を読んだ本物と DTW で比べて採点する(dtw_ruler.py)。
元 = data/male_tts_v4l_par/<M>/fK.wav(男声が女声コーパスと同じ文を読んだもの)・目標 A = 女声 TTS(neutral の 10 文がある話者・種 80)・基準 = A の tK_neutral。
登録 = A の文 tK 以外の neutral 9 文(Rust artic-enroll)・操作者 = M の data/male_tts_v4l の文 22s 以上(Rust artic-enroll)。
条件 = --conds "名前=Rust の追加引数(空白区切り)"(; 区切り)。共通 = artic --mix 0 --per-p75 --med-from-operator --warp --alpha 1.3 --operator-enroll <M>。
同時に測る: FLOOR_<style> = A の同じ文の別の話し方の本物(自然なばらつき)・SRC = 元の男声・SELFCOPY = A の tK_neutral を製品経路で写す(操作者 = A・γ 1・α 1)。
出力 = <out>/<M>_<A>_tK_<cond>.wav・results/conv_p0/<tag>.json。
    uv run python d80_parallel_conv.py --out_dir <scratchpad>/d80 --tag d80 --conds "TAB=--resid 0;W07=--resid 0.7;W10=--resid 1.0"
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from d6_unit_cov import load48

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target/release/lightvc-app"
FEM = ROOT / "data/female_tts_corpus"
MALE = ROOT / "data/male_tts_v4l"
MPAR = ROOT / "data/male_tts_v4l_par"
STYLES = ["breathy", "low_tension", "soft", "warm"]


def run(cmd: list) -> None:
    r = subprocess.run([str(c) for c in cmd], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"FAIL {cmd} {r.stderr[-400:]}")


def enroll(wavs: list, out: Path, need: float) -> Path:
    if (out / "enroll.json").exists():
        return out
    xs, tot = [], 0.0
    for w in wavs:
        if tot >= need:
            break
        x = load48(w); xs.append(x); tot += len(x) / 48000
    assert tot >= 20.0, f"{out} 参照が足りない {tot:.1f}s"
    out.parent.mkdir(parents=True, exist_ok=True)
    rp = out.parent / f"{out.name}.wav"
    sf.write(rp, np.concatenate(xs).astype(np.float32), 48000)
    run([BIN, "artic-enroll", "-r", rp, "-o", out])
    return out


def w48(src: Path, work: Path) -> Path:
    out = work / "48k" / f"{src.parent.name}_{src.name}"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, load48(src).astype(np.float32), 48000)
    return out


def job_enroll(j: tuple) -> str:
    wavs, out, need = j
    return str(enroll(wavs, Path(out), need))


def job_conv(j: tuple) -> str:
    cmd, outp = j
    if not Path(outp).exists():
        run(cmd)
    return outp


def job_score(j: tuple) -> list:
    import dtw_ruler as DR
    from d77_ruler import rd
    pair, refp, items = j
    ax = DR.analyse(rd(Path(refp)))
    return [{"pair": pair, "cond": c, **DR.dist(DR.analyse(rd(Path(p))), ax)} for c, p in items]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", default="d80")
    ap.add_argument("--conds", default="TAB=--resid 0;W07=--resid 0.7;W10=--resid 1.0")
    ap.add_argument("--n_male", type=int, default=24)
    ap.add_argument("--texts", default="0,1,2,3,4")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--extra", default="", help="全条件に足す Rust の引数(例 --rvoc results/diag_rvtab/export)")
    a = ap.parse_args()
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    work = od / "_work"
    conds = {c.split("=", 1)[0]: c.split("=", 1)[1].split() for c in a.conds.split(";") if c}
    males = sorted(d.name for d in MPAR.iterdir() if d.is_dir() and len(list(d.glob("f*.wav"))) == 10)[: a.n_male]
    fems = sorted(d.name for d in FEM.iterdir() if d.is_dir() and all((d / f"t{i:02d}_{s}.wav").exists() for i in range(10) for s in ["neutral"] + STYLES))
    random.Random(80).shuffle(fems)
    texts = [int(t) for t in a.texts.split(",")]
    en_jobs = []
    for mi, M in enumerate(males):
        A = fems[mi]
        en_jobs.append((sorted(MALE.glob(f"{M}/t*.wav")), str(work / f"op_{M}"), 22.0))
        for k in texts:
            en_jobs.append(([FEM / A / f"t{i:02d}_neutral.wav" for i in range(10) if i != k], str(work / f"en_{A}_t{k:02d}"), 30.0))
    with Pool(a.workers) as pool:
        list(pool.imap_unordered(job_enroll, en_jobs))
    print("enrolled", len(en_jobs), flush=True)
    conv_jobs, score_jobs = [], []
    for mi, M in enumerate(males):
        A = fems[mi]
        opd = enroll(sorted(MALE.glob(f"{M}/t*.wav")), work / f"op_{M}", 22.0)
        for k in texts:
            tk = f"t{k:02d}"
            end = enroll([FEM / A / f"t{i:02d}_neutral.wav" for i in range(10) if i != k], work / f"en_{A}_{tk}", 30.0)
            src = MPAR / M / f"f{k:02d}.wav"
            ref = w48(FEM / A / f"{tk}_neutral.wav", work)
            pair = f"{M}_{A}_{tk}"
            items = [("SRC", str(src))] + [(f"FLOOR_{s}", str(w48(FEM / A / f"{tk}_{s}.wav", work))) for s in STYLES]
            sc = od / f"{pair}_SELFCOPY.wav"
            selfop = enroll([FEM / A / f"t{i:02d}_neutral.wav" for i in range(10) if i != k], work / f"en_{A}_{tk}", 30.0)
            conv_jobs.append(([BIN, "artic", "-i", ref, "-o", sc, "--enroll", end, "--operator-enroll", selfop, "--mix", "0", "--per-p75", "--med-from-operator", "--warp", "--alpha", "1.0", "--resid", "1.0"] + a.extra.split(), str(sc)))
            items.append(("SELFCOPY", str(sc)))
            for c, args in conds.items():
                outp = od / f"{pair}_{c}.wav"
                al = [] if "--alpha" in args else ["--alpha", "1.3"]
                conv_jobs.append(([BIN, "artic", "-i", src, "-o", outp, "--enroll", end, "--operator-enroll", opd, "--mix", "0", "--per-p75", "--med-from-operator", "--warp"] + al + args + a.extra.split(), str(outp)))
                items.append((c, str(outp)))
            score_jobs.append((pair, str(ref), items))
    print("conv", len(conv_jobs), "pairs", len(score_jobs), flush=True)
    with Pool(a.workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(job_conv, conv_jobs)):
            if i % 50 == 0:
                print("conv", i, flush=True)
    rows = []
    with Pool(a.workers) as pool:
        for r in pool.imap_unordered(job_score, score_jobs):
            rows.extend(r)
    allc = ["SRC"] + [f"FLOOR_{s}" for s in STYLES] + ["SELFCOPY"] + list(conds)
    summ = {}
    for m in ["env", "lm", "lm_v"]:
        summ[m] = {c: round(float(np.nanmean([r[m] for r in rows if r["cond"] == c])), 4) for c in allc}
        fl = [r[m] for r in rows if r["cond"].startswith("FLOOR_")]
        summ[m]["FLOOR_p90"] = round(float(np.nanpercentile(fl, 90)), 4)
        print(m, json.dumps(summ[m]), flush=True)
    (ROOT / f"results/conv_p0/{a.tag}.json").write_text(json.dumps({"summary": summ, "conds": a.conds, "extra": a.extra, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
