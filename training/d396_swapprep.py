"""diag_ppgvoc31 の事後の入れ替えの対(results/diag_ppgvoc31/prereg.yaml)。VCTK の全話者共通の文(001〜024)を話者ごとに順に削ってつないだ流れ(stream)を作り、
学習の目標の女声 TF(21 人 = 評価の組 devvctk・D395 の保留に入らない女声)と源(学習の交差の標本の男声 30 人 + 他の TF)の対ごとに、文ごとの C1 の事後の DTW から
滑らかな時間写像(F のフレーム → 源のフレームの小数の位置・D395b: 写しの痕跡の判別 0.85 → 0.63〜0.67)を作る。逆向き(源 → F)も保存(自己の往復の対照用)。
最後に学習と同じ形の前向き(区間 2.8 s)で、Rust の書き出しのフレームと実録音のずれ OFF(サンプル)を log-mel L1 の走査で決める。
    uv run python d396_swapprep.py --out <dev の親>/d396
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from d273_data import BIN, job
from d274_evaldump import COM
from d326_vctkset import CONDGAN, V, run, trim
from d6_unit_cov import load48

HOP = 240
H = Path("/home/kojirotanaka/lightvc_scratch_20261013")


def speakers() -> tuple:
    rows = [l.split() for l in (V / "speaker-info.txt").read_text().splitlines()[1:] if l.strip()]
    gender = {("p" + r[0]) if not r[0].startswith("p") else r[0]: r[2] for r in rows}
    used = {x["spk"] for x in json.loads(CONDGAN.read_text())["male"]}
    have = {d.name for d in (V / "wav48").iterdir() if d.is_dir()}
    fsp = sorted(s for s, g in gender.items() if g == "F" and s in have)
    f326 = list(fsp)
    random.Random(326).shuffle(f326)
    f326 = set(f326[:30])
    t395 = sorted({t["F"] for t in json.loads((H / "d395/set.json").read_text())["triples"]})
    free = [f for f in fsp if f not in f326 and f not in t395]
    only395 = [f for f in t395 if f not in f326]
    tf = sorted(free + only395[:10])
    held = sorted(f for f in t395 if f not in tf)
    males = sorted(s for s in used if s in have and gender.get(s) == "M")
    return tf, males, held


def smooth_map(jb: np.ndarray, n_src: int, win: int = 25) -> np.ndarray:
    k = np.ones(win) / win
    jp = np.convolve(np.pad(jb, (win // 2, win - 1 - win // 2), mode="edge"), k, "valid")
    return np.clip(np.maximum.accumulate(jp), 0, n_src - 1)


def pair_maps(args: tuple) -> tuple:
    f, x, out = args
    o = Path(out)
    if (o / "maps" / f"{f}__{x}.npz").exists():
        return f, x, "skip"
    sm = lambda s: (lambda l: np.exp(l - l.max(1, keepdims=True)) / np.exp(l - l.max(1, keepdims=True)).sum(1, keepdims=True))(np.fromfile(o / "dump" / f"{s}.front", "<f4").reshape(-1, 224)[:, :200].astype(np.float64))
    pf, px = sm(f), sm(x)
    sf_, sx_ = json.loads((o / "wav" / f"{f}.json").read_text()), json.loads((o / "wav" / f"{x}.json").read_text())
    fx = np.full(len(pf), np.nan)
    xf = np.full(len(px), np.nan)
    for sid, (a0, a1) in sf_.items():
        if sid not in sx_:
            continue
        b0, b1 = sx_[sid]
        fa, fb = a0 // HOP, min(a1 // HOP, len(pf))
        xa, xb = b0 // HOP, min(b1 // HOP, len(px))
        if fb - fa < 10 or xb - xa < 10:
            continue
        C = 1.0 - np.sqrt(pf[fa:fb]) @ np.sqrt(px[xa:xb]).T
        _, wp = librosa.sequence.dtw(C=C, backtrack=True)
        for src, dst, n_src, n_dst, s0, d0, ii, jj in ((fx, None, xb - xa, fb - fa, xa, fa, 0, 1), (xf, None, fb - fa, xb - xa, fa, xa, 1, 0)):
            acc = np.zeros(n_dst)
            cnt = np.zeros(n_dst)
            for row in wp:
                acc[row[ii]] += row[jj]
                cnt[row[ii]] += 1
            src[d0:d0 + n_dst] = smooth_map(acc / cnt, n_src) + s0
    for m in (fx, xf):
        ok = np.flatnonzero(~np.isnan(m))
        m[:] = np.interp(np.arange(len(m)), ok, m[ok])
    np.savez(o / "maps" / f"{f}__{x}.npz", fx=fx.astype(np.float32), xf=xf.astype(np.float32))
    return f, x, "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=str(Path(__file__).resolve().parent.parent / "results/diag_ppgvoc27/snap/ema_002500.pt"))
    a = ap.parse_args()
    O = Path(a.out)
    for d in ("wav", "en", "dump", "maps", "ref"):
        (O / d).mkdir(parents=True, exist_ok=True)
    tf, males, held = speakers()
    print("TF", len(tf), tf, "| 源の男声", len(males), "| D395 の保留の目標", len(held), flush=True)
    for s in tf + males:
        if (O / "wav" / f"{s}.json").exists():
            continue
        ws = sorted(w for w in (V / "wav48" / s).glob("*.wav") if w.stem.split("_")[1] <= "024")
        xs, bnd, pos = [], {}, 0
        for w in ws:
            x = trim(load48(w)).astype(np.float32)
            bnd[w.stem.split("_")[1]] = [pos, pos + len(x)]
            pos += len(x)
            xs.append(x)
        sf.write(O / "wav" / f"{s}.wav", np.concatenate(xs), 48000)
        (O / "wav" / f"{s}.json").write_text(json.dumps(bnd))
    for f in tf:
        en = O / "en" / f
        if (en / "enroll.json").exists():
            continue
        ws = sorted(w for w in (V / "wav48" / f).glob("*.wav") if w.stem.split("_")[1] > "024")
        ref, tl = [], 0.0
        while tl < 24.0 and ws:
            x = trim(load48(ws.pop()))
            ref.insert(0, x)
            tl += len(x) / 48000
        rw = O / "ref" / f"ref_{f}.wav"
        sf.write(rw, np.concatenate(ref).astype(np.float32), 48000)
        run([BIN, "artic-enroll", "-r", rw, "-o", en])
    jobs = []
    for s in tf + males:
        en = O / "en" / (s if s in tf else tf[0])
        jobs.append(([BIN, "artic", "--enroll", en, "--operator-enroll", en, "--med-from-operator"] + COM + ["-i", O / "wav" / f"{s}.wav"], O / "dump" / f"{s}.cond", O / "dump" / f"{s}.front"))
    with Pool(6) as pl:
        list(pl.imap_unordered(job, jobs))
    print("dumped", sum(Path(c).exists() for _, c, _ in jobs), "/", len(jobs), flush=True)
    pairs = [(f, x, str(O)) for f in tf for x in males + [g for g in tf if g != f]]
    with Pool(6) as pl:
        res = list(pl.imap_unordered(pair_maps, pairs, chunksize=4))
    print("maps", sum(r[2] in ("ok", "skip") for r in res), "/", len(pairs), flush=True)
    import torch
    import nvoc as N
    import train_ppgvoc as P
    import train_rvoc as TR
    dev = "cuda"
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=False)
    P.slow_from_ckpt(st)
    g = P.init_gen(dev, "table")
    g.load_state_dict(st["ema"])
    g.eval()
    mels = [P.build_mel(nf, hm, nm).to(dev) for nf, hm, nm in P.MEL_SPECS]
    SEG = 33600 * 4
    T = SEG // HOP
    scores: dict = {}
    for f in tf[:6]:
        c, lg = P.load_dump(O / "dump", f)
        real, _ = sf.read(O / "wav" / f"{f}.wav", dtype="float32")
        spk = P.spk_tensor(P.load_enroll(O / "en" / f), dev)
        for t0 in (400, 2000, 4000):
            if t0 + T > len(c):
                continue
            ct = torch.from_numpy(c[t0:t0 + T].T.copy()).float().to(dev)[None]
            p = torch.from_numpy(lg[t0:t0 + T].T.copy()).float().to(dev)[None].softmax(1)
            f0 = torch.where(ct[:, 26] > 0.5, 200.0 * torch.exp(ct[:, 25]), torch.zeros_like(ct[:, 25]))
            with torch.no_grad():
                y = g(g.make(p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], spk), TR.excitation(f0, SEG, torch.Generator().manual_seed(0)), spk)[0]
                yw = y[TR.WARM:]
                for off in range(0, 1201, 24):
                    s0 = t0 * HOP - off + TR.WARM
                    if s0 < 0:
                        continue
                    rr = torch.from_numpy(real[s0:s0 + len(yw)]).to(dev)
                    if len(rr) < len(yw):
                        continue
                    yy = yw * torch.sqrt((rr ** 2).mean() / ((yw ** 2).mean() + 1e-12))
                    scores.setdefault(off, []).append(float(P.logmel_l1(yy[None, None], rr[None, None], mels)))
    ms = {o: float(np.mean(v)) for o, v in scores.items()}
    off = min(ms, key=ms.get)
    print("ずれ OFF の走査(平均 log-mel L1)", {o: round(v, 4) for o, v in sorted(ms.items()) if o % 120 == 0 or o == off}, "→ OFF", off, flush=True)
    (O / "pool.json").write_text(json.dumps({"tf": tf, "males": males, "held_d395": held, "off": off, "seg": SEG, "scan": ms}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
