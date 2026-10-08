"""c5_1 の前計算(results/c5_1/prereg.yaml): 女声の話者ごとに 参照(20s・表 T と参照の残差 = z の入力)/ 学習用の発話(包絡の教師・単位・f0・c0)を
ファイル単位で素に分けて npz に保存。男声(TTS 男声・VCTK 男声 train)は 単位・f0・c0・有声 のみ。
除外: 評価話者(held)・その TTS 複製・LDA 物差しの評価プール(d4b の学習話者以外 = D11b/D13 の評価候補)。女声は d4b の学習話者(1,200 人の範囲)から使う。
    uv run python prep_c5.py --out ../data/c5 --n_fem 1000
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from d6_unit_cov import TTS_LEAK, load48

ROOT = Path(__file__).resolve().parent.parent


def analyse(path: str):
    import f0hi as H
    import nvoc as N
    import pae as PA
    try:
        x = load48(path)[: 12 * 48000]
        n = len(x) // N.HOP
        if n < 200:
            return None
        x = x[: n * N.HOP]
        f0, _ = H.teacher_f0(x, n)
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        E = PA.envelope(xa, f0)
        e = np.pad(E, ((0, 0), (1, 1)), mode="edge")
        E = 0.25 * e[:, :-2] + 0.5 * e[:, 1:-1] + 0.25 * e[:, 2:]
        return path, x, E[:, :n].astype(np.float32)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_fem", type=int, default=1000)
    ap.add_argument("--max_train_utt", type=int, default=8)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--n_male", type=int, default=3000)
    a = ap.parse_args()
    import c1_content as CC
    import conv_c0 as C0
    import f0est as FE
    import lvl as LV
    import nvoc as N
    import train_c1 as T1
    import train_c3 as C3
    import train_f0est as TF
    import train_rvoc as TR
    dev = "cuda"
    out = Path(a.out); (out / "fem").mkdir(parents=True, exist_ok=True); (out / "male").mkdir(parents=True, exist_ok=True)
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev); c1.load_state_dict(st1["net"]); c1.eval()
    mfront = CC.MelFront().to(dev)
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()
    K = Cb.shape[0]
    sf1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev); f1.load_state_dict(sf1["net"]); f1.eval()
    f1front = FE.Front().to(dev)
    sf2 = torch.load(ROOT / "results/f2_2/last.pt", map_location="cpu", weights_only=False)
    f2 = LV.Lvl(sf2["cfg"]["ch"], tuple(sf2["cfg"]["dils"])).to(dev); f2.load_state_dict(sf2["net"]); f2.eval()

    @torch.no_grad()
    def prod(x: np.ndarray) -> dict:
        n = len(x) // N.HOP
        xt = torch.from_numpy(x)[None].to(dev)
        u = c1(mfront(torch.from_numpy(T1.prime(x))[None].to(dev)))[..., -n:].argmax(1)[0].cpu().numpy()
        f0 = TF.infer(f1front, f1, x, dev)
        f0 = np.pad(f0, (0, max(0, n - len(f0))))[:n]
        c0 = f2(mfront(torch.cat([torch.zeros(1, N.WIN - N.HOP, device=dev), xt], 1)))[:, None]
        c0 = C3.causal_avg(c0, 3)[0, 0].cpu().numpy()[:n]
        u = np.pad(u, (0, max(0, n - len(u))), mode="edge")[:n]
        return {"u": u.astype(np.int16), "f0": f0.astype(np.float32), "c0n": c0.astype(np.float32)}

    hs = TR.held_speakers()
    allspk = sorted(p.name for p in (ROOT / "female-dataset").iterdir() if p.is_dir() and p.name not in (hs | TTS_LEAK))
    lda_tr = [allspk[i] for i in np.random.RandomState(0).permutation(len(allspk))[:1200]]
    fem = lda_tr[: a.n_fem]
    index = {"fem": [], "male": []}
    with Pool(a.procs) as pool:
        for si, s in enumerate(fem):
            ws = sorted((ROOT / "female-dataset" / s).glob("*.wav"))
            ref, tot = [], 0.0
            for w in ws:
                if tot >= 20.0:
                    break
                ref.append(w); tot += sf.info(str(w)).duration
            rest = [w for w in ws if w not in ref][: a.max_train_utt]
            if tot < 18.0 or len(rest) < 2:
                continue
            res = [r for r in pool.map(analyse, [str(w) for w in ref + rest]) if r is not None]
            rr = [r for r in res if Path(r[0]) in set(ref)]
            ro = [r for r in res if Path(r[0]) not in set(ref)]
            if not rr or not ro:
                continue
            pr = [prod(r[1]) for r in rr]
            Er = np.concatenate([r[2] for r in rr], 1)
            ur = np.concatenate([p["u"] for p in pr]).astype(int)
            T = C0.table(Er[1:25], ur, K, Cb).astype(np.float32)
            rres = (Er[1:25] - T[:, ur]).astype(np.float16)
            vo_r = np.concatenate([p["f0"] for p in pr]) > 0
            mu = float(np.median(np.log(np.concatenate([p["f0"] for p in pr])[vo_r]))) if vo_r.any() else 5.6
            utts = []
            for r in ro:
                p = prod(r[1])
                utts.append({"env": r[2][1:25].astype(np.float16), "c0_real": r[2][0].astype(np.float16), **p})
            np.savez(out / "fem" / f"{s}.npz", T=T, ref_res=rres, ref_u=ur.astype(np.int16), mu=mu, cnt=np.bincount(ur, minlength=K).astype(np.int32),
                     **{f"u{j}_{k}": v for j, d in enumerate(utts) for k, v in d.items()}, n_utt=len(utts))
            index["fem"].append({"spk": s, "n_utt": len(utts)})
            if si % 50 == 0:
                print("fem", si, len(index["fem"]), flush=True)
    mrows = [r for r in json.loads((ROOT / "data/f0est_hi/manifest.json").read_text())["rows"] if r["ok"] and r["src"] in ("tts_m", "vctk_m") and r["split"] == "train"]
    rng = np.random.default_rng(0)
    pick = [mrows[i] for i in rng.permutation(len(mrows))[: a.n_male]]
    for i, r in enumerate(pick):
        try:
            x = load48(r["wav"])[: 12 * 48000]
            x = x[: len(x) // N.HOP * N.HOP]
            if len(x) < 200 * N.HOP:
                continue
            p = prod(x)
            np.savez(out / "male" / f"m{i:05d}.npz", **p)
            index["male"].append({"id": f"m{i:05d}", "spk": r["spk"], "src": r["src"]})
        except Exception:
            continue
        if i % 500 == 0:
            print("male", i, flush=True)
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print("done fem", len(index["fem"]), "male", len(index["male"]), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
