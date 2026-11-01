"""D395(results/conv_p0/d395_prereg.yaml): VCTK の女声 F を F 自身の条件(音量・高さ・有声・F の登録の表)で描画し、事後だけを
SELF = F 自身 / SWAP_F = 別の女声 F2 の同じ文 / SWAP_M = 男声 M の同じ文(C1 の事後の DTW で F の時刻へ写す)に替える。
log-mel L1 と WavLM-SV の本人らしさを F の実録音と比べ、SWAP_M 対 SWAP_F の 2 標本の判別(d326_pair.py)と v1⊥(eval_v1perp.py)を測る。
    uv run python d395_swap.py --set <dev の親>/d395 --ckpt ../results/diag_ppgvoc27/product_002500_m0.pt --out ../results/conv_p0/d395
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P

ROLES = ("SELF", "SELF_RT", "SWAP_F", "SWAP_M")


def warp(lf: np.ndarray, lx: np.ndarray, rt: bool = False) -> tuple[np.ndarray, float]:
    """X の事後を F の時刻へ写す: Hellinger の親和度のコストの DTW → F のフレームごとに対応する X のフレームの確率の平均 → log。
    rt = True: F 自身の事後を F → X → F の往復で写す(同じ経路で X の時刻へ写してから戻す = 写しの痕跡をそろえた自己の対照・D395b)。"""
    pf = torch.from_numpy(lf).softmax(1).numpy().astype(np.float64)
    px = torch.from_numpy(lx).softmax(1).numpy().astype(np.float64)
    C = 1.0 - np.sqrt(pf) @ np.sqrt(px).T
    _, wp = librosa.sequence.dtw(C=C, backtrack=True)
    src = px
    if rt:
        acc = np.zeros_like(px)
        cnt = np.zeros(len(px))
        for i, j in wp:
            acc[j] += pf[i]
            cnt[j] += 1
        src = acc / cnt[:, None]
    acc = np.zeros_like(pf)
    cnt = np.zeros(len(pf))
    for i, j in wp:
        acc[i] += src[j]
        cnt[i] += 1
    q = acc / cnt[:, None]
    agree = float((q.argmax(1) == pf.argmax(1)).mean())
    return np.log(q + 1e-8).astype(np.float32), agree


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--perm", type=int, default=200)
    ap.add_argument("--held", default="", help="pool.json(d396_swapprep.py)の held_d395 の目標だけを使う(学習の目標と重ならない保留)")
    a = ap.parse_args()
    dev = "cuda"
    S, O = Path(a.set), Path(a.out)
    for r in ROLES + ("REAL",):
        (O / r).mkdir(parents=True, exist_ok=True)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    P.slow_from_ckpt(st)
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    mels = [P.build_mel(nf, hm, nm).to(dev) for nf, hm, nm in P.MEL_SPECS]
    from transformers import WavLMForXVector
    sv = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(dev).eval()

    @torch.no_grad()
    def sv_emb(x48: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(resample_poly(x48, 16000, 48000).astype(np.float32))[None].to(dev)
        return torch.nn.functional.normalize(sv(input_values=x).embeddings, dim=-1)[0].cpu().numpy()

    trip = json.loads((S / "set.json").read_text())["triples"]
    if a.held:
        hs = set(json.loads(Path(a.held).read_text())["held_d395"])
        trip = [t for t in trip if t["F"] in hs]
        print("保留の目標だけ", len(hs), "人・組", len(trip), flush=True)
    ref_e: dict = {}
    rows = []
    for n, t in enumerate(trip):
        k = t["key"]
        cF, lF = P.load_dump(S / "dump", f"{k}_F")
        _, l2 = P.load_dump(S / "dump", f"{k}_F2")
        _, lM = P.load_dump(S / "dump", f"{k}_M")
        w2, ag2 = warp(lF, l2)
        wM, agM = warp(lF, lM)
        wR, agR = warp(lF, l2, rt=True)
        spk = P.load_enroll(S / "en" / t["F"])
        real, _ = sf.read(S / "wav" / f"{k}_F.wav", dtype="float32")
        nm = f"p{n:04d}_t{t['ti']:03d}"
        sf.write(O / "REAL" / f"{nm}_REAL.wav", real, N.SR)
        if t["F"] not in ref_e:
            rr, _ = sf.read(S / "ref" / f"ref_{t['F']}.wav", dtype="float32")
            ref_e[t["F"]] = sv_emb(rr)
        row = {"key": k, "ti": t["ti"], "F": t["F"], "M": t["M"], "agree_F2": ag2, "agree_M": agM, "agree_RT": agR}
        for role, lg in zip(ROLES, (lF, wR, w2, wM)):
            y = P.render(g, cF, lg, spk, dev)
            sf.write(O / role / f"{nm}_{role}.wav", y, N.SR)
            L = min(len(y), len(real))
            yy, rr = y[4800:L], real[4800:L]
            yy = yy * np.sqrt((rr ** 2).mean() / ((yy ** 2).mean() + 1e-12))
            with torch.no_grad():
                lm = float(P.logmel_l1(torch.from_numpy(yy)[None, None].to(dev), torch.from_numpy(rr)[None, None].to(dev), mels))
            row[f"lm_{role}"] = lm
            row[f"sv_{role}"] = float(sv_emb(y) @ ref_e[t["F"]])
        if n < 3:
            y = P.render(g, cF, lF, spk, dev)
            L = min(len(y), len(real)) - 2000
            sc = {}
            for sh in range(-960, 961, 120):
                yy, rr = y[4800 + sh:L + sh], real[4800:L]
                yy = yy * np.sqrt((rr ** 2).mean() / ((yy ** 2).mean() + 1e-12))
                with torch.no_grad():
                    sc[sh] = round(float(P.logmel_l1(torch.from_numpy(yy.copy())[None, None].to(dev), torch.from_numpy(rr.copy())[None, None].to(dev), mels)), 4)
            print("SELF 対 実録音の log-mel L1 のずれの走査(出力を前へ sh サンプル)", sc, flush=True)
        rows.append(row)
        if n % 20 == 0:
            print(n, {kk: round(v, 4) for kk, v in row.items() if isinstance(v, float)}, flush=True)
    rep: dict = {"n": len(rows)}
    tg = np.array([r["ti"] for r in rows])
    us = sorted(set(tg))
    idx = {u: np.flatnonzero(tg == u) for u in us}
    rng = np.random.default_rng(395)

    def boot(d: np.ndarray) -> dict:
        bs = [d[np.concatenate([idx[u] for u in rng.choice(us, len(us))])].mean() for _ in range(2000)]
        return {"diff": round(float(d.mean()), 4), "ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]}

    for m in ("lm", "sv"):
        rep[m] = {r: round(float(np.mean([x[f"{m}_{r}"] for x in rows])), 4) for r in ROLES}
        for a_, b_ in (("SWAP_M", "SWAP_F"), ("SWAP_F", "SELF_RT"), ("SWAP_M", "SELF_RT"), ("SELF_RT", "SELF"), ("SWAP_F", "SELF")):
            rep[m][f"{a_}-{b_}"] = boot(np.array([x[f"{m}_{a_}"] - x[f"{m}_{b_}"] for x in rows]))
    rep["agree"] = {k_: round(float(np.mean([r[f"agree_{k_}"] for r in rows])), 4) for k_ in ("F2", "M", "RT")}
    rep["agree"]["M-F2"] = boot(np.array([r["agree_M"] - r["agree_F2"] for r in rows]))
    (O / "rows.json").write_text(json.dumps(rows, indent=1))
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    T = Path(__file__).parent
    r = subprocess.run([sys.executable, str(T / "d326_pair.py"), "--pairs", f"SWAPM_vs_SWAPF={O}/SWAP_F/*.wav|{O}/SWAP_M/*.wav", f"SWAPF_vs_SELFRT={O}/SELF_RT/*.wav|{O}/SWAP_F/*.wav", f"SWAPM_vs_SELFRT={O}/SELF_RT/*.wav|{O}/SWAP_M/*.wav", f"SELFRT_vs_SELF={O}/SELF/*.wav|{O}/SELF_RT/*.wav",
                        "--perm", str(a.perm), "--out", str(O / "pair.json")], capture_output=True, text=True)
    print(r.stdout[-2000:], r.stderr[-1500:] if r.returncode else "", flush=True)
    r = subprocess.run([sys.executable, str(T / "eval_v1perp.py"), "--groups", *[f"{x}={O}/{x}/*.wav" for x in ("REAL",) + ROLES], "--out", str(O / "v1perp.json")], capture_output=True, text=True)
    print(r.stdout[-2500:], r.stderr[-1500:] if r.returncode else "", flush=True)
    rep["pair"] = json.loads((O / "pair.json").read_text()) if (O / "pair.json").exists() else None
    rep["v1perp"] = {k: v["v1perp"]["logit"] for k, v in json.loads((O / "v1perp.json").read_text()).items()} if (O / "v1perp.json").exists() else None
    (O / "summary.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
