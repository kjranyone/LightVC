"""P0b(synth_nam_vc §8・学習なし): 描画器 rvoc3hi(リセット前の自前の道具・因果・48 kHz・界面 = 包絡 25 + log f0 + 有声 + 周期性 4)を P0a の前段の条件で測り直す。
条件 = PQ0(P0a の前段そのもの)・PQP(包絡に因果の 3 点の平滑 = ckpt の env_smooth)・PQH(帰属: PQP の因果の時刻のまま高さだけ harvest)・TCH(参考値: harvest の f0 + 中心揃えの CheapTrick と周期性 = rvoc3hi の学習の条件)。
組 = T(A0F の目標 24 声 × j 3〜5・本人らしさの写しの天井)・V(E1 の検証の 30 声 × t09)・R(本物の男声 12 × 10 s)・N(N0r の女声の音素の無い発声 84 本)。
物差し = ECAPA + LDA(T の top-1・元との cos)・UTMOS の対の差・large-v3 の仮名の CER(V・R)・TEX(N)・因果 log-mel の L1。高さの較正 = 自前の f0 対 harvest(参考値)。
事前登録 = results/snv_p0b/prereg.yaml。
P0d の測定(results/snv_p0d_rvoc/prereg.yaml): p0d=<ckpt> で rvoc3hi の PQP・PQ0 と P0d の EMA の PQ0(D0)を同じ組で並べる。vset=p1val は V を P1 の検証の 36 声の t09 にする(E1 の最後の 30 声は P0d の学習の声と重なる)。
    cd training && uv run --no-sync python snv_p0b_render.py
    cd training && uv run --no-sync python snv_p0b_render.py p0d=results/snv_p0d_rvoc/snap/ema_5k.pt vset=p1val
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

import nvoc as N
import pae as PA
import rvoc as R
import snv_p0_front as P
import train_rvoc as TR

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/snv_p0b"
A0 = ROOT / "results/snv_a0"
A1 = ROOT / "results/snv_a1"
E1 = ROOT / "data/snv_e1"
N0R = ROOT / "results/snv_n0r/wav"
CK = ROOT / "results/rvoc3hi/snap/ema_330k.pt"
CONDS = ("PQ0", "PQP", "PQH", "TCH")
DEV = "cuda"
SR = P.SR


def load48(p: Path) -> np.ndarray:
    return librosa.load(str(p), sr=SR, mono=True)[0].astype(np.float32)


def real_male() -> dict:
    from snv_a1_convert import segments
    out = {}
    for name, p in (("uirou", ROOT / "外郎売.mp3"), ("namikawa", ROOT / "namikawa.mp3")):
        _, tests, sr = segments(p)
        for i, seg in enumerate(tests):
            out[f"R|{name}_{i}"] = librosa.resample(seg.astype(np.float32), orig_sr=sr, target_sr=SR)
    return out


def harvest48(x: np.ndarray, T: int) -> np.ndarray:
    import pyworld
    x16 = librosa.resample(x.astype(np.float64), orig_sr=SR, target_sr=16000)
    f0, t = pyworld.harvest(x16, 16000, f0_floor=60, f0_ceil=1000, frame_period=P.H / SR * 1000)
    f0 = pyworld.stonemask(x16, f0, t, 16000)
    return np.pad(f0[:T], (0, max(0, T - len(f0)))).astype(np.float32)


def feats(job: tuple) -> tuple:
    key, x = job
    torch.set_num_threads(1)
    T = P.n_frames(len(x))
    x = x[: T * P.H]
    own = P.analyze(x)
    f0h = harvest48(x, T)
    xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
    env_t = PA.envelope(xa, f0h.astype(np.float64))[:, :T].astype(np.float32)
    per_t = PA.periodicity(torch.from_numpy(xa)[None], torch.from_numpy(f0h)[None])[0].numpy()[:, :T]
    hyb = {"f0": f0h, "env": P.envelope(x, f0h.astype(np.float64)), "per": P.periodicity(torch.from_numpy(x), torch.from_numpy(f0h)).numpy()}
    return key, x, own, {"f0": f0h, "env": env_t, "per": per_t, "hyb": hyb}


def center_smooth(e: np.ndarray, w: float) -> np.ndarray:
    q = np.pad(e, ((0, 0), (1, 1)), mode="edge")
    return (w * q[:, :-2] + (1 - 2 * w) * q[:, 1:-1] + w * q[:, 2:]).astype(np.float32)


def f0_calib(own: np.ndarray, ref: np.ndarray) -> dict:
    best = None
    for lag in range(0, 9):
        a, b = own[lag:], ref[: len(ref) - lag] if lag else ref
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        both = (a > 0) & (b > 0)
        vde = float(np.mean((a > 0) != (b > 0)))
        gpe = float(np.mean(np.abs(a[both] / b[both] - 1) > 0.2)) if both.any() else 1.0
        cents = float(np.median(np.abs(1200 * np.log2(a[both] / b[both])))) if both.any() else 0.0
        row = {"lag": lag, "vde": vde, "gpe": gpe, "cents_med": cents, "n_both": int(both.sum())}
        if best is None or vde + gpe < best["vde"] + best["gpe"]:
            best = row
    return best


def main() -> int:
    smoke = "smoke" in sys.argv[1:]
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    p0d, vset = opt.get("p0d"), opt.get("vset", "e1last30")
    out_dir = ROOT / "results/snv_p0d_rvoc" if p0d else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = [("PQP", CK, "PQP"), ("PQ0", CK, "PQ0"), ("D0", ROOT / p0d, "PQ0")] if p0d else [(c, CK, c) for c in CONDS]
    labels = tuple(r[0] for r in runs)
    meta = json.loads((A1 / "convert_meta.json").read_text())
    tgts = meta["targets"]
    voices = sorted(p.name for p in E1.iterdir() if p.is_dir())
    val = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())["val"] if vset == "p1val" else voices[-30:]
    n0r = [it for it in json.loads((N0R / "items.json").read_text()) if it["sex"] == "F"][::4]
    jobs = {}
    for v in tgts:
        for j in (3, 4, 5):
            jobs[f"T|{v}_{j}"] = load48(A0 / f"wav/{v}_{j}.wav")
    for v in val:
        jobs[f"V|{v}_t09"] = load48(E1 / v / "t09.wav")
    jobs.update(real_male())
    for it in n0r:
        n = f"{it['voice']}_{it['cat']}_{it['k']}"
        jobs[f"N|{n}"] = load48(N0R / f"{n}.wav")
    if smoke:
        keep = [k for k in jobs if k.startswith("T|pf000") or k.startswith("R|uirou_0") or k in list(jobs)[72:74] or k.startswith("N|pf000")][:12]
        jobs = {k: jobs[k] for k in keep}
    print("items", {p: sum(k.startswith(p + "|") for k in jobs) for p in "TVRN"}, flush=True)
    with ProcessPoolExecutor(12) as ex:
        F = {k: (x, own, tch) for k, x, own, tch in ex.map(feats, list(jobs.items()))}
    print("features done", flush=True)
    calib = {}
    for part in "TVR":
        rows = [f0_calib(F[k][1]["f0"], F[k][2]["f0"]) for k in F if k.startswith(part + "|")]
        if rows:
            calib[part] = {m: round(float(np.mean([r[m] for r in rows])), 4) for m in ("lag", "vde", "gpe", "cents_med")}
    print("f0 calib (own vs harvest)", calib, flush=True)

    smw = TR.ckpt_env_smooth(torch.load(CK, map_location="cpu", weights_only=False), None)
    W: dict = {}
    for k in F:
        W[(k, "ORI")] = F[k][0][: len(F[k][0]) - N.DELAY]
    for label, ck, kind in runs:
        st = torch.load(ck, map_location="cpu", weights_only=False)
        cfg = st["cfg"]
        gen = R.RVoc(ch=cfg["ch"], kernels=tuple(cfg["kernels"]), dils=tuple(cfg["dils"]), d_cond=cfg["d_cond"]).to(DEV)
        gen.load_state_dict(st["ema"])
        gen.eval().requires_grad_(False)

        @torch.no_grad()
        def render(c: np.ndarray, f0: np.ndarray) -> np.ndarray:
            ct = torch.from_numpy(c)[None].to(DEV)
            ex = TR.excitation(torch.from_numpy(f0.astype(np.float32))[None].to(DEV), ct.shape[-1] * N.HOP, torch.Generator().manual_seed(0))
            return gen(ct, ex)[0].cpu().numpy()[N.DELAY:].astype(np.float32)

        for k, (x, own, tch) in F.items():
            if kind == "PQ0":
                W[(k, label)] = render(P.cond(own), own["f0"])
            elif kind == "PQP":
                W[(k, label)] = render(P.cond(own, smooth=smw), own["f0"])
            elif kind == "PQH":
                W[(k, label)] = render(P.cond(tch["hyb"], smooth=smw), tch["f0"])
            else:
                W[(k, label)] = render(P.cond({"f0": tch["f0"], "env": center_smooth(tch["env"], smw), "per": tch["per"]}), tch["f0"])
        del gen
        torch.cuda.empty_cache()
    wd = out_dir / "wav"
    wd.mkdir(exist_ok=True)
    for k in [k for k in F if k.endswith("_3") or k.endswith("uirou_0") or k.endswith("namikawa_0")][:6]:
        for c in ("ORI",) + labels:
            sf.write(wd / f"{k.replace('|', '__')}__{c}.wav", W[(k, c)], SR)

    from snv_a0_measure import cer, kana, to16
    from snv_a0b_measure import proj
    from snv_n0_measure import trim
    from snv_n2c_tex import tex_dist, tex_stats
    from speechbrain.inference.speaker import EncoderClassifier
    y16 = {kc: to16(w.astype(np.float64), SR).astype(np.float32) for kc, w in W.items()}
    cm = N.CausalMel()
    lm = {}
    with torch.no_grad():
        for k in F:
            mo = cm(torch.from_numpy(W[(k, "ORI")])[None])
            for c in labels:
                mc = cm(torch.from_numpy(W[(k, c)])[None])
                n = min(mo.shape[-1], mc.shape[-1])
                lm[(k, c)] = float((mo[..., :n] - mc[..., :n]).abs().mean())
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})
    Tz = np.load(ROOT / "data/snv_e1_feat/ecapa_lda_T.npz")
    Tt = (Tz["mu"], Tz["scalings"])

    def E(w: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(w).float()[None].to(DEV)).reshape(-1)
        return proj(torch.nn.functional.normalize(e, dim=0).cpu().numpy(), Tt)

    Et = {v: E(np.concatenate([to16(load48(A0 / f"wav/{v}_{j}.wav").astype(np.float64), SR) for j in (0, 1, 2)]).astype(np.float32)) for v in tgts}
    Ew = {kc: E(trim(w)) for kc, w in y16.items() if not kc[0].startswith("N|")}
    del ec
    torch.cuda.empty_cache()
    top = lambda e: max(tgts, key=lambda u: float(e @ Et[u]))
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(DEV).eval()
    mos = {kc: float(utmos(torch.from_numpy(w).float()[None].to(DEV), 16000).item()) for kc, w in y16.items() if not kc[0].startswith("N|")}
    del utmos
    torch.cuda.empty_cache()
    from faster_whisper import WhisperModel
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16")

    def tr(w: np.ndarray) -> str:
        segs, _ = asr.transcribe(w, language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(s.text for s in segs))

    txt = {kc: tr(w) for kc, w in y16.items() if kc[0][0] in "VR"}
    del asr
    scale = np.load(ROOT / "results/snv_n2c/tex_scale.npy")
    tst = {k: tex_stats(y16[(k, "ORI")]) for k in F if k.startswith("N|")}
    res: dict = {"prereg": "results/snv_p0d_rvoc/prereg.yaml" if p0d else "results/snv_p0b/prereg.yaml", "runs": [[l_, str(Path(c_).relative_to(ROOT)), k_] for l_, c_, k_ in runs],
                 "vset": vset, "env_smooth": smw, "f0_calib_vs_harvest": calib, "n": {p: sum(k.startswith(p + "|") for k in F) for p in "TVRN"}, "by_cond": {}}
    for c in ("ORI",) + labels:
        row: dict = {}
        tk = [k for k in F if k.startswith("T|")]
        row["T_top1"] = round(float(np.mean([top(Ew[(k, c)]) == k[2:].rsplit("_", 1)[0] for k in tk])), 3)
        for part in "TVR":
            ks = [k for k in F if k.startswith(part + "|")]
            if c != "ORI":
                row[f"{part}_cos_to_orig"] = round(float(np.mean([Ew[(k, c)] @ Ew[(k, "ORI")] / (np.linalg.norm(Ew[(k, c)]) * np.linalg.norm(Ew[(k, "ORI")]) + 1e-9) for k in ks])), 3)
                row[f"{part}_utmos_diff"] = round(float(np.mean([mos[(k, c)] - mos[(k, "ORI")] for k in ks])), 3)
                row[f"{part}_logmel_l1"] = round(float(np.mean([lm[(k, c)] for k in ks])), 4)
            else:
                row[f"{part}_utmos"] = round(float(np.mean([mos[(k, c)] for k in ks])), 3)
            if part in "VR" and c != "ORI":
                row[f"{part}_cer"] = round(float(np.mean([cer(txt[(k, "ORI")], txt[(k, c)]) for k in ks])), 4)
        if c != "ORI":
            nk = [k for k in F if k.startswith("N|")]
            row["N_tex"] = round(float(np.mean([tex_dist(tex_stats(y16[(k, c)]), tst[k], scale) for k in nk])), 4)
        res["by_cond"][c] = row
        print(c, row, flush=True)
    name = (f"p0d_{Path(p0d).stem}_{vset}" if p0d else "p0b") + ("_smoke" if smoke else "")
    (out_dir / f"{name}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
