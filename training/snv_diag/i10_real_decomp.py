"""段 I10 の後の本物の音声の崩れの分解(学習なし)。mode ごとに results/snv_i10/decomp_<mode>.json へ。
ood = 推論の E の入力 xin(y, v) の 1 近傍の距離(参照 = 学習の声の発話 400 本 の元の音 / rec = 同じ発話の録音の版 k0・k1 + 元の音 130 本 = e_rec が見た分布)を有声と無声のフレームごとに・JVNV・本物の区間・検証の Irodori。
cpc = P3 の演奏の表現(results/snv_p2b_cpc の CPC の文脈 256)で同じ 1 近傍の距離(参照 300 本)。
voice = ECAPA(spkrec-ecapa-voxceleb・LDA なし)で学習の声の最も近い声の余弦(各声 t03〜t05)・学習の男声の数 25 / 50 / 100 を 20 回引いた平均。
blur = 物差しの脆さ: 写しの描画の包絡 25 を時間に幅 w フレームで平らにしたときの CER(元の写しの書き起こしに対して)・w = 3 / 5 / 9 / 15。
rate = 有声の 1 秒あたりの c0 の山の数・有声の終わりから c0 が 10 下がるまでの ms。
estruct = JVNV → A0F の目標 k0 で、推論の E の操作者の構造だけを 自分の登録 / 別の JVNV の話者 / 検証の Irodori の男声 2 声 に替える(段 I8・段 I9 の乱しあり)。
resid = 推論の E の内容の出力の声ごとの有声の平均を、学習の男声 114 声の分布で標準化した z(本物の声の外れ)と、検証の Irodori の男声に入力側の乱し(伸縮・傾き・音色の余弦 3 つ・周期性・ケプストラムの重み・音量)を掛けたときの z の動きの向きで、本物の声の外れを最小二乗で説明する率。
rec= = 録音の版の特徴の場所(既定 data/snv_p_feat_rec・段 I12 の本物の音源の族 = data/snv_p_feat_rec2)。
    cd training && uv run --no-sync python snv_diag/i10_real_decomp.py mode=ood|cpc|voice|blur|rate|estruct|resid|warpfit [rec=data/snv_p_feat_rec2] [out=...]
"""
from __future__ import annotations

import glob
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_snv_p1 as T

ROOT = Path(__file__).resolve().parents[2]
REC = ROOT / "data/snv_p_feat_rec"
JV = ROOT / "data/jvnv/jvnv_v1"
EMOS = ("anger", "disgust", "fear", "happy", "sad", "surprise")
REAL = [f"uirou_{i}" for i in range(6)] + ["uirou_enroll"] + [f"namikawa_{i}" for i in range(6)] + ["namikawa_enroll"]


def split() -> dict:
    return json.loads((ROOT / "results/snv_i8_m13/split.json").read_text())


def groups() -> dict:
    sp = split()
    val = [p for v in sp["val"][:12] for p in sorted((T.FEAT / "e1" / v).glob("t*.npz"))[3:5]]
    return {"irodori_val": [T.load_utt(p) for p in val],
            "irodori_val_rec_k0": [T.load_utt(REC / "e1" / p.parent.name / f"{p.stem}_k0.npz") for p in val],
            "jvnv_M": [T.load_utt(p) for p in sorted((T.FEAT / "jvnv").glob("M*_regular_01.npz"))]} | {n: [T.load_utt(T.FEAT / f"real/{n}.npz")] for n in REAL}


def knn_table(feat, ref_utts: list[dict]) -> dict:
    import i10_real_ood as O
    R = torch.cat([feat(u) for u in ref_utts])
    mu, sd = R.mean(0), R.std(0) + 1e-3
    R = (R - mu) / sd
    out = {}
    for g, us in groups().items():
        r = []
        for u in us:
            d = O.nn_dist((feat(u) - mu) / sd, R).cpu().numpy()
            vv = u["v"] > 0.5
            r.append([float(np.median(d)), float(np.median(d[vv])) if vv.any() else np.nan, float(np.median(d[~vv])) if (~vv).any() else np.nan,
                      float(np.median(u["y"][0][~vv])) if (~vv).any() else np.nan])
        out[g] = dict(zip(("all", "voiced", "unvoiced", "c0_unvoiced"), [round(float(np.nanmedian([x[i] for x in r])), 3) for i in range(4)]))
        print(g, out[g], flush=True)
    return out


def mode_ood() -> dict:
    import i10_real_ood as O
    rng = random.Random(0)
    tr = [p for v in split()["train"] for p in sorted((T.FEAT / "e1" / v).glob("t*.npz"))]
    pick = rng.sample(tr, 400)
    feat = lambda u: torch.from_numpy(O.xin_np(u)).to(O.DEV)
    clean = knn_table(feat, [T.load_utt(p) for p in pick])
    recd = knn_table(feat, [T.load_utt(REC / p.relative_to(T.FEAT).with_name(f"{p.stem}_k{i % 2}.npz")) for i, p in enumerate(pick)] + [T.load_utt(p) for p in pick[:130]])
    return {"ref_clean": clean, "ref_rec": recd}


def mode_cpc() -> dict:
    import i10_real_ood as O
    import train_snv_p2_cpc as CP
    st = torch.load(ROOT / "results/snv_p2b_cpc/last.pt", map_location="cpu", weights_only=False)
    cpc = CP.CPC()
    cpc.load_state_dict(st["cpc"])
    cpc = cpc.to(O.DEV).eval()
    sig = torch.from_numpy(np.asarray(st["sig"], np.float32)).to(O.DEV)

    @torch.no_grad()
    def feat(u: dict) -> torch.Tensor:
        yv = torch.from_numpy(np.concatenate([u["y"], u["v"][None]], 0))[None].to(O.DEV)
        return cpc.encode(CP.x_torch(yv, sig))[1][0].T

    rng = random.Random(0)
    tr = [p for v in split()["train"] for p in sorted((T.FEAT / "e1" / v).glob("t*.npz"))]
    return {"ref_clean": knn_table(feat, [T.load_utt(p) for p in rng.sample(tr, 300)])}


def mode_voice() -> dict:
    import snv_p1_eval as EV
    from snv_a0_measure import load, to16
    from speechbrain.inference.speaker import EncoderClassifier
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": EV.DEV})
    lw = lambda p: to16(*load(p)).astype(np.float32)

    def emb(ws: list[np.ndarray]) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(np.concatenate(ws)).float()[None].to(EV.DEV)).reshape(-1)
        return torch.nn.functional.normalize(e, dim=0).cpu().numpy()

    sp = split()
    tr = {v: emb([lw(ROOT / f"data/snv_e1/{v}/t0{j}.wav") for j in (3, 4, 5)]) for v in sp["train"]}
    va = {v: emb([lw(ROOT / f"data/snv_e1/{v}/t0{j}.wav") for j in (3, 4, 5)]) for v in sp["val"]}
    q = {f"jvnv/{s}": emb([lw(p) for p in sorted(glob.glob(str(JV / s / "*" / "regular" / "*_03.wav")))[:4]]) for s in ("M1", "M2", "F1", "F2")}
    q |= {f"real/{s}_enroll": emb([lw(ROOT / f"data/snv_p_feat/real/{s}_enroll.wav")]) for s in ("uirou", "namikawa")}
    mn = [n for n in tr if n.startswith("em")]
    M = np.stack([tr[n] for n in mn])
    out: dict = {"n_train_male": len(mn), "val_male_nearest_male": round(float(np.median([(M @ e).max() for v, e in va.items() if v.startswith("em")])), 3),
                 "nearest_male": {k: round(float((M @ e).max()), 3) for k, e in q.items()}, "subsets": {}}
    rng = random.Random(0)
    for n in (25, 50, 100):
        acc: dict = {}
        for _ in range(20):
            Ms = np.stack([tr[x] for x in rng.sample(mn, n)])
            for k, e in q.items():
                acc.setdefault(k, []).append(float((Ms @ e).max()))
            acc.setdefault("val_male_median", []).append(float(np.median([(Ms @ e).max() for v, e in va.items() if v.startswith("em")])))
        out["subsets"][n] = {k: round(float(np.mean(v)), 3) for k, v in acc.items()}
    print(json.dumps(out, indent=1), flush=True)
    return out


def mode_blur() -> dict:
    from scipy.ndimage import uniform_filter1d
    import nvoc as N
    import snv_p1_eval as EV
    import train_rvoc as TR
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, kana, to16
    asr = WhisperModel("large-v3", device=EV.DEV, compute_type="float16" if EV.DEV == "cuda" else "int8")
    tr = lambda w: kana("".join(x.text for x in asr.transcribe(to16(w.astype(np.float64), EV.SR).astype(np.float32), language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)[0]))
    gen = EV.load_gen()

    @torch.no_grad()
    def copy(u: dict, w: int = 1) -> np.ndarray:
        y = u["y"].copy()
        if w > 1:
            y[:25] = uniform_filter1d(y[:25], w, axis=1, mode="nearest")
        y, lf, v = torch.from_numpy(y)[None].to(EV.DEV), torch.from_numpy(u["lf"])[None].to(EV.DEV), torch.from_numpy(u["v"])[None].to(EV.DEV)
        cond = torch.cat([y[:, :25], lf[:, None], v[:, None], y[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
        f0 = torch.from_numpy(u["f0"])[None].to(EV.DEV)
        return gen(cond, TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:].astype(np.float32)

    sp = split()
    gs = {"irodori_val": [T.load_utt(p) for v in sp["val"][:12] for p in sorted((T.FEAT / "e1" / v).glob("t*.npz"))[3:4]],
          "jvnv": [T.load_utt(p) for p in sorted((T.FEAT / "jvnv").glob("*_regular_01.npz"))],
          "uirou": [T.load_utt(T.FEAT / f"real/uirou_{i}.npz") for i in range(6)],
          "namikawa_3to5": [T.load_utt(T.FEAT / f"real/namikawa_{i}.npz") for i in (3, 4, 5)]}
    out = {}
    for g, us in gs.items():
        refs = [tr(copy(u)) for u in us]
        out[g] = {}
        for w in (3, 5, 9, 15):
            c = [cer(r, tr(copy(u, w))) for r, u in zip(refs, us)]
            out[g][w] = {"median": round(float(np.median(c)), 3), "mean": round(float(np.mean(c)), 3)}
        print(g, out[g], flush=True)
    return out


def mode_rate() -> dict:
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks

    def m(u: dict) -> tuple[float, float]:
        c0, v = u["y"][0] * 10, u["v"] > 0.5
        pk, _ = find_peaks(np.where(v, uniform_filter1d(c0, 5), -1e9), prominence=3.0, distance=12)
        sl = []
        for t in np.flatnonzero(v[:-1] & ~v[1:]):
            seg = c0[t:t + 30]
            if len(seg) == 30 and seg[0] - seg.min() > 10:
                sl.append(float(np.argmax(seg <= seg[0] - 10)) * 5.0)
        return len(pk) / (v.sum() / 200.0 + 1e-9), float(np.median(sl)) if sl else float("nan")

    out = {}
    for g, us in groups().items():
        if g == "irodori_val_rec_k0":
            continue
        r = [m(u) for u in us]
        out[g] = {"peaks_per_voiced_s": round(float(np.median([x[0] for x in r])), 2), "offset_fall_ms": round(float(np.nanmedian([x[1] for x in r])), 1)}
        print(g, out[g], flush=True)
    return out


def mode_estruct() -> dict:
    import nvoc as N
    import snv_p1_eval as EV
    import train_rvoc as TR
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, kana, to16
    asr = WhisperModel("large-v3", device=EV.DEV, compute_type="float16" if EV.DEV == "cuda" else "int8")
    tr = lambda w: kana("".join(x.text for x in asr.transcribe(to16(w.astype(np.float64), EV.SR).astype(np.float32), language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)[0]))
    gen = EV.load_gen()
    meta = json.loads((EV.A1 / "convert_meta.json").read_text())
    ftg = meta["targets"][:2]
    males = [v for v in split()["val"] if v.startswith("em")]
    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in ftg}
    irm = {v: [EV.utt(f"e1/{v}/{p.stem}") for p in sorted((T.FEAT / "e1" / v).glob("t*.npz"))[:3]] for v in males[2:4]}
    jv = {s: [T.load_utt(T.FEAT / f"jvnv/{s}_{e}_regular_02.npz") for e in EMOS] for s in ("M1", "M2")}
    systems = [EV.System(t, gen) for t in ("snv_i8_m13", "snv_i9_aug")]
    rows = []
    for s_ in ("M1", "M2"):
        for e in EMOS:
            u = T.load_utt(T.FEAT / f"jvnv/{s_}_{e}_regular_01.npz")
            y, lf, v = (torch.from_numpy(u[k])[None].to(EV.DEV) for k in ("y", "lf", "v"))
            with torch.no_grad():
                cond = torch.cat([y[:, :25], lf[:, None], v[:, None], y[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
                ref = tr(gen(cond, TR.excitation(torch.from_numpy(u["f0"])[None].to(EV.DEV), cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:].astype(np.float32))
            for s in systems:
                for name, st in (("own", jv[s_]), ("other_jvnv", jv["M2" if s_ == "M1" else "M1"]), ("irodori_m_a", irm[males[2]]), ("irodori_m_b", irm[males[3]])):
                    for tgt in ftg:
                        rows.append({"sys": s.tag, "estruct": name, "src": f"{s_}_{e}", "tgt": tgt, "cer": cer(ref, tr(s.convert(u, st, s.structure(f"t|{tgt}", trefs[tgt]), 0)))})
    out = {s.tag: {n: {"median": round(float(np.median([r["cer"] for r in rows if r["sys"] == s.tag and r["estruct"] == n])), 3),
                       "mean_cap1": round(float(np.mean([min(r["cer"], 1.0) for r in rows if r["sys"] == s.tag and r["estruct"] == n])), 3)}
                   for n in ("own", "other_jvnv", "irodori_m_a", "irodori_m_b")} for s in systems} | {"rows": rows}
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1), flush=True)
    return out


def mode_resid(opt_tags: tuple = ("snv_i8_m13", "snv_i9_aug", "snv_i10_e_rec")) -> dict:
    import snv_p1_eval as EV
    import train_snv_p11 as P11
    import train_snv_p2_cpc as P2
    dev = EV.DEV
    Ms, tv = torch.from_numpy(P2.warp_mats()).to(dev), torch.from_numpy(P2.tilt_vec()).to(dev)
    eqb = torch.from_numpy(P2.eq_basis()).to(dev)
    alphas = np.asarray(P2.ALPHAS)

    def cat(us: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(np.concatenate([u["y"] for u in us], 1))[None].to(dev), torch.from_numpy(np.concatenate([u["v"] for u in us]))[None].to(dev)

    def pert(y: torch.Tensor, v: torch.Tensor, kind: str, p: float) -> torch.Tensor:
        if kind == "warp":
            return P2.augment(torch.cat([y, v[:, None]], 1), Ms, tv, None, fixed=(torch.tensor([int(np.argmin(np.abs(alphas - p)))], device=dev), torch.zeros(1, device=dev)))[:, :T.D_Y]
        if kind == "tilt":
            return P2.augment(torch.cat([y, v[:, None]], 1), Ms, tv, None, fixed=(torch.tensor([int(np.argmin(np.abs(alphas - 1.0)))], device=dev), torch.full((1,), p, device=dev)))[:, :T.D_Y]
        if kind.startswith("eq"):
            env = torch.cat([y[:, :1] * 10 + T.PA.C0_SIL, y[:, 1:25] * 10], 1) + p * eqb[int(kind[2:])][None, :, None]
            return torch.cat([(env[:, :1] - T.PA.C0_SIL) / 10, env[:, 1:25] / 10, y[:, 25:]], 1)
        if kind == "per":
            return torch.cat([y[:, :25], y[:, 25:] * p], 1)
        if kind == "lifter":
            w = 1.0 + (p - 1.0) * torch.linspace(0, 1, 24, device=dev)
            return torch.cat([y[:, :1], y[:, 1:25] * w[None, :, None], y[:, 25:]], 1)
        if kind == "level":
            return torch.cat([y[:, :1] + p, y[:, 1:]], 1)
        raise ValueError(kind)

    out: dict = {}
    for tag in opt_tags:
        st = torch.load(ROOT / "results" / tag / "last.pt", map_location="cpu", weights_only=False)
        E = P11.EncS().to(dev).eval()
        E.load_state_dict(st["Es"])

        @torch.no_grad()
        def vmean(us: list[dict], kind: str | None = None, p: float = 0.0) -> np.ndarray:
            y, v = cat(us)
            if kind is not None:
                y = pert(y, v, kind, p)
            x = P11.xin(y, v)
            a = E(x, E.structure(x, torch.ones(v.shape, dtype=torch.bool, device=dev)))[0][0, :59]
            return a[:, v[0] > 0.5].mean(1).cpu().numpy()

        sp = split()
        pop = np.stack([vmean([T.load_utt(p) for p in sorted((T.FEAT / "e1" / v).glob("t*.npz"))[:4]]) for v in sp["train"] if v.startswith("em")])
        mu, sd = pop.mean(0), pop.std(0) + 1e-4
        z = lambda m: (m - mu) / sd
        valm = [v for v in sp["val"] if v.startswith("em")]
        val_us = {v: [T.load_utt(p) for p in sorted((T.FEAT / "e1" / v).glob("t*.npz"))[:4]] for v in valm}
        real = {f"jvnv/{s}": [T.load_utt(T.FEAT / f"jvnv/{s}_{e}_regular_02.npz") for e in EMOS] for s in ("M1", "M2")} | {f"real/{s}_enroll": [T.load_utt(T.FEAT / f"real/{s}_enroll.npz")] for s in ("uirou", "namikawa")}
        zr = {k: z(vmean(us)) for k, us in real.items()}
        base = {v: vmean(us) for v, us in val_us.items()}
        kinds = [("warp", 0.85), ("warp", 1.18), ("tilt", -1.0), ("tilt", 1.0), ("eq0", 0.5), ("eq1", 0.5), ("eq2", 0.5), ("per", 0.6), ("lifter", 0.8), ("lifter", 1.2), ("level", -1.0)]
        dirs = {f"{k}{p}": np.mean([(vmean(val_us[v], k, p) - base[v]) / sd for v in valm], 0) for k, p in kinds}
        Dm = np.stack(list(dirs.values()), 1)
        res = {"z_norm_val_male_median": round(float(np.median([np.linalg.norm(z(b)) for b in base.values()])), 3), "real": {}}
        for k, d in zr.items():
            beta, *_ = np.linalg.lstsq(Dm, d, rcond=None)
            r2 = 1 - float(((d - Dm @ beta) ** 2).sum() / (d ** 2).sum())
            r2wt = 0.0
            sub = [i for i, n in enumerate(dirs) if n.startswith(("warp", "tilt"))]
            b2, *_ = np.linalg.lstsq(Dm[:, sub], d, rcond=None)
            r2wt = 1 - float(((d - Dm[:, sub] @ b2) ** 2).sum() / (d ** 2).sum())
            res["real"][k] = {"z_norm": round(float(np.linalg.norm(d)), 3), "r2_all_dirs": round(r2, 3), "r2_warp_tilt": round(r2wt, 3),
                              "cos": {n: round(float(d @ dv / (np.linalg.norm(d) * np.linalg.norm(dv) + 1e-9)), 3) for n, dv in dirs.items()}}
        res["dir_norms"] = {n: round(float(np.linalg.norm(dv)), 3) for n, dv in dirs.items()}
        out[tag] = res
        print(tag, json.dumps({k: v for k, v in res.items()}, ensure_ascii=False), flush=True)
    return out


def feats_jv(name: str) -> dict:
    import librosa
    import snv_p0_front as P
    dst = T.FEAT / "jvnv" / f"{name}.npz"
    spk, emo = name.split("_")[:2]
    x = librosa.load(str(JV / spk / emo / "regular" / f"{name}.wav"), sr=P.SR, mono=True)[0].astype(np.float32)
    n = P.n_frames(len(x))
    f = P.analyze(x[: n * P.H])
    np.savez(dst, **{k: v.astype(np.float32) for k, v in f.items()})
    return T.load_utt(dst)


def mode_warpfit(opt: dict) -> dict:
    import nvoc as N
    import snv_p1_eval as EV
    import train_rvoc as TR
    import train_snv_p11 as P11
    import train_snv_p2_cpc as P2
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, kana, to16
    dev = EV.DEV
    lo, hi = float(opt.get("alo", 0.8)), float(opt.get("ahi", 1.25))
    Ms, tv = torch.from_numpy(P2.warp_mats(lo, hi)).to(dev), torch.from_numpy(P2.tilt_vec()).to(dev)
    alphas = np.asarray(P2.ALPHAS) if (lo, hi) == (0.8, 1.25) else np.exp(np.linspace(np.log(lo), np.log(hi), 33))
    tgrid = tuple(np.round(np.arange(-float(opt.get("tmax", 1.0)), float(opt.get("tmax", 1.0)) + 1e-6, 0.5), 2).tolist())
    lgrid = tuple(np.round(np.arange(-float(opt.get("lmax", 1.0)), float(opt.get("lmax", 1.0)) + 1e-6, 0.5), 2).tolist())
    rounds = int(opt.get("rounds", 1))
    popsex = opt.get("pop", "em")
    tags = opt.get("tags", "snv_i8_m13,snv_i9_aug").split(",")
    asr = WhisperModel("large-v3", device=dev, compute_type="float16" if dev == "cuda" else "int8")
    tr = lambda w: kana("".join(x.text for x in asr.transcribe(to16(w.astype(np.float64), EV.SR).astype(np.float32), language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)[0]))
    gen = EV.load_gen()

    def tf(y: torch.Tensor, v: torch.Tensor, p: tuple) -> torch.Tensor:
        ia, tau, lev = p
        yw = P2.augment(torch.cat([y, v[:, None]], 1), Ms, tv, None, fixed=(torch.tensor([ia], device=dev), torch.full((1,), tau, device=dev)))[:, :T.D_Y]
        return torch.cat([yw[:, :1] + lev, yw[:, 1:]], 1)

    def tt(u: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(u["y"])[None].to(dev), torch.from_numpy(u["v"])[None].to(dev)

    def cat(us: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(np.concatenate([u["y"] for u in us], 1))[None].to(dev), torch.from_numpy(np.concatenate([u["v"] for u in us]))[None].to(dev)

    @torch.no_grad()
    def render(y: torch.Tensor, lf: torch.Tensor, v: torch.Tensor, f0: np.ndarray) -> np.ndarray:
        cond = torch.cat([y[:, :25], lf[:, None], v[:, None], y[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
        return gen(cond, TR.excitation(torch.from_numpy(f0)[None].to(dev), cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:].astype(np.float32)

    sp = split()
    meta = json.loads((EV.A1 / "convert_meta.json").read_text())
    males = [v for v in sp["val"] if v.startswith("em")]
    tg = {"F": meta["targets"][:2], "M": males[:2]}
    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tg["F"]} | {v: [EV.utt(f"e1/{v}/{p.stem}") for p in sorted((T.FEAT / "e1" / v).glob("t*.npz"))[:3]] for v in tg["M"]}
    i1 = int(np.argmin(np.abs(alphas - 1.0)))
    out: dict = {}
    for tag in tags:
        s = EV.System(tag, gen)
        E = s.Es

        @torch.no_grad()
        def emean(us: list[dict], p: tuple) -> np.ndarray:
            y, v = cat(us)
            x = P11.xin(tf(y, v, p), v)
            a = E(x, E.structure(x, torch.ones(v.shape, dtype=torch.bool, device=dev)))[0][0, :59]
            return a[:, v[0] > 0.5].mean(1).cpu().numpy()

        pop = np.stack([emean([T.load_utt(p) for p in sorted((T.FEAT / "e1" / v).glob("t*.npz"))[:4]], (i1, 0.0, 0.0)) for v in sp["train"] if v.startswith(popsex)])
        mu, sd = pop.mean(0), pop.std(0) + 1e-4
        zn = lambda us, p: float(np.linalg.norm((emean(us, p) - mu) / sd))
        res: dict = {}
        rows = []
        for spk in ("M1", "M2"):
            en = [T.load_utt(T.FEAT / f"jvnv/{spk}_{e}_regular_{opt.get('enr', '02')}.npz") if (T.FEAT / f"jvnv/{spk}_{e}_regular_{opt.get('enr', '02')}.npz").exists() else feats_jv(f"{spk}_{e}_regular_{opt.get('enr', '02')}") for e in EMOS]
            ia, tau, lev = i1, 0.0, 0.0
            for _ in range(rounds):
                ia = min(range(len(alphas)), key=lambda i: zn(en, (i, tau, lev)))
                tau = min(tgrid, key=lambda t: zn(en, (ia, t, lev)))
                lev = min(lgrid, key=lambda l: zn(en, (ia, tau, l)))
            ia_w = min(range(len(alphas)), key=lambda i: zn(en, (i, 0.0, 0.0)))
            fits = {"base": (i1, 0.0, 0.0), "warp": (ia_w, 0.0, 0.0), "all": (ia, tau, lev)}
            res[spk] = {"alpha": round(float(alphas[ia]), 3), "alpha_warp_only": round(float(alphas[ia_w]), 3), "tilt": tau, "level_y0": lev, "z_base": round(zn(en, fits["base"]), 3), "z_warp": round(zn(en, fits["warp"]), 3), "z_all": round(zn(en, fits["all"]), 3)}
            print(tag, spk, res[spk], flush=True)
            for e in EMOS:
                u = T.load_utt(T.FEAT / f"jvnv/{spk}_{e}_regular_01.npz")
                y, v = tt(u)
                lf = torch.from_numpy(u["lf"])[None].to(dev) * v
                ref = tr(render(y, lf, v, u["f0"]))
                for name, p in fits.items():
                    ry, rv = cat(en)
                    rx = P11.xin(tf(ry, rv, p), rv)
                    with torch.no_grad():
                        a = E(P11.xin(tf(y, v, p), v), E.structure(rx, torch.ones(rv.shape, dtype=torch.bool, device=dev)))[0]
                        for side, ts in tg.items():
                            for t in ts:
                                yo = s.C(a, lf, v, s.structure(f"t|{t}", trefs[t]))
                                rows.append({"spk": spk, "src": e, "fit": name, "side": side, "tgt": t, "cer": cer(ref, tr(render(yo, lf, v, u["f0"])))})
        for side in ("F", "M"):
            for name in ("base", "warp", "all"):
                c = [r["cer"] for r in rows if r["side"] == side and r["fit"] == name]
                res[f"{side}_k0|{name}"] = {"median": round(float(np.median(c)), 3), "mean_cap1": round(float(np.mean(np.minimum(c, 1.0))), 3)}
        print(tag, {k: v for k, v in res.items() if "|" in k}, flush=True)
        out[tag] = res | {"rows": rows}
    return out


def main() -> int:
    global REC
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    REC = ROOT / opt.get("rec", "data/snv_p_feat_rec")
    mode = opt.get("mode", "ood")
    res = {"ood": mode_ood, "cpc": mode_cpc, "voice": mode_voice, "blur": mode_blur, "rate": mode_rate, "estruct": mode_estruct, "resid": mode_resid, "warpfit": lambda: mode_warpfit(opt)}[mode]()
    (ROOT / opt.get("out", f"results/snv_i10/decomp_{mode}.json")).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
