"""P0c(synth_nam_vc §8.4・学習なし): 信号処理の一律のずらしの錨を運用の条件の test の物差しで測る(results/snv_p0c/prereg.yaml)。
錨 W = 操作者の包絡を周波数の一律の伸縮 α で目標へ(α = 操作者の登録の平均の包絡を目標の登録の平均へ当てはめる・格子 0.80〜1.50)+ 高さ = 操作者 × 2^(k/12)。
錨 WMS = W の後に平均を目標の登録の平均へずらす。描画 = rvoc3hi(P1 の物差しと同じ)。
物差し = snv_p1_eval.py test と同じ(本物の男声 12 区間 → A0F 24 声・ECAPA + LDA の 1 位・M_alt・源に近い率・UTMOS・k 0 / 7 × 4 声の large-v3 の仮名の CER)。
固定の α(alpha=1.12,1.20)= 当てはめを使わず一律の伸縮の典型値で(Graillon 型の錨・results/snv_p0c/anchor_fixed.json)。
    cd training && uv run --no-sync python snv_p0c_anchor.py [alpha=1.12,1.20]
"""
from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import torch

import nvoc as N
import pae as PA
import snv_p1_eval as EV
import train_snv_p1 as T

ROOT = Path(__file__).resolve().parent.parent
DEV = "cuda"
ALPHAS = np.round(np.arange(0.80, 1.501, 0.01), 2)
FIT_HZ = 8000.0


def mel_hz() -> np.ndarray:
    return librosa.mel_frequencies(n_mels=N.N_MEL + 2, fmin=0.0, fmax=N.FMAX)[1:-1]


def warp_env(c: np.ndarray, alpha: float) -> np.ndarray:
    _, dct = PA._np_consts()
    L = dct[:25].T @ c.astype(np.float64)
    fm = mel_hz()
    idx = np.interp(fm / alpha, fm, np.arange(len(fm)))
    i0 = np.clip(np.floor(idx).astype(int), 0, len(fm) - 2)
    w = (idx - i0)[:, None] if L.ndim == 2 else idx - i0
    Lw = L[i0] * (1 - w) + L[i0 + 1] * w
    return (dct[:25] @ Lw).astype(np.float32)


def fit_alpha(mu_op: np.ndarray, mu_t: np.ndarray) -> float:
    _, dct = PA._np_consts()
    band = mel_hz() <= FIT_HZ
    Lt = (dct[:25].T @ mu_t.astype(np.float64))[band]
    best, ba = None, 1.0
    for a in ALPHAS:
        Lw = (dct[:25].T @ warp_env(mu_op, float(a)).astype(np.float64))[band]
        d = float(np.mean(((Lw - Lw.mean()) - (Lt - Lt.mean())) ** 2))
        if best is None or d < best:
            best, ba = d, float(a)
    return ba


def selftest() -> None:
    c = np.random.default_rng(0).normal(0, 1, (25, 7)).astype(np.float32)
    assert np.allclose(warp_env(c, 1.0), c, atol=1e-4), "α = 1 で包絡が変わる"
    assert fit_alpha(c[:, 0], c[:, 0]) == 1.0, "同じ包絡に α ≠ 1"
    t = warp_env(c[:, 0], 1.2)
    assert abs(fit_alpha(c[:, 0], t) - 1.2) <= 0.02, "伸縮の当てはめが戻らない"


def main() -> int:
    import sys
    selftest()
    fixed = [float(x) for x in dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a).get("alpha", "").split(",") if x]
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, kana, load, to16
    from snv_a0b_measure import proj
    from snv_n0_measure import trim
    from speechbrain.inference.speaker import EncoderClassifier
    gen = EV.load_gen()
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})
    Tz = np.load(ROOT / "data/snv_e1_feat/ecapa_lda_T.npz")
    Tt = (Tz["mu"], Tz["scalings"])

    def emb(w16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(w16).float()[None].to(DEV)).reshape(-1)
        return proj(torch.nn.functional.normalize(e, dim=0).cpu().numpy(), Tt)

    w16 = lambda w: to16(w.astype(np.float64), EV.SR).astype(np.float32)
    lw = lambda p: to16(*load(p)).astype(np.float32)

    @torch.no_grad()
    def render(env: np.ndarray, u: dict, k: int) -> np.ndarray:
        f0 = u["f0"] * 2 ** (k / 12)
        v = (f0 > 0).astype(np.float32)
        y = T.to_y(env, np.clip(u["y"][25:29], 0, 1.5), f0)
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
        cond = np.concatenate([y[:25], lf[None], v[None], y[25:29]], 0)
        ct = torch.from_numpy(cond)[None].to(DEV)
        ex = EV.TR.excitation(torch.from_numpy(f0.astype(np.float32))[None].to(DEV), ct.shape[-1] * N.HOP, torch.Generator().manual_seed(0))
        return gen(ct, ex)[0].cpu().numpy()[N.DELAY:].astype(np.float32)

    meta = json.loads((EV.A1 / "convert_meta.json").read_text())
    tgts, srcs = meta["targets"], meta["sources"]
    Et = {v: emb(np.concatenate([lw(EV.A0 / f"wav/{v}_{j}.wav") for j in (0, 1, 2)])) for v in tgts}
    Et2 = {v: emb(np.concatenate([lw(EV.A0 / f"wav/{v}_{j}.wav") for j in (6, 7)])) for v in tgts}
    Es = {sp: emb(lw(T.FEAT / f"real/{sp}_enroll.wav")) for sp in ("uirou", "namikawa")}
    top = lambda e, ref: max(tgts, key=lambda u: float(e @ ref[u]))
    mu_t = {v: EV.mu_of([EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)])[0].cpu().numpy() for v in tgts}
    mu_o = {sp: EV.mu_of([EV.utt(f"real/{sp}_enroll")])[0].cpu().numpy() for sp in ("uirou", "namikawa")}
    alpha = {(sp, v): fit_alpha(mu_o[sp], mu_t[v]) for sp in mu_o for v in tgts}
    names = ("W", "WMS")
    if fixed:
        names = tuple(f"W{a:.2f}" for a in fixed)
    src_u = {s_: EV.utt(f"real/{s_}") for s_ in srcs}
    raw = {s_: np.load(T.FEAT / f"real/{s_}.npz")["env"] for s_ in srcs}
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16")

    def tr(w: np.ndarray) -> str:
        segs, _ = asr.transcribe(w, language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(x.text for x in segs))

    ref_txt = {s_: tr(lw(T.FEAT / f"real/{s_}.wav")) for s_ in srcs}
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(DEV).eval()
    mos = lambda w: float(utmos(torch.from_numpy(w).float()[None].to(DEV), 16000).item())
    res: dict = {"prereg": "results/snv_p0c/prereg.yaml", "alpha": {f"{sp}|{v}": a for (sp, v), a in alpha.items()},
                 "alpha_mean": {sp: round(float(np.mean([alpha[(sp, v)] for v in tgts])), 3) for sp in mu_o}, "systems": {}}
    print("alpha mean", res["alpha_mean"], flush=True)
    wd = ROOT / "results/snv_p0c/wav"
    wd.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    for name in names:
        al = (lambda sp, v: float(name[1:])) if fixed else (lambda sp, v: alpha[(sp, v)])
        row: dict = {}
        trials: dict = {}
        for k in EV.KS:
            hit, hit2, grp, sc, ms_ = [], [], [], [], []
            for s_ in srcs:
                sp = s_.rsplit("_", 1)[0]
                for v in tgts:
                    env = warp_env(raw[s_], al(sp, v))
                    if name == "WMS":
                        env = env - warp_env(mu_o[sp], al(sp, v))[:, None] + mu_t[v][:, None]
                    w = render(env, src_u[s_], k)
                    y16 = w16(w)
                    e = emb(trim(y16))
                    hit.append(top(e, Et) == v), hit2.append(top(e, Et2) == v), grp.append(v)
                    sc.append(float(e @ Es[sp]) > float(e @ Et[v]))
                    ms_.append(mos(y16))
                    if s_ in srcs[:1] and v in tgts[:2]:
                        sf.write(wd / f"{name}__{s_}__{v}__k{k}.wav", w, EV.SR)
            trials[f"k{k}"] = {"hit": [int(h) for h in hit], "target": grp}
            row[f"M_k{k}"] = round(float(np.mean(hit)), 4)
            row[f"M_alt_k{k}"] = round(float(np.mean(hit2)), 4)
            row[f"utmos_k{k}"] = round(float(np.mean(ms_)), 3)
            row[f"src_closer_k{k}"] = round(float(np.mean(sc)), 4)
        row["M_mean_4k"] = round(float(np.mean([row[f"M_k{k}"] for k in EV.KS])), 4)
        cs = []
        for k in (0, 7):
            for s_ in srcs:
                sp = s_.rsplit("_", 1)[0]
                for v in tgts[:4]:
                    env = warp_env(raw[s_], al(sp, v))
                    if name == "WMS":
                        env = env - warp_env(mu_o[sp], al(sp, v))[:, None] + mu_t[v][:, None]
                    cs.append(cer(ref_txt[s_], tr(w16(render(env, src_u[s_], k)))))
        row["cer_k0_k7"] = round(float(np.mean(cs)), 4)
        res["systems"][name] = row
        res.setdefault("trials", {})[name] = trials
        print(name, json.dumps(row), flush=True)
    (ROOT / ("results/snv_p0c/anchor_fixed.json" if fixed else "results/snv_p0c/anchor.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
