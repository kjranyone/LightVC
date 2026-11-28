"""段 P115(学習なし・耳の材料): 描画器を通さない描き方 = 操作者の音そのもの(高さだけ別の道具で上げたもの)に、「変換した声の形 ÷ その音の声の形」の利得を周波数ごとに掛ける。励起(声の鳴り・息・割れ)は操作者の音のまま残る。
比べる形: prod = 今の製品(K3G → 描画器)・shift = 高さだけ上げた操作者の音(位相ボコーダ・基準)・srcexc = shift に変換した声の形を掛けた音。本物の男声と合成の男声は半音 +12・悲鳴と笑いと喘ぎは +0(元から高い)。
    cd training && SNV_ESM=1 SNV_ENVC=1 SNV_WF16=1 SNV_KV16=1 SNV_DEV=cuda uv run --no-sync python snv_diag/p115_srcexc.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import p98_probe as PB
import p107_edges as E
import p114_hnr as H
import pae as PA
import snv_out_guard as OG
import snv_p1_eval as EV
import train_snv_p1 as T

ROOT = Path(__file__).resolve().parent.parent.parent
NFFT = 2048


def swap(xs: np.ndarray, env_s: np.ndarray, env_c: np.ndarray) -> np.ndarray:
    fb, dct = PA._np_consts()
    n = min(env_s.shape[1], env_c.shape[1])
    d = (env_c[:, :n] - env_s[:, :n]).astype(np.float64)
    d = np.concatenate([d[:, :1], d[:, :1], d], 1)
    d = (d[:, 2:] + d[:, 1:-1] + d[:, :-2]) / 3.0
    g_mel = np.exp(np.clip(dct[:25].T @ d, -3.5, 3.5))
    den = fb.sum(0)[:, None]
    g = np.where(den > 1e-9, (fb.T @ g_mel) / np.maximum(den, 1e-9), 1.0)
    X = librosa.stft(xs.astype(np.float32), n_fft=NFFT, hop_length=240, window="hann", center=True)
    m = min(X.shape[1], g.shape[1])
    Y = X.copy()
    Y[:, :m] = X[:, :m] * g[:, :m]
    Y[:, m:] = X[:, m:] * g[:, m - 1 : m]
    return librosa.istft(Y, hop_length=240, window="hann", center=True, length=len(xs)).astype(np.float32)


def main() -> None:
    out = ROOT / "results/snv_p115/ear"
    out.mkdir(parents=True, exist_ok=True)
    rd = lambda p: np.fromfile(p, np.float32)
    lw = lambda p: librosa.load(p, sr=EV.SR, mono=True)[0].astype(np.float32)
    S = EV.Ens(E.SYS, EV.load_gen())
    tg = [PB.utt(rd(p)) for p in sorted((PB.IN / "tg").glob("*.f32"))]
    st = S.structure("t", tg)
    p_out = OG.speech_level([u["wav"] for u in tg])
    sp3 = lambda v: [lw(ROOT / f"data/snv_e1/{v}/t0{j}.wav") for j in (0, 1, 2)]
    src = {f"real{i + 1}": (rd(PB.IN / sp / "in.f32"), [rd(p) for p in sorted((PB.IN / sp / "op").glob("*.f32"))], 12) for i, sp in enumerate(PB.SPS)}
    src["fam_em595"] = (lw(ROOT / "data/snv_e1/em595/t05.wav"), sp3("em595"), 7)
    for v, c in (("em595", "scream"), ("em596", "laugh"), ("em595", "moan"), ("em597", "cry")):
        src[f"{c}_{v}"] = (lw(ROOT / f"data/snv_e1nv/{v}/{c}_0.wav"), sp3(v), 0)
    res: dict = {}
    for name, (x, ops, k) in src.items():
        u, enr = PB.utt(x), [PB.utt(o) for o in ops]
        lv = OG.speech_level([e["wav"] for e in enr])
        w, yo = S.convert(u, enr, st, k, with_y=True)
        prod = OG.apply(u["wav"], np.asarray(w, np.float32), lv, p_out)
        xs = librosa.effects.pitch_shift(u["wav"], sr=EV.SR, n_steps=k, res_type="soxr_hq").astype(np.float32) if k else u["wav"].copy()
        us = PB.utt(xs)
        env_s = T.y_env(torch.from_numpy(us["y"][None, :25]))[0].numpy()
        env_c = T.y_env(yo[:, :25].detach().cpu())[0].numpy()
        y = swap(us["wav"], env_s, env_c)
        rms = lambda z: float(np.sqrt(np.mean(z ** 2)) + 1e-9)
        y = y * (rms(prod) / rms(y))
        sh = us["wav"] * (rms(prod) / rms(us["wav"]))
        r = {"k": k}
        for nm, z in (("prod", prod), ("shift", sh), ("srcexc", y)):
            z = np.clip(z, -0.99, 0.99)
            p = out / f"{name}_{nm}.wav"
            sf.write(p, z, EV.SR, subtype="PCM_16")
            r[nm] = H.measure(str(p))["hnr_med"]
        sf.write(out / f"{name}_src.wav", u["wav"], EV.SR, subtype="PCM_16")
        res[name] = r
        print(name, json.dumps(r), flush=True)
    sf.write(out / "target_ref.wav", tg[0]["wav"][: EV.SR * 12], EV.SR, subtype="PCM_16")
    (ROOT / "results/snv_p115/srcexc.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
