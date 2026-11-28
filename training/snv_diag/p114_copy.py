"""段 P114(学習なし): 描画器の写しの誤差を、「倍の周期の成分が強いフレーム」とそうでないフレームで分けて測る。高さを前段の読み(製品)と pYIN(オフライン・診断)の 2 通りで描く。
誤差 = 元の音と写しの対数メル(80 帯・hop 240)の差の絶対値の平均(dB)。合成の検証の声(男 4・女 4)× 通常の発話・笑い・喘ぎ・泣き・悲鳴。
    cd training && SNV_ESM=1 SNV_DEV=cuda uv run --no-sync python snv_diag/p114_copy.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import p98_probe as PB
import snv_p0_front as P
import snv_p1_eval as EV

ROOT = Path(__file__).resolve().parent.parent.parent


def lmel(w: np.ndarray) -> np.ndarray:
    m = librosa.feature.melspectrogram(y=w.astype(np.float32), sr=EV.SR, n_fft=2048, hop_length=240, win_length=1200, n_mels=80, fmax=12000, center=False)
    return 10 * np.log10(m + 1e-9)


def acf_err(x: np.ndarray, w: np.ndarray, r: np.ndarray, n: int) -> np.ndarray:
    e = np.full(n, np.nan, np.float32)
    for t in range(4, n - 4):
        if r[t] <= 0:
            continue
        T0 = int(round(EV.SR / r[t]))
        L = int(2.6 * T0)
        c = t * 240 + 120
        a, b = c - 2 * L, c + 2 * L
        if a < 0 or b > min(len(x), len(w)) or T0 < 16:
            continue
        def acf(z: np.ndarray) -> np.ndarray:
            z = z[a:b] - z[a:b].mean()
            f = np.fft.rfft(z, 2 * len(z))
            q = np.fft.irfft(f * f.conj())[: L + 1]
            return q[T0 // 3 :] / (q[0] + 1e-12)
        qa, qb = acf(x), acf(w)
        if np.std(qa) < 1e-6 or np.std(qb) < 1e-6:
            continue
        e[t] = 1.0 - float(np.corrcoef(qa, qb)[0, 1])
    return e


def main() -> None:
    gen = EV.load_gen()
    vs = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())["val"]
    voices = [v for v in vs if v.startswith("em")][:4] + [v for v in vs if v.startswith("ef")][:4]
    acc: dict = {}
    for v in voices:
        for kind, p in [("speech", ROOT / f"data/snv_e1/{v}/t03.wav")] + [(c, ROOT / f"data/snv_e1nv/{v}/{c}_0.wav") for c in ("laugh", "moan", "cry", "scream")]:
            x = librosa.load(p, sr=EV.SR, mono=True)[0].astype(np.float32)
            u = PB.utt(x)
            x = u["wav"]
            n = len(u["f0"])
            r, vf, _ = librosa.pyin(librosa.resample(x, orig_sr=EV.SR, target_sr=16000), fmin=60, fmax=1600 if kind != "speech" else 700, sr=16000, frame_length=1024, hop_length=80, center=True)
            r = np.where(vf, np.nan_to_num(r), 0.0)[:n].astype(np.float32)
            r = np.concatenate([r, np.zeros(n - len(r), np.float32)])
            S = P.power_spec(torch.from_numpy(x))[:n]
            shr = P.odd_even(S, (r / 2)[:, None])[:, 0]
            sub = (r > 0) & np.isfinite(shr) & (shr > -10.0)
            plain = (r > 0) & np.isfinite(shr) & (shr <= -10.0)
            mo = lmel(x)
            for nm, f0 in (("前段の高さ", u["f0"].astype(np.float32)), ("pYIN の高さ", r)):
                vv = (f0 > 0).astype(np.float32)
                lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
                cond = torch.from_numpy(np.concatenate([u["y"][:25], lf[None], vv[None], np.clip(u["y"][25:29], 0, 1.5) * vv[None]], 0))[None].to(EV.DEV)
                cond = torch.cat([EV.smooth_env(gen, cond[:, :25]), cond[:, 25:]], 1)
                w = np.asarray(EV.draw(gen, cond, torch.from_numpy(f0)[None].to(EV.DEV), x, f0), np.float32)
                w = np.concatenate([w[gen.delay :], np.zeros(gen.delay, np.float32)])[: len(x)]
                mc = lmel(np.concatenate([w, np.zeros(max(0, len(x) - len(w)), np.float32)]))
                T = n - 8
                e = acf_err(x, np.concatenate([w, np.zeros(max(0, len(x) - len(w)), np.float32)]), r, n)[2 : 2 + T]
                idx = np.arange(T) + 2
                for cls, m in (("成分が強い", sub), ("成分が弱い", plain)):
                    q = m[idx] & np.isfinite(e)
                    a = acc.setdefault((("通常の発話" if kind == "speech" else "音素の無い発声"), nm, cls), [0.0, 0])
                    a[0] += float(e[q].sum()); a[1] += int(q.sum())
    out = {}
    for (k, nm, cls), (s, c) in sorted(acc.items()):
        out[f"{k}|{nm}|{cls}"] = [round(s / max(1, c), 3), c]
        print(k, nm, cls, "周期の構造のずれ(1 − 相関)", round(s / max(1, c), 3), "フレーム", c, flush=True)
    (ROOT / "results/snv_p110/copy_acf.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
