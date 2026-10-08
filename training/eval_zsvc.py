"""ZS-VC の評価(学習中にも呼ぶ)。

  recon : 除外女性話者 6 発話の自己再構成(参照 = 同じ話者の別発話)の log-mel L1 と、nvoc で描いた PESQ
  conv  : 除外男声 4 話者 × 除外女性 3 話者 = 12 変換。出力を nvoc で描き
          tgt  = ECAPA cos(出力, 目標話者の「参照に使っていない」別発話)  目安: 本人の別発話 0.663・別の実女性 0.421
          leak = ECAPA cos(出力, 元の男声の別発話)                         目安: 元の男声どうし ~0.6
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import zsvc as Z

ROOT = Path(__file__).resolve().parent.parent
CTX = N.WIN - N.HOP


def _logf0(fz) -> tuple[float, float]:
    f0 = np.load(fz)["f0"]
    v = f0[f0 > 0]
    lv = np.log(v) if len(v) else np.array([np.log(200.0)])
    return float(np.median(lv)), float(max(np.std(lv), 0.05))


def build_evalset(spk: dict, ev: list, males_held: list, f0_adv: int = 0) -> dict:
    from train_ddsp_vc import load48
    fem = [k for k in ev if k.startswith("real_female") and len(spk[k]) >= 3][:9]
    recon = []
    for k in fem[:6]:
        z, w, _ = spk[k][0]
        x = load48(w)[:6 * N.SR]
        recon.append({"x": x[:len(x) // N.HOP * N.HOP], "f0": np.load(z)["f0"].astype(np.float32)[f0_adv:], "ref": load48(spk[k][1][1])[:3 * N.SR]})
    tg = []
    for k in [k for k in ev if k.startswith("real_female") and len(spk[k]) >= 2 and k not in fem[:6]]:
        st = _logf0(spk[k][0][0])
        if 250 <= np.exp(st[0]) <= 400:
            tg.append({"ref": load48(spk[k][0][1])[:3 * N.SR], "stat": st, "other": load48(spk[k][1][1])[:6 * N.SR]})
        if len(tg) == 3:
            break
    by = {}
    for z, w in males_held:
        by.setdefault(z.parent.name, []).append((z, w))
    src = []
    for s, lst in sorted(by.items()):
        z, w = next((p for p in lst if "neutral" in p[0].stem), lst[0])
        x = load48(w)[:6 * N.SR]
        z2, w2 = next(p for p in lst if p[0] != z)
        import f0_fix as FX
        f0f = FX.fix_f0(np.load(z)["f0"].astype(np.float32))[0]
        src.append({"x": x[:len(x) // N.HOP * N.HOP], "f0": f0f, "stat": FX.logf0_stats(f0f), "other": load48(w2)[:6 * N.SR]})
    return {"recon": recon, "tgt": tg, "src": src}


def contrast(y: np.ndarray, f0: np.ndarray) -> float:
    """倍音間コントラスト(有声・500–3000Hz・log|STFT| の p90 − p10・nfft 4096)。耳の較正: 元音声 3.27・BigVGAN 3.23・Y-S1(不合格)3.06。"""
    import scipy.signal as ss
    f, _, Zs = ss.stft(y, N.SR, nperseg=4096, noverlap=4096 - N.HOP)
    L = np.log(np.abs(Zs) + 1e-6)
    band = (f >= 500) & (f <= 3000)
    T = min(L.shape[1], len(f0))
    v = f0[:T] > 0
    if v.sum() < 5:
        return float("nan")
    Lb = L[band][:, :T][:, v]
    return float(np.mean(np.percentile(Lb, 90, axis=0) - np.percentile(Lb, 10, axis=0)))


def render(model: Z.ZSVC, voc: N.NVoc, x: np.ndarray, f0: np.ndarray, ref: np.ndarray, dev: str) -> tuple[np.ndarray, torch.Tensor]:
    xin = torch.cat([torch.zeros(1, CTX), torch.from_numpy(x.astype(np.float32))[None]], -1).to(dev)
    rin = torch.cat([torch.zeros(1, CTX), torch.from_numpy(ref.astype(np.float32))[None]], -1).to(dev)
    with torch.no_grad():
        mel = N.NVoc.mel_ctx(model, xin)
        s = model.spk(N.NVoc.mel_ctx(model, rin))
        T = mel.shape[-1]
        f = torch.from_numpy(f0[:T]).to(dev)[None]
        f = torch.nn.functional.pad(f, (0, T - f.shape[-1]))
        y = model(mel, model.level(mel), f, s)
        g = torch.Generator(device="cpu").manual_seed(0)
        exc = torch.stack([torch.zeros(1, T * N.HOP), torch.randn(1, T * N.HOP, generator=g)], 1).to(dev)
        w = voc.generate(y, exc)[0, N.DELAY:].cpu().numpy()
    return w, y


def evaluate(model: Z.ZSVC, voc: N.NVoc, emb, es: dict, dev: str) -> dict:
    import eval_nvoc as E
    model.eval()
    out: dict = {}
    lms, pqs, ams = [], [], []
    for it in es["recon"]:
        f0 = it["f0"]
        w, y = render(model, voc, it["x"], f0, it["ref"], dev)
        xin = torch.cat([torch.zeros(1, CTX), torch.from_numpy(it["x"].astype(np.float32))[None]], -1).to(dev)
        with torch.no_grad():
            mel = N.NVoc.mel_ctx(model, xin)
        lms.append(float((y - mel)[..., 20:].abs().mean()))
        mm = E.metrics(np.pad(w, (N.DELAY, 0)), it["x"], N.DELAY, dev)
        pqs.append(mm["pesq"])
        ams.append(mm["am_db"])
    out["recon_mel_l1"] = round(float(np.mean(lms)), 4)
    import artic_dsp as AD
    fol = []
    for it in es["recon"][:3]:
        w5, _ = render(model, voc, it["x"], (it["f0"] * 2 ** (5 / 12)).astype(np.float32), it["ref"], dev)
        fo, _ = AD.causal_yin(w5.astype(np.float64), voi_max=0.45)
        k = min(len(fo), len(it["f0"]))
        vv = (fo[:k] > 0) & (it["f0"][:k] > 0)
        if vv.sum() > 20:
            fol.append(float(np.median(12 * np.log2(fo[:k][vv] / it["f0"][:k][vv]))))
    out["f0_follow_+5st"] = round(float(np.median(fol)), 2) if fol else None
    out["recon_pesq"] = round(float(np.nanmean(pqs)), 3)
    out["recon_am_db"] = round(float(np.mean(ams)), 2)

    def e(v: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            return emb(torch.from_numpy(v.astype(np.float32))[None].to(dev))[0]
    tg_other = [e(t["other"]) for t in es["tgt"]]
    tgt_s, leak_s, devs, cons, cams = [], [], [], [], []
    for s in es["src"]:
        e_src_other = e(s["other"])
        for t, eo in zip(es["tgt"], tg_other):
            lf = np.log(np.maximum(s["f0"], 1.0))
            f0m = np.where(s["f0"] > 0, np.exp(t["stat"][0] + (lf - s["stat"][0])), 0.0).astype(np.float32)
            w, _ = render(model, voc, s["x"], f0m, t["ref"], dev)
            import artic_dsp as AD
            fo, _ = AD.causal_yin(w.astype(np.float64), voi_max=0.45)
            k = min(len(fo), len(f0m))
            vv = (fo[:k] > 0) & (f0m[:k] > 0)
            if vv.sum() > 20:
                devs.append(float(np.median(12 * np.log2(fo[:k][vv] / f0m[:k][vv]))))
            cons.append(contrast(w, f0m))
            cams.append(E.amline(w))
            ew = e(w)
            tgt_s.append(float((ew * eo).sum()))
            leak_s.append(float((ew * e_src_other).sum()))
    out["conv_tgt_ecapa"] = round(float(np.mean(tgt_s)), 4)
    out["conv_src_leak"] = round(float(np.mean(leak_s)), 4)
    out["conv_f0_dev_st"] = round(float(np.median(devs)), 2) if devs else None
    out["conv_contrast"] = round(float(np.nanmean(cons)), 3)
    out["conv_am_db"] = round(float(np.mean(cams)), 2)
    return out
