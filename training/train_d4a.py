"""D4a: 条件付きNAM vocoder学習(定理M1検証・同話者再構成)。

条件 mel80@48k+lf0+energy+spk(100fps) → 48kHz波形。teacher-forced並列学習。
損失=S1-3安定レシピ(15·lm+2·ms+1·wl)。evalはheld再構成+chorus代理指標。

    CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
      uv run python train_d4a.py --tag d4a_namvoc --steps 40000
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile
import torch

sys.path.insert(0, str(Path(__file__).parent))
from d4_vocoder import D4Voc, causal_mel48, HOP
from train_s1_1 import MEL_SPECS, build_mel, logmel_l1, mrstft
from train_s1_3 import build_index, load_wav

ROOT = Path(__file__).resolve().parent.parent
F0FIX = ROOT / "data/female_real_f0fix"
FEAT = ROOT / "data/female_real_feat"
CROP_F = 100
CROP_N = CROP_F * HOP
F0_SRC_FPS = 44100 / 512.0
MEL_SCALE = 8.0


def f0_at_100(wav_path: Path, T: int) -> torch.Tensor:
    fp = F0FIX / wav_path.parent.name / (wav_path.stem + ".pt")
    f0 = torch.load(fp, map_location="cpu", weights_only=False)["f0"].float()
    idx = ((torch.arange(T, dtype=torch.float64) + 1.0) * 100.0 / F0_SRC_FPS
           - 1.0).floor().clamp(0, f0.shape[0] - 1).long()
    return f0[idx]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--every", type=int, default=1000)
    ap.add_argument("--noise-in", type=float, default=0.005,
                    help="入力波形への学習時ノイズ(AR暴露バイアス対策)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(a.seed)

    tr, held = build_index()
    tr = [p for p in tr
          if (F0FIX / p.parent.name / (p.stem + ".pt")).exists()
          and (FEAT / p.parent.name / (p.stem + ".pt")).exists()]
    rng.shuffle(tr)
    spk_emb = torch.load(ROOT / "data/ecapa_spk_mean_full.pt",
                         map_location="cpu", weights_only=False)
    print(f"  train {len(tr)} / held {len(held)}", flush=True)

    m = D4Voc().to(dev)
    params = [q for q in m.parameters() if q.requires_grad]
    print(f"  params {sum(q.numel() for q in params)/1e6:.2f}M", flush=True)
    mels = [build_mel(nf, h, nm).to(dev) for nf, h, nm in MEL_SPECS]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps)
    out_dir = ROOT / "results" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cache: dict = {}

    def get(p: Path):
        if p not in cache:
            if len(cache) > 300:
                cache.pop(next(iter(cache)))
            try:
                feat = torch.load(FEAT / p.parent.name / (p.stem + ".pt"),
                                  map_location="cpu", weights_only=False)
                w = load_wav(p).astype(np.float32)
                if len(w) < CROP_N + HOP:
                    cache[p] = None
                    return cache[p]
                wv = torch.from_numpy(w[: len(w) // HOP * HOP])
                T = wv.shape[0] // HOP
                mel = causal_mel48(wv[None])[0]
                f0g = f0_at_100(p, T)
                en = feat["energy"].float()
                i_e = ((torch.arange(T, dtype=torch.float64) + 1.0) * 100.0
                       / F0_SRC_FPS - 1.0).floor().clamp(0, en.shape[0] - 1).long()
                lf0 = torch.log(f0g.clamp(min=50.0) / 200.0)
                enl = torch.log(en[i_e].clamp(min=1e-4))
                s_ = spk_emb.get(feat.get("speaker"), torch.zeros(192))
                cond = torch.cat([(mel / MEL_SCALE).clamp(-6, 6), lf0[None],
                                  enl[None], s_[:, None].expand(-1, T)], 0)
                cache[p] = (wv, cond)
            except Exception:  # noqa: BLE001
                cache[p] = None
        return cache[p]

    from eval_d4a_freerun import freerun as _freerun

    def freerun_held0():
        m.eval()
        p = held[0]
        it = get(p)
        if it is None:
            return 9.9, {}
        wv, cond = it
        n = min(wv.shape[0], 192000)
        with torch.no_grad():
            y = _freerun(m, cond[None, :, : n // HOP].to(dev), n, dev)
        import librosa as _lb
        import pyworld as _pw
        w44 = _lb.resample(y[0].cpu().numpy().astype(np.float64),
                           orig_sr=48000, target_sr=44100)
        f0o, t_ = _pw.harvest(w44, 44100, f0_floor=65, f0_ceil=1000,
                              frame_period=512 / 44100 * 1000)
        v = f0o > 60
        S = np.abs(_lb.stft(w44, n_fft=2048, hop_length=512)) ** 2
        f_ = _lb.fft_frequencies(sr=44100, n_fft=2048)
        hi = float(S[(f_ > 2000) & (f_ < 6000)].sum() / (S.sum() + 1e-12))
        penalty = 0.0 if v.sum() > 200 else 3.0
        if v.sum() > 200 and abs(float(np.median(f0o[v])) - 100.0) < 2.0:
            penalty += 1.5
        if hi < 0.005:
            penalty += 1.0
        m.train()
        return penalty, {"fr_f0": round(float(np.median(f0o[v])) if v.sum() else 0.0, 1),
                         "fr_voiced": round(float((f0o > 60).mean()), 3),
                         "fr_hi_mid": round(hi, 4)}

    def eval_held():
        m.eval()
        tot, n = 0.0, 0
        extra = {}
        with torch.no_grad():
            for p in held[:4]:
                it = get(p)
                if it is None:
                    continue
                wv, cond = it
                n_ut = min(wv.shape[0], 192000)
                xt = wv[None, :n_ut].to(dev)
                ct = cond[None, :, : n_ut // HOP].to(dev)
                y = m(xt, ct)
                mn = min(y.shape[-1], xt.shape[-1])
                seg = slice(4800, mn)
                tot += float(logmel_l1(y[:, None, seg], xt[:, None, seg], mels))
                n += 1
                if n == 1:
                    import librosa as _lb
                    import pyworld as _pw
                    w44 = _lb.resample(y[0, :mn].cpu().numpy().astype(np.float64),
                                       orig_sr=48000, target_sr=44100)
                    f0o, t_ = _pw.harvest(w44, 44100, f0_floor=65, f0_ceil=1000,
                                          frame_period=512 / 44100 * 1000)
                    apo = _pw.d4c(w44, f0o, t_, 44100, fft_size=2048)
                    nn_ = min(len(f0o), apo.shape[1]); v = f0o[:nn_] > 60
                    extra["aperiod"] = round(float(apo[:, :nn_][:, v].mean(0).mean()), 3) if v.sum() > 10 else 1.0
                    extra["voiced"] = round(float((f0o > 60).mean()), 3)
                    S = np.abs(_lb.stft(w44, n_fft=2048, hop_length=512)) ** 2
                    f_ = _lb.fft_frequencies(sr=44100, n_fft=2048)
                    tot_s = S.sum() + 1e-12
                    extra["hi_mid"] = round(float(S[(f_ > 2000) & (f_ < 6000)].sum() / tot_s), 4)
                    soundfile.write(out_dir / "held0_d4a.wav",
                                    np.clip(y[0, :mn].cpu().numpy(), -1, 1), 48000)
        m.train()
        return tot / max(n, 1), extra

    best = 1e9
    t0 = time.time()
    step = 0
    while step < a.steps:
        ws, cs = [], []
        while len(ws) < a.batch:
            p = rng.choice(tr)
            it = get(p)
            if it is None:
                continue
            wv, cond = it
            T = min(wv.shape[0] // HOP, cond.shape[1])
            if T <= CROP_F + 4:
                continue
            s0 = rng.randrange(0, T - CROP_F)
            ws.append(wv[s0 * HOP:(s0 + CROP_F) * HOP])
            cs.append(cond[:, s0:s0 + CROP_F])
        step += 1
        xb = torch.stack(ws).to(dev)
        cb = torch.stack(cs).to(dev)
        x_in = xb.clone()
        for bi in range(xb.shape[0]):
            r = rng.random()
            if r < 0.35:
                x_in[bi] = 0.0
            elif r < 0.70:
                x_in[bi] = x_in[bi] + torch.randn_like(x_in[bi]) * (
                    0.002 + 0.018 * rng.random())
        y = m(x_in, cb)
        loss = 15 * logmel_l1(y[:, None], xb[:, None], mels) + 2 * mrstft(
            y[:, None], xb[:, None]) + 1 * torch.nn.functional.l1_loss(y, xb)
        if not torch.isfinite(loss) or float(loss.detach()) > 1e6:
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sch.step()
        if step % a.every == 0 or step == a.steps:
            evl, extra = eval_held()
            fr_pen = 0.0
            fr_extra = {}
            if step % 2000 == 0 or step == a.steps:
                fr_pen, fr_extra = freerun_held0()
            score = evl + fr_pen
            print(f"  step {step:6d}  loss {float(loss.detach()):.4f}"
                  f"  held-mel {evl:.4f}  {fr_extra}  ({time.time()-t0:.0f}s)",
                  flush=True)
            torch.save({"net": m.state_dict(), "step": step, "cli": vars(a),
                        "held_mel": evl, "extra": extra},
                       out_dir / f"{a.tag}_last.pt")
            if score < best:
                best = evl
                torch.save({"net": m.state_dict(), "step": step, "cli": vars(a),
                            "held_mel": evl, "extra": extra},
                           out_dir / f"{a.tag}_best.pt")
    print(f"\n{a.tag}: best held-mel {best:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

def held_paths():
    _, held = build_index()
    return held


def get_utt_cond(p: Path):
    """1発話の(wav48, cond274)を返す(train_d4a.getと同一規約)。"""
    feat = torch.load(FEAT / p.parent.name / (p.stem + ".pt"),
                      map_location="cpu", weights_only=False)
    w = load_wav(p).astype(np.float32)
    wv = torch.from_numpy(w[: len(w) // HOP * HOP])
    T = wv.shape[0] // HOP
    mel = causal_mel48(wv[None])[0]
    f0 = torch.load(F0FIX / p.parent.name / (p.stem + ".pt"),
                    map_location="cpu", weights_only=False)["f0"].float()
    idx = ((torch.arange(T, dtype=torch.float64) + 1.0) * 100.0 / F0_SRC_FPS
           - 1.0).floor().clamp(0, f0.shape[0] - 1).long()
    f0g = f0[idx]
    en = feat["energy"].float()
    i_e = ((torch.arange(T, dtype=torch.float64) + 1.0) * 100.0 / F0_SRC_FPS
           - 1.0).floor().clamp(0, en.shape[0] - 1).long()
    lf0 = torch.log(f0g.clamp(min=50.0) / 200.0)
    enl = torch.log(en[i_e].clamp(min=1e-4))
    spk_emb = torch.load(ROOT / "data/ecapa_spk_mean_full.pt",
                         map_location="cpu", weights_only=False)
    s_ = spk_emb.get(feat.get("speaker"), torch.zeros(192))
    cond = torch.cat([(mel / MEL_SCALE).clamp(-6, 6), lf0[None], enl[None],
                      s_[:, None].expand(-1, T)], 0)
    return wv, cond
