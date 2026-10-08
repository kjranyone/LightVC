"""decoder v2 の対照腕学習(prereg: results/<tag>/prereg.yaml)。--arch aa(候補A・対照) / a2(候補B・本命)。

共通: encoder は c32 EMA で凍結(潜在ABI不変)。データ=female_real 全train話者(f0fix付き・held末尾24話者除外)。
crop は左文脈1s + 損失区間1.28s・gain aug 50%[-30,0]dB・quiet 30%(RMS下位25%)。損失は codec Phase G と同一
(15 logmel + 2 mrstft + 1 wave L1 + adv + 2 FM)、判別器は MPD + MRD + MS-SB-CQT(設計書§4.3)。
A: DecoderAA(taps8) を c32 decoder EMA から warm-start。B: DecoderA2(C12) をスクラッチ(Phase R→G)。
ckpt に optimizer/判別器/EMA/step を保存し --resume で再開できる。

    CUDA_VISIBLE_DEVICES=0 uv run python train_dec2.py --arch a2 --tag diag_dec2_a2 --r-steps 20000 --g-steps 60000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from causal_codec import CausalCodec, SAMPLE_RATE, HOP_LENGTH
from decoder_aa import DecoderAA
from decoder_a2 import DecoderA2
from train_s1_1 import MEL_SPECS, build_mel, logmel_l1, mrstft, LOSS_FRAMES, CTX_FRAMES
from train_s1_g import MRD
from train_d1 import build_index
from train_cfmys import F0FIX, F0_FPS, LAT_FPS

ROOT = Path(__file__).resolve().parent.parent
PAD_FR = CTX_FRAMES + LOSS_FRAMES


def load48(path: str) -> np.ndarray:
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SAMPLE_RATE:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=SAMPLE_RATE)
    return x.astype(np.float32)


def _rms(path: str) -> float:
    try:
        x = load48(path)
        return float(np.sqrt((x ** 2).mean())) if len(x) > SAMPLE_RATE else -1.0
    except Exception:
        return -1.0


def f0_latent(f0_raw: np.ndarray, T: int) -> np.ndarray:
    i_f = np.floor((np.arange(T) + 1.0) * F0_FPS / LAT_FPS - 1.0).clip(0, len(f0_raw) - 1).astype(int)
    return f0_raw[i_f].astype(np.float32)


class CropStream(torch.utils.data.IterableDataset):
    """無限ランダムcrop(worker毎に独立RNG・独立キャッシュ)。quiet 30%・gain aug 50%[-30,0]dB。"""

    def __init__(self, quiet: list, loud: list, seed: int) -> None:
        self.quiet, self.loud, self.seed = quiet, loud, seed

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 1000 + (wi.id if wi else 0))
        cache: dict = {}
        while True:
            fp, path, _ = (self.quiet[rng.randrange(len(self.quiet))] if rng.random() < 0.3
                           else self.loud[rng.randrange(len(self.loud))])
            if fp not in cache:
                if len(cache) > 200:
                    cache.pop(next(iter(cache)))
                f = Path(fp)
                cache[fp] = (load48(path), torch.load(F0FIX / f.parent.name / f.name, map_location="cpu",
                                                      weights_only=False)["f0"].numpy())
            wav, f0r = cache[fp]
            T = len(wav) // HOP_LENGTH
            if T <= LOSS_FRAMES + 1:
                continue
            s = rng.randrange(1, T - LOSS_FRAMES)
            seg = np.zeros(PAD_FR * HOP_LENGTH, dtype=np.float32)
            lo_fr = max(0, s - CTX_FRAMES)
            take = wav[lo_fr * HOP_LENGTH:(s + LOSS_FRAMES) * HOP_LENGTH]
            seg[-len(take):] = take
            f0_all = f0_latent(f0r, s + LOSS_FRAMES)
            f0seg = np.zeros(PAD_FR, dtype=np.float32)
            f0seg[PAD_FR - (s + LOSS_FRAMES - lo_fr):] = f0_all[lo_fr:]
            if rng.random() < 0.5:
                seg = seg * np.float32(10 ** (rng.uniform(-30.0, 0.0) / 20))
            yield torch.from_numpy(seg)[None], torch.from_numpy(f0seg)


def build_decoder(arch: str, dev: str, codec: CausalCodec):
    if arch == "aa":
        d = DecoderAA(taps=8).to(dev)
        d.load_state_dict(codec.decoder.state_dict(), strict=True)
        return d
    return DecoderA2(channels=12).to(dev)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["aa", "a2"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--r-steps", type=int, default=0)
    ap.add_argument("--d-warm", type=int, default=1000)
    ap.add_argument("--g-steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--every", type=int, default=1000)
    ap.add_argument("--tripwire", type=float, default=1.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cqt-filters", type=int, default=32,
                    help="MS-SB-CQTの基本filter数(BigVGAN v2は128だが20GBでは batch8 で13GB→32=3.6GB)")
    ap.add_argument("--dec-ctx", type=int, default=None,
                    help="decoderに渡す左文脈フレーム数(encoderは常に1s文脈)。既定 aa=64 / a2=40")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.dec_ctx is None:
        a.dec_ctx = 64 if a.arch == "aa" else 40
    c0 = CTX_FRAMES - a.dec_ctx
    rng = random.Random(a.seed)
    torch.manual_seed(a.seed)
    out_dir = ROOT / "results" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs, _, held_spk = build_index(0)
    hset = set(held_spk)
    items = []
    for f in pairs:
        if f.parent.name in hset:
            continue
        items.append((str(f), f.parent.name))
    items.sort()
    ridx = ROOT / "results/ys1_dec2/rms_index.json"
    if ridx.exists():
        rms = json.loads(ridx.read_text())
    else:
        paths = []
        for fp, _ in items:
            paths.append(torch.load(fp, map_location="cpu", weights_only=False)["path"])
        with ProcessPoolExecutor(10) as ex:
            vals = list(ex.map(_rms, paths, chunksize=64))
        rms = {fp: (p, v) for (fp, _), p, v in zip(items, paths, vals)}
        ridx.parent.mkdir(parents=True, exist_ok=True)
        ridx.write_text(json.dumps(rms))
    ok = [(fp, rms[fp][0], rms[fp][1]) for fp, _ in items if rms.get(fp, (None, -1))[1] > 0]
    th = float(np.percentile([v for _, _, v in ok], 25))
    quiet = [x for x in ok if x[2] < th]
    loud = [x for x in ok if x[2] >= th]
    print(f"  train utts {len(ok)} (quiet {len(quiet)} / loud {len(loud)} th {th:.4f})", flush=True)

    ck = torch.load(ROOT / "results/s1_3_c32/s1_3_c32_last.pt", map_location=dev, weights_only=False)
    codec = CausalCodec(latent_dim=32, channels=(32, 64, 128, 256, 512)).to(dev)
    codec.load_state_dict(ck["ema"])
    codec.eval()
    for p in codec.parameters():
        p.requires_grad_(False)
    dec = build_decoder(a.arch, dev, codec)
    print(f"  {a.tag}: arch {a.arch} params {sum(p.numel() for p in dec.parameters())/1e6:.3f}M", flush=True)

    from bigvgan.discriminators import MultiPeriodDiscriminator, MultiScaleSubbandCQTDiscriminator
    from types import SimpleNamespace
    mpd = MultiPeriodDiscriminator(SimpleNamespace(mpd_reshapes=[2, 3, 5, 7, 11], use_spectral_norm=False,
                                                   discriminator_channel_mult=1)).to(dev)
    mrd = MRD().to(dev)
    cqt = MultiScaleSubbandCQTDiscriminator({"sampling_rate": SAMPLE_RATE, "cqtd_filters": a.cqt_filters,
                                             "cqtd_max_filters": 1024, "cqtd_filters_scale": 1,
                                             "cqtd_dilations": [1, 2, 4], "cqtd_hop_lengths": [512, 256, 256],
                                             "cqtd_n_octaves": [9, 9, 9],
                                             "cqtd_bins_per_octaves": [24, 36, 48]}).to(dev)
    dparams = list(mpd.parameters()) + list(mrd.parameters()) + list(cqt.parameters())
    dopt = torch.optim.AdamW(dparams, lr=2e-4, betas=(0.8, 0.99))
    opt = torch.optim.AdamW(dec.parameters(), lr=a.lr, betas=(0.8, 0.99))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup))
    ema = {k: v.detach().clone() for k, v in dec.state_dict().items()}
    mels = [build_mel(nf, hm, nm).to(dev) for nf, hm, nm in MEL_SPECS]
    step = 0
    ckp = out_dir / f"{a.tag}_last.pt"
    if a.resume and ckp.exists():
        st = torch.load(ckp, map_location=dev, weights_only=False)
        dec.load_state_dict(st["dec"])
        ema = st["ema"]
        opt.load_state_dict(st["opt"])
        sch.load_state_dict(st["sch"])
        for m, k in ((mpd, "mpd"), (mrd, "mrd"), (cqt, "cqt")):
            m.load_state_dict(st[k])
        dopt.load_state_dict(st["dopt"])
        step = int(st["step"])
        rng.setstate(st["rng"])
        print(f"  resumed at step {step}", flush=True)

    held_items = []
    for f in sorted(p for p in pairs if p.parent.name in hset)[:3]:
        d = torch.load(f, map_location="cpu", weights_only=False)
        x = load48(d["path"])
        T = min(len(x) // HOP_LENGTH, 800)
        f0r = torch.load(F0FIX / f.parent.name / f.name, map_location="cpu", weights_only=False)["f0"].numpy()
        held_items.append((torch.from_numpy(x[:T * HOP_LENGTH]).to(dev), torch.from_numpy(f0_latent(f0r, T)).to(dev)))

    def run_dec(m, z, f0):
        return m(z) if a.arch == "aa" else m(z, f0)

    def held_eval(state=None) -> float:
        old = None
        if state is not None:
            old = {k: v.detach().clone() for k, v in dec.state_dict().items()}
            dec.load_state_dict(state)
        dec.eval()
        vals = []
        with torch.no_grad():
            for x, f0 in held_items:
                z = codec.encode(x[None, None])
                y = run_dec(dec, z, f0[None])[..., :x.shape[-1]]
                vals.append(float(logmel_l1(y, x[None, None], mels)))
        dec.train()
        if old is not None:
            dec.load_state_dict(old)
        return float(np.mean(vals))

    base = held_eval()
    print(f"  held3 logmel at start: {base:.4f}", flush=True)
    loader = iter(torch.utils.data.DataLoader(
        CropStream(quiet, loud, a.seed + step), batch_size=a.batch, num_workers=a.workers,
        persistent_workers=True, prefetch_factor=4, pin_memory=True))

    def crop_batch():
        xs, fs = next(loader)
        return xs.to(dev, non_blocking=True), fs.to(dev, non_blocking=True)

    def self_crop(x):
        return x[..., :(x.shape[-1] // 2730) * 2730]

    def d_losses(real, fake):
        dl = 0.0
        r, f_, _, _ = mpd(self_crop(real), self_crop(fake))
        dl = dl + sum(((x - 1) ** 2).mean() + (y ** 2).mean() for x, y in zip(r, f_))
        dr, fr = mrd(self_crop(real))[0], mrd(self_crop(fake))[0]
        dl = dl + ((dr - 1) ** 2).mean() + (fr ** 2).mean()
        cr, cf, _, _ = cqt(real, fake)
        dl = dl + sum(((x - 1) ** 2).mean() + (y ** 2).mean() for x, y in zip(cr, cf))
        return dl

    def g_losses(real, fake):
        _, f2, fm_r, fm_f = mpd(self_crop(real), self_crop(fake))
        adv = sum(((f - 1) ** 2).mean() for f in f2)
        fm = sum(F.l1_loss(x.detach(), y) for A, B in zip(fm_r, fm_f) for x, y in zip(A, B))
        dr = mrd(self_crop(real))[0]
        fr2 = mrd(self_crop(fake))[0]
        adv = adv + ((fr2 - 1) ** 2).mean()
        fm = fm + F.l1_loss(dr.detach(), fr2)
        _, cf, cfm_r, cfm_f = cqt(real, fake)
        adv = adv + sum(((f - 1) ** 2).mean() for f in cf)
        fm = fm + sum(F.l1_loss(x.detach(), y) for A, B in zip(cfm_r, cfm_f) for x, y in zip(A, B))
        return adv, fm

    L = LOSS_FRAMES * HOP_LENGTH
    total = a.r_steps + a.d_warm + a.g_steps
    t0 = time.time()
    acc: dict = {"lm": [], "adv": [], "dl": []}
    while step < total:
        step += 1
        phase = "R" if step <= a.r_steps else ("D" if step <= a.r_steps + a.d_warm else "G")
        xb, fb = crop_batch()
        with torch.no_grad():
            z = codec.encode(xb)[..., c0:]
        fb = fb[:, c0:]
        if phase == "D":
            with torch.no_grad():
                yb = run_dec(dec, z, fb)
        else:
            yb = run_dec(dec, z, fb)
        y_l, t_l = yb[..., -L:], xb[..., -L:]
        if phase in ("D", "G"):
            dopt.zero_grad(set_to_none=True)
            dl = d_losses(t_l, y_l.detach())
            dl.backward()
            torch.nn.utils.clip_grad_norm_(dparams, 1.0)
            dopt.step()
            acc["dl"].append(float(dl))
        if phase in ("R", "G"):
            lm = logmel_l1(y_l, t_l, mels)
            loss = 15 * lm + 2 * mrstft(y_l, t_l) + F.l1_loss(y_l, t_l)
            if phase == "G":
                adv, fm = g_losses(t_l, y_l)
                loss = loss + adv + 2.0 * fm
                acc["adv"].append(float(adv))
            if not torch.isfinite(loss):
                print(f"  non-finite loss at {step} -> abort", flush=True)
                return 1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
            opt.step()
            sch.step()
            with torch.no_grad():
                for k, v in dec.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(0.999).add_(v.detach(), alpha=0.001)
            acc["lm"].append(float(lm))
        if step % a.every == 0 or step == total:
            ev = held_eval(ema) if phase != "D" else float("nan")
            msg = {k: round(float(np.mean(v)), 4) for k, v in acc.items() if v}
            acc = {k: [] for k in acc}
            print(f"  {phase} {step:6d}  {json.dumps(msg)}  held3_ema {ev:.4f}  ({time.time()-t0:.0f}s)", flush=True)
            torch.save({"dec": dec.state_dict(), "ema": ema, "opt": opt.state_dict(), "sch": sch.state_dict(),
                        "mpd": mpd.state_dict(), "mrd": mrd.state_dict(), "cqt": cqt.state_dict(),
                        "dopt": dopt.state_dict(), "step": step, "rng": rng.getstate(), "arch": a.arch,
                        "cli": vars(a), "base_held3": base}, ckp)
            if a.arch == "aa" and phase == "G" and step - a.r_steps - a.d_warm >= 2000 \
                    and ev > base * a.tripwire:
                print(f"  TRIPWIRE: held3 {ev:.4f} > {a.tripwire}×base {base:.4f} -> abort", flush=True)
                return 2
    print(f"\n{a.tag}: done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
