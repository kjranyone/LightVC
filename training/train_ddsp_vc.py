"""DDSP-VC の学習(女声フルコーパス・自己再構成・ゼロショット)。事前登録 results/ddsp_vc/prereg.yaml。

  入力 x → 因果 log-mel → 周波数伸縮 α と利得の摂動(content 側だけ・入力側)→ content 符号器
  参照 = 同じ話者の別発話(≤3s)→ 話者符号器    f0 = artic_feat の因果 YIN(フレーム t は x[:(t+1)·240])
  出力 y[n] ≈ x[n − DELAY](DELAY=480=10ms)。損失: 15·多尺度 logmel + 2·mrstft + ContentVec 補助(cos)
  段 G(r_steps 以降): + LSGAN(MPD+MRD+CQT)+ 2·FM + CIPT(出力と参照の ECAPA コサイン)

    CUDA_VISIBLE_DEVICES=0 uv run python train_ddsp_vc.py --tag ddsp_vc --steps 90000
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

sys.path.insert(0, str(Path(__file__).parent))
import ddsp_vc as V
from train_s1_1 import MEL_SPECS, build_mel, logmel_l1, mrstft
from train_s1_2 import MRD

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data/artic_feat"
SEG = 57600
W0 = 12000
REF_MAX = 144000
SRCS = {"real_female": ("female-dataset", "data/female_real_feat"),
        "tts_base": ("data/female_tts_corpus", "data/female_tts_feat"),
        "tts_emotional_live": ("data/female_tts_corpus", "data/female_tts_feat")}


def load48(path: Path) -> np.ndarray:
    from scipy.signal import resample_poly
    x, sr = soundfile.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != V.SR:
        g = math.gcd(sr, V.SR)
        x = resample_poly(x, V.SR // g, sr // g).astype(np.float32)
    return x


def index() -> tuple[dict, list, list]:
    import train_phys_e1 as TP
    spk = {}
    for st, (wd, cd) in SRCS.items():
        for d in sorted((FEAT / st).iterdir()):
            items = []
            for f in sorted(d.glob("*.npz")):
                items.append((f, ROOT / wd / d.name / (f.stem + ".wav"), ROOT / cd / d.name / (f.stem + ".pt")))
            if len(items) >= 2:
                spk[f"{st}/{d.name}"] = items
    _, _, ev = TP.speakers()
    ev = [k for k in ev if k in spk]
    tr = [k for k in spk if k not in set(ev)]
    return spk, tr, ev


class DS(torch.utils.data.IterableDataset):
    def __init__(self, spk: dict, keys: list, seed: int):
        self.spk, self.keys, self.seed = spk, keys, seed
        self.flat = [(k, i) for k in keys for i in range(len(spk[k]))]
        self.kid = {k: j for j, k in enumerate(keys)}

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 1009 + (wi.id if wi else 0))
        while True:
            k, i = self.flat[rng.randrange(len(self.flat))]
            fz, fw, fc = self.spk[k][i]
            try:
                x = load48(fw)
                z = np.load(fz)
                f0 = z["f0"].astype(np.float32)
                if len(x) < SEG + V.DELAY + V.HOP:
                    continue
                s = V.DELAY + V.HOP * rng.randrange(0, (len(x) - SEG - V.DELAY) // V.HOP)
                seg = x[s - V.DELAY:s + SEG]
                if np.sqrt((seg ** 2).mean()) < 1e-3:
                    continue
                t0 = s // V.HOP
                ff = f0[t0:t0 + SEG // V.HOP]
                if len(ff) < SEG // V.HOP:
                    ff = np.pad(ff, (0, SEG // V.HOP - len(ff)))
                cv = np.zeros((SEG // 960, 768), np.float32)
                has_cv = 0.0
                if fc.exists():
                    c = torch.load(fc, map_location="cpu", weights_only=False)["content"].numpy()
                    j0 = s // 960
                    if j0 + SEG // 960 <= len(c):
                        cv, has_cv = c[j0:j0 + SEG // 960].astype(np.float32), 1.0
                others = [j for j in range(len(self.spk[k])) if j != i]
                xr = load48(self.spk[k][rng.choice(others)][1])
                if len(xr) > REF_MAX:
                    r0 = rng.randrange(0, len(xr) - REF_MAX)
                    xr = xr[r0:r0 + REF_MAX]
                ref = np.zeros(REF_MAX, np.float32)
                ref[:len(xr)] = xr
                yield (torch.from_numpy(seg), torch.from_numpy(ff), torch.from_numpy(cv), has_cv,
                       torch.from_numpy(ref), len(xr), self.kid[k])
            except Exception:
                continue


class GradRev(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class SpkAdv(torch.nn.Module):
    """content → 話者(平均+標準偏差プーリング → MLP)。符号器へは勾配反転で話者情報を消す方向に働く。"""

    def __init__(self, cin: int, n_spk: int):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(2 * cin, 512), torch.nn.ReLU(), torch.nn.Linear(512, n_spk))

    def forward(self, c: torch.Tensor, lam: float) -> torch.Tensor:
        c = GradRev.apply(c, lam)
        return self.net(torch.cat([c.mean(-1), c.std(-1)], -1))


def ecapa_model(dev):
    from speechbrain.inference.speaker import EncoderClassifier
    m = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                       savedir=str(ROOT / "pretrained_models/spkrec-ecapa-voxceleb"), run_opts={"device": dev})
    for p in m.mods.parameters():
        p.requires_grad_(False)
    m.mods.eval()

    def emb(y48: torch.Tensor) -> torch.Tensor:
        e = m.encode_batch(AF.resample(y48, V.SR, 16000)).squeeze(1)
        return e / (e.norm(dim=-1, keepdim=True) + 1e-6)
    return emb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ddsp_vc")
    ap.add_argument("--steps", type=int, default=90000)
    ap.add_argument("--r_steps", type=int, default=15000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--cipt", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=2500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--norm", action="store_true")
    ap.add_argument("--grl", type=float, default=0.0)
    ap.add_argument("--spk_distill", type=float, default=0.0)
    ap.add_argument("--init", default=None, help="この ckpt の model/ema/opt/adv_clf から開始(step は 0 から数える)")
    a = ap.parse_args()
    dev = "cuda"
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    print("prereg:", ROOT / "results" / a.tag / "prereg.yaml", flush=True)
    torch.manual_seed(0)
    spk, tr, ev = index()
    print(f"speakers train {len(tr)} eval {len(ev)} utts {sum(len(spk[k]) for k in tr)}", flush=True)
    model = V.DDSPVC(norm=a.norm).to(dev)
    adv_clf = SpkAdv(192, len(tr)).to(dev) if a.grl > 0 else None
    spk_head = torch.nn.Linear(256, 192).to(dev) if a.spk_distill > 0 else None
    from bigvgan.discriminators import MultiPeriodDiscriminator, MultiScaleSubbandCQTDiscriminator
    from types import SimpleNamespace
    mpd = MultiPeriodDiscriminator(SimpleNamespace(mpd_reshapes=[2, 3, 5, 7, 11], use_spectral_norm=False,
                                                   discriminator_channel_mult=1)).to(dev)
    mrd = MRD().to(dev)
    cqt = MultiScaleSubbandCQTDiscriminator({"sampling_rate": V.SR, "cqtd_filters": 32, "cqtd_max_filters": 1024,
                                             "cqtd_filters_scale": 1, "cqtd_dilations": [1, 2, 4],
                                             "cqtd_hop_lengths": [512, 256, 256], "cqtd_n_octaves": [9, 9, 9],
                                             "cqtd_bins_per_octaves": [24, 36, 48]}).to(dev)
    dparams = list(mpd.parameters()) + list(mrd.parameters()) + list(cqt.parameters())
    gparams = list(model.parameters()) + (list(adv_clf.parameters()) if adv_clf is not None else []) \
        + (list(spk_head.parameters()) if spk_head is not None else [])
    opt = torch.optim.AdamW(gparams, lr=a.lr, betas=(0.8, 0.99))
    dopt = torch.optim.AdamW(dparams, lr=a.lr, betas=(0.8, 0.99))
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    mels = [build_mel(nf, hm, nm).to(dev) for nf, hm, nm in MEL_SPECS]
    emb = ecapa_model(dev)
    step = 0
    ckp = out / "last.pt"
    if a.resume and ckp.exists():
        st = torch.load(ckp, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        ema = st["ema"]
        if adv_clf is not None and "adv_clf" in st:
            adv_clf.load_state_dict(st["adv_clf"])
        opt.load_state_dict(st["opt"])
        dopt.load_state_dict(st["dopt"])
        for m, k in ((mpd, "mpd"), (mrd, "mrd"), (cqt, "cqt")):
            m.load_state_dict(st[k])
        step = int(st["step"])
        print("resumed at", step, flush=True)
    if a.init:
        st = torch.load(a.init, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        ema = st["ema"]
        if spk_head is None:
            opt.load_state_dict(st["opt"])
        if adv_clf is not None and st.get("adv_clf") is not None:
            adv_clf.load_state_dict(st["adv_clf"])
        print("init from", a.init, "(step", st["step"], ")", flush=True)
    loader = iter(torch.utils.data.DataLoader(DS(spk, tr, 1 + step), batch_size=a.bs, num_workers=a.workers,
                                              persistent_workers=True, prefetch_factor=4, pin_memory=True))
    held = []
    hr = random.Random(5)
    for k in [k for k in ev if k.startswith("real_female")][:6]:
        its = spk[k]
        fz, fw, _ = its[0]
        x = load48(fw)[:6 * V.SR]
        x = x[:(len(x) // V.HOP) * V.HOP]
        f0 = np.load(fz)["f0"].astype(np.float32)[:len(x) // V.HOP]
        f0 = np.pad(f0, (0, len(x) // V.HOP - len(f0)))
        xr = load48(its[1][1])[:REF_MAX]
        held.append((torch.from_numpy(x).to(dev), torch.from_numpy(f0).to(dev), torch.from_numpy(xr).to(dev)))

    def infer(m, x, f0, xr, alpha=None):
        mel = m.front(x[None])
        mel_c = mel if alpha is None else V.warp_mel(mel, torch.tensor([alpha], device=dev), m.centers)
        s = m.spk(m.front(xr[None]))
        return m(mel_c, m.level(mel), f0[None], s, x.shape[-1], gen=torch.Generator(device=dev).manual_seed(0))[0]

    def held_eval() -> float:
        cur = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema)
        model.eval()
        vals = []
        with torch.no_grad():
            for x, f0, xr in held:
                y = infer(model, x, f0, xr)
                vals.append(float(logmel_l1(y[None, None, V.DELAY:], x[None, None, :-V.DELAY], mels)))
        model.load_state_dict(cur)
        model.train()
        return float(np.mean(vals))

    def self_crop(z):
        return z[..., :(z.shape[-1] // 2730) * 2730]

    def d_losses(real, fake):
        r, f_, _, _ = mpd(self_crop(real), self_crop(fake))
        dl = sum(((x - 1) ** 2).mean() + (y ** 2).mean() for x, y in zip(r, f_))
        dr, fr = mrd(self_crop(real)), mrd(self_crop(fake))
        dl = dl + ((dr - 1) ** 2).mean() + (fr ** 2).mean()
        cr, cf, _, _ = cqt(real, fake)
        return dl + sum(((x - 1) ** 2).mean() + (y ** 2).mean() for x, y in zip(cr, cf))

    def g_losses(real, fake):
        _, f2, fm_r, fm_f = mpd(self_crop(real), self_crop(fake))
        adv = sum(((f - 1) ** 2).mean() for f in f2)
        fm = sum(F.l1_loss(x.detach(), y) for A, B in zip(fm_r, fm_f) for x, y in zip(A, B))
        fr2 = mrd(self_crop(fake))
        adv = adv + ((fr2 - 1) ** 2).mean()
        _, cf, cfm_r, cfm_f = cqt(real, fake)
        adv = adv + sum(((f - 1) ** 2).mean() for f in cf)
        fm = fm + sum(F.l1_loss(x.detach(), y) for A, B in zip(cfm_r, cfm_f) for x, y in zip(A, B))
        return adv, fm

    log = open(out / "train.jsonl", "a")
    base = held_eval()
    print(f"held6 logmel at step {step}: {base:.4f}(錨 BigVGAN 0.195 は held24・参考)", flush=True)
    t0 = time.time()
    acc: dict = {"lm": [], "aux": [], "adv": [], "cipt": [], "dl": [], "spk_ce": [], "spk_acc": [], "spk_dist": []}
    total = 40 if a.smoke else a.steps
    n_bad = 0
    while step < total:
        step += 1
        seg, ff, cv, has_cv, ref, nref, sid = next(loader)
        sid = sid.to(dev)
        seg, ff, cv, ref = (t.to(dev, non_blocking=True) for t in (seg, ff, cv, ref))
        has_cv = has_cv.to(dev).float()
        x_in, tgt = seg[:, V.DELAY:], seg[:, :-V.DELAY]
        with torch.no_grad():
            mel = model.front(x_in)
            B = mel.shape[0]
            alpha = torch.exp(torch.empty(B, device=dev).uniform_(math.log(0.8), math.log(1.25)))
            gain = torch.empty(B, 1, 1, device=dev).uniform_(-1.4, 1.4)
            mel_p = V.warp_mel(mel, alpha, model.centers) + gain
            rmel = model.front(ref)
            rmask = (torch.arange(rmel.shape[-1], device=dev)[None] < (nref.to(dev) // V.HOP)[:, None]).float()
        s = model.spk(rmel, rmask)
        y, parts = model(mel_p, model.level(mel), ff, s, SEG, return_parts=True)
        yl, tl = y[:, W0:], tgt[:, W0:]
        phase_g = step > a.r_steps
        if phase_g:
            dopt.zero_grad(set_to_none=True)
            dl = d_losses(tl[:, None], yl.detach()[:, None])
            dl.backward()
            torch.nn.utils.clip_grad_norm_(dparams, 1.0)
            dopt.step()
            acc["dl"].append(float(dl))
        lm = logmel_l1(yl[:, None], tl[:, None], mels)
        c50 = F.avg_pool1d(parts["content"], 4)
        auxp = model.content.aux(c50).transpose(1, 2)
        aux = ((1 - F.cosine_similarity(auxp, cv, dim=-1)).mean(-1) * has_cv).sum() / has_cv.sum().clamp(min=1)
        loss = 15 * lm + 2 * mrstft(yl[:, None], tl[:, None]) + aux
        if spk_head is not None:
            with torch.no_grad():
                e_ref = emb(ref[:, :min(REF_MAX, int(nref.min()))])
            sd = (1 - F.cosine_similarity(spk_head(s), e_ref, dim=-1)).mean()
            loss = loss + a.spk_distill * sd
            acc["spk_dist"].append(float(sd))
        if adv_clf is not None:
            logits = adv_clf(parts["content_n"], a.grl)
            ce = F.cross_entropy(logits, sid)
            loss = loss + ce
            acc["spk_ce"].append(float(ce))
            acc["spk_acc"].append(float((logits.argmax(-1) == sid).float().mean()))
        if phase_g:
            adv, fm = g_losses(tl[:, None], yl[:, None])
            e_out = emb(yl)
            with torch.no_grad():
                e_ref = emb(ref[:, :min(REF_MAX, int(nref.min()))])
            cipt = (1 - (e_out * e_ref).sum(-1)).mean()
            loss = loss + adv + 2 * fm + a.cipt * cipt
            acc["adv"].append(float(adv))
            acc["cipt"].append(float(cipt))
        if not torch.isfinite(loss):
            print("non-finite loss at", step, "→ skip", flush=True)
            opt.zero_grad(set_to_none=True)
            n_bad += 1
            if n_bad >= 20:
                raise RuntimeError(f"non-finite 20 steps in a row at {step}; last finite save is last.pt")
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(gparams, 1.0)
        if not torch.isfinite(gn):
            print("non-finite grad norm at", step, "→ skip", flush=True)
            opt.zero_grad(set_to_none=True)
            n_bad += 1
            if n_bad >= 20:
                raise RuntimeError(f"non-finite 20 steps in a row at {step}; last finite save is last.pt")
            continue
        n_bad = 0
        opt.step()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(0.999).add_(v.detach(), alpha=0.001)
                else:
                    ema[k].copy_(v)
        acc["lm"].append(float(lm))
        acc["aux"].append(float(aux))
        if step % 100 == 0 or a.smoke:
            r = {"step": step, "phase": "G" if phase_g else "R", "min": round((time.time() - t0) / 60, 1)}
            r.update({k: round(float(np.mean(v)), 4) for k, v in acc.items() if v})
            acc = {k: [] for k in acc}
            print(json.dumps(r), flush=True)
            log.write(json.dumps(r) + "\n")
            log.flush()
        if step % a.eval_every == 0 or step == total:
            he = held_eval()
            r = {"step": step, "held6_logmel": round(he, 4)}
            print(json.dumps(r), flush=True)
            log.write(json.dumps(r) + "\n")
            log.flush()
            torch.save({"model": model.state_dict(), "ema": ema, "opt": opt.state_dict(), "dopt": dopt.state_dict(),
                        "norm": a.norm, "adv_clf": adv_clf.state_dict() if adv_clf is not None else None,
                        "mpd": mpd.state_dict(), "mrd": mrd.state_dict(), "cqt": cqt.state_dict(), "step": step}, ckp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
