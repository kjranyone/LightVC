from __future__ import annotations

import sys
import argparse
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import librosa

sys.path.insert(0, str(Path(__file__).parent))
from nsf_hn import NsfHifiGan
from train_m1 import mel_of, mrstft_loss, fm_loss, SR, HOP, DEV, N_MELS
from train_m2 import TimbreEncoder, ContentScrub, Discriminator2, grad_reverse
from train_m3 import M3Set, EMB_DIM, ECAPA_PATH
from render_m2 import load_cv, content_of, harvest_f0

REF_SEC = 2.0


class MaleCrossSet(Dataset):
    """male content X -> random female target Z. srcshift F0 to Z register. no GT audio."""
    def __init__(self, male_root: str, female_root: str, ecapa: dict, seg_frames: int = 128) -> None:
        self.files = sorted(Path(male_root).rglob("*.pt"))
        fem = sorted(Path(female_root).rglob("*.pt"))
        self.fem_by_spk = defaultdict(list)
        for f in fem:
            self.fem_by_spk[f.parent.name].append(f)
        self.fem_spks = list(self.fem_by_spk)
        self.seg = seg_frames
        self.ecapa = ecapa
        self._tmed = {}

    def __len__(self) -> int:
        return len(self.files)

    def _wav(self, path):
        w, _ = sf.read(path, dtype="float32")
        if w.ndim > 1:
            w = w.mean(1)
        return torch.from_numpy(np.ascontiguousarray(w)).float()

    def _target_median(self, zf):
        if zf not in self._tmed:
            zd = torch.load(zf, weights_only=False)
            f0 = zd["f0"].float().numpy()
            v = f0[f0 > 1]
            self._tmed[zf] = float(np.median(v)) if len(v) else 200.0
        return self._tmed[zf]

    def __getitem__(self, i):
        try:
            d = torch.load(self.files[i], weights_only=False)
            if "f0" not in d or "content" not in d or "energy" not in d:
                raise KeyError("incomplete")
        except Exception:
            return self.__getitem__((i + 1) % len(self.files))
        content = d["content"].float()
        f0 = d["f0"].float()
        energy = d["energy"].float()
        tmel = f0.shape[0]
        c = F.interpolate(content.t().unsqueeze(0), size=tmel, mode="linear",
                          align_corners=False).squeeze(0)
        smed = float(np.median(f0.numpy()[f0.numpy() > 1]) or 120.0)

        zd = None
        for _ in range(8):
            zspk = random.choice(self.fem_spks)
            zf = random.choice(self.fem_by_spk[zspk])
            try:
                zc = torch.load(zf, weights_only=False)
                if "f0" in zc and "path" in zc:
                    zd = zc; break
            except Exception:
                continue
        if zd is None:
            return self.__getitem__((i + 1) % len(self.files))
        zf0 = zd["f0"].float().numpy(); zv = zf0[zf0 > 1]
        tmed = float(np.median(zv)) if len(zv) else 200.0
        f0s = f0.clone()
        vm = f0s > 1
        f0s[vm] = f0s[vm] * (tmed / max(smed, 1e-3))

        if tmel <= self.seg:
            c = F.pad(c, (0, self.seg - tmel)); f0s = F.pad(f0s, (0, self.seg - tmel))
            energy = F.pad(energy, (0, self.seg - tmel)); s0 = 0
        else:
            s0 = random.randint(0, tmel - self.seg)
        c = c[:, s0:s0 + self.seg]; f0s = f0s[s0:s0 + self.seg]; energy = energy[s0:s0 + self.seg]
        logf0 = torch.log(f0s.clamp(min=1.0)) / 7.0
        eng = torch.log(energy.clamp(min=1e-4)) * 0.2

        rw = self._wav(zd["path"])
        rn = int(REF_SEC * SR)
        rw = rw[:rn] if rw.shape[0] >= rn else F.pad(rw, (0, rn - rw.shape[0]))
        emb = self.ecapa.get(str(zd["path"]))
        ev = 0.0 if emb is None else 1.0
        if emb is None:
            emb = np.zeros(EMB_DIM, np.float32)
        return (c, f0s, logf0.unsqueeze(0), eng.unsqueeze(0), rw,
                torch.from_numpy(np.asarray(emb, np.float32)), ev)


def build_eval(cv, t, ecapa_emb, pairs):
    import pyworld
    ev = []
    for male, tgt in pairs:
        m44, _ = librosa.load(male, sr=SR, mono=True)
        tmel = len(m44) // HOP; m44 = m44[:tmel * HOP]
        m16 = librosa.resample(m44, orig_sr=SR, target_sr=16000)
        c = F.interpolate(content_of(cv, m16).float().t().unsqueeze(0), size=tmel,
                          mode="linear", align_corners=False).to(DEV)
        r44, _ = librosa.load(tgt, sr=SR, mono=True)
        r16, _ = librosa.load(tgt, sr=16000, mono=True)
        rf0, _ = pyworld.harvest(r44.astype(np.float64), SR); tmed = np.median(rf0[rf0 > 1])
        sf0 = harvest_f0(m44)[:tmel]; smed = np.median(sf0[sf0 > 1])
        f0n = sf0.copy(); f0n[sf0 > 1] = sf0[sf0 > 1] * (tmed / smed)
        f0 = torch.from_numpy(f0n).float().to(DEV).clamp(0, 1000)
        seg = m44.reshape(tmel, HOP); e = np.sqrt((seg ** 2).mean(1) + 1e-9).astype(np.float32)
        eng = (torch.log(torch.from_numpy(e).clamp(min=1e-4)) * 0.2).to(DEV)
        s = t(mel_of(torch.from_numpy(r44[:3 * SR]).float().unsqueeze(0).to(DEV)))
        with torch.no_grad():
            tg = ecapa_emb(torch.from_numpy(r16).float().unsqueeze(0).to(DEV))[0]
        ev.append((c, f0, eng, s, tg))
    return ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffeat", default="../data/rcav_feat")
    ap.add_argument("--mfeat", default="../data/male_feat")
    ap.add_argument("--out", default="checkpoints/cipt2")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--fbatch", type=int, default=8)
    ap.add_argument("--xbatch", type=int, default=4)
    ap.add_argument("--xseg", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--init", default="checkpoints/m3b_base.pt")
    ap.add_argument("--w-idout", type=float, default=3.0)
    ap.add_argument("--w-cc", type=float, default=6.0)
    ap.add_argument("--w-xadv", type=float, default=1.0)
    ap.add_argument("--w-breath", type=float, default=0.0)
    ap.add_argument("--aa", action="store_true")
    ap.add_argument("--bigv", action="store_true")
    ap.add_argument("--freeze-te", action="store_true")
    ap.add_argument("--scratch-g", action="store_true")
    ap.add_argument("--recon-only", action="store_true")
    ap.add_argument("--w-sil", type=float, default=0.0)
    ap.add_argument("--noise-std", type=float, default=-1.0)
    ap.add_argument("--w-mod", type=float, default=0.0)
    ap.add_argument("--afhn", action="store_true")
    ap.add_argument("--w-gan", type=float, default=1.0)
    ap.add_argument("--fseg", type=int, default=32)
    ap.add_argument("--uv-noise", type=float, default=-1.0)
    ap.add_argument("--w-mel", type=float, default=45.0)
    ap.add_argument("--nsf2", action="store_true")
    ap.add_argument("--nsf2-ch", type=int, default=64)
    ap.add_argument("--nsf3", action="store_true")
    ap.add_argument("--nsf3-ch", type=int, default=64)
    ap.add_argument("--nsf3-ctrl", type=int, default=128)
    ap.add_argument("--nsf3-base-noise", type=float, default=-1.0)
    ap.add_argument("--nsf3-tilt", type=float, default=0.0)
    ap.add_argument("--mel-cond", action="store_true",
                    help="mel-ceiling test: condition decoder on GT mel instead of content")
    ap.add_argument("--content-raw", action="store_true",
                    help="mel-ceiling arm C: raw ContentVec (bypass ContentScrub) — tests scrub over-removal")
    ap.add_argument("--w-hf", type=float, default=0.0)
    ap.add_argument("--preemph", type=float, default=0.0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ecapa = torch.load(ECAPA_PATH, weights_only=False)
    fds = M3Set(args.ffeat, ecapa, args.fseg)
    fdl = DataLoader(fds, batch_size=args.fbatch, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
    if args.recon_only:
        xds = []
        xdl = None
    else:
        xds = MaleCrossSet(args.mfeat, args.ffeat, ecapa, args.xseg)
        xdl = DataLoader(xds, batch_size=args.xbatch, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)

    ck = torch.load(args.init, map_location=DEV, weights_only=False)
    if args.bigv:
        from nsf_hn_bigv import NsfBigVGAN
        g = NsfBigVGAN(cond_dim=770, timbre_dim=EMB_DIM).to(DEV)
        if args.scratch_g:
            print(f"BigVGAN(snake+AA) | RANDOM init (from-scratch g, {sum(p.numel() for p in g.parameters())/1e6:.1f}M)", flush=True)
        else:
            miss, unexp = g.load_state_dict(ck["g"], strict=False)
            miss = [k for k in miss if "filt" not in k and "log_alpha" not in k and "log_beta" not in k]
            assert not miss and not unexp, (miss[:5], list(unexp)[:5])
            print(f"BigVGAN(snake+AA) | warm-start convs OK ({sum(p.numel() for p in g.parameters())/1e6:.1f}M)", flush=True)
    elif args.aa:
        from nsf_hn_aa import NsfHifiGanAA
        g = NsfHifiGanAA(cond_dim=770, timbre_dim=EMB_DIM).to(DEV)
        miss, unexp = g.load_state_dict(ck["g"], strict=False)
        miss = [k for k in miss if "filt" not in k]
        assert not miss and not unexp, (miss, list(unexp))
        print(f"AA generator | warm-start OK ({sum(p.numel() for p in g.parameters())/1e6:.1f}M)", flush=True)
    elif args.nsf3:
        from nsf_hn3 import NsfHn3
        _cdim = (N_MELS + 2) if args.mel_cond else 770
        g = NsfHn3(cond_dim=_cdim, timbre_dim=EMB_DIM, ch=args.nsf3_ch, ctrl_ch=args.nsf3_ctrl,
                   tilt=args.nsf3_tilt).to(DEV)
        print(f"nsf3 cond_dim={_cdim} ({'MEL-cond ceiling test' if args.mel_cond else 'content'}) "
              f"tilt={args.nsf3_tilt}", flush=True)
        if not args.scratch_g and any(k.startswith("blocks") for k in ck["g"]):
            miss, unexp = g.load_state_dict(ck["g"], strict=False)
            loaded = len(g.state_dict()) - len(miss)
            mb = sum(p.numel() for p in g.parameters()) / 1e6
            if not miss and not unexp:
                print(f"NSF-HN3 | full warm-start ({mb:.2f}M, {loaded} tensors, HN3->HN3)", flush=True)
            else:
                print(f"NSF-HN3 | warm-start ({mb:.2f}M): only {loaded} tensors loaded "
                      f"(weight_norm renames conv keys -> non-HN3 plain weights DROPPED; "
                      f"effectively scratch+FiLM). miss{len(miss)} unexp{len(unexp)}. "
                      f"Use --scratch-g unless a key-conversion loader is added.", flush=True)
        else:
            print(f"NSF-HN3 | RANDOM init from-scratch ({sum(p.numel() for p in g.parameters())/1e6:.2f}M)", flush=True)
    elif args.nsf2:
        from nsf_hn2 import NsfHn2
        g = NsfHn2(cond_dim=770, timbre_dim=EMB_DIM, ch=args.nsf2_ch).to(DEV)
        if not args.scratch_g and any(k.startswith("blocks") for k in ck["g"]):
            g.load_state_dict(ck["g"])
            print(f"NSF-HN2 | warm-start ({sum(p.numel() for p in g.parameters())/1e6:.2f}M)", flush=True)
        else:
            print(f"NSF-HN2 | RANDOM init from-scratch ({sum(p.numel() for p in g.parameters())/1e6:.2f}M)", flush=True)
    elif args.afhn:
        from afhn import AFHN
        g = AFHN(cond_dim=770, timbre_dim=EMB_DIM).to(DEV)
        if not args.scratch_g and any(k.startswith("conv_pre") for k in ck["g"]):
            miss, unexp = g.load_state_dict(ck["g"], strict=False)
            print(f"AFHN | partial warm-start ({sum(p.numel() for p in g.parameters())/1e6:.2f}M) miss{len(miss)} unexp{len(unexp)}", flush=True)
        else:
            print(f"AFHN | RANDOM init from-scratch ({sum(p.numel() for p in g.parameters())/1e6:.2f}M)", flush=True)
    else:
        g = NsfHifiGan(cond_dim=770, timbre_dim=EMB_DIM).to(DEV); g.load_state_dict(ck["g"])
    _has_src = hasattr(g, "m_source")
    if args.noise_std >= 0 and _has_src:
        g.m_source.sine_gen.noise_std = args.noise_std
    if _has_src and args.uv_noise >= 0:
        g.m_source.sine_gen.uv_noise = args.uv_noise
    if (args.nsf2 or args.nsf3) and args.uv_noise >= 0:
        g.uv_noise = args.uv_noise
    if args.nsf3 and args.nsf3_base_noise >= 0:
        g.base_noise = args.nsf3_base_noise
        print(f"nsf3 base_noise floor -> {g.base_noise}", flush=True)
    BN = g.base_noise if args.nsf3 else -1.0
    UVN = (g.uv_noise if (args.nsf2 or args.nsf3) else
           (g.m_source.sine_gen.uv_noise if _has_src else args.uv_noise))
    NS = g.m_source.sine_gen.noise_std if _has_src else float(getattr(getattr(g, "exc", None), "noise_std", -1.0))
    t = TimbreEncoder(dim=EMB_DIM).to(DEV); t.load_state_dict(ck["t"])
    scrub = ContentScrub().to(DEV); scrub.load_state_dict(ck["scrub"])
    cemb_pred = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256), nn.LeakyReLU(0.1),
                              nn.Linear(256, EMB_DIM)).to(DEV)
    d = Discriminator2().to(DEV)
    cv = load_cv()
    for p in cv.parameters():
        p.requires_grad_(False)
    from speechbrain.inference.speaker import EncoderClassifier
    sb = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                        savedir="hf_models/spkrec-ecapa", run_opts={"device": str(DEV)})
    enet = sb.mods.to(DEV)
    for p in enet.parameters():
        p.requires_grad_(False)

    def ecapa_emb(w16):
        feats = enet.compute_features(w16)
        feats = enet.mean_var_norm(feats, torch.ones(w16.shape[0], device=DEV))
        e = enet.embedding_model(feats, torch.ones(w16.shape[0], device=DEV)).squeeze(1)
        return e / (e.norm(dim=-1, keepdim=True) + 1e-6)

    def to16(y):
        return F.interpolate(y.unsqueeze(1), scale_factor=16000 / SR, mode="linear",
                             align_corners=False).squeeze(1)

    def cvec(w16, T):
        h = cv(w16).last_hidden_state
        return F.interpolate(h.transpose(1, 2), size=T, mode="linear", align_corners=False)

    if args.freeze_te:
        for p in t.parameters():
            p.requires_grad = False
        for p in scrub.parameters():
            p.requires_grad = False
        t.eval(); scrub.eval()
        gp = list(g.parameters())
        print("frozen: TimbreEncoder + ContentScrub (train g only)", flush=True)
    else:
        gp = list(g.parameters()) + list(t.parameters()) + list(scrub.parameters()) + list(cemb_pred.parameters())
    og = torch.optim.AdamW(gp, args.lr, betas=(0.8, 0.99))
    od = torch.optim.AdamW(d.parameters(), args.lr, betas=(0.8, 0.99))
    if not args.scratch_g and "d" in ck:
        try:
            d.load_state_dict(ck["d"]); od.load_state_dict(ck["od"]); og.load_state_dict(ck["og"])
            for opt in (og, od):                       # resume moments but honor current --lr
                for grp in opt.param_groups:
                    grp["lr"] = args.lr
            print(f"resumed discriminator + optimizer state (lr override -> {args.lr})", flush=True)
        except Exception as ex:
            print(f"opt/d resume skipped: {type(ex).__name__}", flush=True)

    pairs = [("../data/male_tts_corpus/male_p255/t00_calm_low.wav",
              "/home/kojirotanaka/kjranyone/LightVC/data/female_tts_corpus/0be279f19f4bdda1/t05_intimate_close.wav"),
             ("../data/male_tts_corpus/male_p226/t00_calm_low.wav",
              "/home/kojirotanaka/kjranyone/LightVC/data/female_tts_corpus/07556647e8c0c25e/t19_cute_high.wav")]
    evset = build_eval(cv, t, ecapa_emb, pairs)

    _win = torch.hann_window(2048, device=DEV)
    _freqs = torch.linspace(0, SR / 2, 1025, device=DEV)
    _hi = _freqs >= 4000

    def spec_flat(y):
        S = torch.stft(y, n_fft=2048, hop_length=512, window=_win, return_complex=True).abs() ** 2
        P = S[(_freqs >= 2000) & (_freqs < 8000)].mean(1) + 1e-10
        return float(torch.exp(torch.log(P).mean()) / P.mean())

    def breath_db(y, eng):
        S = torch.stft(y, n_fft=2048, hop_length=512, window=_win, return_complex=True).abs()
        Tf = S.shape[1]
        e = F.interpolate(eng.view(1, 1, -1), size=Tf, mode="linear", align_corners=False).squeeze()
        m = e <= (e.mean() - 1.0 * e.std())
        if m.sum() < 1:
            return 0.0
        P = (S[_hi][:, m] ** 2).mean()
        return float(10 * torch.log10(P + 1e-10))

    def sil_db(y, eng):
        nfr = y.shape[-1] // HOP
        yr = y[:nfr * HOP].reshape(nfr, HOP).pow(2).mean(-1).add(1e-9).sqrt()
        e = eng.view(-1)[:nfr]
        m = e <= torch.quantile(e, 0.15)
        if m.sum() < 1:
            return 0.0
        return float(20 * torch.log10(yr[m].mean() + 1e-9))

    def mod_metric(y, f0):
        K = 16
        nf = y.shape[-1] // HOP
        yb = y[:nf * HOP].reshape(nf, K, HOP // K)
        subp = yb.pow(2).mean(-1)
        prof = subp / (subp.mean(-1, keepdim=True) + 1e-9)
        rms = yb.pow(2).mean((-1, -2)).add(1e-9).sqrt()
        uv = (f0[:nf] < 10) & (rms > 0.03 * rms.max())
        if uv.sum() < 1:
            return 0.0
        return float(prof[uv].mean(0).var())

    @torch.no_grad()
    def eval_secs():
        if args.mel_cond:
            return [0.0], [0.0], [0.0], [0.0], [0.0]   # cross-eval N/A for mel-ceiling test
        g.eval(); t.eval(); scrub.eval(); res = []; fl = []; br = []; sl = []; md = []
        for c, f0, eng, s, tg in evset:
            logf0 = (torch.log(f0.clamp(min=1.0)) / 7.0).view(1, 1, -1)
            cond = torch.cat([scrub(c), logf0, eng.view(1, 1, -1)], 1)
            y = g(cond, f0.view(1, -1), s).squeeze(1)
            res.append(float((ecapa_emb(to16(y))[0] * tg).sum()))
            fl.append(spec_flat(y[0])); br.append(breath_db(y[0], eng)); sl.append(sil_db(y[0], eng))
            md.append(mod_metric(y[0], f0))
        g.train(); t.train(); scrub.train(); return res, fl, br, sl, md

    base, basef, baseb, bases, basem = eval_secs()
    print(f"CIPT-A2 | DEV={DEV} | female {len(fds)} / male {len(xds)} | idout={args.w_idout} xadv={args.w_xadv} wbreath={args.w_breath} | "
          f"SECS {[f'{v:+.3f}' for v in base]} flat {[f'{v:.3f}' for v in basef]} sil_dB {[f'{v:.1f}' for v in bases]} mod {[f'{v:.4f}' for v in basem]}", flush=True)

    xit = iter(xdl) if xdl is not None else None
    _z = torch.zeros((), device=DEV)
    step = 0
    while step < args.steps:
        for cond, f0, logf0, voiced, y, rw, emb, ev in fdl:
            cond, f0, y, rw, emb, ev = cond.to(DEV), f0.to(DEV), y.to(DEV), rw.to(DEV), emb.to(DEV), ev.to(DEV)
            voiced = voiced.to(DEV)
            if not args.recon_only:
                try:
                    xc, xf0, xlogf0, xeng, xrw, xemb, xev = next(xit)
                except StopIteration:
                    xit = iter(xdl); xc, xf0, xlogf0, xeng, xrw, xemb, xev = next(xit)
                xc, xf0, xlogf0, xeng, xrw, xemb, xev = (xc.to(DEV), xf0.to(DEV), xlogf0.to(DEV), xeng.to(DEV),
                                                         xrw.to(DEV), xemb.to(DEV), xev.to(DEV))

            # ---- self-recon (female, with GT) ----
            s = t(mel_of(rw))
            id_loss = ((1.0 - F.cosine_similarity(s, emb, dim=-1)) * ev).sum() / (ev.sum() + 1e-6)
            if args.mel_cond:
                # mel-ceiling test: condition on GT mel (full spectral envelope) instead
                # of content. Same decoder/loss/D/steps -> the recon-quality gap vs the
                # content arm attributes the muffle to conditioning (mel) vs decoder.
                melc = mel_of(y.squeeze(1))                              # (B, N_MELS, Tm)
                if melc.shape[-1] != cond.shape[-1]:
                    melc = F.interpolate(melc, size=cond.shape[-1], mode="linear", align_corners=False)
                c_adv = _z
                y_hat = g(torch.cat([melc, cond[:, 768:]], 1), f0, s)[..., : y.shape[-1]]
            elif args.content_raw:
                # arm C: raw ContentVec, no scrub, no speaker-adversarial (tests scrub over-removal)
                cs = cond[:, :768]
                c_adv = _z
                y_hat = g(torch.cat([cs, cond[:, 768:]], 1), f0, s)[..., : y.shape[-1]]
            else:
                cs = scrub(cond[:, :768])
                c_adv = (F.mse_loss(cemb_pred(grad_reverse(cs.mean(-1), 0.5)) * ev.unsqueeze(-1),
                                    emb * ev.unsqueeze(-1), reduction="sum") / (ev.sum() * EMB_DIM + 1e-6))
                y_hat = g(torch.cat([cs, cond[:, 768:]], 1), f0, s)[..., : y.shape[-1]]
            if args.preemph > 0:
                # pre-emphasis before mel/mrs: boosts HF so the recon loss stops
                # tolerating the bass-heavy tilt (muffled). y'[n]=y[n]-a*y[n-1].
                ymel = y.squeeze(1); yhmel = y_hat.squeeze(1)
                ymel = ymel - args.preemph * F.pad(ymel, (1, 0))[:, :-1]
                yhmel = yhmel - args.preemph * F.pad(yhmel, (1, 0))[:, :-1]
                mel_l = F.l1_loss(mel_of(yhmel), mel_of(ymel))
                mrs = mrstft_loss(yhmel, ymel)
            else:
                mel_l = F.l1_loss(mel_of(y_hat.squeeze(1)), mel_of(y.squeeze(1)))
                mrs = mrstft_loss(y_hat.squeeze(1), y.squeeze(1))
            # silence: where GT is silent, push output amplitude toward 0 (kill floor noise)
            yh1 = y_hat.squeeze(1); y1 = y.squeeze(1)
            # HF-emphasis: log-mag STFT L1 above 4kHz (mel-L1 under-weights HF -> muffled)
            if args.w_hf > 0:
                Syh = torch.stft(yh1, 2048, 512, window=_win, return_complex=True).abs()
                Syt = torch.stft(y1, 2048, 512, window=_win, return_complex=True).abs()
                hf_l = F.l1_loss(torch.log(Syh[:, _hi] + 1e-5), torch.log(Syt[:, _hi] + 1e-5))
            else:
                hf_l = _z
            nfr = yh1.shape[-1] // HOP
            yhr = yh1[:, :nfr * HOP].reshape(yh1.shape[0], nfr, HOP).pow(2).mean(-1).add(1e-9).sqrt()
            gtr = y1[:, :nfr * HOP].reshape(y1.shape[0], nfr, HOP).pow(2).mean(-1).add(1e-9).sqrt()
            silm = (gtr < 0.02 * gtr.amax(-1, keepdim=True)).float()
            sil_loss = (yhr * silm).sum() / (silm.sum() + 1e-6)
            # anti frame-rate (86Hz) modulation of unvoiced/breath noise (the "jiri-jiri")
            K = 16
            yb = yh1[:, :nfr * HOP].reshape(yh1.shape[0], nfr, K, HOP // K)
            subp = yb.pow(2).mean(-1)
            prof = subp / (subp.mean(-1, keepdim=True) + 1e-9)
            vfr = voiced[:, :nfr] if voiced.shape[-1] >= nfr else F.pad(voiced, (0, nfr - voiced.shape[-1]))
            uvm = ((vfr < 0.5) & (gtr > 0.03 * gtr.amax(-1, keepdim=True))).float()
            avgp = (prof * uvm.unsqueeze(-1)).sum((0, 1)) / (uvm.sum() + 1e-6)
            mod_loss = avgp.var()

            # ---- cross (male -> female) ----
            if args.recon_only:
                idout = cc = breath = gx_adv = _z
            else:
                xs = t(mel_of(xrw))
                xy = g(torch.cat([scrub(xc), xlogf0, xeng], 1), xf0, xs).squeeze(1)
                xw16 = to16(xy)
                idout = ((1.0 - F.cosine_similarity(ecapa_emb(xw16), xemb, dim=-1)) * xev).sum() / (xev.sum() + 1e-6)
                cc = F.l1_loss(cvec(xw16, xc.shape[-1]), xc)
                # breath cleanliness: where input energy is low, suppress high-freq magnitude
                Sx = torch.stft(xy, n_fft=2048, hop_length=512, window=_win, return_complex=True).abs()
                hi_e = Sx[:, _hi, :].mean(1)
                eln = xeng.squeeze(1)
                if hi_e.shape[1] != eln.shape[1]:
                    hi_e = F.interpolate(hi_e.unsqueeze(1), size=eln.shape[1], mode="linear", align_corners=False).squeeze(1)
                lowm = (eln <= (eln.mean(1, keepdim=True) - 1.0 * eln.std(1, keepdim=True))).float()
                breath = (hi_e * lowm).sum() / (lowm.sum() + 1e-6)

            # ---- discriminator (reals=female y; fakes=self y_hat [+ cross xy]) ----
            od.zero_grad()
            dr, _ = d(y)
            dg1, _ = d(y_hat.detach())
            d_loss = sum(((r - 1) ** 2).mean() for r in dr) + sum((gg ** 2).mean() for gg in dg1)
            if not args.recon_only:
                dg2, _ = d(xy.detach().unsqueeze(1))
                d_loss = d_loss + sum((gg ** 2).mean() for gg in dg2)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(d.parameters(), 10.0)
            od.step()

            og.zero_grad()
            dg, fg = d(y_hat); dr, fr = d(y)
            g_adv = sum(((gg - 1) ** 2).mean() for gg in dg)
            if not args.recon_only:
                dgx, _ = d(xy.unsqueeze(1))
                gx_adv = sum(((gg - 1) ** 2).mean() for gg in dgx)
            g_loss = (args.w_mel * mel_l + 2.0 * mrs + 3.0 * id_loss + 0.1 * c_adv + args.w_gan * g_adv + args.w_gan * 2.0 * fm_loss(fr, fg)
                      + args.w_idout * idout + args.w_cc * cc + args.w_xadv * gx_adv + args.w_breath * breath
                      + args.w_sil * sil_loss + args.w_mod * mod_loss + args.w_hf * hf_l)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(gp, 10.0)
            og.step()

            if step % 50 == 0:
                print(f"step {step} mel {mel_l.item():.3f} id {id_loss.item():.3f} "
                      f"idout {idout.item():.3f} cc {cc.item():.3f} gxadv {gx_adv.item():.2f} "
                      f"breath {breath.item():.4f} sil {sil_loss.item():.5f} mod {mod_loss.item():.4f} "
                      f"hf {hf_l.item():.3f}", flush=True)
            if step % 500 == 0 and step > 0:
                es, esf, esb, ess, esm = eval_secs()
                print(f"  [eval {step}] SECS {[f'{v:+.3f}' for v in es]} flat {[f'{v:.3f}' for v in esf]} "
                      f"sil_dB {[f'{v:.1f}' for v in ess]} mod {[f'{v:.4f}' for v in esm]}  (base sil {[f'{v:.1f}' for v in bases]} mod {[f'{v:.4f}' for v in basem]})", flush=True)
                snap = {"g": g.state_dict(), "t": t.state_dict(), "scrub": scrub.state_dict(), "step": step,
                        "noise_std": NS, "uv_noise": UVN, "nsf2_ch": args.nsf2_ch,
                        "nsf3_ch": args.nsf3_ch, "nsf3_ctrl": args.nsf3_ctrl, "nsf3_base_noise": BN,
                        "nsf3_tilt": args.nsf3_tilt, "mel_cond": args.mel_cond, "content_raw": args.content_raw,
                        "d": d.state_dict(), "og": og.state_dict(), "od": od.state_dict()}
                torch.save(snap, out / "last.pt")
                if step % 1000 == 0:
                    torch.save(snap, out / f"snap_{step}.pt")
            step += 1
            if step >= args.steps:
                break
    torch.save({"g": g.state_dict(), "t": t.state_dict(), "scrub": scrub.state_dict(), "step": step,
                "noise_std": NS, "uv_noise": UVN, "nsf2_ch": args.nsf2_ch,
                "nsf3_ch": args.nsf3_ch, "nsf3_ctrl": args.nsf3_ctrl, "nsf3_base_noise": BN,
                "nsf3_tilt": args.nsf3_tilt, "mel_cond": args.mel_cond, "content_raw": args.content_raw,
                "d": d.state_dict(), "og": og.state_dict(), "od": od.state_dict()}, out / "last.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
