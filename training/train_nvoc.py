"""nvoc の学習(女声フルコーパスのコピー合成)。事前登録 results/<tag>/prereg.yaml。

  入力: 因果 log-mel(x・左文脈つき)+ 因果 YIN f0(data/artic_feat)。出力 y[m] ≈ x[m − DELAY]。
  損失: 15·多尺度 logmel + LSGAN + 2·FM。判別器は BigVGAN v2 44kHz の学習済み MPD + CQTD(MIT)を初期値にし、
  出力と正解を 44.1kHz へ再標本化・同じ利得(正解の尖頭値を 0.95 に)で 16384 サンプル切り出して入れる。
  判別器の計算だけ bf16(生成器は fp32: 出力の量子化雑音を判別器に読ませない)。

    CUDA_VISIBLE_DEVICES=0 uv run python train_nvoc.py --tag nvoc1 --steps 400000
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
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
from train_s1_1 import MEL_SPECS, build_mel, logmel_l1, mrstft

ROOT = Path(__file__).resolve().parent.parent
SEG = 57600
W0 = 12000
CTX = N.WIN - N.HOP
BIGVGAN = Path.home() / ".cache/huggingface/hub/models--nvidia--bigvgan_v2_44khz_128band_512x/snapshots"


class DS(torch.utils.data.IterableDataset):
    def __init__(self, spk: dict, keys: list, seed: int):
        self.flat = [spk[k][i] for k in keys for i in range(len(spk[k]))]
        self.seed = seed

    def __iter__(self):
        from train_ddsp_vc import load48
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 1009 + (wi.id if wi else 0))
        T = SEG // N.HOP
        while True:
            fz, fw, _ = self.flat[rng.randrange(len(self.flat))]
            try:
                x = load48(fw)
                if len(x) < SEG + 8 * N.HOP:
                    continue
                u = rng.randrange(4, (len(x) - SEG) // N.HOP)
                s = u * N.HOP
                xin = x[s - CTX:s + SEG]
                tgt = x[s - N.DELAY:s + SEG - N.DELAY]
                if np.sqrt((tgt[W0:] ** 2).mean()) < 1e-3:
                    continue
                f0 = np.load(fz)["f0"].astype(np.float32)[u:u + T]
                f0 = np.pad(f0, (0, T - len(f0)))
                g = 2.0 ** rng.uniform(-1.0, 1.0)
                pk = max(float(np.abs(xin).max()), 1e-6)
                g = min(g, 0.99 / pk)
                yield torch.from_numpy(xin * g), torch.from_numpy(tgt * g), torch.from_numpy(f0)
            except Exception:
                continue


def check_alignment() -> None:
    T = SEG // N.HOP
    x = np.arange(SEG + 20 * N.HOP, dtype=np.float32)
    u = 5
    s = u * N.HOP
    xin, tgt = x[s - CTX:s + SEG], x[s - N.DELAY:s + SEG - N.DELAY]
    for m in (0, 1, 777, SEG - 1):
        assert tgt[m] == xin[CTX + m - N.DELAY], "target alignment"
    for t in (0, 3, T - 1):
        assert xin[t * N.HOP + N.WIN - 1] == s + (t + 1) * N.HOP - 1, "frame end alignment"


def discriminators(dev: str, pretrained: bool):
    from types import SimpleNamespace
    from bigvgan.discriminators import MultiPeriodDiscriminator, MultiScaleSubbandCQTDiscriminator
    snap = sorted(BIGVGAN.iterdir())[-1]
    h = json.loads((snap / "config.json").read_text())
    mpd = MultiPeriodDiscriminator(SimpleNamespace(mpd_reshapes=h["mpd_reshapes"], use_spectral_norm=h["use_spectral_norm"],
                                                   discriminator_channel_mult=h["discriminator_channel_mult"]))
    cqt = MultiScaleSubbandCQTDiscriminator(h)
    if pretrained:
        d = torch.load(snap / "bigvgan_discriminator_optimizer.pt", map_location="cpu", weights_only=False)
        mpd.load_state_dict(d["mpd"])
        cqt.load_state_dict(d["mrd"])
    return mpd.to(dev), cqt.to(dev)


class MRDLog(torch.nn.Module):
    """対数振幅 STFT 上の多解像度判別器(全帯域を同じ重みで見る・48kHz 入力)。UnivNet の MRD の層構成で入力だけ対数にしたもの。"""
    RES = ((512, 128), (1024, 256), (2048, 512))

    def __init__(self):
        super().__init__()
        from torch.nn.utils.parametrizations import weight_norm as wn
        self.ds = torch.nn.ModuleList()
        for _ in self.RES:
            self.ds.append(torch.nn.ModuleList(
                [wn(torch.nn.Conv2d(1, 32, (3, 9), padding=(1, 4)))]
                + [wn(torch.nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))) for _ in range(3)]
                + [wn(torch.nn.Conv2d(32, 32, (3, 3), padding=(1, 1))), wn(torch.nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))]))

    def spec(self, y: torch.Tensor, nf: int, hop: int) -> torch.Tensor:
        S = torch.stft(y.float(), nf, hop, nf, torch.hann_window(nf, device=y.device), return_complex=True).abs()
        return torch.log(S.clamp(min=1e-5)).transpose(1, 2)[:, None]

    def one(self, layers, x: torch.Tensor):
        fm = []
        for i, l in enumerate(layers):
            x = l(x)
            if i < len(layers) - 1:
                x = F.leaky_relu(x, 0.1)
                fm.append(x)
        return x.flatten(1), fm

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        rs, gs, frs, fgs = [], [], [], []
        for (nf, hop), layers in zip(self.RES, self.ds):
            a, b = self.spec(y, nf, hop), self.spec(y_hat, nf, hop)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                r, fr = self.one(layers, a)
                g, fg = self.one(layers, b)
            rs.append(r)
            gs.append(g)
            frs.append(fr)
            fgs.append(fg)
        return rs, gs, frs, fgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="nvoc1")
    ap.add_argument("--steps", type=int, default=400000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--dcrop", type=int, default=16384, help="判別器に入れる 44.1kHz の切り出し長")
    ap.add_argument("--lr_g", type=float, default=2e-4)
    ap.add_argument("--lr_d", type=float, default=1e-4)
    ap.add_argument("--decay", type=float, default=0.999998)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--snap_every", type=int, default=50000)
    ap.add_argument("--fresh_d", action="store_true")
    ap.add_argument("--nogan", action="store_true", help="診断: 判別器なし(15·logmel + 2·mrstft)")
    ap.add_argument("--r_steps", type=int, default=0, help="この step までは判別器なしの再構成(段 R)・以降 GAN(段 G)")
    ap.add_argument("--d_warm", type=int, default=0, help="段 G の最初のこの step 数は判別器だけを学習(生成器は再構成損失のまま)")
    ap.add_argument("--init_g", default=None, help="生成器の初期値(EMA を含む ckpt)")
    ap.add_argument("--init_d", default=None, help="判別器の初期値(mpd/cqt を含む ckpt)")
    ap.add_argument("--w_mel", type=float, default=15.0)
    ap.add_argument("--w_fm", type=float, default=2.0)
    ap.add_argument("--mrd_log", action="store_true", help="対数振幅の多解像度判別器を足す(新規初期化)")
    ap.add_argument("--d_hfview", type=float, default=0.0, help="判別器に高域強調(1 − c·z⁻¹)の版も並べて入れる(c・0 で無効)")
    ap.add_argument("--noncausal", action="store_true", help="診断: 生成器の畳み込みを中心揃え(未来を見る)")
    ap.add_argument("--ch", type=int, default=256)
    ap.add_argument("--nosrc", action="store_true", help="診断: 調波源を 0 にする(雑音のみ)")
    ap.add_argument("--aa", action="store_true", help="診断: 段 2・3 の活性化を因果の折り返し防止版にする(nvoc_aa.NVocAA)")
    ap.add_argument("--f0_adv", type=int, default=0, help="診断: f0 を k フレーム先取りさせる(未来を見る・製品不可)")
    ap.add_argument("--delay", type=int, default=N.DELAY, help="出力の遅れ(サンプル・HOP の倍数)。y[m] ≈ x[m − delay]")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    assert a.delay % N.HOP == 0 and a.delay >= N.HOP
    N.DELAY = a.delay
    check_alignment()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    dev = "cuda"
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"prereg: results/{a.tag}/prereg.yaml", flush=True)
    from train_ddsp_vc import index
    import eval_nvoc as E
    spk, tr, ev = index()
    print(f"train speakers {len(tr)} (utts {sum(len(spk[k]) for k in tr)}) | held-out speakers {len(ev)}", flush=True)
    loader = iter(torch.utils.data.DataLoader(DS(spk, tr, 1 if not a.resume else int(time.time()) % 100000), batch_size=a.bs,
                                              num_workers=a.workers, pin_memory=True, persistent_workers=True, prefetch_factor=4))
    if a.aa:
        import nvoc_aa as NA
        mk = lambda: NA.NVocAA(ch=a.ch, causal=not a.noncausal)
    else:
        mk = lambda: N.NVoc(ch=a.ch, causal=not a.noncausal)
    model = mk().to(dev)
    model.cfg["delay"] = N.DELAY
    print("cfg", json.dumps(model.cfg), "| GMAC/s", round(N.macs_per_second(model) / 1e9, 2),
          "| params (M)", round(sum(p.numel() for p in model.parameters()) / 1e6, 2), flush=True)
    evm = mk().to(dev)
    mpd, cqt = discriminators(dev, not a.fresh_d)
    if a.init_d:
        sd = torch.load(a.init_d, map_location=dev, weights_only=False)
        mpd.load_state_dict(sd["mpd"])
        cqt.load_state_dict(sd["cqt"])
        print("discriminators init from", a.init_d, flush=True)
    if a.nogan:
        mpd.requires_grad_(False)
        cqt.requires_grad_(False)
    gparams = list(model.parameters())
    mrd = MRDLog().to(dev) if a.mrd_log else None
    if mrd is not None and a.init_d:
        sd_ = torch.load(a.init_d, map_location=dev, weights_only=False)
        if sd_.get("mrd") is not None:
            mrd.load_state_dict(sd_["mrd"])
            print("mrd init from", a.init_d, flush=True)
    dparams = list(mpd.parameters()) + list(cqt.parameters()) + (list(mrd.parameters()) if mrd is not None else [])
    opt = torch.optim.AdamW(gparams, lr=a.lr_g, betas=(0.8, 0.99))
    dopt = torch.optim.AdamW(dparams, lr=a.lr_d, betas=(0.8, 0.99))
    sch = torch.optim.lr_scheduler.ExponentialLR(opt, a.decay)
    dsch = torch.optim.lr_scheduler.ExponentialLR(dopt, a.decay)
    if a.init_g:
        model.load_state_dict(torch.load(a.init_g, map_location=dev, weights_only=False)["ema"])
        print("generator init from", a.init_g, flush=True)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    mels = [build_mel(nf, hm, nm).to(dev) for nf, hm, nm in MEL_SPECS]
    step = 0
    ckp = out / "last.pt"
    if a.resume and ckp.exists():
        st = torch.load(ckp, map_location=dev, weights_only=False)
        model.load_state_dict(st["model"])
        ema = st["ema"]
        opt.load_state_dict(st["opt"])
        dopt.load_state_dict(st["dopt"])
        mpd.load_state_dict(st["mpd"])
        cqt.load_state_dict(st["cqt"])
        sch.load_state_dict(st["sch"])
        dsch.load_state_dict(st["dsch"])
        step = int(st["step"])
        print("resumed at", step, flush=True)
    items = E.held_items()
    if a.nosrc:
        items = [{**it, "f0": np.zeros_like(it["f0"])} for it in items]
    if a.f0_adv:
        items = [{**it, "f0": np.concatenate([it["f0"][a.f0_adv:], np.repeat(it["f0"][-1:], a.f0_adv)])} for it in items]
    log = open(out / "train.jsonl", "a")

    def evaluate() -> dict:
        evm.load_state_dict(ema)
        r = {"step": step, "held": E.held_eval(evm, dev, items[:6] if a.smoke else items)}
        model.train()
        return r

    if step == 0:
        r0 = evaluate()
        print("held at step 0", json.dumps(r0["held"]), "| 錨 BigVGAN: results/nvoc1/anchor_bigvgan.json", flush=True)
        log.write(json.dumps(r0) + "\n")
        log.flush()

    def d_in(y: torch.Tensor, g: torch.Tensor, off: int, g2: torch.Tensor | None = None):
        z = AF.resample(y * g, N.SR, 44100)
        z = z[:, off:off + a.dcrop] if z.shape[-1] > a.dcrop else z
        if a.d_hfview <= 0:
            return z[:, None], None
        h = F.pad(z[:, 1:] - a.d_hfview * z[:, :-1], (1, 0))
        if g2 is None:
            g2 = (0.95 / h.abs().amax(-1, keepdim=True).clamp(min=1e-4)).clamp(max=50.0)
        return torch.cat([z, h * g2], 0)[:, None], g2

    def feat_loss(fr, fg):
        return sum(F.l1_loss(x.detach().float(), y.float()) for A, B in zip(fr, fg) for x, y in zip(A, B))

    t0 = time.time()
    acc: dict = {"lm": [], "adv": [], "fm": [], "dl": [], "gn": [], "dr": [], "df": [], "mr": [], "mf": []}
    total = 30 if a.smoke else a.steps
    n_bad = 0
    model.train()
    while step < total:
        step += 1
        xin, tgt, f0 = (t.to(dev, non_blocking=True) for t in next(loader))
        if a.f0_adv:
            f0 = torch.cat([f0[:, a.f0_adv:], f0[:, -1:].repeat(1, a.f0_adv)], 1)
        with torch.no_grad():
            mel = model.mel_ctx(xin)
            exc = torch.stack([N.harmonic_source(f0 * (0.0 if a.nosrc else 1.0)), torch.randn(xin.shape[0], SEG, device=dev)], 1)
        y = model.generate(mel, exc)
        yl, tl = y[:, W0:], tgt[:, W0:]
        phase_g = not a.nogan and step > a.r_steps
        if not phase_g:
            lm = logmel_l1(yl[:, None], tl[:, None], mels)
            adv = fm = dl = torch.zeros((), device=dev)
            loss = 15 * lm + 2 * mrstft(yl[:, None], tl[:, None])
        else:
            with torch.no_grad():
                g = (0.95 / tl.abs().amax(-1, keepdim=True).clamp(min=1e-3)).clamp(max=20.0)
                n44 = int(math.floor(tl.shape[-1] * 44100 / N.SR))
                off = random.randrange(0, max(1, n44 - a.dcrop))
                r44, g2 = d_in(tl, g, off)
            f44, _ = d_in(yl, g, off, g2)
            dopt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                yr, yg, _, _ = mpd(r44, f44.detach())
                cr, cg, _, _ = cqt(r44, f44.detach())
            dl = sum(((p.float() - 1) ** 2).mean() + (q.float() ** 2).mean() for p, q in zip(yr + cr, yg + cg))
            if mrd is not None:
                moff = random.randrange(0, max(1, tl.shape[-1] - 32768))
                tlc = (tl * g)[:, moff:moff + 32768]
                mr, mg, _, _ = mrd(tlc, (yl * g)[:, moff:moff + 32768].detach())
                dl = dl + sum(((p.float() - 1) ** 2).mean() + (q.float() ** 2).mean() for p, q in zip(mr, mg))
                acc["mr"].append(float(sum(p.float().mean() for p in mr) / len(mr)))
                acc["mf"].append(float(sum(q.float().mean() for q in mg) / len(mg)))
            if torch.isfinite(dl):
                dl.backward()
                dn = torch.nn.utils.clip_grad_norm_(dparams, 500.0)
                if torch.isfinite(dn):
                    dopt.step()
            acc["dr"].append(float(sum(p.float().mean() for p in yr + cr) / len(yr + cr)))
            acc["df"].append(float(sum(q.float().mean() for q in yg + cg) / len(yg + cg)))
            lm = logmel_l1(yl[:, None], tl[:, None], mels)
            for p in dparams:
                p.requires_grad_(False)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, yg, fr, fg = mpd(r44, f44)
                _, cg, cfr, cfg = cqt(r44, f44)
            adv = sum(((q.float() - 1) ** 2).mean() for q in yg + cg)
            fm = feat_loss(fr, fg) + feat_loss(cfr, cfg)
            if mrd is not None:
                _, mg, mfr, mfg = mrd(tlc, (yl * g)[:, moff:moff + 32768])
                adv = adv + sum(((q.float() - 1) ** 2).mean() for q in mg)
                fm = fm + feat_loss(mfr, mfg)
            loss = a.w_mel * lm + adv + a.w_fm * fm
            if step <= a.r_steps + a.d_warm:
                loss = 15 * lm + 2 * mrstft(yl[:, None], tl[:, None])
            for p in dparams:
                p.requires_grad_(True)
        opt.zero_grad(set_to_none=True)
        bad = not torch.isfinite(loss)
        if not bad:
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(gparams, 500.0)
            bad = not torch.isfinite(gn)
        if bad:
            print("non-finite at", step, "→ skip", flush=True)
            opt.zero_grad(set_to_none=True)
            n_bad += 1
            if n_bad >= 20:
                raise RuntimeError(f"non-finite 20 steps in a row at {step}; last finite save is last.pt")
            continue
        n_bad = 0
        opt.step()
        sch.step()
        if phase_g:
            dsch.step()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(0.999).add_(v.detach(), alpha=0.001)
                else:
                    ema[k].copy_(v)
        for k, v in (("lm", lm), ("adv", adv), ("fm", fm), ("dl", dl), ("gn", gn)):
            acc[k].append(float(v))
        if step % 100 == 0 or a.smoke:
            r = {"step": step, "phase": "G" if phase_g else "R", "min": round((time.time() - t0) / 60, 1)}
            r.update({k: round(float(np.mean(v)), 4) for k, v in acc.items() if v})
            acc = {k: [] for k in acc}
            print(json.dumps(r), flush=True)
            log.write(json.dumps(r) + "\n")
            log.flush()
        if step % a.eval_every == 0 or (a.smoke and step == total):
            r = evaluate()
            print("held", json.dumps(r), flush=True)
            log.write(json.dumps(r) + "\n")
            log.flush()
            torch.save({"model": model.state_dict(), "ema": ema, "opt": opt.state_dict(), "dopt": dopt.state_dict(),
                        "mpd": mpd.state_dict(), "cqt": cqt.state_dict(), "mrd": mrd.state_dict() if mrd is not None else None, "sch": sch.state_dict(), "dsch": dsch.state_dict(),
                        "step": step, "cfg": model.cfg}, out / "last.tmp")
            (out / "last.tmp").replace(ckp)
            if step % a.snap_every == 0 or step == a.r_steps:
                (out / "snap").mkdir(exist_ok=True)
                torch.save({"ema": ema, "step": step, "cfg": model.cfg}, out / "snap" / f"ema_{step // 1000}k.pt")
    print("done", step, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
