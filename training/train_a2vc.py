"""A2-VC 段 S1: 回路の写し合成(女声フルコーパスの自己再構成)。current/a2vc.md・事前登録 results/<tag>/prereg.yaml。

  c_in   = 正解 x のピッチ(RRPS・1〜14 半音)と声道長(LPC 包絡の伸縮 α)を乱数で嘘にし、因果の 3 次櫛形で周期成分を除いたもの(息・過渡・子音だけ)。
           推論では入力そのものに同じ櫛形をかける(aperiodic・先読み 0)
  c_pulse= 正解の f0(2 フレーム先取り)の帯域制限パルス列 = 周期の唯一の正しい源
  c_noise= 白色雑音
  制御   = 正解の因果 log-mel の DCT 低次 12(声道包絡 = 包絡の唯一の正しい源)+ 4 帯の調波性
  出力 y[m] ≈ x[m − DELAY]。損失 = 15·多尺度 logmel + 2·mrstft + w_pitch·周期性一致 |r_y(τ) − r_x(τ)|(τ = 目標 f0 の周期)(段 R)
         → + LSGAN + 2·FM(新規 MPD + MRD-log・48kHz)(段 G)
  a2vc_s1(w_pitch 0)は 5k で出力ピッチが c_in に従った(−12.08 半音)= 写しのチート。mel/STFT 振幅損失はピッチのずれに鈍い。
  a2vc_s1b(w_pitch 5・櫛形なし)も 5k で −11.32 半音 = c_in の周期が残る → c_in を非周期成分に限る(a2vc_s1c)。

    CUDA_VISIBLE_DEVICES=0 uv run python train_a2vc.py --tag a2vc_s1
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import a2vc as A
import nvoc as N
import zsvc as Z
from train_nvoc import MRDLog
from train_s1_1 import MEL_SPECS, build_mel, logmel_l1, mrstft

ROOT = Path(__file__).resolve().parent.parent
SEG = 57600
T = SEG // N.HOP
CTX = N.WIN - N.HOP
W0 = 4800
F0_ADV = 2
HBANDS = ((300, 1000), (1000, 2500), (2500, 5000), (5000, 10000))
D_CTRL = N.N_MEL + len(HBANDS)
EVAL_COND = (("m12_a085", -12.0, 0.85), ("p5_a110", 5.0, 1.10))
CTRL = {"mode": "env12"}


def _frac_delay(x: np.ndarray, P: np.ndarray) -> np.ndarray:
    p = np.arange(len(x)) - P
    i = np.floor(p).astype(np.int64)
    a = p - i
    i0, i1 = np.clip(i, 0, len(x) - 1), np.clip(i + 1, 0, len(x) - 1)
    return np.where(i >= 0, x[i0] * (1 - a) + x[i1] * a, 0.0)


def aperiodic(x: np.ndarray) -> np.ndarray:
    """因果の周期成分除去: サンプル n で既知の最新の因果 YIN の周期 P で 3 次櫛形 (1 − z^−P)^3 / 8。無声はそのまま・切替 5ms の因果移動平均。
    held 10 発話の実測(2026-10-01): 周期性 r(τ) 0.74 → −0.04・有声の 5–16kHz(息)−5.2dB・YIN が正しいピッチを拾う割合 100% → 22%。"""
    import artic_dsp as D
    xd = x.astype(np.float64)
    f0, _ = D.causal_yin(xd, voi_max=0.45)
    j = (np.arange(len(xd)) + 1) // N.HOP - 1
    fk = np.where(j >= 0, f0[np.clip(j, 0, len(f0) - 1)], 0.0)
    v = fk > 0
    P = np.where(v, N.SR / np.maximum(fk, 1.0), 0.0)
    y = 0.125 * (xd - 3 * _frac_delay(xd, P) + 3 * _frac_delay(xd, 2 * P) - _frac_delay(xd, 3 * P))
    L = N.SR * 5 // 1000
    w = np.convolve(v.astype(np.float64), np.ones(L) / L)[:len(xd)]
    return (w * y + (1 - w) * xd).astype(np.float32)


def perturb(x: np.ndarray, st: float, alpha: float) -> np.ndarray:
    import artic_dsp as D
    xd = x.astype(np.float64)
    lar, a_sub, e = D.analyze(xd, 24)
    f0p, _ = D.causal_yin(xd, voi_max=0.45)
    e, _ = D.rrps(e, f0p, 2 ** (st / 12), 480, voiced=D.voiced_known(len(xd), f0p, 480, D.F0_HOP))
    return aperiodic(D.synthesize(e, D.coef_schedule(D.warp_lar(lar, alpha), len(xd))))


def sample_st(rng: random.Random) -> float:
    while True:
        st = rng.uniform(-14.0, 8.0)
        if abs(st) >= 1.0:
            return st


class DS(torch.utils.data.IterableDataset):
    def __init__(self, spk: dict, keys: list, seed: int):
        self.flat = [spk[k][i][:2] for k in keys for i in range(len(spk[k]))]
        self.seed = seed

    def __iter__(self):
        from train_ddsp_vc import load48
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 1009 + (wi.id if wi else 0))
        while True:
            fz, fw = self.flat[rng.randrange(len(self.flat))]
            try:
                x = load48(fw)
                if len(x) < SEG + 8 * N.HOP:
                    continue
                u = rng.randrange(4, (len(x) - SEG) // N.HOP)
                s = u * N.HOP
                seg = x[s - CTX:s + SEG]
                if np.sqrt((seg[CTX:] ** 2).mean()) < 1e-3:
                    continue
                f0 = np.load(fz)["f0"].astype(np.float32)[u + F0_ADV:u + F0_ADV + T]
                f0 = np.pad(f0, (0, T - len(f0)))
                pert = perturb(seg, sample_st(rng), math.exp(rng.uniform(math.log(0.8), math.log(1.25))))
                if not np.isfinite(pert).all():
                    continue
                g = 2.0 ** rng.uniform(-0.5, 0.5)
                g = min(g, 0.99 / max(float(np.abs(seg).max()), 1e-6))
                gp = g * min(1.0, 0.99 / max(g * float(np.abs(pert).max()), 1e-6))
                yield torch.from_numpy(seg * g), torch.from_numpy(pert * gp), torch.from_numpy(f0)
            except Exception:
                continue


def controls(front: Z.ZSVC, seg: torch.Tensor) -> torch.Tensor:
    """seg [B, CTX+n] → [B, 128+4, n/HOP]: 声道包絡(DCT 低次 12)と 4 帯の調波性(倍音の縞の RMS)。"""
    mel = N.NVoc.mel_ctx(front, seg)
    env = Z.lifter_env(mel, front.dct, 12)
    res = mel - env
    c = front.centers
    hs = [res[:, (c >= lo) & (c < hi)].pow(2).mean(1, keepdim=True).sqrt() for lo, hi in HBANDS]
    main = mel if CTRL["mode"] == "mel" else Z.lifter_env(mel, front.dct, int(CTRL["mode"][3:])) if CTRL["mode"] != "env12" else env
    return torch.cat([main, *hs], 1)


def acf_at_f0(wav: torch.Tensor, f0: torch.Tensor, frame0: int) -> tuple[torch.Tensor, torch.Tensor]:
    """各フレーム(窓 1024・HOP)の正規化自己相関を目標 f0 の周期 τ で読む → (r(τ) [B,Tn], 有声 [B,Tn])。wav のサンプル i はフレーム frame0 + i // HOP。"""
    W = 1024
    Tn = (wav.shape[-1] - W) // N.HOP
    fr = wav.unfold(-1, W, N.HOP)[:, :Tn] * torch.hann_window(W, device=wav.device)
    X = torch.fft.rfft(fr, n=2 * W)
    r = torch.fft.irfft(X.abs() ** 2, n=2 * W)[..., :W]
    r = r / r[..., :1].clamp(min=1e-8)
    ff = f0[:, frame0 + W // (2 * N.HOP):frame0 + W // (2 * N.HOP) + Tn]
    Tn = min(Tn, ff.shape[-1])
    r, ff = r[:, :Tn], ff[:, :Tn]
    tau = (N.SR / ff.clamp(min=50.0)).clamp(max=W - 2)
    lo = tau.floor().long()
    a = tau - lo
    rt = torch.gather(r, 2, lo[..., None])[..., 0] * (1 - a) + torch.gather(r, 2, (lo + 1)[..., None])[..., 0] * a
    return rt, (ff > 0).float()


def periodicity_loss(y: torch.Tensor, t: torch.Tensor, f0: torch.Tensor, frame0: int) -> torch.Tensor:
    """|r_y(τ) − r_x(τ)|(有声フレーム平均)。正解と同じ周期性に合わせる(1 − r_y は本物より周期的な機械声へ押すので使わない)。"""
    ry, v = acf_at_f0(y, f0, frame0)
    with torch.no_grad():
        rx, _ = acf_at_f0(t, f0, frame0)
    return ((ry - rx).abs() * v).sum() / v.sum().clamp(min=1)


def carriers(pert: torch.Tensor, f0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """pert [B, CTX+n](seg と同じ時刻)→ c_in[m] = pert[CTX + m − DELAY]・c_pulse・c_noise → [B,3,n]。"""
    n = f0.shape[-1] * N.HOP
    cin = pert[:, CTX - N.DELAY:CTX - N.DELAY + n]
    return torch.stack([cin, N.harmonic_source(f0) * 0.1, noise[:, :n] * 0.01], 1)


def run(model: A.A2Circuit, front: Z.ZSVC, seg: torch.Tensor, pert: torch.Tensor, f0: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    ctl = controls(front, seg)
    return model(carriers(pert, f0[:, :ctl.shape[-1]], noise), ctl, torch.zeros(seg.shape[0], 256, device=seg.device))


def eval_set(items: list) -> list:
    out = []
    for it in items:
        x = it["x"].astype(np.float32)
        Tn = len(x) // N.HOP
        seg = np.concatenate([np.zeros(CTX, np.float32), x])
        f0 = it["f0"][F0_ADV:F0_ADV + Tn]
        f0 = np.pad(f0, (0, Tn - len(f0)))
        perts = {name: perturb(seg, st, al) for name, st, al in EVAL_COND}
        out.append({"x": x, "f0": it["f0"], "seg": seg, "f0a": f0, "perts": perts})
    return out


def pitch_dev(y: np.ndarray, f0: np.ndarray) -> float:
    """出力の因果 YIN(2 フレーム遅れを戻す)と正解 f0 の対数比の中央値(半音)。c_in に従えば m12 条件で −12。"""
    import artic_dsp as D
    f, _ = D.causal_yin(y.astype(np.float64), voi_max=0.45)
    f = f[F0_ADV:]
    n = min(len(f), len(f0))
    v = (f[:n] > 0) & (f0[:n] > 0)
    return float(np.median(12 * np.log2(f[:n][v] / f0[:n][v]))) if v.sum() > 20 else float("nan")


def evaluate(model: A.A2Circuit, front: Z.ZSVC, ev: list, dev: str) -> dict:
    import eval_nvoc as E
    from eval_zsvc import contrast
    model.eval()
    out: dict = {}
    for name, _, _ in EVAL_COND:
        rows = []
        for it in ev:
            g = torch.Generator().manual_seed(0)
            nz = torch.randn(1, len(it["x"]), generator=g).to(dev)
            with torch.no_grad():
                y = run(model, front, torch.from_numpy(it["seg"])[None].to(dev), torch.from_numpy(it["perts"][name])[None].to(dev),
                        torch.from_numpy(it["f0a"])[None].to(dev), nz)[0].cpu().numpy()
            m = E.metrics(y, it["x"], N.DELAY, dev)
            m["contrast"] = contrast(y[N.DELAY:], it["f0"])
            m["pitch_dev_st"] = pitch_dev(y[N.DELAY:], it["f0"])
            rows.append(m)
        out[name] = {k: round(float(np.nanmean([r[k] for r in rows])), 3) for k in rows[0]}
    model.train()
    return out


def reference(ev: list) -> dict:
    import eval_nvoc as E
    from eval_zsvc import contrast
    return {"orig_am_db": round(float(np.mean([E.amline(it["x"]) for it in ev])), 3),
            "orig_contrast": round(float(np.nanmean([contrast(it["x"], it["f0"]) for it in ev])), 3),
            "cin_m12_pesq": round(float(np.nanmean([E.metrics(np.concatenate([np.zeros(N.DELAY, np.float32), it["perts"]["m12_a085"][CTX:]]),
                                                             it["x"], N.DELAY, "cpu")["pesq"] for it in ev])), 3)}


def ship_gate(model: A.A2Circuit, front: Z.ZSVC) -> bool:
    """推論経路(c_in = 入力そのもの・c_pulse = 因果 YIN・制御 = 左寄せ mel)を実音声で未来不変性検査。制御経路の感度を出すため最終層を乱数化した複製で測る。"""
    import copy
    import artic_dsp as D
    import ship_check as SC
    m = copy.deepcopy(model).cpu().eval()
    torch.manual_seed(1)
    for p in m.ctrl[-1].parameters():
        torch.nn.init.normal_(p, 0, 0.05)
    fr = copy.deepcopy(front).cpu()

    def fn(x: torch.Tensor) -> torch.Tensor:
        n = len(x) // N.HOP * N.HOP
        xx = x[:n].float()
        seg = torch.cat([torch.zeros(CTX), xx])[None]
        f0, _ = D.causal_yin(xx.double().numpy(), voi_max=0.45)
        f0 = torch.from_numpy(f0[:n // N.HOP].astype(np.float32))[None]
        nz = torch.randn(1, n, generator=torch.Generator().manual_seed(0))
        cin = torch.from_numpy(aperiodic(seg[0].numpy()))[None]
        y = run(m, fr, seg, cin, f0, nz)[0]
        return y.reshape(-1, N.HOP).T

    la = SC.future_invariance(fn, hop=N.HOP, n=2 * N.SR, sr=N.SR, quantity=False, n_edit=8, male=2)
    return SC.ledger([("A2 推論経路(mel+因果YIN+櫛形+回路)実音声", la), ("出力遅延 DELAY 240", 5.0), ("ブロック HOP 240", 5.0)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="a2vc_s1")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--r_steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_end", type=float, default=None, help="生成器の学習率を lr → lr_end へ指数減衰(全 step で)")
    ap.add_argument("--ch", type=int, default=24)
    ap.add_argument("--w_pitch", type=float, default=5.0)
    ap.add_argument("--ctrl", default="env12", help="env12 | env24 | env40 | mel(診断: 制御の解像度)")
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--init", default=None, help="last.pt から生成器(model・ema)を引き継ぐ(判別器・最適化器は新規)")
    a = ap.parse_args()
    CTRL["mode"] = a.ctrl
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"prereg: results/{a.tag}/prereg.yaml", flush=True)
    import eval_nvoc as E
    from train_ddsp_vc import index
    front = Z.ZSVC().to(dev).eval()
    model = A.A2Circuit(ch=a.ch, d_ctrl=D_CTRL).to(dev)
    print("GMAC/s", round(A.macs_per_second(model) / 1e9, 2), "| params (M)", round(sum(p.numel() for p in model.parameters()) / 1e6, 3), flush=True)
    if not ship_gate(model, front):
        print("出荷ゲート FAIL: 起動しない", flush=True)
        return 1
    spk, tr, _ = index()
    print("speakers", len(tr), "utterances", sum(len(spk[k]) for k in tr), flush=True)
    loader = iter(torch.utils.data.DataLoader(DS(spk, tr, 21), batch_size=a.bs, num_workers=a.workers, pin_memory=True,
                                              persistent_workers=True, prefetch_factor=4))
    if a.init:
        st0 = torch.load(a.init, map_location="cpu", weights_only=False)
        model.load_state_dict(st0["model"])
        print("init from", a.init, "step", st0["step"], flush=True)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if a.init:
        ema = {k: v.to(dev) for k, v in st0["ema"].items()}
    evm = A.A2Circuit(ch=a.ch, d_ctrl=D_CTRL).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.8, 0.99))
    from bigvgan.discriminators import MultiPeriodDiscriminator
    mpd = MultiPeriodDiscriminator(SimpleNamespace(mpd_reshapes=[2, 3, 5, 7, 11], use_spectral_norm=False, discriminator_channel_mult=1)).to(dev)
    mrd = MRDLog().to(dev)
    dparams = list(mpd.parameters()) + list(mrd.parameters())
    dopt = torch.optim.AdamW(dparams, lr=2e-4, betas=(0.8, 0.99))
    mels = [build_mel(nf, hm, nm).to(dev) for nf, hm, nm in MEL_SPECS]
    items = E.held_items()
    ev = eval_set(items[:3] if a.smoke else items)
    log = open(out / "train.jsonl", "a")
    ref = {"step": -1, "held_n": len(ev), **reference(ev)}
    print("ref", json.dumps(ref), flush=True)
    log.write(json.dumps(ref) + "\n")

    def run_eval(step: int) -> None:
        evm.load_state_dict(ema)
        r = {"step": step, **evaluate(evm, front, ev, dev)}
        print("eval", json.dumps(r), flush=True)
        log.write(json.dumps(r) + "\n")
        log.flush()

    run_eval(0)
    acc: dict = {}
    t0 = time.time()
    total = 30 if a.smoke else a.steps
    r_steps = 15 if a.smoke else a.r_steps
    for step in range(1, total + 1):
        seg, pert, f0 = (t.to(dev, non_blocking=True) for t in next(loader))
        B = seg.shape[0]
        y = run(model, front, seg, pert, f0, torch.randn(B, SEG, device=dev))
        tgt = seg[:, CTX - N.DELAY:CTX - N.DELAY + SEG]
        yl, tl = y[:, W0:], tgt[:, W0:]
        lm = logmel_l1(yl[:, None], tl[:, None], mels)
        loss = 15 * lm + 2 * mrstft(yl[:, None], tl[:, None])
        rec = {"lm": lm}
        if a.w_pitch > 0:
            pl = periodicity_loss(yl, tl, f0, W0 // N.HOP)
            loss = loss + a.w_pitch * pl
            rec["pitch"] = pl
        if step > r_steps:
            gg = (0.95 / tl.abs().amax(-1, keepdim=True).clamp(min=1e-3)).clamp(max=20.0)
            off = random.randrange(0, tl.shape[-1] - 16384)
            r, f = (tl * gg)[:, off:off + 16384], (yl * gg)[:, off:off + 16384]
            dopt.zero_grad(set_to_none=True)
            a1, b1, _, _ = mpd(r[:, None], f.detach()[:, None])
            a2, b2, _, _ = mrd(r, f.detach())
            dl = sum(((p.float() - 1) ** 2).mean() + (q.float() ** 2).mean() for p, q in zip(a1 + a2, b1 + b2))
            if torch.isfinite(dl):
                dl.backward()
                torch.nn.utils.clip_grad_norm_(dparams, 500.0)
                dopt.step()
            for p_ in dparams:
                p_.requires_grad_(False)
            _, b1, f1r, f1g = mpd(r[:, None], f[:, None])
            _, b2, f2r, f2g = mrd(r, f)
            for p_ in dparams:
                p_.requires_grad_(True)
            adv = sum(((q.float() - 1) ** 2).mean() for q in b1 + b2)
            fm = sum(F.l1_loss(u.detach().float(), v.float()) for xr, xg in zip(f1r + f2r, f1g + f2g) for u, v in zip(xr, xg))
            loss = loss + adv + 2 * fm
            rec.update({"adv": adv, "fm": fm, "dl": dl})
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        if not torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 500.0)):
            opt.zero_grad(set_to_none=True)
            continue
        if a.lr_end:
            for g_ in opt.param_groups:
                g_["lr"] = a.lr * (a.lr_end / a.lr) ** (step / total)
        opt.step()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(0.999).add_(v.detach(), alpha=0.001)
        for k, v in rec.items():
            acc.setdefault(k, []).append(float(v.detach() if torch.is_tensor(v) else v))
        if step % 100 == 0 or a.smoke:
            rr = {"step": step, "phase": "G" if step > r_steps else "R", "min": round((time.time() - t0) / 60, 1)}
            rr.update({k: round(float(np.mean(v)), 4) for k, v in acc.items()})
            acc = {}
            print(json.dumps(rr), flush=True)
            log.write(json.dumps(rr) + "\n")
            log.flush()
        if step % a.eval_every == 0 or (a.smoke and step == total):
            run_eval(step)
            (out / "snap").mkdir(exist_ok=True)
            torch.save({"ema": ema, "step": step, "ch": a.ch}, out / "snap" / f"ema_{step // 1000}k.pt")
            torch.save({"model": model.state_dict(), "ema": ema, "opt": opt.state_dict(), "mpd": mpd.state_dict(), "mrd": mrd.state_dict(),
                        "dopt": dopt.state_dict(), "step": step, "ch": a.ch}, out / "last.pt")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
