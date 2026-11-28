from __future__ import annotations

import os

os.environ["MPLBACKEND"] = "Agg"

import argparse
import copy
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

import eval_nvoc as EN
import nvoc as N
import rvoc as R
import snv_p0_front as P
from am_loss import am_penalty
from train_nvoc import MRDLog
from train_rvoc import atomic_save, excitation
from train_s1_1 import MEL_SPECS, build_mel, logmel_l1

ROOT = Path(__file__).resolve().parent.parent
CK = ROOT / "results/rvoc3hi/last.pt"
WAV = ROOT / "data/snv_p97_wav"
FEAT_H = ROOT / "data/snv_p122_feat"
SEG = 33600
WARM = 9600
NF = SEG // N.HOP
SHIFTS = tuple(range(240, 721, 60))
ANCHOR_VOICES = ("ef100", "ef032")

ENV_SLICE = slice(0, 25)
LF0_ROW = 25
VUV_ROW = 26
PER_SLICE = slice(27, 31)


def delay_1f(a: np.ndarray) -> np.ndarray:
    return np.concatenate([a[..., :1], a[..., :-1]], axis=-1)


def voice_per0_med(v: str) -> float:
    f = FEAT_H / f"e1/{v}/t05.npz"
    if not f.exists():
        return 0.0
    d = np.load(f)
    m = d["f0"] > 0
    return float(np.median(d["per"][0, m])) if m.any() else 0.0


def pools_for(voices: list[str], clean_th: float) -> tuple[list[tuple[Path, Path, bool]], list[tuple[Path, Path, bool]], int]:
    e1_list: list[tuple[Path, Path, bool]] = []
    nv_list: list[tuple[Path, Path, bool]] = []
    clean_voices = 0
    for v in voices:
        is_clean = voice_per0_med(v) >= clean_th
        if is_clean:
            clean_voices += 1
        for w in sorted((WAV / "e1" / v).glob("t*.flac")):
            f = FEAT_H / f"e1/{v}/{w.stem}.npz"
            if f.exists():
                e1_list.append((w, f, is_clean))
        for w in sorted((WAV / "e1nv" / v).glob("*.flac")):
            f = FEAT_H / f"e1nv/{v}/{w.stem}.npz"
            if f.exists():
                nv_list.append((w, f, False))
    return e1_list, nv_list, clean_voices


def feat_copy(f: Path, smooth: float, sync_f0: bool) -> dict:
    d = np.load(f)
    f0 = d["f0"].astype(np.float32)
    per = d["per"].astype(np.float32)
    if sync_f0:
        f0 = delay_1f(f0)
        per = delay_1f(per)
    c = P.cond({"f0": f0, "env": d["env"], "per": per}, smooth=smooth)
    return {"f0": f0, "c": c}


class DS(torch.utils.data.IterableDataset):
    def __init__(
        self,
        e1_items: list[tuple[Path, Path, bool]],
        nv_items: list[tuple[Path, Path, bool]],
        seed: int,
        smooth: float,
        sync_f0: bool,
        delay: int,
        nv_prob: float,
    ) -> None:
        self.e1_items = e1_items
        self.nv_items = nv_items
        self.seed = seed
        self.smooth = smooth
        self.sync_f0 = sync_f0
        self.delay = delay
        self.nv_prob = nv_prob

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (wi.id if wi else 0) * 1009)
        s_min = max(1, (self.delay + N.HOP - 1) // N.HOP)
        while True:
            use_nv = bool(self.nv_items) and rng.random() < self.nv_prob
            pool = self.nv_items if use_nv else self.e1_items
            w, f, is_clean = pool[rng.randrange(len(pool))]
            d = feat_copy(f, self.smooth, self.sync_f0)
            x, sr = sf.read(str(w), dtype="float32")
            assert sr == P.SR
            tn = min(len(d["f0"]), len(x) // N.HOP)
            if tn < NF + s_min + 1:
                continue
            s = rng.randint(s_min, tn - NF)
            tgt = x[s * N.HOP - self.delay : s * N.HOP - self.delay + SEG]
            if len(tgt) < SEG:
                continue
            yield (
                torch.from_numpy(d["c"][:, s : s + NF].copy()),
                torch.from_numpy(d["f0"][s : s + NF].copy()),
                torch.from_numpy(tgt.astype(np.float32)),
                torch.tensor(bool(is_clean), dtype=torch.bool),
            )


def load_utt_item(w: Path, f: Path, smooth: float, sync_f0: bool) -> dict:
    d = feat_copy(f, smooth, sync_f0)
    x, _ = sf.read(str(w), dtype="float32")
    n = min(len(d["f0"]), len(x) // N.HOP)
    return {"name": f"{w.parent.name}/{w.stem}", "x": x[: n * N.HOP].astype(np.float32), "c": d["c"][:, :n], "f0": d["f0"][:n]}


def val_items(voices: list[str], smooth: float, sync_f0: bool) -> list[dict]:
    out = []
    for v in voices:
        w = WAV / "e1" / v / "t09.flac"
        f = FEAT_H / "e1" / v / "t09.npz"
        if w.exists() and f.exists():
            out.append(load_utt_item(w, f, smooth, sync_f0))
    return out


def anchor_items(smooth: float, sync_f0: bool) -> list[dict]:
    out = []
    for v in ANCHOR_VOICES:
        w = WAV / "e1" / v / "t05.flac"
        f = FEAT_H / "e1" / v / "t05.npz"
        if w.exists() and f.exists():
            out.append(load_utt_item(w, f, smooth, sync_f0))
    return out


def render(gen: R.RVoc, it: dict, dev: str) -> np.ndarray:
    c = torch.from_numpy(it["c"])[None].to(dev)
    f0 = torch.from_numpy(it["f0"])[None].to(dev)
    return gen(c, excitation(f0, c.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()


def run_unit_tests(gen: R.RVoc, vals: list[dict], smooth: float, sync_f0: bool, delay: int, dev: str) -> None:
    fr_ramp = np.arange(300, dtype=np.float32)[None]
    sm_ramp = P.causal_smooth(fr_ramp, smooth)
    f_ramp = delay_1f(fr_ramp[0]) if sync_f0 else fr_ramp[0]
    s = 100
    expected_frame = float(s - 1) if (smooth == 0.25 and sync_f0) else float(s)
    assert abs(float(sm_ramp[0, s]) - expected_frame) < 1e-5, (float(sm_ramp[0, s]), expected_frame)
    assert abs(float(f_ramp[s]) - expected_frame) < 1e-5, (float(f_ramp[s]), expected_frame)

    wav_ramp = np.arange(5000 * N.HOP, dtype=np.float32)
    seg = wav_ramp[s * N.HOP - delay : s * N.HOP - delay + SEG]
    assert len(seg) == SEG
    assert float(seg[N.DELAY]) == expected_frame * N.HOP, (float(seg[N.DELAY]), expected_frame * N.HOP)

    c0 = vals[0]["c"]
    f0 = vals[0]["f0"]
    assert c0.shape[0] == 31
    assert c0[ENV_SLICE].shape[0] == 25 and c0[PER_SLICE].shape[0] == 4
    v_row = c0[VUV_ROW]
    assert set(np.unique(v_row)).issubset({0.0, 1.0})
    vm = v_row > 0.5
    assert vm.any()
    med_cond_hz = float(200.0 * np.exp(np.median(c0[LF0_ROW, vm])))
    med_true_hz = float(np.median(f0[f0 > 0]))
    assert abs(med_cond_hz - med_true_hz) < 1e-2, (med_cond_hz, med_true_hz)

    with torch.no_grad():
        y0 = render(gen, vals[0], dev)[delay:]
    x0 = vals[0]["x"][: len(y0)]
    rms_db = 20.0 * float(np.log10((np.sqrt((y0 ** 2).mean()) + 1e-9) / (np.sqrt((x0 ** 2).mean()) + 1e-9)))
    assert abs(rms_db) < 4.0, rms_db
    print(f"unit_tests OK: sync_frame={expected_frame} f0_med={med_cond_hz:.2f}Hz rms_diff={rms_db:+.2f}dB", flush=True)


@torch.no_grad()
def evaluate(gen: R.RVoc, items: list[dict], delay: int, mels: list, dev: str) -> dict:
    rows, sh = [], {d: [] for d in SHIFTS}
    for it in items:
        y = render(gen, it, dev)
        rows.append(EN.metrics(y, it["x"], delay, dev))
        yt, xt = torch.from_numpy(y).to(dev), torch.from_numpy(it["x"]).to(dev)
        for d in SHIFTS:
            n = min(len(y) - d, len(it["x"]))
            sh[d].append(float(logmel_l1(yt[None, None, d : d + n], xt[None, None, :n], mels)))
    out = {k: round(float(np.nanmean([r[k] for r in rows])), 4) for k in rows[0]}
    m = {d: round(float(np.mean(v)), 4) for d, v in sh.items()}
    out["logmel_by_shift"] = m
    out["align_best"] = min(m, key=m.get)
    return out


@torch.no_grad()
def evaluate_anchors(gen: R.RVoc, anchors: list[dict], delay: int, dev: str) -> dict:
    dfix, bests = [], []
    per_voice = {}
    for it in anchors:
        y = render(gen, it, dev)
        p_fix = float(EN.metrics(y, it["x"], delay, dev)["pesq"])
        bp = p_fix
        for d in SHIFTS:
            bp = max(bp, float(EN.metrics(y, it["x"], d, dev)["pesq"]))
        dfix.append(p_fix)
        bests.append(bp)
        per_voice[it["name"]] = {"pesq_fix": round(p_fix, 4), "pesq_best": round(bp, 4)}
    return {
        "anchor_pesq_fix": round(float(np.mean(dfix)), 4),
        "anchor_pesq_best": round(float(np.mean(bests)), 4),
        "anchors": per_voice,
    }


def ship_gate(gen: R.RVoc, items: list[dict], delay: int) -> bool:
    import ship_check as SC
    prev_det = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        g = copy.deepcopy(gen).cpu().eval()
        worst = 0
        with torch.no_grad():
            ce = []
            for it in items[:2]:
                f = torch.from_numpy(it["f0"])[None]
                c = torch.from_numpy(it["c"])[None]
                ce.append((c, excitation(f, c.shape[-1] * N.HOP, torch.Generator().manual_seed(0))))
            (c1, e1), (c2, e2) = ce
            n = min(c1.shape[-1], c2.shape[-1], 600)
            c1, e1, c2, e2 = c1[..., :n], e1[..., : n * N.HOP], c2[..., :n], e2[..., : n * N.HOP]
            y = g(c1, e1)[0]
            for k in (97, 211, 350, 503):
                ca, ea = c1.clone(), e1.clone()
                ca[..., k:], ea[..., k * N.HOP :] = c2[..., k:], e2[..., k * N.HOP :]
                d = (g(ca, ea)[0] - y).abs()
                nz = (d > 0).nonzero()
                assert len(nz) > 0
                worst = max(worst, k * N.HOP - int(nz[0]))
        la = max(0, worst)
        print(f"  生成器の未来不変性(検証の声の条件・励起・1 bit): 先読み {la} sample", flush=True)
        return la == 0 and SC.ledger([
            ("rvoc 生成器(条件・励起の書き換え・検証の声)", la / N.SR * 1000),
            (f"因果平滑 + rvoc 時刻規約 delay {delay}", delay / N.SR * 1000),
            ("ブロック HOP 240", 5.0),
        ])
    finally:
        torch.use_deterministic_algorithms(prev_det)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prereg", default="results/snv_p123/prereg.yaml")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smooth", type=float, default=0.25)
    ap.add_argument("--sync_f0", type=int, default=1)
    ap.add_argument("--delay", type=int, default=480)
    ap.add_argument("--nv_prob", type=float, default=0.10)
    ap.add_argument("--clean_d", type=float, default=0.0)
    ap.add_argument("--snap_every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    assert 0 <= a.delay <= 2 * N.DELAY
    prereg_path = ROOT / a.prereg
    if not a.smoke and not prereg_path.exists():
        print(f"{a.prereg} が無い: 起動しない", flush=True)
        return 1
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"prereg: {a.prereg}", flush=True)
    print("args", json.dumps(vars(a)), flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    dev = "cuda"
    split = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())
    train_voices = [v for v in split["train"] if v not in ANCHOR_VOICES]
    assert all(v not in train_voices for v in ANCHOR_VOICES)
    sync_flag = bool(a.sync_f0)
    e1_items, nv_items, n_clean = pools_for(train_voices[:20] if a.smoke else train_voices, a.clean_d)
    vals = val_items(split["val"][:4] if a.smoke else split["val"], a.smooth, sync_flag)
    anchors = anchor_items(a.smooth, sync_flag)
    assert len(e1_items) > 0 and len(vals) > 0 and len(anchors) == len(ANCHOR_VOICES)
    print(
        "e1_items", len(e1_items), "nv_items", len(nv_items), "voices", len(train_voices),
        "clean_d_voices", n_clean, "val", len(vals), "anchors", len(anchors), flush=True,
    )

    resume = a.resume and (out / "last.pt").exists()
    st = torch.load(out / "last.pt" if resume else CK, map_location="cpu", weights_only=False)
    it0 = int(st.get("it", 0)) if resume else 0
    base = int(st["step"]) - it0
    cfg = st["cfg"]
    gen = R.RVoc(ch=cfg["ch"], kernels=tuple(cfg["kernels"]), dils=tuple(cfg["dils"]), d_cond=cfg["d_cond"]).to(dev)
    gen.load_state_dict(st["gen"])
    from bigvgan.discriminators import MultiPeriodDiscriminator
    mpd = MultiPeriodDiscriminator(SimpleNamespace(mpd_reshapes=[2, 3, 5, 7, 11], use_spectral_norm=False, discriminator_channel_mult=1)).to(dev)
    mrd = MRDLog().to(dev)
    mpd.load_state_dict(st["mpd"]), mrd.load_state_dict(st["mrd"])
    dparams = list(mpd.parameters()) + list(mrd.parameters())
    opt_g = torch.optim.AdamW(gen.parameters(), lr=2e-4, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(dparams, lr=2e-4, betas=(0.8, 0.99))
    opt_g.load_state_dict(st["opt_g"]), opt_d.load_state_dict(st["opt_d"])
    ema = {k: v.to(dev) for k, v in st["ema"].items()}
    print("start from", f"results/{a.tag}/last.pt" if resume else "results/rvoc3hi/last.pt", "base step", base, "it", it0, "lr", opt_g.param_groups[0]["lr"], flush=True)

    run_unit_tests(gen, vals, a.smooth, sync_flag, a.delay, dev)
    if not ship_gate(gen, vals, a.delay):
        print("出荷ゲート FAIL: 起動しない", flush=True)
        return 1

    mels = [build_mel(nf, hm, nm).to(dev) for nf, hm, nm in MEL_SPECS]
    evg = R.RVoc(ch=cfg["ch"], kernels=tuple(cfg["kernels"]), dils=tuple(cfg["dils"]), d_cond=cfg["d_cond"]).to(dev).eval()
    log = open(out / "train.jsonl", "a")

    def run_eval(it: int) -> dict:
        evg.load_state_dict(ema)
        r = {"eval_it": it, **evaluate(evg, vals, a.delay, mels, dev), **evaluate_anchors(evg, anchors, a.delay, dev)}
        print("eval", json.dumps(r), flush=True)
        log.write(json.dumps(r) + "\n")
        log.flush()
        return r

    def save(it: int, snap: bool) -> None:
        meta = {
            "step": base + it,
            "it": it,
            "cfg": gen.cfg,
            "cond": "pae",
            "env_smooth": a.smooth,
            "sync_f0": sync_flag,
            "delay": a.delay,
            "nv_prob": a.nv_prob,
            "clean_d": a.clean_d,
            "front": "snv_p123 harvest_sync+v3.1_left",
        }
        if snap:
            (out / "snap").mkdir(exist_ok=True)
            atomic_save({"ema": ema, **meta}, out / "snap" / f"ema_{it // 1000}k.pt")
        atomic_save({"gen": gen.state_dict(), "ema": ema, "mpd": mpd.state_dict(), "mrd": mrd.state_dict(), "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(), **meta}, out / "last.pt")

    if it0 == 0:
        run_eval(0)
    ds = DS(e1_items, nv_items, 7 + it0, a.smooth, sync_flag, a.delay, a.nv_prob)
    loader = iter(torch.utils.data.DataLoader(ds, batch_size=a.bs, num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers > 0 else None))
    acc: dict = {}
    t0 = time.time()
    total = it0 + 30 if a.smoke else a.steps
    for it in range(it0 + 1, total + 1):
        cond, f0, tgt, is_clean = (t.to(dev, non_blocking=True) for t in next(loader))
        with torch.no_grad():
            exc = excitation(f0, SEG)
        y = gen(cond, exc)
        yl, tl = y[:, WARM:], tgt[:, WARM:]
        lm = logmel_l1(yl[:, None], tl[:, None], mels)
        pen, _ = am_penalty(yl, tl, 0.3, 0.0, True)
        loss = 45.0 * lm + 0.5 * pen

        d_idx = torch.where(is_clean)[0] if a.clean_d > 0 else torch.arange(yl.shape[0], device=dev)
        adv = torch.tensor(0.0, device=dev)
        fm = torch.tensor(0.0, device=dev)
        dl = torch.tensor(0.0, device=dev)
        if d_idx.numel() > 0:
            off = random.randrange(0, yl.shape[-1] - 16384)
            r_ = tl[d_idx, off : off + 16384]
            f_ = yl[d_idx, off : off + 16384]
            opt_d.zero_grad(set_to_none=True)
            a1, b1, _, _ = mpd(r_[:, None], f_.detach()[:, None])
            a2, b2, _, _ = mrd(r_, f_.detach())
            dl = sum(((p.float() - 1) ** 2).mean() + (q.float() ** 2).mean() for p, q in zip(a1 + a2, b1 + b2))
            if torch.isfinite(dl):
                dl.backward()
                torch.nn.utils.clip_grad_norm_(dparams, 500.0)
                opt_d.step()
            for p_ in dparams:
                p_.requires_grad_(False)
            _, b1, f1r, f1g = mpd(r_[:, None], f_[:, None])
            _, b2, f2r, f2g = mrd(r_, f_)
            for p_ in dparams:
                p_.requires_grad_(True)
            adv = sum(((q.float() - 1) ** 2).mean() for q in b1 + b2)
            fm = sum(F.l1_loss(u.detach().float(), v.float()) for xr, xg in zip(f1r + f2r, f1g + f2g) for u, v in zip(xr, xg))
            loss = loss + adv + 2.0 * fm

        opt_g.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        if not torch.isfinite(torch.nn.utils.clip_grad_norm_(gen.parameters(), 500.0)):
            opt_g.zero_grad(set_to_none=True)
            continue
        opt_g.step()
        with torch.no_grad():
            for k, v in gen.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(0.999).add_(v.detach(), alpha=0.001)
        for k_, v_ in {"lm": lm, "am_pen": pen, "adv": adv, "fm": fm, "dl": dl}.items():
            acc.setdefault(k_, []).append(float(v_.detach()))
        if it % 200 == 0 or a.smoke:
            rr = {"it": it, "min": round((time.time() - t0) / 60, 1), **{k_: round(float(np.mean(v_)), 4) for k_, v_ in acc.items()}}
            acc = {}
            print(json.dumps(rr), flush=True)
            log.write(json.dumps(rr) + "\n")
            log.flush()
        snap = it % a.snap_every == 0 or it == total
        ev_res = None
        if snap:
            ev_res = run_eval(it)
        if snap or it % 2500 == 0:
            save(it, snap)
        if not a.smoke and it == 5000 and ev_res is not None:
            if ev_res["anchor_pesq_best"] < 3.45 or ev_res["am_db"] > 1.5:
                stop_info = {"it": it, "reason": "5k_tripwire", "anchor_pesq_best": ev_res["anchor_pesq_best"], "am_db": ev_res["am_db"]}
                (out / "STOP.json").write_text(json.dumps(stop_info, indent=1))
                print("TRIPWIRE STOP", json.dumps(stop_info), flush=True)
                return 2
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
