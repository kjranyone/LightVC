from __future__ import annotations

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
YC = ROOT / "data/snv_p97_y"
SEG = 33600
WARM = 9600
NF = SEG // N.HOP
SHIFTS = tuple(range(0, 601, 60))
ANCHOR_VOICES = ("ef100", "ef032")

ENV_SLICE = slice(0, 25)
LF0_ROW = 25
VUV_ROW = 26
PER_SLICE = slice(27, 31)


def items_for(voices: list[str], nv: bool = True) -> list[tuple[Path, Path]]:
    out = []
    for v in voices:
        for w in sorted((WAV / "e1" / v).glob("t*.flac")):
            out.append((w, FEAT_H / f"e1/{v}/{w.stem}.npz"))
        if nv:
            for w in sorted((WAV / "e1nv" / v).glob("*.flac")):
                out.append((w, FEAT_H / f"e1nv/{v}/{w.stem}.npz"))
    return [(w, f) for w, f in out if f.exists()]


def feat_copy(f: Path, smooth: float) -> dict:
    d = np.load(f)
    f0 = d["f0"].astype(np.float32)
    c = P.cond({"f0": f0, "env": d["env"], "per": d["per"]}, smooth=smooth)
    return {"f0": f0, "c": c}


def feat_y(f: Path, smooth: float) -> dict:
    dh = np.load(f)
    dy = np.load(YC / f.relative_to(FEAT_H))
    n = min(len(dh["f0"]), dy["y"].shape[1])
    f0 = dh["f0"][:n].astype(np.float32)
    y = dy["y"][:, :n].astype(np.float32)
    v = (f0 > 0).astype(np.float32)[None]
    lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)[None]
    env = P.causal_smooth(y[:25], smooth) if smooth > 0 else y[:25]
    c = np.concatenate([env, lf, v, np.clip(y[25:29], 0.0, 1.5) * v], 0).astype(np.float32)
    return {"f0": f0, "c": c}


class DS(torch.utils.data.IterableDataset):
    def __init__(self, items: list[tuple[Path, Path]], seed: int, smooth: float, delay: int, ycond: float = 0.0) -> None:
        self.items, self.seed, self.smooth, self.delay, self.ycond = items, seed, smooth, delay, ycond

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (wi.id if wi else 0) * 1000)
        while True:
            w, f = self.items[rng.randrange(len(self.items))]
            d = feat_y(f, self.smooth) if (self.ycond > 0 and rng.random() < self.ycond) else feat_copy(f, self.smooth)
            x, sr = sf.read(str(w), dtype="float32")
            assert sr == P.SR
            tn = min(len(d["f0"]), len(x) // N.HOP)
            if tn < NF + 2:
                continue
            s = rng.randint(1, tn - NF)
            tgt = x[s * N.HOP - self.delay : s * N.HOP - self.delay + SEG]
            if len(tgt) < SEG:
                continue
            yield (
                torch.from_numpy(d["c"][:, s : s + NF].copy()),
                torch.from_numpy(d["f0"][s : s + NF].copy()),
                torch.from_numpy(tgt.astype(np.float32)),
            )


def load_utt_item(w: Path, f: Path, smooth: float, y: bool = False) -> dict:
    d = feat_y(f, smooth) if y else feat_copy(f, smooth)
    x, _ = sf.read(str(w), dtype="float32")
    n = min(len(d["f0"]), len(x) // N.HOP)
    return {"name": f"{w.parent.name}/{w.stem}", "x": x[: n * N.HOP].astype(np.float32), "c": d["c"][:, :n], "f0": d["f0"][:n]}


def val_items(voices: list[str], smooth: float, y: bool = False) -> list[dict]:
    out = []
    for w, f in items_for(voices, nv=False):
        if w.stem == "t09":
            out.append(load_utt_item(w, f, smooth, y))
    return out


def anchor_items(smooth: float) -> list[dict]:
    out = []
    for v in ANCHOR_VOICES:
        w = WAV / "e1" / v / "t05.flac"
        f = FEAT_H / "e1" / v / "t05.npz"
        if w.exists() and f.exists():
            out.append(load_utt_item(w, f, smooth, False))
    return out


def render(gen: R.RVoc, it: dict, dev: str) -> np.ndarray:
    c = torch.from_numpy(it["c"])[None].to(dev)
    f0 = torch.from_numpy(it["f0"])[None].to(dev)
    return gen(c, excitation(f0, c.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()


def run_unit_tests(gen: R.RVoc, vals: list[dict], delay: int, dev: str) -> None:
    ramp = np.arange(5000 * N.HOP, dtype=np.float32)
    s = 100
    seg = ramp[s * N.HOP - delay : s * N.HOP - delay + SEG]
    assert len(seg) == SEG and float(seg[0]) == float(s * N.HOP - delay) and float(seg[delay]) == float(s * N.HOP)

    c0 = vals[0]["c"]
    f0 = vals[0]["f0"]
    assert c0.shape[0] == 31
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
    print(f"unit_tests OK: f0_med={med_cond_hz:.2f}Hz rms_diff={rms_db:+.2f}dB", flush=True)


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
    d240, bests = [], []
    per_voice = {}
    for it in anchors:
        y = render(gen, it, dev)
        p_fix = float(EN.metrics(y, it["x"], delay, dev)["pesq"])
        bp = p_fix
        for d in SHIFTS:
            bp = max(bp, float(EN.metrics(y, it["x"], d, dev)["pesq"]))
        d240.append(p_fix)
        bests.append(bp)
        per_voice[it["name"]] = {"pesq_d240": round(p_fix, 4), "pesq_best": round(bp, 4)}
    return {
        "anchor_pesq_d240": round(float(np.mean(d240)), 4),
        "anchor_pesq_best": round(float(np.mean(bests)), 4),
        "anchors": per_voice,
    }


def ship_gate(gen: R.RVoc, items: list[dict], delay: int) -> bool:
    import ship_check as SC
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
    return la == 0 and SC.ledger([("rvoc 生成器(条件・励起の書き換え・検証の声)", la / N.SR * 1000), (f"出力の時刻の規約 delay {delay}", delay / N.SR * 1000), ("ブロック HOP 240", 5.0)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--prereg", default="results/snv_p122/prereg.yaml")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smooth", type=float, default=0.25)
    ap.add_argument("--delay", type=int, default=N.DELAY)
    ap.add_argument("--ycond", type=float, default=0.0)
    ap.add_argument("--snap_every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    assert 0 <= a.delay <= N.DELAY
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
    items = items_for(train_voices[:20] if a.smoke else train_voices)
    vals = val_items(split["val"][:4] if a.smoke else split["val"], a.smooth)
    valy = val_items(split["val"][:4] if a.smoke else split["val"], a.smooth, True) if a.ycond > 0 else []
    anchors = anchor_items(a.smooth)
    assert len(items) > 0 and len(vals) > 0 and len(anchors) == len(ANCHOR_VOICES)
    if a.ycond > 0:
        miss = [f for _, f in items if not (YC / f.relative_to(FEAT_H)).exists()]
        assert not miss, f"模型の出力の条件が無い発話 {len(miss)} 本"
    print("items", len(items), "voices", len(train_voices), "val", len(vals), "anchors", len(anchors), flush=True)

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

    run_unit_tests(gen, vals, a.delay, dev)
    if not ship_gate(gen, vals, a.delay):
        print("出荷ゲート FAIL: 起動しない", flush=True)
        return 1

    mels = [build_mel(nf, hm, nm).to(dev) for nf, hm, nm in MEL_SPECS]
    evg = R.RVoc(ch=cfg["ch"], kernels=tuple(cfg["kernels"]), dils=tuple(cfg["dils"]), d_cond=cfg["d_cond"]).to(dev).eval()
    log = open(out / "train.jsonl", "a")

    def run_eval(it: int) -> dict:
        evg.load_state_dict(ema)
        r = {"eval_it": it, **evaluate(evg, vals, a.delay, mels, dev), **evaluate_anchors(evg, anchors, a.delay, dev)}
        if valy:
            r["y"] = evaluate(evg, valy, a.delay, mels, dev)
        print("eval", json.dumps(r), flush=True)
        log.write(json.dumps(r) + "\n")
        log.flush()
        return r

    def save(it: int, snap: bool) -> None:
        meta = {"step": base + it, "it": it, "cfg": gen.cfg, "cond": "pae", "env_smooth": a.smooth, "delay": a.delay, "ycond": a.ycond, "front": "snv_p122 harvest+v3.1_left"}
        if snap:
            (out / "snap").mkdir(exist_ok=True)
            atomic_save({"ema": ema, **meta}, out / "snap" / f"ema_{it // 1000}k.pt")
        atomic_save({"gen": gen.state_dict(), "ema": ema, "mpd": mpd.state_dict(), "mrd": mrd.state_dict(), "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(), **meta}, out / "last.pt")

    if it0 == 0:
        run_eval(0)
    loader = iter(torch.utils.data.DataLoader(DS(items, 7 + it0, a.smooth, a.delay, a.ycond), batch_size=a.bs, num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers > 0 else None))
    acc: dict = {}
    t0 = time.time()
    total = it0 + 30 if a.smoke else a.steps
    for it in range(it0 + 1, total + 1):
        cond, f0, tgt = (t.to(dev, non_blocking=True) for t in next(loader))
        with torch.no_grad():
            exc = excitation(f0, SEG)
        y = gen(cond, exc)
        yl, tl = y[:, WARM:], tgt[:, WARM:]
        lm = logmel_l1(yl[:, None], tl[:, None], mels)
        pen, _ = am_penalty(yl, tl, 0.3, 0.0, True)
        loss = 45.0 * lm + 0.5 * pen
        off = random.randrange(0, yl.shape[-1] - 16384)
        r_, f_ = tl[:, off : off + 16384], yl[:, off : off + 16384]
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
