"""C4: 自己再構成の動的補正 G(検証器なし・results/c4_1/prereg.yaml・current/converter.md §4e-7)。
実の話者の発話から『その話者の表 + 発話の硬い単位列 + f0・c0』→『その発話の実の包絡の軌道』を L1 で回帰する。
凍結の前段は train_c3.Chain.prep(推論と同じ経路)。出力 Δc(24)・Δ周期性(4)は ConvG(one-hot 単位入力)。
    uv run python train_c4.py --tag c4_1 --steps 20000
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import c1_content as CC
import f0est as FE
import lvl as LV
import nvoc as N
import pae as PA
import train_c3 as C3
import train_rvoc as TR

ROOT = Path(__file__).resolve().parent.parent
SEG_F = 800
CTX_F = 200
TTS_LEAK = {"fe9565ca1f33bf20", "ffb9b5647612b32b"}


def sm3(E: np.ndarray, w: float = 0.25) -> np.ndarray:
    e = np.pad(E, ((0, 0), (1, 1)), mode="edge")
    return w * e[:, :-2] + (1 - 2 * w) * e[:, 1:-1] + w * e[:, 2:]


def utt_rows() -> dict:
    """(real|tts, spk) → [(wav, f0path, sr, dur)]。評価話者の実音声・評価の TTS 複製は除く。"""
    hx, hs = TR.hi_eval_paths(), TR.held_speakers()
    rv = json.loads((ROOT / "data/rvoc_f0hi/manifest.json").read_text())["rows"]
    fx = json.loads((ROOT / "data/f0est_hi/manifest.json").read_text())["rows"]
    out: dict = {}
    for r in rv:
        if r["keep"] and not TR.eval_excluded(r, hx, hs) and r["spk"] not in TTS_LEAK and r["dur"] >= 5.5:
            kind = "real" if r["src"] == "real_female" else "tts"
            out.setdefault((kind, r["spk"]), []).append((r["wav"], str(ROOT / r["f0"]), r["sr"], r["dur"]))
    for r in fx:
        if r["ok"] and r["src"].startswith("female_extra") and not TR.eval_excluded(r, hx, hs) and r["spk"] not in TTS_LEAK and r["dur"] >= 5.5:
            kind = "real" if r["src"].endswith("real") else "tts"
            out.setdefault((kind, r["spk"]), []).append((r["wav"], str(ROOT / r["f0"]), r["sr"], r["dur"]))
    return out


class DS(torch.utils.data.IterableDataset):
    def __init__(self, utts: dict, rows_of: dict, keys: list, seed: int, gain: float = 12.0, n: int = 0):
        self.utts, self.rows_of, self.keys, self.seed, self.gain, self.n = utts, rows_of, keys, seed, gain, n

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 977 + (wi.id if wi else 0))
        cnt = 0
        while self.n == 0 or cnt < self.n:
            key = self.keys[rng.randrange(len(self.keys))]
            wav, f0p, sr, dur = rng.choice(self.utts[key])
            try:
                n48 = int(dur * N.SR)
                tot = SEG_F + CTX_F
                if n48 < (tot + 10) * N.HOP:
                    continue
                s0 = rng.randrange(0, n48 - tot * N.HOP) // N.HOP * N.HOP
                x = TR.read_span(wav, sr, s0, tot * N.HOP)
                if len(x) < tot * N.HOP:
                    continue
                g = 10 ** (rng.uniform(-self.gain, self.gain) / 20)
                x = np.clip(x * g, -1, 1).astype(np.float32)
                f0 = np.load(f0p).astype(np.float32)
                i0 = s0 // N.HOP
                f0c = np.zeros(tot, np.float32)
                seg = f0[i0:i0 + tot]
                f0c[:len(seg)] = seg
                vo = f0c > 0
                if vo.mean() < 0.15:
                    continue
                med = float(np.median(np.log(f0c[vo])))
                xs = x[CTX_F * N.HOP:]
                fs = f0c[CTX_F:]
                xa = np.concatenate([np.zeros(PA.A, np.float32), xs, np.zeros(PA.A, np.float32)])
                env = sm3(PA.envelope(xa, fs))
                per = PA.periodicity(torch.from_numpy(xa)[None], torch.from_numpy(fs)[None])[0].numpy()
                c0n = ((env[0] - PA.C0_SIL) / 10).astype(np.float32)
                row = rng.choice(self.rows_of[key])
                cnt += 1
                yield (torch.from_numpy(x), torch.from_numpy((env[1:25] / 10).astype(np.float32)), torch.from_numpy(per.astype(np.float32)),
                       torch.from_numpy(c0n), torch.tensor(row), torch.tensor(med, dtype=torch.float32))
            except Exception:
                continue


def losses(chain, G, batch, dev, drop, zero=False):
    x, e_real, per_real, c0_real, idx, med = (t.to(dev) for t in batch)
    q = chain.prep(x, idx, med, drop=drop)
    dc, dp = G(q["e_s"], q["p"], q["lf"], q["c0n"], q["vo"], q["film"])
    T = e_real.shape[-1]
    e_s = q["e_s"][..., -T:]
    dc, dp = dc[..., -T:], dp[..., -T:]
    act = (c0_real > 1.0).float()[:, None]
    vo = q["vo"][..., -T:]
    pm = q["pm"][..., -T:]
    den = act.sum().clamp(min=1) * e_real.shape[1]
    l_env = ((e_s + dc - e_real).abs() * act).sum() / den
    l_env0 = ((e_s - e_real).abs() * act).sum() / den
    dv = (vo * act).sum().clamp(min=1) * per_real.shape[1]
    pr = (pm + dp).clamp(0, 1.5)
    l_per = ((pr - per_real).abs() * vo * act).sum() / dv
    l_per0 = ((pm.clamp(0, 1.5) - per_real).abs() * vo * act).sum() / dv
    return l_env, l_env0, l_per, l_per0, dc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--g_ch", type=int, default=192)
    ap.add_argument("--g_dils", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32])
    ap.add_argument("--drop", type=float, default=0.05)
    ap.add_argument("--l_per", type=float, default=0.5)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--targets", nargs="+", default=[str(ROOT / "data/c3" / f) for f in ("targets.npz", "targets_v1.npz", "targets_v2.npz", "targets_v3.npz")])
    ap.add_argument("--rvoc", default=str(ROOT / "results/rvoc3hi/snap/ema_330k.pt"))
    ap.add_argument("--f1", default=str(ROOT / "results/f0est3/last.pt"))
    ap.add_argument("--f2", default=str(ROOT / "results/f2_2/last.pt"))
    ap.add_argument("--c1", default=str(ROOT / "results/c1_1"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    if not a.smoke and not (out / "prereg.yaml").exists():
        print(f"results/{a.tag}/prereg.yaml が無い: 起動しない", flush=True)
        return 1
    print(f"prereg: results/{a.tag}/prereg.yaml", flush=True)
    sr_ = torch.load(a.rvoc, map_location="cpu", weights_only=False)
    sm_w = TR.ckpt_env_smooth(sr_, None)
    sf1 = torch.load(a.f1, map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev)
    f1.load_state_dict(sf1["net"]); f1.eval().requires_grad_(False)
    f1front = FE.Front().to(dev)
    sf2 = torch.load(a.f2, map_location="cpu", weights_only=False)
    f2 = LV.Lvl(sf2["cfg"]["ch"], tuple(sf2["cfg"]["dils"])).to(dev)
    f2.load_state_dict(sf2["net"]); f2.eval().requires_grad_(False)
    c1d = Path(a.c1)
    st1 = torch.load(c1d / "last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev)
    c1.load_state_dict(st1["net"]); c1.eval().requires_grad_(False)
    mfront = CC.MelFront().to(dev)
    Cb = torch.load(c1d / "codebook.pt").to(dev)
    K = Cb.shape[0]
    zs = [np.load(p) for p in a.targets]
    Z = {k: np.concatenate([z[k] for z in zs]) for k in zs[0].files}
    Tt = torch.from_numpy(Z["T"].astype(np.float32)).to(dev) / 10.0
    mu = torch.from_numpy(Z["mu"]).to(dev)
    per_t = torch.from_numpy(Z["per"]).to(dev)
    summ = torch.cat([Tt.mean(2), mu[:, None], per_t], 1)
    utts = utt_rows()
    rows_of: dict = {}
    for i, (s_, sp) in enumerate(zip(Z["src"].tolist(), Z["spk"].tolist())):
        key = ("real" if s_ == "real" else "tts" if s_ == "tts" else s_, sp)
        if key in utts and sp not in TTS_LEAK:
            rows_of.setdefault(key, []).append(i)
    keys = sorted(rows_of)
    rs = np.random.default_rng(0).permutation(len(keys))
    val_keys = [keys[i] for i in rs[:max(40, len(keys) // 25)]]
    tr_keys = [k for k in keys if k not in set(val_keys)]
    print("話者", len(keys), "学習", len(tr_keys), "検証", len(val_keys), "表の行", len(Z["spk"]), flush=True)
    chain = C3.Chain(c1, mfront, Cb, f1, f1front, f2, None, Tt, mu, per_t, summ, dev)
    chain.onehot = True
    G = C3.ConvG(K, a.g_ch, tuple(a.g_dils)).to(dev)
    with torch.no_grad():
        tr_rows = torch.tensor([r for k in tr_keys for r in rows_of[k]], device=dev)
        G.film_mu.copy_(summ[tr_rows].mean(0))
        G.film_sd.copy_(summ[tr_rows].std(0) + 1e-6)
    print("G params (M)", round(sum(p.numel() for p in G.parameters()) / 1e6, 3), flush=True)
    if not a.smoke:
        chain.G = G
        chain.onehot = True
        if not C3.ship_gate(chain, dev):
            print("Shipping Gate FAIL: 起動しない", flush=True)
            return 1
    chain.G = G
    opt = torch.optim.AdamW(G.parameters(), lr=a.lr, weight_decay=0.01)
    total = 30 if a.smoke else a.steps
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=total, pct_start=0.05)
    loader = iter(torch.utils.data.DataLoader(DS(utts, rows_of, tr_keys, 3), batch_size=a.bs, num_workers=a.workers, persistent_workers=True, prefetch_factor=4))
    vds = DS(utts, rows_of, val_keys, 11, gain=0.0, n=a.bs * 12)
    val_batches = None
    t0 = time.time()
    best = 1e9
    logf = open(out / "train.jsonl", "a")
    for step in range(1, total + 1):
        batch = next(loader)
        G.train()
        l_env, l_env0, l_per, l_per0, dc = losses(chain, G, batch, dev, a.drop)
        loss = l_env + a.l_per * l_per
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(G.parameters(), 5.0))
        opt.step(); sch.step()
        if step % 100 == 0 or a.smoke:
            rec = {"step": step, "min": round((time.time() - t0) / 60, 1), "l_env": round(float(l_env), 4), "l_env0": round(float(l_env0), 4), "l_per": round(float(l_per), 4), "l_per0": round(float(l_per0), 4), "gn": round(gn, 2),
                   "dc_rms_over_scale": round(float(dc.pow(2).mean().sqrt() / 0.5), 3)}
            print(json.dumps(rec), flush=True); logf.write(json.dumps(rec) + "\n"); logf.flush()
        if step % a.eval_every == 0 or step == total:
            if val_batches is None:
                val_batches = list(torch.utils.data.DataLoader(vds, batch_size=a.bs, num_workers=2))
            G.eval()
            with torch.no_grad():
                acc = np.zeros(6)
                for vb in val_batches:
                    le, le0, lp, lp0, _ = losses(chain, G, vb, dev, 0.0)
                    x, e_real, per_real, c0_real, idx, med = (t.to(dev) for t in vb)
                    sh = idx[torch.randperm(len(idx), device=dev)]
                    shuf = (x, e_real, per_real, c0_real, sh, med)
                    ls, ls0, _, _, _ = losses(chain, G, shuf, dev, 0.0)
                    acc += np.array([float(le), float(le0), float(lp), float(lp0), float(ls), float(ls0)])
                acc /= len(val_batches)
            G.train()
            ev = {"step": step, "val_l_env": round(acc[0], 4), "val_l_env0": round(acc[1], 4), "gain_pct": round(100 * (1 - acc[0] / acc[1]), 2),
                  "val_l_per": round(acc[2], 4), "val_l_per0": round(acc[3], 4), "shuf_l_env": round(acc[4], 4), "shuf_l_env0": round(acc[5], 4),
                  "shuf_gain_pct": round(100 * (1 - acc[4] / acc[5]), 2)}
            print("eval", json.dumps(ev), flush=True); logf.write("eval " + json.dumps(ev) + "\n"); logf.flush()
            sd = {"G": G.state_dict(), "cfg": G.cfg, "sm_w": sm_w, "W_SM": C3.W_SM, "onehot": True, "step": step}
            C3.atomic_save(sd, out / "last.pt")
            if acc[0] / acc[1] < best:
                best = acc[0] / acc[1]
                C3.atomic_save(sd, out / "best.pt")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
