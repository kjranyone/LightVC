"""F2(lvl.py)の学習: 教師 = 出力部の条件の c0(CheapTrick・教師の f0・[.25,.5,.25] 平滑)。男声(f0est の train)と女声(rvoc の train)を 1:1・利得 ±18dB・評価は男声の評価話者と女声の学習話者の外。
    uv run python train_lvl.py --tag f2_1 --steps 4000
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

sys.path.insert(0, str(Path(__file__).parent))
import c1_content as CC
import lvl as LV
import nvoc as N
import pae as PA
import train_f0est as TF
import train_rvoc as TR

ROOT = Path(__file__).resolve().parent.parent
CTX = N.WIN - N.HOP
SEG_F = 800


def sm3(E: np.ndarray, w: float = 0.25) -> np.ndarray:
    e = np.pad(E, ((0, 0), (1, 1)), mode="edge")
    return w * e[:, :-2] + (1 - 2 * w) * e[:, 1:-1] + w * e[:, 2:]


class DS(torch.utils.data.IterableDataset):
    def __init__(self, male: list, fem: list, seed: int, gain: float = 18.0):
        self.male, self.fem, self.seed, self.gain = male, fem, seed, gain

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 977 + (wi.id if wi else 0))
        while True:
            pool = self.male if rng.random() < 0.5 else self.fem
            wav, f0p, sr, dur = pool[rng.randrange(len(pool))]
            try:
                n48 = int(dur * N.SR)
                if n48 < (SEG_F + 20) * N.HOP:
                    continue
                s = rng.randrange(0, n48 - SEG_F * N.HOP) // N.HOP * N.HOP
                a = s - CTX
                lo = max(a, 0)
                x = TR.read_span(wav, sr, lo, a + CTX + SEG_F * N.HOP - lo)
                x = np.concatenate([np.zeros(lo - a, np.float32), x]) if lo > a else x
                if len(x) < CTX + SEG_F * N.HOP:
                    continue
                x = x[:CTX + SEG_F * N.HOP]
                g = 10 ** (rng.uniform(-self.gain, self.gain) / 20)
                x = np.clip(x * g, -1, 1).astype(np.float32)
                f0 = np.load(f0p).astype(np.float32)
                f0c = np.zeros(SEG_F, np.float32)
                i0 = s // N.HOP
                seg = f0[i0:i0 + SEG_F]
                f0c[:len(seg)] = seg
                xs = x[CTX:]
                xa = np.concatenate([np.zeros(PA.A, np.float32), xs, np.zeros(PA.A, np.float32)])
                c0n = ((sm3(PA.envelope(xa, f0c)[:1])[0] - PA.C0_SIL) / 10).astype(np.float32)
                yield torch.from_numpy(x), torch.from_numpy(c0n)
            except Exception:
                continue


def ship_gate(net, front, row, dev) -> None:
    """未来不変性(実音声): 入力の t 以降を書き換えて、t 以前を担当する出力フレームが変わらないこと。変わらず・編集が出力を動かす(INCONCLUSIVE でない)の両方を要る。"""
    wav, f0p, sr, dur = row
    x = TR.read_span(wav, sr, 0, int(min(dur, 6.0) * N.SR))
    x = torch.from_numpy(x.astype(np.float32))[None].to(dev)
    net.eval()
    pad = torch.zeros(1, CTX, device=dev)
    with torch.no_grad():
        ref = net(front(torch.cat([pad, x], 1)))[0]
        t = x.shape[1] // 2 // N.HOP * N.HOP
        x2 = x.clone()
        g = torch.Generator(device="cpu").manual_seed(0)
        x2[:, t:] = torch.randn(x2[:, t:].shape, generator=g).to(dev) * x.std() * 3
        out = net(front(torch.cat([pad, x2], 1)))[0]
    k = t // N.HOP
    before = float((ref[:k] - out[:k]).abs().max())
    after = float((ref[k + 2:] - out[k + 2:]).abs().max())
    print(f"ship_gate: 未来を書き換えたとき t 以前の出力の最大差 {before:.2e}・以後の最大差 {after:.2e}", flush=True)
    assert before < 1e-4 and after > 1e-3, "Shipping Gate FAIL(未来不変でない または 編集が出力を動かさず INCONCLUSIVE)"
    net.train()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--dils", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    a = ap.parse_args()
    dev = "cuda"
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    male = TF.male_rows("train")
    real, tts = TF.female_rows()
    fem = real
    ev_male = random.Random(0).sample(TF.male_rows("eval"), 150)
    print("male", len(male), "female", len(fem), "eval male", len(ev_male), flush=True)
    front = CC.MelFront().to(dev)
    net = LV.Lvl(a.ch, tuple(a.dils)).to(dev)
    ship_gate(net, front, ev_male[0], dev)
    print("params (M)", round(sum(p.numel() for p in net.parameters()) / 1e6, 4), "RF frames", net.rf, flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.1)
    loader = iter(torch.utils.data.DataLoader(DS(male, fem, 3), batch_size=a.bs, num_workers=a.workers, persistent_workers=True, prefetch_factor=4))
    evl = iter(torch.utils.data.DataLoader(DS(ev_male, random.Random(1).sample(fem, 150), 9, gain=6.0), batch_size=a.bs, num_workers=2))
    t0 = time.time()
    for step in range(1, a.steps + 1):
        x, y = (t.to(dev) for t in next(loader))
        p = net(front(x))
        loss = torch.nn.functional.smooth_l1_loss(p, y, beta=0.1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sch.step()
        if step % 200 == 0:
            print(json.dumps({"step": step, "min": round((time.time() - t0) / 60, 1), "loss": round(float(loss), 4)}), flush=True)
        if step % 1000 == 0 or step == a.steps:
            with torch.no_grad():
                net.eval()
                r = []
                for _ in range(8):
                    xe, ye = (t.to(dev) for t in next(evl))
                    pe = net(front(xe))
                    act = ye > 1.0
                    r.append(((pe - ye)[act]).cpu().numpy())
                r = np.concatenate(r)
                net.train()
            print("eval residual (active frames) mean %.3f std %.3f |  p95 abs %.3f" % (r.mean(), r.std(), np.percentile(np.abs(r), 95)), flush=True)
            torch.save({"net": net.state_dict(), "cfg": net.cfg, "step": step}, out / "last.pt")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
