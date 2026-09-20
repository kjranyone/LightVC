"""V2-2: P = 残差 prosody 写像の学習（same_text ペア・source f0 輪郭 -> Δlf0/vuv/en）。

入力: source lf0/vuv/energy（因果）+ causal content（E1 凍結）+ target prosody-style
統計ベクトル（話者×caption 定数）+ GUI ノブ（tension など・学習時はランダム摂動）。
出力: Δlf0（source への残差）・vuv'・Δen。絶対 F0 を全面生成しない＝source の
タイミング・抑揞を保持（vc_fm.md §3 rev2）。

教師: same_text 男女ペアを content DTW で target->source warp し、warp 後の
target lf0 - source lf0 = Δlf0 GT。timing そのものは回帰対象にしない。

    CUDA_VISIBLE_DEVICES=0 uv run python train_p.py --steps 20000 --tag v22_p
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import ship_front as SF
from train_vc_g import CausalBlock
from train_vc_e import E1

ROOT = Path(__file__).resolve().parent.parent
PAIRS = ROOT / "data/same_text_pairs"
F0_FPS = 44100 / 512


class P1(nn.Module):
    """残差 prosody 写像。全左 pad・先読み 0。RTF 予算 0.01（小型）。"""

    def __init__(self, dim: int = 96, layers: int = 5):
        super().__init__()
        self.dim, self.layers = dim, layers
        self.inp = nn.Conv1d(3 + 16 + 2, dim, 3)     # lf0,vuv,en + style16 + gui2
        self.blocks = nn.ModuleList(
            [CausalBlock(dim, 3, 2 ** (i // 2)) for i in range(layers)])
        self.out = nn.Linear(dim, 3)                 # Δlf0, vuv_logit, Δen

    @property
    def ctx(self) -> int:
        return 2 + sum((3 - 1) * (2 ** (i // 2)) for i in range(self.layers))

    def forward(self, x):                            # [B, 21, T] -> [B, 3, T]
        x = self.inp(F.pad(x, (2, 0)))
        for b in self.blocks:
            x = b(x)
        return self.out(x.transpose(1, 2)).transpose(1, 2)

    def arch(self) -> dict:
        return {"arch": "p1", "dim": self.dim, "L": self.layers, "ctx": self.ctx}


def f0v_of(path: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import soundfile as sf_
    w, sr = sf_.read(str(path), dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    if sr != 44100:
        import librosa
        w = librosa.resample(np.asarray(w), orig_sr=sr, target_sr=44100)
    x = torch.from_numpy(np.ascontiguousarray(w)) * 32768.0
    f0, _ = SF.causal_f0(x)
    n = x.shape[-1]
    hop = 512
    nf = n // hop
    rms = torch.sqrt(((x[: nf * hop] / 32768.0).reshape(nf, hop) ** 2).mean(-1) + 1e-12)
    return f0, (f0 > 50).float(), torch.log(rms.clamp(min=1e-4))


def lf0_prep(f0: torch.Tensor) -> torch.Tensor:
    v = f0 > 50
    lf = torch.log2(f0.clamp(min=50.0) / 200.0)
    out = torch.where(v, lf, torch.zeros_like(lf))
    # 直近有声値で埋める（因果 fill）
    for i in range(1, out.shape[-1]):
        if not v[i]:
            out[i] = out[i - 1]
    return out


def dtw_warp(src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor | None:
    """target を source フレームグリッドへ warp（単調・局所）。軽量: sax+粗 DTW。"""
    n, m = src.shape[-1], tgt.shape[-1]
    if abs(n - m) > 0.6 * max(n, m):
        return None
    # 粗視化（8 フレーム平均）で DTW
    r = 8
    ns, ms = n // r, m // r
    if ns < 2 or ms < 2:
        return None
    s_ = src[..., : ns * r].reshape(ns, r).mean(-1).numpy().astype(float)
    t_ = tgt[..., : ms * r].reshape(ms, r).mean(-1).numpy().astype(float)
    D = np.full((ns + 1, ms + 1), np.inf)
    D[0, 0] = 0
    P = np.zeros((ns + 1, ms + 1), dtype=np.int8)
    for i in range(1, ns + 1):
        si = float(s_[i - 1])
        for j in range(1, ms + 1):
            tj = float(t_[j - 1])
            c = abs(si - tj)
            best, bk = np.inf, 0
            for di, dj, tag in ((1, 1, 1), (1, 0, 2), (0, 1, 3)):
                v = D[i - di, j - dj]
                if v < best:
                    best, bk = v, tag
            D[i, j] = best + c
            P[i, j] = bk
    # 経路から src 各粗フレーム -> tgt 粗フレーム index
    i, j = ns, ms
    map_ = np.full(ns, -1, dtype=np.int64)
    while i > 0:
        if map_[i - 1] < 0:
            map_[i - 1] = j - 1
        tag = P[i, j]
        if tag == 1:
            i, j = i - 1, j - 1
        elif tag == 2:
            i -= 1
        else:
            j -= 1
    fine = (torch.from_numpy(map_.clip(0)).float() * r).clamp(0, max(m - r, 0)).long()
    # ファイングリッドは線形伸縮で射影（粗 DTW の区分定数 + 区間内線形）
    out = torch.zeros_like(src)
    for k in range(ns):
        lo, hi = k * r, min((k + 1) * r, n)
        tlo = fine[k].item()
        seg = tgt[..., tlo: tlo + max(hi - lo, 1)]
        if seg.shape[-1] < hi - lo:
            seg = F.interpolate(seg[None], size=hi - lo, mode="linear",
                                align_corners=False)[0]
        out[..., lo:hi] = seg
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=344)   # 2s @172fps mel 相当 -> f0 86fps
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--e", default=str(ROOT / "results/diag_e2/diag_e2_best.pt"))
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(a.seed)

    male_dirs = sorted([d for d in PAIRS.iterdir() if d.name.startswith(("m", "n"))])
    fem_dirs = sorted([d for d in PAIRS.iterdir() if d.name.startswith("f")])
    print(f"  males {len(male_dirs)}  females {len(fem_dirs)}", flush=True)
    style = torch.load(ROOT / "data/prosody_style_table.pt",
                       map_location="cpu", weights_only=False)
    style_mat, style_keys = [], []
    for k, v in style.items():
        style_mat.append(v)
        style_keys.append(k)
    style_mat = torch.stack(style_mat)                # [N, 6]

    ek = torch.load(a.e, map_location=dev)
    enet = E1(dim=ek["args"]["dim"], layers=ek["args"]["L"],
              look=ek["args"].get("look", 0)).to(dev).eval()
    enet.load_state_dict(ek["net"])
    for q in enet.parameters():
        q.requires_grad_(False)

    net = P1().to(dev)
    print(f"  params {sum(p.numel() for p in net.parameters())/1e6:.3f}M ctx {net.ctx}",
          flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.99))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps)
    out_dir = ROOT / "results" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cache: dict = {}

    def get_pair(md: Path, fd: Path, tid: str):
        key = (md.name, fd.name, tid)
        if key in cache:
            return cache[key]
        mf, ff = md / f"{tid}.wav", fd / f"{tid}.wav"
        if not mf.exists() or not ff.exists():
            cache[key] = None
            return None
        f0m, vuvm, enm = f0v_of(mf)
        f0f, vuvf, enf = f0v_of(ff)
        lm, lf = lf0_prep(f0m), lf0_prep(f0f)
        lf_w = dtw_warp(lm, lf)
        if lf_w is None or lf_w.shape[-1] < 100:
            cache[key] = None
            return None
        t = min(lm.shape[-1], lf_w.shape[-1], enm.shape[-1])
        cache[key] = (lm[:t], vuvm[:t], enm[:t], lf_w[:t], vuvf[:t], enf[:t])
        if len(cache) > 300:
            cache.pop(next(iter(cache)))
        return cache[key]

    tids = sorted(p.stem for p in (PAIRS / fem_dirs[0]).glob("*.wav"))
    print(f"  {len(tids)} texts", flush=True)

    print("  pre-computing pairs (DTW warp) ...", flush=True)
    items: list = []
    for md in male_dirs:
        for fd in fem_dirs:
            for tid in tids:
                it = get_pair(md, fd, tid)
                if it is not None:
                    lm, vuvm, enm, lf_w, vuvf, enf = it
                    items.append((lm, vuvm, enm, lf_w, vuvf, enf, fd.name))
        print(f"    {md.name}: {len(items)} pairs", flush=True)
    print(f"  {len(items)} usable pairs", flush=True)
    cache.clear()

    def style_vec(fd_name: str) -> torch.Tensor:
        cands = [i for i, k in enumerate(style_keys)
                 if k.split("|")[1] == fd_name or fd_name in k]
        v = style_mat[rng.choice(cands) if cands else rng.randrange(len(style_mat))]
        return v

    best = 1e9
    step = 0
    t0 = time.time()
    while step < a.steps:
        xs, ys = [], []
        while len(xs) < a.batch:
            lm, vuvm, enm, lf_w, vuvf, enf, fdn = items[rng.randrange(len(items))]
            if lm.shape[-1] <= a.crop + net.ctx + 4:
                continue
            s = rng.randrange(net.ctx, lm.shape[-1] - a.crop)
            sv = torch.cat([style_vec(fdn), torch.randn(10), torch.rand(2) * 2 - 1])
            seg = a.crop + net.ctx
            x = torch.cat([lm[None, s - net.ctx: s + a.crop],
                           vuvm[None, s - net.ctx: s + a.crop],
                           enm[None, s - net.ctx: s + a.crop],
                           sv[:, None].expand(-1, seg)], 0)
            t_ = min(a.crop, lf_w.shape[-1] - s, vuvf.shape[-1] - s, enf.shape[-1] - s)
            if t_ < a.crop:
                continue
            y = torch.stack([lf_w[s: s + t_] - lm[s: s + t_],
                             vuvf[s: s + t_],
                             enf[s: s + t_] - enm[s: s + t_]])
            xs.append(x)
            ys.append(y)
        step += 1
        xb = torch.stack(xs).to(dev)
        yb = torch.stack(ys).to(dev)
        pred = net(xb)[:, :, net.ctx:]
        dlf0_gt = yb[:, 0]
        m = (yb[:, 1] > 0.5)
        l_dlf0 = F.l1_loss(pred[:, 0][m], dlf0_gt[m]) if m.any() else xb.sum() * 0
        l_vuv = F.binary_cross_entropy_with_logits(pred[:, 1], yb[:, 1])
        l_den = F.l1_loss(pred[:, 2], yb[:, 2])
        loss = l_dlf0 + 0.5 * l_vuv + 0.25 * l_den
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sch.step()
        if step % a.every == 0 or step == a.steps:
            print(f"  step {step:6d}  dlf0 {float(l_dlf0):.4f}  vuv {float(l_vuv):.4f}"
                  f"  den {float(l_den):.4f}  ({time.time()-t0:.0f}s)", flush=True)
            ck = {"net": net.state_dict(), "args": net.arch(), "step": step,
                  "cli": vars(a)}
            torch.save(ck, out_dir / f"{a.tag}_last.pt")
            if float(l_dlf0) < best:
                best = float(l_dlf0)
                torch.save(ck, out_dir / f"{a.tag}_best.pt")
    print(f"\n{a.tag}: best dlf0-L1 {best:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
