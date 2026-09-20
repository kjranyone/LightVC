"""R-E1: 因果 E — mel80(172fps) -> ContentVec 768d(50fps) の蒸留。

`current/vc_eg.md` のラダー第 2 段。**表現監督**（CLAUDE.md 許可）であり
VC teacher 蒸留ではない。教師特徴は data/*_feat に焼き込み済み。

入力は**製品 front-end の mel80**（causal_mel 左寄せ・先読み 0）——
「学習で見た値」と「製品が作る値」を最初から一致させる。

判定（凍結・vc_eg.md）: held-out cos-sim >= 0.90。

    uv run python train_vc_e.py --steps 20000 --tag diag_e1
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import ship_front as SF
from train_vc_g import CausalBlock, load_item, MEL_FPS, CV_FPS, FEATS

# E は universal content 抽出器なので男声も学習に含める（G の FEATS は女声のみで正しい）。
# 初版が女声のみだったのは設計ミス: male mel が OOD になり cos 0.73→0.64 に劣化
# （R-X 切り分け 2026-08-20）。
from train_vc_g import ROOT as _ROOT
E_FEATS = FEATS + [_ROOT / "data/male_feat"]

ROOT = Path(__file__).resolve().parent.parent


class E1(nn.Module):
    """mel80 -> 768d content。既定は左パディングのみ＝先読み 0。

    `--look R` で右文脈 R フレーム（R×5.8ms）を許す。**出荷ゲートの
    「content encoder 予備 10ms」の枠内でのみ使う**（R=1 → 5.8ms）。
    教師 ContentVec は双方向なので、因果では写せない未来依存成分がある
    （dim 256/384/512 で cos 0.728/0.728/0.731 ＝ 容量では解けないと実測）。

    出力は mel グリッド（172fps）のまま。教師 50fps へは**因果対応**で
    「content フレーム j を、その窓が閉じた直後の mel フレームに割り当てる」。
    """

    def __init__(self, dim: int = 256, layers: int = 8, look: int = 0):
        super().__init__()
        self.dim, self.layers, self.look = dim, layers, look
        self.inp = nn.Conv1d(SF.N_MEL, dim, 3)
        self.blocks = nn.ModuleList(
            [CausalBlock(dim, 3, 2 ** (i // 2)) for i in range(layers)])
        self.out = nn.Linear(dim, 768)

    @property
    def ctx(self) -> int:
        return 2 + sum((3 - 1) * (2 ** (i // 2)) for i in range(self.layers))

    def forward(self, mel):                            # [B, 80, T] -> [B, 768, T]
        # look>0: 入力を look フレームだけ左へずらす＝各出力が look 先の mel まで見る。
        # 出力時刻は保存される（右端は複製で埋める＝ストリームでは look 分の遅延）。
        if self.look:
            mel = torch.cat([mel[:, :, self.look:],
                             mel[:, :, -1:].expand(-1, -1, self.look)], -1)
        x = self.inp(F.pad(mel, (2, 0)))
        for b in self.blocks:
            x = b(x)
        return self.out(x.transpose(1, 2)).transpose(1, 2)

    def arch(self) -> dict:
        return {"arch": "e1", "dim": self.dim, "L": self.layers, "ctx": self.ctx,
                "look": self.look}


def teacher_index(t_mel: int) -> torch.Tensor:
    """mel フレーム t に対応する教師フレーム（因果: 窓が閉じた最新のもの）。"""
    return ((torch.arange(t_mel, dtype=torch.float64) + 1.0)
            * CV_FPS / MEL_FPS - 1.0).floor().clamp(min=0).long()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=344)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--every", type=int, default=1000)
    ap.add_argument("--snap", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--look", type=int, default=0,
                    help="右文脈フレーム数。1=5.8ms（予備 10ms の枠内のみ）")
    ap.add_argument("--limit-utts", type=int, default=0)
    ap.add_argument("--tag", type=str, required=True)
    a = ap.parse_args()
    if a.limit_utts and not a.tag.startswith("diag_"):
        sys.exit("部分集合は diag_ タグでのみ許可（CLAUDE.md Data 規則）")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(a.seed)
    files: list[Path] = []
    for root in E_FEATS:
        for spk in sorted(root.iterdir()):
            if spk.is_dir():
                files += sorted(spk.glob("*.pt"))
    rng.shuffle(files)
    # held は**女声**末尾 24 名で不変に保つ（male_* は 'm'>hex で末尾に並ぶため、
    # 素朴な [-24:] だと held が男声に化けて cos の基準が壊れる）
    spk_all = sorted({f.parent.name for f in files if not f.parent.name.startswith("male_")})
    held = set(spk_all[-24:])
    tr = [f for f in files if f.parent.name not in held]
    ev = [f for f in files if f.parent.name in held][:24]
    if a.limit_utts:
        tr = tr[:a.limit_utts]
    print(f"  train {len(tr)} utts / {len(spk_all) - 24} spk   eval {len(ev)}", flush=True)

    if a.look * 256 / 44.1 > 10.0:
        sys.exit(f"look={a.look} は content encoder 予備 10ms を超える（出荷ゲート）")
    net = E1(dim=a.dim, layers=a.layers, look=a.look).to(dev)
    print(f"  params {sum(p.numel() for p in net.parameters())/1e6:.2f}M  ctx {net.ctx}",
          flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.99))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps)
    out_dir = ROOT / "results" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cache: dict = {}

    def get(f: Path):
        if f not in cache:
            if len(cache) > 3000:
                cache.pop(next(iter(cache)))
            try:
                cache[f] = load_item(f)
            except Exception:                          # noqa: BLE001
                cache[f] = None
        return cache[f]

    def pair(d, mel):
        t = mel.shape[-1]
        idx = teacher_index(t)
        c = d["content"].T.float()                    # [768, Tc]
        idx = idx.clamp(max=c.shape[-1] - 1)
        return mel, c[:, idx]                          # [80,T], [768,T]

    def evaluate() -> float:
        net.eval()
        cs = []
        with torch.no_grad():
            for f in ev:
                it = get(f)
                if it is None:
                    continue
                d, mel = it
                x, y = pair(d, mel)
                p = net(x.to(dev)[None])[0].cpu()
                m = min(p.shape[-1], y.shape[-1])
                cs.append(float(F.cosine_similarity(p[:, :m], y[:, :m], dim=0).mean()))
        net.train()
        return sum(cs) / max(len(cs), 1)

    t0 = time.time()
    best = -1.0
    step = 0
    while step < a.steps:
        xs, ys = [], []
        while len(xs) < a.batch:
            it = get(rng.choice(tr))
            if it is None:
                continue
            d, mel = it
            x, y = pair(d, mel)
            t = x.shape[-1]
            if t <= a.crop + net.ctx + 4:
                continue
            s = rng.randrange(net.ctx, t - a.crop)
            xs.append(x[:, s - net.ctx: s + a.crop])
            ys.append(y[:, s: s + a.crop])
        step += 1
        xb = torch.stack(xs).to(dev)
        yb = torch.stack(ys).to(dev)
        p = net(xb)[:, :, net.ctx:]
        # cos 蒸留 ＋ L2（`e1` の実績構成に合わせる）
        loss = (1.0 - F.cosine_similarity(p, yb, dim=1).mean()) \
            + 0.1 * F.mse_loss(p, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sch.step()
        if step % a.every == 0 or step == a.steps:
            cs = evaluate()
            print(f"  step {step:6d}  loss {float(loss):.4f}  eval-cos {cs:.4f}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
            ck = {"net": net.state_dict(), "args": net.arch(), "step": step,
                  "eval_cos": cs, "cli": vars(a)}
            torch.save(ck, out_dir / f"{a.tag}_last.pt")
            if cs > best:
                best = cs
                torch.save(ck, out_dir / f"{a.tag}_best.pt")
            if a.snap and step % a.snap == 0:
                torch.save(ck, out_dir / f"{a.tag}_s{step}.pt")
    print(f"\n{a.tag}: best eval-cos {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
