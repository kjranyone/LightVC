"""D4v2: 話者重心正規化mel条件 + spk条件ドロップ(CFG)のカテゴリカルWaveNet。

定理E1処方(prereg: results/d4v2/prereg.yaml)。d4bの生成基盤(M1)の上に
「melから話者情報を重心減算で除去→話者identityの供給源をspk_emb唯一」を載せる。
学習=同話者再構成のみ、推論時に「source重心で正規化+target spk_emb」を差し込む
(train/test差が効くか=本腕の問い)。CFG: spkドロップ15%・推論外挿w=1.5。

    CUDA_VISIBLE_DEVICES=0 PYTORCH_ALLOC_CONF=expandable_segments:True \
      uv run python train_d4v2.py --tag d4v2_spknorm --steps 40000
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile
import torch

sys.path.insert(0, str(Path(__file__).parent))
from d4_vocoder import causal_mel48, HOP
from train_d4a import F0FIX, FEAT, F0_SRC_FPS, MEL_SCALE
from train_d4b import D4Cat, sample_from, mulaw_decode, mulaw_encode
from train_s1_3 import build_index, load_wav

ROOT = Path(__file__).resolve().parent.parent
SPK_ROWS = (82, 274)          # cond内 spk_emb の行範囲(mel80+lf0+enlのあと)
CFG_DROP = 0.15
CFG_W = 1.5


def load_cent():
    return torch.load(ROOT / "data/spk_mel_centroids.pt",
                      map_location="cpu", weights_only=False)


def get_utt_cond_v2(p: Path, cent: dict, spk_override: str | None = None):
    """d4bのget_utt_condと同一規約・melのみ重心減算に変更。"""
    feat = torch.load(FEAT / p.parent.name / (p.stem + ".pt"),
                      map_location="cpu", weights_only=False)
    w = load_wav(p).astype(np.float32)
    wv = torch.from_numpy(w[: len(w) // HOP * HOP])
    T = wv.shape[0] // HOP
    mel = causal_mel48(wv[None])[0]
    f0 = torch.load(F0FIX / p.parent.name / (p.stem + ".pt"),
                    map_location="cpu", weights_only=False)["f0"].float()
    idx = ((torch.arange(T, dtype=torch.float64) + 1.0) * 100.0 / F0_SRC_FPS
           - 1.0).floor().clamp(0, f0.shape[0] - 1).long()
    f0g = f0[idx]
    en = feat["energy"].float()
    i_e = ((torch.arange(T, dtype=torch.float64) + 1.0) * 100.0 / F0_SRC_FPS
           - 1.0).floor().clamp(0, en.shape[0] - 1).long()
    lf0 = torch.log(f0g.clamp(min=50.0) / 200.0)
    enl = torch.log(en[i_e].clamp(min=1e-4))
    spk_emb = torch.load(ROOT / "data/ecapa_spk_mean_full.pt",
                         map_location="cpu", weights_only=False)
    key = spk_override or feat.get("speaker")
    s_ = spk_emb.get(key, torch.zeros(192))
    c = cent.get(p.parent.name)
    if c is None:
        return None
    mel_norm = ((mel - c[:, None]) / MEL_SCALE).clamp(-6, 6)
    cond = torch.cat([mel_norm, lf0[None], enl[None],
                      s_[:, None].expand(-1, T)], 0)
    return wv, cond


def sample_cfg(m, cond, n, dev, temp=0.9, w=CFG_W, seed=3):
    """CFG外挿サンプリング: logits = u + w*(c-u)。u=spk行ゼロの無条件。"""
    g = torch.Generator(device=dev).manual_seed(seed)
    gen = torch.zeros(1, n, device=dev)
    rf = 3540
    hist = torch.zeros(1, rf, device=dev)
    cu = cond.clone()
    cu[:, SPK_ROWS[0]:SPK_ROWS[1]] = 0
    pos = 0
    while pos < n:
        nb = min(HOP, n - pos)
        ctx = torch.cat([hist, torch.zeros(1, nb, device=dev)], -1)
        with torch.no_grad():
            nf = (pos + nb) // HOP + 1
            lg_c = m.logits(ctx, cond[:, :, :nf])[:, :, -nb:]
            lg_u = m.logits(ctx, cu[:, :, :nf])[:, :, -nb:]
            lg = lg_u + w * (lg_c - lg_u)
        pr = torch.softmax(lg / temp, 1)
        bins = torch.multinomial(pr.transpose(1, 2).reshape(-1, 256), 1,
                                 generator=g)
        lv = bins.float() / 255 * 2 - 1
        y = mulaw_decode(lv).reshape(1, nb)
        gen[:, pos:pos + nb] = y
        new = gen[:, max(0, pos + nb - rf):pos + nb]
        hist = torch.cat([torch.zeros(1, rf - new.shape[-1], device=dev), new], -1) \
            if new.shape[-1] < rf else new
        pos += nb
    return gen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--every", type=int, default=2000)
    ap.add_argument("--cfg-drop", type=float, default=CFG_DROP)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(a.seed)

    cent = load_cent()
    tr, held = build_index()
    tr = [p for p in tr if p.parent.name in cent
          and (F0FIX / p.parent.name / (p.stem + ".pt")).exists()
          and (FEAT / p.parent.name / (p.stem + ".pt")).exists()]
    rng.shuffle(tr)
    print(f"  train {len(tr)}  centroids {len(cent)}", flush=True)

    m = D4Cat().to(dev)
    params = [q for q in m.parameters() if q.requires_grad]
    print(f"  params {sum(q.numel() for q in params)/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.99))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps,
                                              pct_start=0.03)
    out_dir = ROOT / "results" / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cache: dict = {}

    def get(p: Path):
        if p not in cache:
            if len(cache) > 300:
                cache.pop(next(iter(cache)))
            try:
                cache[p] = get_utt_cond_v2(p, cent)
            except Exception:  # noqa: BLE001
                cache[p] = None
        return cache[p]

    best = 1e9
    t0 = time.time()
    step = 0
    if a.resume:
        ck = out_dir / f"{a.tag}_last.pt"
        if ck.exists():
            st = torch.load(ck, map_location=dev, weights_only=False)
            m.load_state_dict(st["net"])
            step = st["step"]
            for _ in range(step):
                sch.step()
            print(f"  resumed at step {step} (opt state reset)", flush=True)

    while step < a.steps:
        ws, cs = [], []
        while len(ws) < a.batch:
            p = rng.choice(tr)
            it = get(p)
            if it is None:
                continue
            wv, cond = it
            T = min(wv.shape[0] // HOP, cond.shape[1])
            if T <= 104:
                continue
            s0 = rng.randrange(0, T - 100)
            ws.append(wv[s0 * HOP:(s0 + 100) * HOP])
            cs.append(cond[:, s0:s0 + 100])
        step += 1
        xb = torch.stack(ws).to(dev)
        cb = torch.stack(cs).to(dev)
        keep = (torch.rand(a.batch, device=dev) >= a.cfg_drop)
        cb[:, SPK_ROWS[0]:SPK_ROWS[1]] *= keep[:, None, None]
        lg = m.logits(xb, cb)
        tgt = ((mulaw_encode(xb) + 1) / 2 * 255).round().long().clamp(0, 255)
        loss = torch.nn.functional.cross_entropy(lg.transpose(1, 2).reshape(-1, 256),
                                                 tgt.reshape(-1))
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sch.step()
        if step % a.every == 0 or step == a.steps:
            from eval_d4a_freerun import metrics
            p = held[0]
            it = get(p)
            n = min(it[0].shape[0], 96000)
            y = sample_from(m, it[1][None, :, : n // HOP].to(dev), n, dev)
            y = y[0].cpu().numpy()
            soundfile.write(out_dir / "freerun.wav", np.clip(y, -1, 1), 48000)
            mm = metrics(y)
            print(f"  step {step:6d}  CE {float(loss.detach()):.4f}  fr {mm}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
            torch.save({"net": m.state_dict(), "step": step, "cli": vars(a)},
                       out_dir / f"{a.tag}_last.pt")
            score = float(loss.detach()) - (1.0 if mm["hi_mid"] > 0.004 else 0.0) \
                    - (1.0 if abs(mm["f0_median"] - 100.0) > 3.0 else 0.0)
            if score < best:
                best = score
                torch.save({"net": m.state_dict(), "step": step, "cli": vars(a)},
                           out_dir / f"{a.tag}_best.pt")
    print(f"\n{a.tag}: best score {best:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
