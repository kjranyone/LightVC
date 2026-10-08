"""構音の逆推定 段 2: 因果の推定器(ネット)を、物理の前向きモデルとピッチ不変性で学習する。current/artic_inv.md §4・事前登録 results/<tag>/prereg.yaml。

入力(100fps・フレーム因果): 因果 log-mel 128(フレーム内の平均を引く)・生の因果 YIN の log f0(無声は 0・YIN の 240 hop の偶数番目 = mel の窓より 5ms 前に終わる値・artic_prep と同じ)・有声フラグ = 推論と同じ入力。
損失側の倍音の位置は精密 f0(harvest・中心揃え)= 損失だけで推論経路に入らない。
出力: q [K](声道の形の成分・tanh で ±2.5・既定 K=16)・src [3](log(fg/f0) ∈ [log .4, log 5]・log fa ∈ [log 300, log 12000]・
      壁の損失(帯域幅)の倍率 log ∈ [log .3, log 3])・log L ∈ [log .11, log .21]。対象の帯域は倍音 ≤ 5kHz(管の物理が成り立つ範囲)。
      5kHz より上の包絡は構音界面の別成分(推定不要・mel から直接)で、この推定器は扱わない(artic_inv.md §6c)。
損失: (1) 物理: real・w0・ws の 3 信号で、前向きモデル(artic_fit.model_db)の倍音の振幅を観測へ(重み付き Huber・利得はフレームごとの閉じた式)。
      (2) 不変性: w0 と ws(同じ包絡・f0 だけ違う)の q と log L の差(両方有声のフレーム)。
      (3) 時間の差分(q)・区間内の log L の分散・q の大きさ(中立への弱い引き)。
データ: artic_prep.py の data/artic_inv(区間ごとに real・w0・ws)。除外 = ファイル番号の末尾 5%。

    CUDA_VISIBLE_DEVICES=0 uv run python train_artic_inv.py --tag artic_inv1
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
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import artic_fit as AF

ROOT = Path(__file__).resolve().parent.parent
SIGS = ("real", "w0", "ws")


class DS(torch.utils.data.Dataset):
    def __init__(self, files: list[Path]):
        self.files = files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int) -> dict:
        d = np.load(self.files[i])
        out = {}
        for s in SIGS:
            out[f"f0_{s}"] = torch.from_numpy(d[f"f0_{s}"].astype(np.float64))
            out[f"yin_{s}"] = torch.from_numpy(d[f"yin_{s}"].astype(np.float64))
            out[f"am_{s}"] = torch.from_numpy(d[f"am_{s}"].astype(np.float64))
            out[f"mel_{s}"] = torch.from_numpy(d[f"mel_{s}"].astype(np.float32))
        return out


class Inv(nn.Module):
    def __init__(self, k: int, ch: int = 256, dils: tuple = (1, 2, 4, 8, 16, 1, 2, 4)):
        super().__init__()
        self.k = k
        self.inp = nn.Conv1d(130, ch, 1)
        self.convs = nn.ModuleList(nn.Conv1d(ch, ch, 3, dilation=d) for d in dils)
        self.dils = dils
        self.head = nn.Conv1d(ch, k + 4, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, mel: torch.Tensor, f0: torch.Tensor) -> tuple:
        """mel [B,128,T]・f0 [B,T] → q [B,T,K]・src [B,T,2]・logL [B,T]。"""
        m = mel - mel.mean(1, keepdim=True)
        v = (f0 > 0).float()
        lf = torch.where(f0 > 0, torch.log(f0.clamp(min=1.0)) - math.log(200.0), torch.zeros_like(f0)).float()
        h = self.inp(torch.cat([m, lf[:, None], v[:, None]], 1))
        for c, d in zip(self.convs, self.dils):
            h = h + F.gelu(c(F.pad(h, (2 * d, 0))))
        o = self.head(h).transpose(1, 2).double()
        q = 2.5 * torch.tanh(o[..., :self.k])
        g = math.log(0.4) + (math.log(5.0) - math.log(0.4)) * torch.sigmoid(o[..., self.k] + 1.0)
        fa = math.log(300.0) + (math.log(12000.0) - math.log(300.0)) * torch.sigmoid(o[..., self.k + 1] + 0.3)
        lL = math.log(0.11) + (math.log(0.21) - math.log(0.11)) * torch.sigmoid(o[..., self.k + 2])
        ls = math.log(0.3) + (math.log(3.0) - math.log(0.3)) * torch.sigmoid(o[..., self.k + 3])
        return q, torch.stack([g, fa, ls], -1), lL


def model_db_frames(q: torch.Tensor, lL: torch.Tensor, src: torch.Tensor, fr: torch.Tensor, f0: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """フレームを平らにして artic_fit の前向きモデルを使う(声道長はフレームごと)。q [M,K]・lL [M]・src [M,2]・fr [M,H]・f0 [M] → [M,H]。"""
    areas = (AF.A0 * torch.exp(q @ B)).clamp(5e-6, 2e-3)
    f = fr.clamp(min=20.0)
    h = AF.transfer_pointwise(areas, torch.exp(lL), f, torch.exp(src[:, 2]))
    env = 20 * torch.log10((2 * math.pi * f) * h.abs() + 1e-12)
    fg = (torch.exp(src[:, 0]) * f0)[:, None]
    fa = torch.exp(src[:, 1])[:, None]
    return env - 20 * torch.log10(1 + (f / fg) ** 2) - 10 * torch.log10(1 + (f / fa) ** 2)


def physics_loss(q, src, lL, am, f0, B) -> tuple[torch.Tensor, torch.Tensor]:
    """q [B,T,K] 等 → 重み付き Huber(利得は閉じた式)と RMS(dB)。有声フレームのみ。"""
    Bn, T, H = am.shape
    v = (f0 > 0).reshape(-1)
    if v.sum() == 0:
        z = q.sum() * 0
        return z, z
    kk = torch.arange(1, H + 1, device=am.device, dtype=am.dtype)
    fr = f0.reshape(-1, 1)[v] * kk
    obs = am.reshape(-1, H)[v]
    mk = (fr <= AF.F_MAX) & (obs > -100)
    mkd = mk.double()
    top = torch.where(mk, obs, torch.full_like(obs, -1e9)).amax(1, keepdim=True)
    w = mkd * torch.sigmoid((obs - (top - 25.0)) / 3.0)
    pred0 = model_db_frames(q.reshape(-1, q.shape[-1])[v], lL.reshape(-1)[v], src.reshape(-1, src.shape[-1])[v], fr, f0.reshape(-1)[v], B)
    gain = ((obs - pred0) * w).sum(1) / w.sum(1).clamp(min=1e-9)
    pred = pred0 + gain[:, None]
    hub = (F.huber_loss(pred, obs, reduction="none", delta=6.0) * w).sum() / w.sum().clamp(min=1e-9)
    rms = torch.sqrt((((pred - obs) ** 2) * w).sum() / w.sum().clamp(min=1e-9))
    return hub, rms


def evaluate(net: Inv, dl, B: torch.Tensor, dev: str) -> dict:
    net.eval()
    rms, moves, nat = [], [], []
    with torch.no_grad():
        for bt in dl:
            bt = {k: v.to(dev) for k, v in bt.items()}
            outs = {s: net(bt[f"mel_{s}"], bt[f"yin_{s}"]) for s in SIGS}
            _, r = physics_loss(*outs["real"][:1], outs["real"][1], outs["real"][2], bt["am_real"], bt["f0_real"], B)
            rms.append(float(r))
            v = (bt["f0_w0"] > 0) & (bt["f0_ws"] > 0)
            if v.sum() > 10:
                q0, q1 = outs["w0"][0][v], outs["ws"][0][v]
                moves.append(((q0 - q1) ** 2).mean(0).sqrt().cpu().numpy())
                nat.append(q0.std(0).cpu().numpy())
    net.train()
    mv = np.mean(moves, 0) / np.maximum(np.mean(nat, 0), 1e-6)
    return {"g0_rms_db_real": round(float(np.mean(rms)), 3), "g1_q_move_over_nat": round(float(np.median(mv)), 3),
            "q_nat_std": [round(float(x), 3) for x in np.mean(nat, 0)]}


def ship_gate(net: Inv) -> bool:
    """推論経路(音声 → 因果 log-mel・生の因果 YIN → 推定器)の実音声の未来不変性(出力層を乱数化した複製・0 判定・男声プローブ込み)。"""
    import copy
    import artic_dsp as D
    import nvoc as N
    import ship_check as SC
    from artic_prep import causal_logmel
    m = copy.deepcopy(net).cpu().float().eval()
    torch.manual_seed(1)
    nn.init.normal_(m.head.weight, 0, 0.05)
    fb = N.mel_fb().numpy().astype(np.float64)
    H = AF.HOP

    def fn(x: torch.Tensor) -> torch.Tensor:
        xx = x.double().numpy()
        n = len(xx) // H * H
        xx = xx[:n]
        T = n // H
        pad = np.concatenate([np.zeros(1024 - H), xx])
        fr = np.lib.stride_tricks.sliding_window_view(pad, 1024)[::H][:T] * np.hanning(1024)
        mel = np.log(np.maximum(np.abs(np.fft.rfft(fr, n=2048, axis=1)) @ fb.T, 1e-5)).T
        y, _ = D.causal_yin(xx, voi_max=0.45)
        y = y[0::2][:T]
        y = np.pad(y, (0, T - len(y)))
        q, src, lL = m(torch.from_numpy(mel).float()[None], torch.from_numpy(y).double()[None])
        return torch.cat([q[0], src[0], lL[0][:, None]], 1).T.float()

    la = SC.future_invariance(fn, hop=H, n=2 * 48000, sr=48000, quantity=False, n_edit=8, male=2)
    return SC.ledger([("構音の推定器(因果 mel・因果 YIN)実音声", la)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="artic_inv1")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--w_inv", type=float, default=2.0)
    ap.add_argument("--w_t", type=float, default=0.5)
    ap.add_argument("--w_q", type=float, default=0.01)
    ap.add_argument("--data", default=str(ROOT / "data/artic_inv"))
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    print(f"prereg: results/{a.tag}/prereg.yaml", flush=True)
    files = sorted(Path(a.data).glob("*.npz"))
    held = [f for f in files if int(f.stem) % 20 == 19]
    train = [f for f in files if int(f.stem) % 20 != 19]
    print("files", len(files), "train", len(train), "held", len(held), flush=True)
    dl = torch.utils.data.DataLoader(DS(train), batch_size=a.bs, shuffle=True, num_workers=6, drop_last=True, persistent_workers=True)
    dlh = torch.utils.data.DataLoader(DS(held[:240] if not a.smoke else held[:24]), batch_size=a.bs, num_workers=2)
    B = AF.basis(AF.N_SEC, a.k).to(dev)
    net = Inv(a.k).to(dev)
    if not a.smoke and not ship_gate(net):
        print("出荷ゲート FAIL: 起動しない", flush=True)
        return 1
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr)
    log = open(out / "train.jsonl", "a")
    r0 = {"step": 0, **evaluate(net, dlh, B, dev)}
    print("eval", json.dumps(r0), flush=True)
    log.write(json.dumps(r0) + "\n")
    step, t0 = 0, time.time()
    total = 50 if a.smoke else a.steps
    acc: dict = {}
    while step < total:
        for bt in dl:
            step += 1
            bt = {k: v.to(dev, non_blocking=True) for k, v in bt.items()}
            outs = {s: net(bt[f"mel_{s}"], bt[f"yin_{s}"]) for s in SIGS}
            phys = sum(physics_loss(outs[s][0], outs[s][1], outs[s][2], bt[f"am_{s}"], bt[f"f0_{s}"], B)[0] for s in SIGS) / 3
            v = ((bt["f0_w0"] > 0) & (bt["f0_ws"] > 0)).double()
            inv = (((outs["w0"][0] - outs["ws"][0]) ** 2).sum(-1) * v).sum() / v.sum().clamp(min=1) \
                + 20 * (((outs["w0"][2] - outs["ws"][2]) ** 2) * v).sum() / v.sum().clamp(min=1)
            qs = [outs[s][0] for s in SIGS]
            tsm = sum(((q[:, 1:] - q[:, :-1]) ** 2).sum(-1).mean() for q in qs) / 3
            lvar = sum(outs[s][2].var(1).mean() for s in SIGS) / 3
            prior = sum((q ** 2).sum(-1).mean() for q in qs) / 3
            loss = phys + a.w_inv * inv + a.w_t * tsm + 50 * lvar + a.w_q * prior
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            for k_, v_ in (("phys", phys), ("inv", inv), ("tsm", tsm), ("lvar", lvar)):
                acc.setdefault(k_, []).append(float(v_.detach()))
            if step % 200 == 0 or a.smoke:
                rr = {"step": step, "min": round((time.time() - t0) / 60, 1), **{k_: round(float(np.mean(v_)), 4) for k_, v_ in acc.items()}}
                acc = {}
                print(json.dumps(rr), flush=True)
                log.write(json.dumps(rr) + "\n")
                log.flush()
            if step % 2000 == 0 or (a.smoke and step == total):
                r = {"step": step, **evaluate(net, dlh, B, dev)}
                print("eval", json.dumps(r), flush=True)
                log.write(json.dumps(r) + "\n")
                log.flush()
                torch.save({"net": net.state_dict(), "k": a.k, "step": step}, out / "last.pt")
            if step >= total:
                break
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
