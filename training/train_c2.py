"""C2 の学習: 因果な連続の内容の特徴(diag_ppgvoc49 の上限 PASS を製品の経路へ・事前登録 results/<tag>/prereg.yaml が無ければ起動しない)。

骨格・データ・入力の摂動・切り出し・開始状態の規約・15 ms の遅れは C1(train_c1.py)と同じ。違いは正解だけ:
教師 = ContentVec(lengyue233/content-vec-best・最終層・L2 正規化・非因果・学習時だけ)を D502 の主成分 64(分散で白色化・results/conv_p0/cv_pca64.npz)へ射影した連続値。
損失 = 白色化した主成分の二乗誤差(区間の後ろ T フレーム)。評価(5k 毎)= 教師との決定係数 R² と余弦(音声区間・日本語女声 held21 / 英語女声 / 英語男声)。

    uv run python train_c2.py --tag c2_1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import c1_content as C
import nvoc as N
import train_c1 as TC
import train_f0est as TF

ROOT = Path(__file__).resolve().parent.parent


class FeatTeacher:
    def __init__(self, dev: str, pca: str):
        from transformers import HubertModel
        self.m = HubertModel.from_pretrained("lengyue233/content-vec-best").to(dev).eval().half()
        z = np.load(pca)
        self.mean = torch.from_numpy(z["mean"]).float().to(dev)
        self.comp = torch.from_numpy(z["comp"] / np.sqrt(z["ev"])[:, None]).float().to(dev)

    @torch.no_grad()
    def feats(self, x16: torch.Tensor) -> torch.Tensor:
        h = F.normalize(self.m(x16.half()).last_hidden_state.float(), dim=-1)
        return (h - self.mean) @ self.comp.T


@torch.no_grad()
def teacher_feat(teacher: FeatTeacher, x: np.ndarray, dev: str) -> torch.Tensor:
    from scipy.signal import resample_poly
    n = len(x) // N.HOP
    xp = np.concatenate([TC.prime(x), np.zeros(TC.POST, np.float32)])
    x16 = torch.from_numpy(resample_poly(xp.astype(np.float64), 1, 3).astype(np.float32))[None].to(dev)
    tall = (TC.PRE + len(x) - TC.CTXM) // N.HOP
    return C.teacher_targets(teacher.feats(x16), pre48=TC.CTXM, T=tall)[..., -n:]


def evaluate(front, net, teacher, items: dict, dev: str) -> dict:
    net.eval()
    out = {}
    for name, its in items.items():
        se, st, cs, ss = 0.0, 0.0, [], []
        S, T_ = [], []
        for it in its:
            if "tf" not in it:
                it["tf"] = teacher_feat(teacher, it["x"], dev)[0].cpu().numpy()
                it["sp"] = TC.speech_mask(it["x"])
            s = TC.student_logits(front, net, it["x"], dev)[0].cpu().numpy()
            m = it["sp"][: s.shape[1]]
            S.append(s[:, m]); T_.append(it["tf"][:, m])
        S, T_ = np.concatenate(S, 1), np.concatenate(T_, 1)
        r2 = 1 - ((S - T_) ** 2).sum() / ((T_ - T_.mean(1, keepdims=True)) ** 2).sum()
        cos = (S * T_).sum(0) / (np.linalg.norm(S, axis=0) * np.linalg.norm(T_, axis=0) + 1e-9)
        out[name] = {"r2": round(float(r2), 4), "cos": round(float(cos.mean()), 4)}
    net.train()
    return out


def ship_gate(front, net) -> bool:
    import copy
    import ship_check as SC
    fr, g = copy.deepcopy(front).cpu(), copy.deepcopy(net).cpu().eval()

    def fn(x: torch.Tensor) -> torch.Tensor:
        n = len(x) // N.HOP
        xx = torch.cat([torch.zeros(TC.PRE), x[:n * N.HOP].float()])[None]
        with torch.no_grad():
            return g(fr(xx))[0, :2, -n:]

    la = SC.future_invariance(fn, hop=N.HOP, n=2 * N.SR, sr=N.SR, quantity=False, n_edit=150, male=2)
    return SC.ledger([("C2 連続の内容(因果 mel・ネット)実音声", la), ("出力部 DELAY 240", 5.0), ("ブロック HOP 240", 5.0)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c2_1")
    ap.add_argument("--pca", default=str(ROOT / "results/conv_p0/cv_pca64.npz"))
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    if not a.smoke and not (out / "prereg.yaml").exists():
        print(f"results/{a.tag}/prereg.yaml が無い: 起動しない", flush=True)
        return 1
    print(f"prereg: results/{a.tag}/prereg.yaml", flush=True)
    real, tts = TF.female_rows()
    male = TF.male_rows("train")
    vf = TC.vctk_female_rows()
    groups, probs = [male, real, tts, vf], [0.5, 0.2, 0.2, 0.1]
    print("files male", len(male), "female real", len(real), "tts", len(tts), "vctk female", len(vf), flush=True)
    front = C.MelFront().to(dev)
    teacher = FeatTeacher(dev, a.pca)
    k = int(teacher.comp.shape[0])
    net = C.C1(k).to(dev)
    print("params (M)", round(sum(p.numel() for p in net.parameters()) / 1e6, 3), "RF", C.RF, "dim", k, flush=True)
    if not a.smoke and not ship_gate(front, net):
        print("出荷ゲート FAIL: 起動しない", flush=True)
        return 1
    items = TC.eval_items()
    if a.smoke:
        items = {kk: v[:3] for kk, v in items.items()}
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    total = 30 if a.smoke else a.steps
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)
    loader = iter(torch.utils.data.DataLoader(TC.DS(groups, probs, 13), batch_size=a.bs, num_workers=a.workers, pin_memory=True,
                                              persistent_workers=True, prefetch_factor=4))
    log = open(out / "train.jsonl", "a")
    r0 = {"step": 0, **evaluate(front, net, teacher, items, dev)}
    print("eval", json.dumps(r0), flush=True)
    log.write(json.dumps(r0) + "\n")
    acc: dict = {}
    t0 = time.time()
    for step in range(1, total + 1):
        xs, x16, alpha = (t.to(dev, non_blocking=True) for t in next(loader))
        with torch.no_grad():
            mel = front(xs, alpha)
            tg = C.teacher_targets(teacher.feats(x16), pre48=TC.CTXM, T=mel.shape[-1])[..., -TC.T:]
        y = net(mel)[..., -TC.T:]
        loss = F.mse_loss(y, tg)
        if step == 1:
            print(f"損失の照合: 初期 {float(loss):.3f}・正解の分散(白色化 = 約 1){float(tg.var()):.3f}", flush=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sch.step()
        acc.setdefault("loss", []).append(float(loss))
        if step % 200 == 0 or a.smoke:
            rr = {"step": step, "min": round((time.time() - t0) / 60, 1), "loss": round(float(np.mean(acc["loss"])), 4)}
            acc = {}
            print(json.dumps(rr), flush=True)
            log.write(json.dumps(rr) + "\n")
            log.flush()
        if step % a.eval_every == 0 or (a.smoke and step == total):
            r = {"step": step, **evaluate(front, net, teacher, items, dev)}
            print("eval", json.dumps(r), flush=True)
            log.write(json.dumps(r) + "\n")
            log.flush()
            torch.save({"net": net.state_dict(), "cfg": net.cfg, "step": step, "pca": a.pca}, out / "last.pt")
            if not a.smoke and step == 10000 and r["female_ja_held21"]["r2"] < 0.2:
                (out / "STOP.json").write_text(json.dumps({"step": step, "why": "10k で日本語女声の R² < 0.2"}, ensure_ascii=False))
                print("中止条件: 10k で日本語女声の R² < 0.2", flush=True)
                return 2
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
