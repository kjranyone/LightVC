"""C1 の学習(converter.md §3 C1・事前登録 results/<tag>/prereg.yaml が無ければ起動しない)。

データ: 男女 1:1。男声 = data/f0est の train(VCTK 評価外の男声 23 人の共通文以外 + その JA TTS クローン)。女声 = 実(data/rvoc_f0 + f0est の female_extra)・
TTS(評価話者のクローンを除く)・VCTK の評価外の女声(共通文以外)を 4:4:2。評価に使う VCTK 話者(差し替えラダーの男 24・女 45)とそのクローンは入れない。
教師: ContentVec(学習時だけ・非因果・区間の前後に余白)→ コードブック(K 個・学習データだけから k-means・results/<tag>/codebook.pt)へ cos/τ の softmax。
生徒: 因果 log-mel(入力側だけ声道長の伸縮 α ∈ [0.8, 1.25]・利得 ±18dB・確率 0.5 で白色雑音 SNR 15〜40dB)→ c1_content.C1。
損失 = 柔らかい正解の交差エントロピー(区間の後ろ T フレーム・前は受容野の暖機)。
切り出し: 30% はファイルの先頭から・残りは任意の位置。ファイルの外は無音(0)で埋める(短いファイルも使う = フルデータ)。
開始状態の規約: 受容野ぶん以上の無音(PRE)を先に流す。学習(ファイル先頭の切り出し)・評価・出荷ゲート・製品(Rust の初期化)で同じ。
評価(5k 毎): 教師の単位(同じ 15ms の遅れ)と生徒の最大の単位の一致率(全フレーム・音声区間だけ)を、日本語女声(held21)・英語女声(VCTK 評価女声の
共通文 7〜8)・英語男声(VCTK 評価男声)で。同じ物差しで E2(同じコードブック・全体で 1 つの遅れ −10〜+25ms の最良)を並べる。

    CUDA_VISIBLE_DEVICES=0 uv run python train_c1.py --tag c1_1
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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import c1_content as C
import nvoc as N
import train_f0est as TF
import train_rvoc as TR

ROOT = Path(__file__).resolve().parent.parent
VC = ROOT / "data/vctk/VCTK-Corpus/VCTK-Corpus"
LADDER = ROOT / "results/conv_c0/ladder_jobs.json"
SEG = 72000
T = SEG // N.HOP
CTXM = N.WIN - N.HOP
PRE = CTXM + (C.RF + 2) * N.HOP + 24000
POST = 24000
TALL = (PRE + SEG - CTXM) // N.HOP


def eval_vctk() -> tuple[set, set]:
    J = json.loads(LADDER.read_text())
    return set(J["fems"]), set(J["males"])


def vctk_female_rows() -> list:
    ev_f, ev_m = eval_vctk()
    info = [l.split() for l in (VC / "speaker-info.txt").read_text().splitlines()[1:] if l.strip()]
    fem = sorted(f"p{r[0]}" for r in info if len(r) > 2 and r[2] == "F" and f"p{r[0]}" not in ev_f)
    rows = []
    for s in fem:
        for w in sorted((VC / "wav48" / s).glob("*.wav")):
            if int(w.stem.split("_")[1]) > 24:
                rows.append((str(w), "", 48000, None))
    return rows


def perturb(x: np.ndarray, rng: random.Random) -> np.ndarray:
    return TF.perturb(x, rng)


def read_pad(wav: str, sr: int, n48_file: int, a48: int, n48: int) -> np.ndarray:
    """48kHz の区間 [a48, a48 + n48) を読み、ファイルの外は 0(無音)で埋める。"""
    lo, hi = max(0, a48), min(n48_file, a48 + n48)
    out = np.zeros(n48, np.float32)
    if hi > lo:
        y = TR.read_span(wav, sr, lo, hi - lo)
        out[lo - a48:lo - a48 + len(y)] = y
    return out


class DS(torch.utils.data.IterableDataset):
    def __init__(self, groups: list, probs: list, seed: int):
        self.groups, self.cum, self.seed = groups, np.cumsum(probs) / np.sum(probs), seed

    def __iter__(self):
        import soundfile as sf
        from scipy.signal import resample_poly
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 6151 + (wi.id if wi else 0))
        while True:
            g = self.groups[int(np.searchsorted(self.cum, rng.random()))]
            wav, _, sr, dur = g[rng.randrange(len(g))]
            try:
                if dur is None:
                    info = sf.info(wav)
                    sr, dur = int(info.samplerate), float(info.duration)
                n48 = int(dur * N.SR)
                if n48 < 2 * N.HOP:
                    continue
                s = 0 if rng.random() < 0.3 else rng.randrange(0, n48 - N.HOP) // N.HOP * N.HOP
                x = read_pad(wav, sr, n48, s - PRE, PRE + SEG + POST)
                if np.sqrt((x[PRE:PRE + SEG] ** 2).mean()) < 1e-4:
                    continue
                x16 = resample_poly(x.astype(np.float64), 1, 3).astype(np.float32)
                xs = perturb(x[:PRE + SEG], rng)
                alpha = math.exp(rng.uniform(math.log(0.8), math.log(1.25)))
                yield torch.from_numpy(xs), torch.from_numpy(x16), torch.tensor(alpha, dtype=torch.float32)
            except Exception:
                continue


class Teacher:
    def __init__(self, dev: str, C_: torch.Tensor, tau: float):
        from transformers import HubertModel
        self.m = HubertModel.from_pretrained("lengyue233/content-vec-best").to(dev).eval().half()
        self.C, self.tau = C_.to(dev), tau

    @torch.no_grad()
    def feats(self, x16: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.m(x16.half()).last_hidden_state.float(), dim=-1)

    @torch.no_grad()
    def post(self, x16: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.feats(x16) @ self.C.T / self.tau, -1)


def build_codebook(teacher: Teacher, groups: list, probs: list, k: int, dev: str, n_seg: int = 3000) -> torch.Tensor:
    import artic_g2_unit as U
    from scipy.signal import resample_poly
    rng = random.Random(5)
    cum = np.cumsum(probs) / np.sum(probs)
    pool = []
    for _ in range(n_seg):
        g = groups[int(np.searchsorted(cum, rng.random()))]
        wav, _, sr, dur = g[rng.randrange(len(g))]
        try:
            import soundfile as sf
            if dur is None:
                info = sf.info(wav)
                sr, dur = int(info.samplerate), float(info.duration)
            n48 = int(dur * N.SR)
            if n48 < 2 * N.SR:
                continue
            s = rng.randrange(0, n48 - 2 * N.SR) // 3 * 3
            x = TR.read_span(wav, sr, s, 2 * N.SR)
            x16 = torch.from_numpy(resample_poly(x.astype(np.float64), 1, 3).astype(np.float32))[None].to(dev)
            pool.append(teacher.feats(x16)[0].cpu().numpy())
        except Exception:
            continue
    return torch.from_numpy(U.kmeans(np.concatenate(pool), k, dev))


def eval_items() -> dict:
    """女声 = held21(日本語)と VCTK 評価女声の共通文 7〜24(英語・各 2 発話)、男声 = VCTK 評価男声(英語・100 発話)。性別と言語を分けて読む。"""
    import eval_nvoc as E
    from train_ddsp_vc import load48
    fem = [{"x": it["x"].astype(np.float32)} for it in E.held_items()]
    ev_f, _ = eval_vctk()
    vf = []
    for s_ in sorted(ev_f):
        for u in (7, 8):
            w = VC / "wav48" / s_ / f"{s_}_{u:03d}.wav"
            if w.exists():
                x = load48(str(w)).astype(np.float32)
                vf.append({"x": x[:len(x) // N.HOP * N.HOP]})
    rows = random.Random(0).sample(TF.male_rows("eval"), 100)
    male = []
    for wav, _, _, _ in rows:
        x = load48(wav).astype(np.float32)
        male.append({"x": x[:len(x) // N.HOP * N.HOP]})
    return {"female_ja_held21": fem, "female_en_vctk": vf, "male_en_vctk": male}


def prime(x: np.ndarray) -> np.ndarray:
    """開始状態の規約: 受容野ぶん以上の無音(PRE)を先に流す(学習のファイル先頭の切り出し・評価・製品で同じ)。"""
    return np.concatenate([np.zeros(PRE, np.float32), x])


@torch.no_grad()
def student_logits(front, net, x: np.ndarray, dev: str) -> torch.Tensor:
    n = len(x) // N.HOP
    return net(front(torch.from_numpy(prime(x))[None].to(dev)))[..., -n:]


@torch.no_grad()
def teacher_post(teacher: Teacher, x: np.ndarray, dev: str) -> torch.Tensor:
    from scipy.signal import resample_poly
    n = len(x) // N.HOP
    xp = np.concatenate([prime(x), np.zeros(POST, np.float32)])
    x16 = torch.from_numpy(resample_poly(xp.astype(np.float64), 1, 3).astype(np.float32))[None].to(dev)
    tall = (PRE + len(x) - CTXM) // N.HOP
    return C.teacher_targets(teacher.post(x16), pre48=CTXM, T=tall)[..., -n:]


def speech_mask(x: np.ndarray) -> np.ndarray:
    n = len(x) // N.HOP
    xp = np.concatenate([np.zeros(N.WIN - N.HOP, np.float32), x])
    e = np.array([float((xp[t * N.HOP:t * N.HOP + N.WIN] ** 2).mean()) for t in range(n)]) + 1e-12
    return 10 * np.log10(e) > 10 * np.log10(e.max()) - 40


def evaluate(front, net, teacher, items: dict, dev: str, e2=None) -> dict:
    net.eval()
    out = {}
    for name, its in items.items():
        a_s, a_sv, e2c = [], [], {}
        for it in its:
            n = len(it["x"]) // N.HOP
            if "tu" not in it:
                it["tu"] = teacher_post(teacher, it["x"], dev).argmax(1)[0].cpu().numpy()
                it["sp"] = speech_mask(it["x"])
            su = student_logits(front, net, it["x"], dev).argmax(1)[0].cpu().numpy()
            a_s.append(float((su == it["tu"]).mean()))
            a_sv.append(float((su == it["tu"])[it["sp"]].mean()))
            if e2 is not None:
                if "e2u" not in it:
                    h = e2(it["x"].astype(np.float64))
                    u = (torch.from_numpy(h).to(dev) @ teacher.C.T).argmax(1).cpu().numpy()
                    t = np.arange(n) * N.HOP / N.SR
                    it["e2u"] = {sh: u[np.clip(np.floor((t + sh * 0.005) * 44100 / 256).astype(int), 0, len(u) - 1)] for sh in range(-2, 6)}
                for sh, uu in it["e2u"].items():
                    e2c.setdefault(sh, []).append(float((uu == it["tu"]).mean()))
        r = {"c1_agree": round(float(np.mean(a_s)), 4), "c1_agree_speech": round(float(np.mean(a_sv)), 4)}
        if e2c:
            best = max(e2c, key=lambda k: np.mean(e2c[k]))
            r.update(e2_agree_global_lag=round(float(np.mean(e2c[best])), 4), e2_lag_ms=best * 5)
        out[name] = r
    net.train()
    return out


def ship_gate(front, net) -> bool:
    import copy
    import ship_check as SC
    fr, g = copy.deepcopy(front).cpu(), copy.deepcopy(net).cpu().eval()

    def fn(x: torch.Tensor) -> torch.Tensor:
        n = len(x) // N.HOP
        xx = torch.cat([torch.zeros(PRE), x[:n * N.HOP].float()])[None]
        with torch.no_grad():
            lg = g(fr(xx))[..., -n:]
        return torch.stack([lg.argmax(1)[0].float(), lg.softmax(1).amax(1)[0]])

    la = SC.future_invariance(fn, hop=N.HOP, n=2 * N.SR, sr=N.SR, quantity=False, n_edit=150, male=2)
    return SC.ledger([("C1 内容符号器(因果 mel・ネット・argmax)実音声", la), ("出力部 DELAY 240", 5.0), ("ブロック HOP 240", 5.0)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="c1_1")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--tau", type=float, default=0.05)
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
    vf = vctk_female_rows()
    groups, probs = [male, real, tts, vf], [0.5, 0.2, 0.2, 0.1]
    print("files male", len(male), "female real", len(real), "tts", len(tts), "vctk female", len(vf), flush=True)
    front = C.MelFront().to(dev)
    tmp = Teacher(dev, torch.zeros(a.k, 768), a.tau)
    cbp = out / "codebook.pt"
    if cbp.exists():
        Cb = torch.load(cbp)
    else:
        Cb = build_codebook(tmp, groups, probs, a.k, dev, 200 if a.smoke else 3000)
        torch.save(Cb, cbp)
    teacher = Teacher.__new__(Teacher)
    teacher.m, teacher.C, teacher.tau = tmp.m, Cb.to(dev), a.tau
    net = C.C1(a.k).to(dev)
    print("params (M)", round(sum(p.numel() for p in net.parameters()) / 1e6, 3), "RF", C.RF, flush=True)
    if not a.smoke and not ship_gate(front, net):
        print("出荷ゲート FAIL: 起動しない", flush=True)
        return 1
    import artic_g2_unit as U
    e2 = U.E2(dev)
    items = eval_items()
    if a.smoke:
        items = {k: v[:3] for k, v in items.items()}
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    total = 30 if a.smoke else a.steps
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)
    loader = iter(torch.utils.data.DataLoader(DS(groups, probs, 13), batch_size=a.bs, num_workers=a.workers, pin_memory=True,
                                              persistent_workers=True, prefetch_factor=4))
    log = open(out / "train.jsonl", "a")
    r0 = {"step": 0, **evaluate(front, net, teacher, items, dev, e2)}
    print("eval", json.dumps(r0), flush=True)
    log.write(json.dumps(r0) + "\n")
    acc: dict = {}
    t0 = time.time()
    for step in range(1, total + 1):
        xs, x16, alpha = (t.to(dev, non_blocking=True) for t in next(loader))
        with torch.no_grad():
            mel = front(xs, alpha)
            tg = C.teacher_targets(teacher.post(x16), pre48=CTXM, T=mel.shape[-1])[..., -T:]
        lg = net(mel)[..., -T:]
        loss = -(tg * F.log_softmax(lg, 1)).sum(1).mean()
        if step == 1:
            ent = float(-(tg * torch.log(tg.clamp(min=1e-12))).sum(1).mean())
            print(f"損失の床の照合: 初期 {float(loss):.3f}(一様 {math.log(a.k):.3f})・教師の事後のエントロピー = 床 {ent:.3f}・"
                  f"教師の最大確率の平均 {float(tg.amax(1).mean()):.3f}", flush=True)
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
            r = {"step": step, **evaluate(front, net, teacher, items, dev, e2)}
            print("eval", json.dumps(r), flush=True)
            log.write(json.dumps(r) + "\n")
            log.flush()
            torch.save({"net": net.state_dict(), "cfg": net.cfg, "step": step, "tau": a.tau}, out / "last.pt")
            sets = [k for k in r if k != "step"]
            why = None
            if not a.smoke and step == 5000 and all(r[k]["c1_agree"] < r[k]["e2_agree_global_lag"] - 0.05 for k in sets):
                why = "5k で全ての評価の一致率が E2 − 0.05 未満"
            if not a.smoke and step == 10000 and all(r[k]["c1_agree"] <= r[k]["e2_agree_global_lag"] for k in sets):
                why = "10k で全ての評価の一致率が E2 以下"
            if why:
                (out / "STOP.json").write_text(json.dumps({"step": step, "why": why}, ensure_ascii=False))
                print("中止条件:", why, flush=True)
                return 2
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
