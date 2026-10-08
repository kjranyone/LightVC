"""F1 の学習(converter.md §3b・事前登録 results/<tag>/prereg.yaml が無ければ起動しない)。

データ: 男女 1:1。女声 = data/rvoc_f0 の採用分 + data/f0est の female_extra(高域の選別で落ちた分 = フルデータ)・実と TTS を 1:1・
評価話者(index の eval)の TTS クローンは除く。男声 = data/f0est の train。教師 = harvest + stonemask(f0_floor 60・フレーム t の中心 = 区間の tH)。
推定器のフレーム t は入力の (t+1)H までだけを見る。区間 = 受容野の暖機 WARM フレーム + 1.5s(損失は後ろの T フレームだけ)。
入力側だけの摂動: 利得 ±18dB・確率 0.5 で白色雑音(SNR 15〜40dB)。教師は元の音声のまま。
損失 = BCE(有声)+ CE(log f0 の 20 cent 刻み・ぼかした正解・有声フレームだけ)。
評価(5k 毎): 男声 = data/f0est の eval(VCTK 評価男声 24 人)・女声 = held21(参照 f0 は教師と同じ作り方 = 元の標本化 → 16k → harvest)・
その雑音版(SNR 20dB)。遅れ k̂(0〜4 フレーム・推定を k だけ前へずらして誤りが最小の k)を推定し、誤りを「そのまま」と「k̂ で補正」の 2 通り、
有声の誤りは参照の切り替わり ±2 フレームを除いた版も。生の因果 YIN を同じ物差しで並べる。+ 出力部 A(diag_rvoc_r の EMA 20k・相対比較の代理)の写し合成 PESQ。

    CUDA_VISIBLE_DEVICES=0 uv run python train_f0est.py --tag f0est1
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
import f0est as FE
import nvoc as N
import train_rvoc as TR

ROOT = Path(__file__).resolve().parent.parent
SEG = 72000
T = SEG // N.HOP
WARM = 64
TT = WARM + T


def eval_real_ids() -> set:
    from train_ddsp_vc import index
    _, _, ev = index()
    return {k.split("/")[1] for k in ev}


def female_rows() -> tuple[list, list]:
    ev = eval_real_ids()
    hx, hs = TR.hi_eval_paths(), TR.held_speakers()
    rv = json.loads((ROOT / "data/rvoc_f0hi/manifest.json").read_text())["rows"]
    fx = json.loads((ROOT / "data/f0est_hi/manifest.json").read_text())["rows"]
    real, tts = [], []
    for r in rv:
        if r["keep"] and not TR.eval_excluded(r, hx, hs):
            (real if r["src"] == "real_female" else tts).append((r["wav"], str(ROOT / r["f0"]), r["sr"], r["dur"], r["spk"]))
    for r in fx:
        if r["ok"] and r["src"].startswith("female_extra") and not TR.eval_excluded(r, hx, hs):
            (real if r["src"].endswith("real") else tts).append((r["wav"], str(ROOT / r["f0"]), r["sr"], r["dur"], r["spk"]))
    tts = [t for t in tts if t[4] not in ev]
    return [t[:4] for t in real], [t[:4] for t in tts]


def hi_rows() -> list:
    """1kHz 超のフレームを含む女声の行(rvoc_f0hi と f0est_hi の台帳・評価話者の実音声と評価区間を除く)。"""
    hx, hs = TR.hi_eval_paths(), TR.held_speakers()
    out = []
    for name, key in (("data/rvoc_f0hi/manifest.json", "keep"), ("data/f0est_hi/manifest.json", "ok")):
        for r in json.loads((ROOT / name).read_text())["rows"]:
            if r.get(key) and r.get("hi_frames", 0) > 0 and (key == "keep" or r["src"].startswith("female_extra")) and not TR.eval_excluded(r, hx, hs):
                out.append((r["wav"], str(ROOT / r["f0"]), r["sr"], r["dur"]))
    return out


def male_rows(split: str) -> list:
    m = json.loads((ROOT / "data/f0est_hi/manifest.json").read_text())
    return [(r["wav"], str(ROOT / r["f0"]), r["sr"], r["dur"]) for r in m["rows"] if r["ok"] and r["split"] == split and r["src"] in ("vctk_m", "tts_m")]


def perturb(x: np.ndarray, rng: random.Random) -> np.ndarray:
    y = x * 10 ** (rng.uniform(-18, 18) / 20)
    if rng.random() < 0.5:
        p = float(np.sqrt((y ** 2).mean()))
        snr = rng.uniform(15, 40)
        y = y + np.random.default_rng(rng.randrange(1 << 30)).standard_normal(len(y)).astype(np.float32) * p * 10 ** (-snr / 20)
    pk = float(np.abs(y).max())
    return (y * (0.99 / pk) if pk > 0.99 else y).astype(np.float32)


class DS(torch.utils.data.IterableDataset):
    def __init__(self, male: list, real: list, tts: list, seed: int, warm: int = WARM, hi: list | None = None, p_hi: float = 0.0):
        self.male, self.real, self.tts, self.seed, self.warm = male, real, tts, seed, warm
        self.hi, self.p_hi = hi or [], p_hi

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 7919 + (wi.id if wi else 0))
        pre = FE.CTX48 + self.warm * N.HOP
        while True:
            u = rng.random()
            use_hi = bool(self.hi) and rng.random() < self.p_hi
            g = self.hi if use_hi else (self.male if u < 0.5 else (self.real if u < 0.75 else self.tts))
            wav, f0p, sr, dur = g[rng.randrange(len(g))]
            try:
                n48 = int(dur * N.SR)
                if n48 < SEG + pre + 2 * N.HOP:
                    continue
                if use_hi:
                    idx = np.where(np.asarray(np.load(f0p, mmap_mode="r")) >= 950.0)[0]
                    if len(idx) == 0:
                        continue
                    c = int(idx[rng.randrange(len(idx))])
                    s = (c - rng.randrange(int(0.1 * T), int(0.9 * T))) * N.HOP
                    if not (pre + N.HOP <= s < n48 - SEG - N.HOP):
                        continue
                else:
                    s = rng.randrange(pre + N.HOP, n48 - SEG - N.HOP) // N.HOP * N.HOP
                x = TR.read_span(wav, sr, s - pre, pre + SEG)
                if len(x) < pre + SEG or np.sqrt((x[pre:] ** 2).mean()) < 1e-4:
                    continue
                f0 = np.load(f0p, mmap_mode="r")[s // N.HOP:s // N.HOP + T].astype(np.float32)
                yield torch.from_numpy(perturb(x, rng)), torch.from_numpy(np.pad(f0, (0, T - len(f0))))
            except Exception:
                continue


def harvest_native(wav: str, n: int, x48: np.ndarray | None = None) -> np.ndarray:
    """教師と同じ作り方(元の標本化 → 16kHz の harvest + stonemask 上限 1000Hz)。x48 を渡せば 1kHz 超の f0hi の置換(f0hi.teacher_f0)を重ねる。"""
    import pyworld
    import soundfile as sf
    from scipy.signal import resample_poly
    x, sr = sf.read(wav, dtype="float64", always_2d=True)
    x = x.mean(1)
    g = math.gcd(sr, 16000)
    x16 = resample_poly(x, 16000 // g, sr // g)
    f0, t = pyworld.harvest(x16, 16000, f0_floor=60, f0_ceil=1000, frame_period=5.0)
    f0 = pyworld.stonemask(x16, f0, t, 16000)[:n]
    f0 = np.pad(f0, (0, n - len(f0))).astype(np.float32)
    if x48 is not None:
        import f0hi as H
        f0, _ = H.teacher_f0(x48, n, f0)
    return f0


def held_female() -> list:
    """held21(eval_nvoc.held_items と同じ発話・先頭 8s)。参照 f0 は教師と同じ作り方。描画用に train_rvoc.held の項目も持つ。"""
    import s0_artic as S
    from train_ddsp_vc import index, load48
    spk, _, ev = index()
    rend = {it["stem"] if "stem" in it else None: it for it in []}
    out = []
    items = TR.held("cpu")
    k = 0
    for it in S.held24():
        key = next((kk for kk in ev if kk.endswith("/" + it["spk"])), None)
        if key is None:
            continue
        z, w, _ = next(r for r in spk[key] if r[0].stem == it["stem"])
        x = load48(w)[:8 * N.SR]
        n = len(x) // N.HOP
        x = x[:n * N.HOP].astype(np.float32)
        ri = items[k]
        assert len(ri["x"]) == len(x) and float(np.abs(ri["x"] - x).max()) == 0.0, "held の並びが eval_nvoc と食い違う"
        out.append({"x": x, "f0": harvest_native(str(w), n, x), "it": ri, "stem": it["stem"]})
        k += 1
    return out


def synth_set() -> list:
    """独立の参照を持つ合成の評価(教師に依らない・循環の回避): 高音 1000〜2000Hz と普通の声 150〜900Hz の調波音(傾き k^-2・振動 2%・SNR 25/12dB)。参照 = 振動つきの f0 そのもの。"""
    import f0hi as H
    items = []
    k = 0
    for f0 in (1000, 1200, 1500, 1800, 2000, 150, 250, 350, 500, 700, 900):
        for snr in (25, 12):
            x = H._tone(f0, 1.6, tilt=2.0, snr_db=snr, seed=k, vib=0.02)
            n = len(x) // N.HOP
            ref = (f0 * (1 + 0.02 * np.sin(2 * np.pi * 5.5 * np.arange(n) * N.HOP / N.SR))).astype(np.float32)
            items.append({"x": x[:n * N.HOP], "f0": ref, "it": None, "stem": f"synth_{f0}_{snr}", "yin": None})
            k += 1
    return items


def eval_sets(n_male: int = 120) -> dict:
    from train_ddsp_vc import load48
    rows = random.Random(0).sample(male_rows("eval"), min(n_male, len(male_rows("eval"))))
    male = []
    for wav, f0p, sr, dur in rows:
        x = load48(wav).astype(np.float32)
        n = len(x) // N.HOP
        f0 = np.load(f0p)[:n].astype(np.float32)
        male.append({"x": x[:n * N.HOP], "f0": np.pad(f0, (0, n - len(f0)))})
    fem = held_female()
    rng = np.random.default_rng(1)
    noisy = lambda items: [{**it, "x": (it["x"] + rng.standard_normal(len(it["x"])).astype(np.float32) * float(np.sqrt((it["x"] ** 2).mean())) * 0.1).astype(np.float32), "yin": None} for it in items]
    sets = {"male": male, "female": fem, "male_n20": noisy(male), "female_n20": noisy(fem)}
    sets["synth"] = synth_set()
    hi_idx = ROOT / "data/hi_eval/index.json"
    if hi_idx.exists():
        hi = []
        for r in json.loads(hi_idx.read_text())["items"]:
            z = np.load(ROOT / "data/hi_eval" / f"{r['name']}.npz")
            hi.append({"x": z["x"].astype(np.float32), "f0": z["f0"].astype(np.float32), "it": None, "stem": r["name"], "yin": None})
        if hi:
            sets["hi"] = hi
    return sets


@torch.no_grad()
def infer(front: FE.Front, net: FE.F0Est, x: np.ndarray, dev: str) -> np.ndarray:
    n = len(x) // N.HOP
    xx = torch.from_numpy(np.concatenate([np.zeros(FE.CTX48, np.float32), x[:n * N.HOP]]))[None].to(dev)
    v, p = net(front(xx, n))
    return FE.decode(v, p, 0.5)[0].cpu().numpy()


def f0_metrics(est: np.ndarray, ref: np.ndarray) -> dict:
    ve, vr = est > 0, ref > 0
    both = ve & vr
    c = np.abs(1200 * np.log2(np.where(both, est, 1) / np.where(both, ref, 1)))[both]
    tr = np.zeros(len(ref), bool)
    ch = np.nonzero(np.diff(vr.astype(int)) != 0)[0]
    for i in ch:
        tr[max(0, i - 1):i + 3] = True
    keep = ~tr
    return {"oct_err": float((c > 600).mean()) if len(c) else float("nan"),
            "gross50": float((c > 50).mean()) if len(c) else float("nan"),
            "med_cent": float(np.median(c[c <= 600])) if (c <= 600).any() else float("nan"),
            "miss": float((vr & ~ve).sum() / max(1, vr.sum())), "false": float((~vr & ve).sum() / max(1, (~vr).sum())),
            "miss_xtr": float((vr & ~ve & keep).sum() / max(1, (vr & keep).sum())), "false_xtr": float((~vr & ve & keep).sum() / max(1, (~vr & keep).sum())),
            "hi_recall": float((((np.abs(1200 * np.log2(np.where(ve & (ref >= 950), est, 1) / np.where(ref >= 950, ref, 1))) < 100) & ve & (ref >= 950)).sum()) / (ref >= 950).sum()) if (ref >= 950).any() else float("nan"),
            "false_hi": float((ve & (est >= 950) & (ref < 800)).sum() / max(1, (ref < 800).sum()))}


def pyin_ref(x: np.ndarray, n: int) -> np.ndarray:
    import librosa
    from scipy.signal import resample_poly
    f, vf, _ = librosa.pyin(resample_poly(x.astype(np.float64), 1, 3), fmin=50, fmax=2400, sr=16000, frame_length=1024, hop_length=80, center=True)
    f = np.nan_to_num(np.where(vf, f, 0.0))[:n]
    return np.pad(f, (0, n - len(f)))


def agreed_metrics(est: np.ndarray, ref: np.ndarray, py: np.ndarray) -> dict:
    """pYIN と harvest が一致するフレーム(割れていない所)だけで数えた誤り。有声: 両者の有声判定が一致するフレーム・オクターブ: 両者が有声で ±0.25 オクターブ以内。"""
    m = min(len(est), len(ref), len(py))
    est, ref, py = est[:m], ref[:m], py[:m]
    vr, vp, ve = ref > 0, py > 0, est > 0
    vag = vr == vp
    both = vr & vp & ve
    fag = both & (np.abs(np.log2(np.where(both, py, 1) / np.where(both, ref, 1))) < 0.25)
    c = np.abs(1200 * np.log2(np.where(fag, est, 1) / np.where(fag, ref, 1)))[fag]
    return {"oct_err": float((c > 600).mean()) if len(c) else float("nan"),
            "miss": float((vag & vr & ~ve).sum() / max(1, (vag & vr).sum())), "false": float((vag & ~vr & ve).sum() / max(1, (vag & ~vr).sum())),
            "agreed_voicing_frac": float(vag.mean())}


def lag_comp(est: np.ndarray, k: int) -> np.ndarray:
    return np.concatenate([est[k:], np.zeros(k, est.dtype)]) if k else est


def score(pairs: list) -> dict:
    """pairs = [(est, ref)] → そのまま・遅れ k̂ で補正の両方。k̂ = 0〜4 で gross50 + oct_err + miss + false の平均が最小。"""
    def agg(k):
        rows = [f0_metrics(lag_comp(e, k)[:len(r)], r[:len(e)]) for e, r in pairs]
        return {kk: round(float(np.nanmean([q[kk] for q in rows])), 4) for kk in rows[0]}
    cand = {k: agg(k) for k in range(5)}
    kh = min(cand, key=lambda k: cand[k]["gross50"] + cand[k]["oct_err"] + cand[k]["miss"] + cand[k]["false"])
    return {"raw": cand[0], "lag_hat": kh, "comp": cand[kh]}


def evaluate(front, net, sets: dict, dev: str, rgen=None, rfront=None) -> dict:
    import eval_nvoc as E
    net.eval()
    out = {}
    for name, items in sets.items():
        pe, py = [], []
        for it in items:
            est = infer(front, net, it["x"], dev)
            if it.get("yin") is None:
                it["yin"] = TR.yin_frames(np.concatenate([np.zeros(TR.YCTX, np.float32), it["x"]]), len(it["f0"]))
            pe.append((est, it["f0"]))
            py.append((it["yin"], it["f0"]))
        out[name] = score(pe)
        out[name + "_yin"] = score(py)
        if not name.endswith("_n20"):
            ag, agy = [], []
            for (est, ref), it in zip(pe, items):
                if it.get("pyin") is None:
                    it["pyin"] = pyin_ref(it["x"], len(it["f0"]))
                ag.append(agreed_metrics(lag_comp(est, out[name]["lag_hat"]), ref, it["pyin"]))
                agy.append(agreed_metrics(it["yin"], ref, it["pyin"]))
            out[name + "_agreed"] = {k: round(float(np.nanmean([q[k] for q in ag])), 4) for k in ag[0]}
            out[name + "_yin_agreed"] = {k: round(float(np.nanmean([q[k] for q in agy])), 4) for k in agy[0]}
    if rgen is not None:
        for comp in (False, True):
            pq = []
            with torch.no_grad():
                for it in sets["female"]:
                    est = infer(front, net, it["x"], dev)
                    if comp:
                        est = lag_comp(est, out["female"]["lag_hat"])
                    f0 = np.pad(est, (0, max(0, len(it["f0"]) - len(est))))[:len(it["f0"])]
                    y = TR.render(rgen, rfront, it["it"], f0.astype(np.float32), dev)
                    pq.append(E.metrics(y, it["it"]["x"], N.DELAY, dev)["pesq"])
            out["female_copysynth_pesq" + ("_comp" if comp else "")] = round(float(np.nanmean(pq)), 4)
    net.train()
    return out


def ship_gate(front, net) -> bool:
    import copy
    import ship_check as SC
    fr, g = copy.deepcopy(front).cpu(), copy.deepcopy(net).cpu().eval()

    def fn(x: torch.Tensor) -> torch.Tensor:
        n = len(x) // N.HOP
        xx = torch.cat([torch.zeros(FE.CTX48), x[:n * N.HOP].float()])[None]
        with torch.no_grad():
            v, p = g(fr(xx, n))
            f0 = FE.decode(v, p)
        return torch.stack([f0[0], torch.sigmoid(v[0])])

    la = SC.future_invariance(fn, hop=N.HOP, n=2 * N.SR, sr=N.SR, quantity=False, n_edit=150, male=2)
    return SC.ledger([("F1 f0・有声推定器(CMNDF 2 窓・mel・ネット・argmax と閾値)実音声", la), ("出力部 DELAY 240", 5.0), ("ブロック HOP 240", 5.0)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="f0est1")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--p_hi", type=float, default=0.0, help="1kHz 超のフレームを含む女声の区間をこの確率で引く(評価話者の実音声を除く)")
    ap.add_argument("--long_rf", action="store_true", help="受容野 1.27s(dil 1〜32 × 2)= オクターブの判断に長い文脈")
    a = ap.parse_args()
    dev = "cuda"
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    if not a.smoke and not (out / "prereg.yaml").exists():
        print(f"results/{a.tag}/prereg.yaml が無い: 起動しない", flush=True)
        return 1
    print(f"prereg: results/{a.tag}/prereg.yaml", flush=True)
    real, tts = female_rows()
    male = male_rows("train")
    print("files female real", len(real), "tts", len(tts), "male", len(male), f"({sum(r[3] for r in male) / 3600:.1f} h)", flush=True)
    m = json.loads((ROOT / "data/f0est_hi/manifest.json").read_text())
    (out / "eval_speakers.json").write_text(json.dumps({"eval_male": m["eval_male_speakers"], "train_male": m["train_male_speakers"],
                                                        "excluded_tts_clone_ids": sorted(eval_real_ids())}, indent=1))
    front = FE.Front().to(dev)
    net = (FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)) if a.long_rf else FE.F0Est()).to(dev)
    global WARM, TT
    rf = 2 + sum(2 * d for d in net.cfg["dils"])
    WARM = max(WARM, rf + 2)
    TT = WARM + T
    print("params (M)", round(sum(p.numel() for p in net.parameters()) / 1e6, 3), flush=True)
    if not a.smoke and not ship_gate(front, net):
        print("出荷ゲート FAIL: 起動しない", flush=True)
        return 1
    import rvoc as R
    st = torch.load(ROOT / "results/rvoc2am2/snap/ema_300k.pt", map_location="cpu", weights_only=False)
    rgen = R.RVoc(ch=st["cfg"]["ch"], d_cond=st["cfg"]["d_cond"]).to(dev)
    rgen.load_state_dict(st["ema"])
    rgen.eval()
    rfront = TR.Front("pae", st["env_smooth"]).to(dev)
    sets = eval_sets(12 if a.smoke else 120)
    if a.smoke:
        sets = {k: v[:3] for k, v in sets.items()}
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    total = 30 if a.smoke else a.steps
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)
    loader = iter(torch.utils.data.DataLoader(DS(male, real, tts, 11, WARM, hi_rows() if a.p_hi > 0 else None, a.p_hi), batch_size=a.bs, num_workers=a.workers, pin_memory=True,
                                              persistent_workers=True, prefetch_factor=4))
    log = open(out / "train.jsonl", "a")
    r0 = {"step": 0, **evaluate(front, net, sets, dev, rgen, rfront)}
    print("eval", json.dumps(r0), flush=True)
    log.write(json.dumps(r0) + "\n")
    acc: dict = {}
    t0 = time.time()
    for step in range(1, total + 1):
        x, f0 = (t.to(dev, non_blocking=True) for t in next(loader))
        with torch.no_grad():
            feat = front(x, TT)
            tb = FE.target_bins(f0)
        v, p = net(feat)
        v, p = v[:, WARM:], p[..., WARM:]
        vt = (f0 > 0).float()
        lv = F.binary_cross_entropy_with_logits(v, vt)
        lp = -(tb * F.log_softmax(p, 1)).sum(1)
        lp = (lp * vt).sum() / vt.sum().clamp(min=1)
        if step == 1:
            ent = -(tb * torch.log(tb.clamp(min=1e-12))).sum(1)
            print(f"損失の床の照合: BCE 初期 {float(lv):.3f}(未学習 ≈ 0.693)・CE 初期 {float(lp):.3f}(一様 {math.log(FE.N_BIN):.3f})・"
                  f"ぼかした正解のエントロピー = CE の床 {float((ent * vt).sum() / vt.sum().clamp(min=1)):.3f}", flush=True)
        loss = lv + lp
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        sch.step()
        acc.setdefault("lv", []).append(float(lv))
        acc.setdefault("lp", []).append(float(lp))
        if step % 200 == 0 or a.smoke:
            rr = {"step": step, "min": round((time.time() - t0) / 60, 1), **{k: round(float(np.mean(v_)), 4) for k, v_ in acc.items()}}
            acc = {}
            print(json.dumps(rr), flush=True)
            log.write(json.dumps(rr) + "\n")
            log.flush()
        if step % a.eval_every == 0 or (a.smoke and step == total):
            r = {"step": step, **evaluate(front, net, sets, dev, rgen, rfront)}
            print("eval", json.dumps(r), flush=True)
            log.write(json.dumps(r) + "\n")
            log.flush()
            torch.save({"net": net.state_dict(), "cfg": net.cfg, "step": step}, out / "last.pt")
            if not a.smoke and step == 5000:
                worse = all(r[s]["raw"][k] >= r[s + "_yin"]["raw"][k] for s in ("male", "female") for k in ("oct_err", "gross50", "miss", "false"))
                if worse:
                    (out / "STOP.json").write_text(json.dumps({"step": step, "why": "5k で全ての誤りが生の YIN より悪い = 実装の誤り"}, ensure_ascii=False))
                    print("中止条件: 5k で全ての誤りが生の YIN より悪い", flush=True)
                    return 2
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
