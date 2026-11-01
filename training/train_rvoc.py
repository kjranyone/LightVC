"""出力部 rvoc の学習(current/renderer.md・事前登録 results/<tag>/prereg.yaml が無ければ起動しない): 女声フルコーパスの写し合成。

データ: data/rvoc_f0/manifest.json(prep_rvoc_f0.py・高域を持つファイルだけ)から実音声と TTS を 1:1。f0 = 前計算の harvest + stonemask
(フレーム t の中心 = 出力ブロック t の内容の終わり = パルス列の補間の端点と一致)。条件は Front(pae: CheapTrick の包絡 DCT c0..c24 + 周期性・
中心揃えの分析 = 推論の経路に無い(製品では変換器が出す)・0 = 無音)。
励起 = 同じ f0 の帯域制限パルス列 + 白色雑音。出力 y[m] ≈ x[m − DELAY]。区間 0.7s の先頭 WARM(200ms > 受容野 174ms)は損失と判別器から外す。
損失 = 45·多尺度 logmel L1 (+ step > gan_from で LSGAN + 2·FM・MPD 周期 2,3,5,7,11 + MRD-log・48kHz)。学習率は半減期 lr_half の指数減衰(総 step によらない)。
評価・耳の判定は EMA(0.999)の重み。

    CUDA_VISIBLE_DEVICES=0 uv run python train_rvoc.py --tag rvoc1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import pae as PA
import rvoc as R
from am_loss import am_penalty
from train_nvoc import MRDLog
from train_s1_1 import MEL_SPECS, build_mel, logmel_l1

ROOT = Path(__file__).resolve().parent.parent
SEG = 33600
WARM = 9600
T = SEG // N.HOP
YCTX = 2400
MANIFEST = Path(os.environ.get("RVOC_MANIFEST", ROOT / "data" / "rvoc_f0" / "manifest.json"))
STOPS = {20000: 2.0, 40000: 2.4, 100000: 2.5, 300000: 3.2}
AM_MAX = 3.0
TARGET = 3.2


def hi_eval_paths() -> set:
    """1kHz 超の評価区間(data/hi_eval・hi_eval_set.py)の元の wav。学習から除く(評価話者の本人の録音)。"""
    idx = ROOT / "data" / "hi_eval" / "index.json"
    return {r["path"] for r in json.loads(idx.read_text())["items"]} if idx.exists() else set()


def held_speakers() -> set:
    """評価話者(held・build_index(0))。この話者の実音声は学習に入れない(TTS の複製は既知の条件として残す)。"""
    from s0_artic import build_index
    return set(build_index(0)[2])


def eval_excluded(r: dict, ex_paths: set, ex_spk: set) -> bool:
    return r["wav"] in ex_paths or (r["src"].startswith("real") and r["spk"] in ex_spk)


def load_manifest() -> tuple[list, list]:
    ex, exs = hi_eval_paths(), held_speakers()
    rows = [r for r in json.loads(MANIFEST.read_text())["rows"] if r["keep"] and not eval_excluded(r, ex, exs)]
    pick = lambda g: [(r["wav"], str(ROOT / r["f0"]), r["sr"], r["dur"]) for r in rows if r["src"].startswith(g)]
    return pick("real_female"), pick("tts")


def load_manifest_spk() -> tuple[list, list]:
    """load_manifest と同じ行に話者を足す(表の包絡の学習用)。"""
    ex, exs = hi_eval_paths(), held_speakers()
    rows = [r for r in json.loads(MANIFEST.read_text())["rows"] if r["keep"] and not eval_excluded(r, ex, exs)]
    pick = lambda g: [(r["wav"], str(ROOT / r["f0"]), r["sr"], r["dur"], r["spk"]) for r in rows if r["src"].startswith(g)]
    return pick("real_female"), pick("tts")


def load_hi_rows() -> list:
    """台帳(prep_f0hi が作った manifest)で 1kHz 超のフレームを含む行(hi_frames > 0・評価話者の実音声と評価区間は除く)。"""
    ex, exs = hi_eval_paths(), held_speakers()
    rows = [r for r in json.loads(MANIFEST.read_text())["rows"] if r.get("keep") and r.get("hi_frames", 0) > 0 and not eval_excluded(r, ex, exs)]
    return [(r["wav"], str(ROOT / r["f0"]), r["sr"], r["dur"]) for r in rows]


def read_span(wav: str, sr: int, a48: int, n48: int) -> np.ndarray:
    """48kHz の区間 [a48, a48 + n48) を元の標本化から切り出して変換。ファイル全体を resample_poly したときと同じ格子(端の過渡は余白で捨てる)。"""
    import soundfile as sf
    from scipy.signal import resample_poly
    g = math.gcd(sr, N.SR)
    u, d = N.SR // g, sr // g
    b0 = max(0, (a48 - 4096) // u)
    b1 = (a48 + n48 + 4096) // u + 1
    x, _ = sf.read(wav, start=b0 * d, stop=b1 * d, dtype="float32", always_2d=True)
    x = x.mean(1)
    y = resample_poly(x, u, d).astype(np.float32) if u != d else x
    return y[a48 - b0 * u:a48 - b0 * u + n48]


def harvest_frames(x_ctx: np.ndarray, n_frames: int) -> np.ndarray:
    """学習の f0(目標の音声に揃った精密な値・オクターブ誤りなし): harvest + stonemask(16kHz・5ms)。
    出力ブロック t は x[tH − DELAY, (t+1)H − DELAY) を作り、その終わり = 区間の tH = harvest のフレーム YCTX/H + t。推論の f0 は変換器から来る(因果性は変換器の側)。"""
    import librosa
    import pyworld
    x16 = librosa.resample(x_ctx.astype(np.float64), orig_sr=N.SR, target_sr=16000)
    f0, t = pyworld.harvest(x16, 16000, f0_floor=60, f0_ceil=1000, frame_period=N.HOP / N.SR * 1000)
    f0 = pyworld.stonemask(x16, f0, t, 16000)
    k0 = YCTX // N.HOP
    f = f0[k0:k0 + n_frames]
    return np.pad(f, (0, n_frames - len(f))).astype(np.float32)


def yin_frames(x_ctx: np.ndarray, n_frames: int) -> np.ndarray:
    """x_ctx = [左文脈 YCTX | 区間] の生の因果 YIN を区間のフレームへ(フレーム t は区間の (t+1)·HOP に終わる窓)。"""
    import artic_dsp as D
    f0, _ = D.causal_yin(x_ctx.astype(np.float64), voi_max=0.45)
    k0 = YCTX // N.HOP
    f = f0[k0:k0 + n_frames]
    return np.pad(f, (0, n_frames - len(f))).astype(np.float32)


class Front(torch.nn.Module):
    """条件(200fps・31 行 / mel は 134 行)。0 = 無音(固定の定数でずらす・発話の統計なし)。
    mode = "pae": CheapTrick の mel 包絡の DCT c0..c24(c0 − √128·log 1e-5)/10・log(f0/200)・有声・4 帯の周期性(倍音の山谷比)。
    mode = "mel": 因果 log-mel 128 帯((mel − log 1e-5)/10)・同じ 6 行(診断: 条件の情報の上限・倍音の縞 = ピッチを含む)。
    入力 xa は左右に PA.A の余白・f0_ana は分析の f0(包絡と周期性)・f0_out は出す f0(log f0 と有声・既定 = f0_ana)。"""

    def __init__(self, mode: str = "pae", env_smooth: float = 0.0):
        super().__init__()
        self.mode = mode
        self.env_smooth = env_smooth
        self.mel = N.CausalMel()

    def forward(self, xa: torch.Tensor, f0_ana: torch.Tensor, env: torch.Tensor, f0_out: torch.Tensor | None = None) -> torch.Tensor:
        T = f0_ana.shape[-1]
        f = f0_ana if f0_out is None else f0_out
        lf = torch.where(f > 0, torch.log(f.clamp(min=1.0) / 200.0), torch.zeros_like(f)).unsqueeze(1)
        v = (f > 0).float().unsqueeze(1)
        per = PA.periodicity(xa, f0_ana)
        if self.mode == "mel":
            seg = xa[:, PA.A - (N.WIN - N.HOP):PA.A + T * N.HOP]
            mel = N.NVoc.mel_ctx(self, seg)[..., :T]
            return torch.cat([(mel - math.log(1e-5)) / 10, lf, v, per], 1)
        cc = env.clone()
        if self.env_smooth:
            k = torch.tensor([self.env_smooth, 1 - 2 * self.env_smooth, self.env_smooth], device=env.device, dtype=env.dtype)[None, None]
            B, D, L = cc.shape
            cc = F.conv1d(F.pad(cc.reshape(B * D, 1, L), (1, 1), mode="replicate"), k).reshape(B, D, L)
        cc[:, 0] = cc[:, 0] - PA.C0_SIL
        return torch.cat([cc / 10, lf, v, per], 1)


def channel_params(rng: random.Random) -> tuple:
    """入力の側の録音の経路の摂動の値(D400・diag_ppgvoc32): (低域通過の遮断 Hz または 0, 2 kHz の高域の棚の利得 dB)。"""
    return (rng.uniform(5000.0, 14000.0) if rng.random() < 0.5 else 0.0, rng.uniform(-9.0, 9.0))


def channel_apply(x: np.ndarray, prm: tuple) -> np.ndarray:
    """ゼロ位相(時刻をずらさない・往復で振幅特性は 2 乗 = 棚は設計の半分の dB で作る)の低域通過(8 次 Butterworth)と高域の棚(RBJ・2 kHz・S = 1)。"""
    from scipy.signal import butter, sosfiltfilt, filtfilt
    if len(x) < 64:
        return x
    y = x.astype(np.float64)
    fc, gdb = prm
    if fc > 0:
        y = sosfiltfilt(butter(8, fc, "low", fs=N.SR, output="sos"), y)
    if abs(gdb) > 0.25:
        A_ = 10 ** (gdb / 2 / 40)
        w0 = 2 * math.pi * 2000.0 / N.SR
        al = math.sin(w0) / 2 * math.sqrt(2)
        cw = math.cos(w0)
        b = [A_ * ((A_ + 1) + (A_ - 1) * cw + 2 * math.sqrt(A_) * al), -2 * A_ * ((A_ - 1) + (A_ + 1) * cw), A_ * ((A_ + 1) + (A_ - 1) * cw - 2 * math.sqrt(A_) * al)]
        a = [(A_ + 1) - (A_ - 1) * cw + 2 * math.sqrt(A_) * al, 2 * ((A_ - 1) - (A_ + 1) * cw), (A_ + 1) - (A_ - 1) * cw - 2 * math.sqrt(A_) * al]
        y = filtfilt(np.array(b) / a[0], np.array(a) / a[0], y)
    return y.astype(np.float32)


def sex_params(rng: random.Random) -> tuple:
    """入力の側の性別の摂動の値(D434・diag_ppgvoc44): (f0 の倍率, 包絡の周波数の倍率)。1/3 は WORLD の再合成だけ(加工の痕跡を摂動の目印にさせない)。"""
    if rng.random() < 1 / 3:
        return (1.0, 1.0)
    return (rng.uniform(0.5, 0.65), 1.0 / rng.uniform(1.08, 1.2))


def sex_apply(x: np.ndarray, prm: tuple) -> np.ndarray:
    """WORLD(dio + stonemask・cheaptrick・d4c)で f0 を kf 倍・包絡と非周期性を周波数軸で ke 倍(ke < 1 = 声道を長く)にして再合成・RMS をそろえる。時刻は保つ。"""
    import pyworld as pw
    kf, ke = prm
    xd = np.ascontiguousarray(x, dtype=np.float64)
    f0, t = pw.dio(xd, N.SR, f0_floor=60, f0_ceil=1000, frame_period=5.0)
    f0 = pw.stonemask(xd, f0, t, N.SR)
    sp = pw.cheaptrick(xd, f0, t, N.SR)
    ap = pw.d4c(xd, f0, t, N.SR)
    if ke != 1.0:
        nb = sp.shape[1]
        src = np.arange(nb) / ke
        sp = np.stack([np.interp(src, np.arange(nb), r) for r in sp])
        ap = np.stack([np.interp(src, np.arange(nb), r) for r in ap])
    y = pw.synthesize(f0 * kf, sp, ap, N.SR, 5.0)[: len(x)]
    y = np.pad(y, (0, len(x) - len(y)))
    return (y * (np.sqrt((xd ** 2).mean()) / (np.sqrt((y ** 2).mean()) + 1e-9))).astype(np.float32)


class DS(torch.utils.data.IterableDataset):
    """実音声と TTS を 1:1(判別器の「本物」の分布を実音声に寄せる)。xa = x[s − A, s + SEG + A](分析の余白)・f0 = 前計算のフレーム s/H + t・
    包絡 = CheapTrick(利得をかけた後)。"""

    def __init__(self, real: list, tts: list, seed: int, f0_acc: tuple | None = None, hi_rows: list | None = None, p_hi: float = 0.0,
                 spk_idx: dict | None = None, ctx: int = 0, pert: float = 0.0, pert_c0_only: bool = False, sexpert: float = 0.0):
        self.real, self.tts, self.seed, self.f0_acc = real, tts, seed, f0_acc
        self.pert, self.pert_c0_only, self.sexpert = pert, pert_c0_only, sexpert
        self.hi_rows, self.p_hi = hi_rows or [], p_hi
        self.spk_idx, self.ctx = spk_idx, ctx

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed * 1009 + (wi.id if wi else 0))
        while True:
            use_hi = bool(self.hi_rows) and rng.random() < self.p_hi
            pool = self.hi_rows if use_hi else (self.real if rng.random() < 0.5 else self.tts)
            row = pool[rng.randrange(len(pool))]
            wav, f0p, sr, dur = row[:4]
            try:
                n48 = int(dur * N.SR)
                if n48 < SEG + 4 * PA.A:
                    continue
                if use_hi:
                    full = np.load(f0p, mmap_mode="r")
                    idx = np.where(np.asarray(full) >= 950.0)[0]
                    if len(idx) == 0:
                        continue
                    c = int(idx[rng.randrange(len(idx))])
                    s = (c - rng.randrange(int(0.4 * T), int(0.9 * T))) * N.HOP
                    if not (PA.A + N.HOP <= s < n48 - SEG - PA.A - N.HOP):
                        continue
                else:
                    s = rng.randrange(PA.A + N.HOP, n48 - SEG - PA.A - N.HOP) // N.HOP * N.HOP
                f0 = np.load(f0p, mmap_mode="r")[s // N.HOP:s // N.HOP + T].astype(np.float32)
                f0 = np.pad(f0, (0, T - len(f0)))
                if self.f0_acc is not None and not use_hi:
                    v = f0[f0 > 0]
                    acc = f0_mean_acceptance(self.f0_acc) if len(v) < 0.3 * T else self.f0_acc[int(np.searchsorted(F0_BINS, float(np.median(v))))]
                    if rng.random() > acc:
                        continue
                xa = read_span(wav, sr, s - PA.A, SEG + 2 * PA.A)
                if len(xa) < SEG + 2 * PA.A or np.sqrt((xa[PA.A:PA.A + SEG] ** 2).mean()) < 1e-3:
                    continue
                g = min(2.0 ** rng.uniform(-0.5, 0.5), 0.99 / max(float(np.abs(xa).max()), 1e-6))
                xa = (xa * g).astype(np.float32)
                if self.ctx:
                    a0 = max(0, s - self.ctx)
                    xl = read_span(wav, sr, a0, s - a0) if s > a0 else np.zeros(0, np.float32)
                    xl = np.concatenate([np.zeros(self.ctx - len(xl), np.float32), xl]) * g
                    if self.pert > 0 or self.sexpert > 0:
                        xp, xlp = xa, xl.astype(np.float32)
                        kf = 1.0
                        if self.pert > 0 and rng.random() < self.pert:
                            prm = channel_params(rng)
                            xp, xlp = channel_apply(xa, prm), channel_apply(xlp, prm)
                        if self.sexpert > 0 and rng.random() < self.sexpert:
                            prm = sex_params(rng)
                            kf = prm[0]
                            z_ = sex_apply(np.concatenate([xlp, xp]), prm)
                            xlp, xp = z_[:len(xlp)], z_[len(xlp):]
                        e_o = PA.envelope(xa, f0)
                        e_p = e_o if xp is xa else PA.envelope(xp, f0 * kf)
                        if self.pert_c0_only:
                            xp, xlp = xa, xl.astype(np.float32)
                        yield (torch.from_numpy(xa), torch.from_numpy(f0), torch.from_numpy(e_p),
                               torch.from_numpy(xlp), torch.tensor(self.spk_idx[row[4]]), torch.from_numpy(xp), torch.from_numpy(e_o))
                        continue
                    yield (torch.from_numpy(xa), torch.from_numpy(f0), torch.from_numpy(PA.envelope(xa, f0)),
                           torch.from_numpy(xl.astype(np.float32)), torch.tensor(self.spk_idx[row[4]]))
                else:
                    yield torch.from_numpy(xa), torch.from_numpy(f0), torch.from_numpy(PA.envelope(xa, f0))
            except Exception:
                continue


F0_BINS = (250.0, 350.0, 450.0, 550.0, 650.0)
F0_TARGET = (0.10, 0.20, 0.22, 0.22, 0.15, 0.11)
F0_MIX = (0.191, 0.327, 0.233, 0.134, 0.073, 0.042)


def f0_acceptance() -> tuple:
    w = [t / m for t, m in zip(F0_TARGET, F0_MIX)]
    return tuple(round(x / max(w), 4) for x in w)


def f0_mean_acceptance(acc: tuple) -> float:
    """有声が 30% 未満の区間(息・無声音)の採択率 = 有声区間の平均採択率(無声主体の区間の割合を変えない)。"""
    return sum(m * a for m, a in zip(F0_MIX, acc))


def excitation(f0: torch.Tensor, n: int, gen: torch.Generator | None = None) -> torch.Tensor:
    h = N.harmonic_source(f0)[:, :n]
    nz = torch.randn(h.shape, generator=gen, device="cpu").to(h.device) if gen is not None else torch.randn_like(h)
    return torch.stack([h * 0.1, nz * 0.01], 1)


def held(dev: str, hi: bool = False) -> list:
    """hi = True: f0_h を f0hi の教師(1kHz 超の叫び・ハイトーンを含む・current/f0_range.md)にする。"""
    import eval_nvoc as E
    out = []
    for it in E.held_items():
        x = it["x"].astype(np.float32)
        n = len(x) // N.HOP
        x = x[:n * N.HOP]
        xx = np.concatenate([np.zeros(YCTX, np.float32), x])
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        f0h = harvest_frames(xx, n)
        if hi:
            import f0hi as H
            f0h, _ = H.teacher_f0(x, n, f0h)
        out.append({"x": x, "xa": xa, "f0_h": f0h, "f0_y": yin_frames(xx, n), "env": PA.envelope(xa, f0h), "f0_feat": it["f0"]})
    return out


def render(gen: R.RVoc, front: Front, it: dict, f0_out: np.ndarray, dev: str) -> np.ndarray:
    """包絡と周期性は harvest の f0 で分析した値に固定し(製品では変換器が出す)、log f0・有声・パルスだけ f0_out にする。"""
    xa = torch.from_numpy(it["xa"])[None].to(dev)
    fa = torch.from_numpy(it["f0_h"])[None].to(dev)
    fo = torch.from_numpy(f0_out.astype(np.float32))[None].to(dev)
    cond = front(xa, fa, torch.from_numpy(it["env"])[None].to(dev), fo)
    return gen(cond, excitation(fo, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()


def f0_authority(gen: R.RVoc, front: Front, items: list, dev: str, semis: tuple = (-4.0, 4.0)) -> float:
    """f0 の権威: 包絡と周期性はそのまま f0 だけ ±4 半音 → 出力の harvest が指示の f0 に従うか(有声で一致したフレームの |cent| の中央値の平均・
    測定の床 = WORLD の理想の追従で 10.5 cent)。"""
    errs = []
    for it in items:
        for st in semis:
            f0 = it["f0_h"] * np.float32(2.0 ** (st / 12))
            y = render(gen, front, it, f0, dev)[N.DELAY:]
            fo = harvest_frames(np.concatenate([np.zeros(YCTX, np.float32), y]), len(f0))
            m = (f0 > 0) & (fo > 0)
            if m.sum() > 20:
                errs.append(float(np.median(np.abs(1200 * np.log2(fo[m] / f0[m])))))
    return round(float(np.mean(errs)), 1) if errs else float("nan")


def evaluate(gen: R.RVoc, front: Front, items: list, dev: str) -> dict:
    """写し合成を 2 通りで: f0 = harvest(出力部の上限・判定に使う)と 生の因果 YIN(製品の f0 推定の誤りを pulse と log f0 に入れる)。+ f0 の権威(6 件)。"""
    import eval_nvoc as E
    from eval_zsvc import contrast
    gen.eval()
    out = {}
    with torch.no_grad():
        for key in ("f0_h", "f0_y"):
            rows = []
            for it in items:
                y = render(gen, front, it, it[key], dev)
                m = E.metrics(y, it["x"], N.DELAY, dev)
                m["contrast"] = contrast(y[N.DELAY:], it["f0_feat"])
                rows.append(m)
            out[key[3:]] = {k: round(float(np.nanmean([r[k] for r in rows])), 4) for k in rows[0]}
        out["f0auth_cent"] = f0_authority(gen, front, items[:6], dev)
    gen.train()
    return out


def C3causal(x: torch.Tensor, w: int) -> torch.Tensor:
    """因果の w フレーム平均(train_c3.causal_avg と同じ: 先頭は最初の値で埋める)。x [B, C, T]。"""
    xp = torch.cat([x[..., :1].expand(-1, -1, w - 1), x], -1)
    return xp.unfold(-1, w, 1).mean(-1)


def ckpt_env_smooth(st: dict, override: float | None) -> float:
    """チェックポイントの env_smooth。無い(旧)ckpt で明示もされなければ止める(平滑ありで学習した重みを平滑なしの条件で測る事故を防ぐ)。"""
    if override is not None:
        return override
    if "env_smooth" not in st:
        raise SystemExit("ckpt に env_smooth が無い: --env_smooth を明示せよ(0 = 平滑なしで学習・0.25 = diag_rvoc1s 以降)")
    return st["env_smooth"]


def atomic_save(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def tripwire_v2(hist: list, drop: float = 0.10) -> str | None:
    """rvoc2 以降(renderer.md R4・PESQ の絶対値の線は耳で較正されていないので外す)。hist = [(step, pesq_h, am_db_h, hf_db_h)]。
    PESQ_h が最良値(再開時の評価を含む)より 0.10 以上落ちる・評価値が非有限・変調の線が 2 回連続で 1.5dB 超・hf が 2.5dB 未満(欠落)のとき止める。"""
    step, pq, am, hf = hist[-1]
    if not all(math.isfinite(v) for v in (pq, am, hf)):
        return f"step {step}: 評価値が非有限 (pesq {pq}, am {am}, hf {hf})"
    best = max(h[1] for h in hist)
    if pq < best - drop:
        return f"step {step}: PESQ_h {pq:.3f} が最良 {best:.3f}(再開時の評価を含む)から {drop:.2f} 以上低下"
    if len(hist) >= 2 and am > 1.5 and hist[-2][2] > 1.5:
        return f"step {step}: 変調の線が 2 回連続で 1.5dB 超({hist[-2][2]:.2f}・{am:.2f})"
    if hf < -2.5:
        return f"step {step}: hf {hf:.2f}dB < -2.5"
    return None


def tripwire(hist: list) -> str | None:
    """事前登録の中止条件(renderer.md §5)。hist = [(step, pesq_h, am_db_h)]。"""
    step, pq, am = hist[-1][:3]
    for s_, lo in STOPS.items():
        if step == s_ and pq < lo:
            return f"step {step}: PESQ {pq:.3f} < {lo}"
    if step >= 100000 and am > AM_MAX:
        return f"step {step}: 変調の線 {am:.2f}dB > {AM_MAX}"
    pts = [(math.log10(s_), p_) for s_, p_, *_ in hist if s_ >= 40000]
    if step >= 100000 and len(pts) >= 4:
        k, b = np.polyfit([u for u, _ in pts], [v for _, v in pts], 1)
        proj = k * math.log10(300000) + b
        if proj < TARGET:
            return f"step {step}: log(step) 外挿で 300k の PESQ {proj:.3f} < {TARGET}"
    return None


def ship_gate(gen: R.RVoc, front: Front, items: list) -> bool:
    """出力部の推論経路 = 生成器(条件と励起 → 波形)。分析(CheapTrick・周期性)は学習の教師側だけで推論経路に無い(製品の条件は変換器が出す)。
    実音声から作った条件と励起で、フレーム k 以降を別の実音声の値に書き換え、k·H より前の出力が 1 bit も変わらないこと。"""
    import copy
    import ship_check as SC
    g = copy.deepcopy(gen).cpu().eval()
    fr = copy.deepcopy(front).cpu()

    def ce(it):
        xa = torch.from_numpy(it["xa"])[None]
        f = torch.from_numpy(it["f0_h"])[None]
        c = fr(xa, f, torch.from_numpy(it["env"])[None])
        return c, excitation(f, c.shape[-1] * N.HOP, torch.Generator().manual_seed(0))

    worst = 0
    with torch.no_grad():
        c1, e1 = ce(items[0])
        c2, e2 = ce(items[1])
        T = min(c1.shape[-1], c2.shape[-1], 600)
        c1, e1, c2, e2 = c1[..., :T], e1[..., :T * N.HOP], c2[..., :T], e2[..., :T * N.HOP]
        y = g(c1, e1)[0]
        for k in (97, 211, 350, 503):
            ca, ea = c1.clone(), e1.clone()
            ca[..., k:], ea[..., k * N.HOP:] = c2[..., k:], e2[..., k * N.HOP:]
            d = (g(ca, ea)[0] - y).abs()
            nz = (d > 0).nonzero()
            assert len(nz) > 0, "書き換えが出力を動かさない = INCONCLUSIVE"
            worst = max(worst, k * N.HOP - int(nz[0]))
    la = max(0, worst)
    print(f"  生成器の未来不変性(実音声由来の条件・励起・1 bit): 先読み {la} sample", flush=True)
    return la == 0 and SC.ledger([("rvoc 生成器(条件・励起の書き換え・実音声由来)", la / N.SR * 1000), ("出力遅延 DELAY 240", 5.0), ("ブロック HOP 240", 5.0)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="rvoc1")
    ap.add_argument("--steps", type=int, default=300000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ch", type=int, default=256)
    ap.add_argument("--kernels", default="3,7,11")
    ap.add_argument("--dils", default="1,3,5")
    ap.add_argument("--eval_every", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cond", default="pae", choices=("pae", "mel"))
    ap.add_argument("--gan_from", type=int, default=0, help="この step から GAN(0 = 最初から・steps 以上 = 回帰のみ)")
    ap.add_argument("--lr_half", type=int, default=300000)
    ap.add_argument("--held_hi", action="store_true", help="評価の f0_h を f0hi の教師にする(1kHz 超を含む)")
    ap.add_argument("--p_hi", type=float, default=0.0, help="1kHz 超のフレームを含む区間をこの確率で引く(台帳に hi_frames のある行から・f0 の寄せの棄却は通さない)")
    ap.add_argument("--am_w", type=float, default=0.0, help="フレーム周期の包絡変調の線の罰則の重み(am_loss.am_penalty・0 = なし)")
    ap.add_argument("--am_margin", type=float, default=0.3, help="バッチ平均の線のパワー(dB)が目標 + margin を超えた分だけ罰する(proxy の単位)")
    ap.add_argument("--am_two_sided", action="store_true", help="線が目標より低く行き過ぎる(谷)のも罰する(rvoc2am の 240k で線 −1.5 まで行き過ぎた)")
    ap.add_argument("--am_margin_bg", type=float, default=0.0, help="近傍の変調パワー(dB)が目標を margin 超えて水増しされた分だけ罰する(片方向)")
    ap.add_argument("--guard", default="v1", choices=("v1", "v2", "v3"), help="v1 = 旧の中止条件(PESQ の絶対値と外挿)・v2 = 最良からの低下 0.10・変調の線・hf・v3 = v2 の低下の許容を 0.30(表の条件の追加学習: 本物の条件の held は設計上下がる・diag_rvtab で 5k に −0.18)")
    ap.add_argument("--f0_balance", action="store_true", help="区間の f0 中央値の分布を F0_TARGET に寄せる(棄却サンプリング・f0 を読んだ後・音声を読む前。実:TTS の実効比は 1:1 から実音声寄りに動く)")
    ap.add_argument("--tab_dir", default="", help="話者ごとの表(build_tables_rvoc.py)。指定時は表のある話者の行だけで学習")
    ap.add_argument("--tab_per", default="const", choices=("const", "f4"), help="表の条件の周期性: const = 話者の 75 パーセンタイルの定数・f4 = 因果な周期性推定器 F4(製品の経路)")
    ap.add_argument("--x_p", type=float, default=0.0, help="話者をまたいだ交差の標本の割合(内容と動き = A・表 = 別の話者 B・f0 / 周期性を B の水準へ)。L1 なし・判別器 + 同一性の損失")
    ap.add_argument("--x_id", type=float, default=0.0, help="交差の標本の同一性の損失(1 − cos(ECAPA(出力), e_B))の重み")
    ap.add_argument("--x_gamma", default="0,0.5,0.7,1", help="交差の標本で足す A の動き(F3 − 表_A)の量 γ の一覧")
    ap.add_argument("--spk_emb", default="", help="話者ごとの ECAPA の埋め込み(build_spk_emb.py)")
    ap.add_argument("--tab_dyn", default="real", choices=("real", "f3"), help="表の条件で γ を掛ける包絡: real = 本物の分析(CheapTrick)・f3 = 製品の因果な包絡推定器 F3(製品の写し合成 / 変換と同じ入力の分布)")
    ap.add_argument("--tab_gamma", default="0", help="表の条件の包絡 = 表 + γ·(本物の包絡 − 表)。γ はサンプルごとにこの一覧から一様に選ぶ(例 0,0.35,0.7)。0 = 表だけ")
    ap.add_argument("--tab_p", type=float, default=0.0, help="サンプルごとにこの確率で条件を製品と同じ作り方にする(包絡 = Σp T_話者・c0 = F2 + 因果 3 平均・周期性 = 話者の 75 パーセンタイルの定数・f0 は教師)")
    ap.add_argument("--slm_w", type=float, default=0.0, help="凍結 SSL の特徴の上の判別器(slm_disc.SLMDisc)の敵対の重み(0 = なし・目標関数 v1 の学習側の近似)")
    ap.add_argument("--lm_w", type=float, default=45.0, help="多尺度 logmel L1 の重み(既定 45 = 従来)。下げると最適解が条件付き中央値から分布の一致の側へ動く")
    ap.add_argument("--lr_reset", action="store_true", help="--resume の後に生成器・判別器の学習率を --lr に置き直す(既定 = 復元値から減衰を続ける)")
    ap.add_argument("--slm_model", default="microsoft/wavlm-base-plus", help="SLM 判別器の凍結 SSL(評価の物差しの WavLM-large とは別にする)")
    ap.add_argument("--env_smooth", type=float, default=0.0, help="教師側の包絡 c0..c24 を中心窓 [w,1-2w,w] でフレーム方向に平滑(0 = なし・0.25 = [.25,.5,.25]・推論の遅延なし・変換器の目標も同じ値)")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    if not a.smoke and not (out / "prereg.yaml").exists():
        print(f"results/{a.tag}/prereg.yaml が無い: 起動しない", flush=True)
        return 1
    print(f"prereg: results/{a.tag}/prereg.yaml", flush=True)
    tabs = None
    if a.tab_dir:
        tfiles = sorted(Path(a.tab_dir).glob("*.npz"))
        spk_idx = {f.stem: i for i, f in enumerate(tfiles)}
        zs = [np.load(f) for f in tfiles]
        tabs = (torch.from_numpy(np.stack([z["T"] for z in zs])).float().to(dev), torch.from_numpy(np.stack([z["per_p75"] for z in zs])).float().to(dev),
                torch.from_numpy(np.array([float(z["mu"]) for z in zs])).float().to(dev))
        real, tts = load_manifest_spk()
        real, tts = [r for r in real if r[4] in spk_idx], [r for r in tts if r[4] in spk_idx]
        print("tables", len(tfiles), "speakers", flush=True)
        import c1_content as CC
        import lvl as LV
        import train_c1 as T1
        st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
        c1n = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev); c1n.load_state_dict(st1["net"]); c1n.eval().requires_grad_(False)
        sf2 = torch.load(ROOT / "results/f2_2/last.pt", map_location="cpu", weights_only=False)
        f2n = LV.Lvl(sf2["cfg"]["ch"], tuple(sf2["cfg"]["dils"])).to(dev); f2n.load_state_dict(sf2["net"]); f2n.eval().requires_grad_(False)
        mfc = CC.MelFront().to(dev)
        f3n = None
        if a.tab_dyn == "f3":
            sf3 = torch.load(ROOT / "results/f3_1/last.pt", map_location="cpu", weights_only=False)
            f3n = LV.EnvEst(sf3["cfg"]["ch"], tuple(sf3["cfg"]["dils"])).to(dev); f3n.load_state_dict(sf3["net"]); f3n.eval().requires_grad_(False)
        f4n = None
        if a.tab_per == "f4" or a.x_p > 0:
            sf4 = torch.load(ROOT / "results/f4_1/last.pt", map_location="cpu", weights_only=False)
            f4n = LV.PerEst(sf4["cfg"]["ch"], tuple(sf4["cfg"]["dils"])).to(dev); f4n.load_state_dict(sf4["net"]); f4n.eval().requires_grad_(False)
        ecx, embx = None, None
        if a.x_p > 0:
            assert f3n is not None and a.x_id > 0 and a.spk_emb, "交差の標本には --tab_dyn f3・--x_id・--spk_emb が要る"
            import idloss as ID
            ecx = ID.Ecapa(dev)
            ze = np.load(a.spk_emb)
            embx = torch.from_numpy(np.stack([ze[f.stem] for f in tfiles])).float().to(dev)
        CTX = T1.PRE
    else:
        real, tts = load_manifest()
    print("files real", len(real), f"({sum(r[3] for r in real) / 3600:.1f} h)", "tts", len(tts), f"({sum(r[3] for r in tts) / 3600:.1f} h)", flush=True)
    KK, DD = tuple(int(v) for v in a.kernels.split(",")), tuple(int(v) for v in a.dils.split(","))
    front = Front(a.cond, a.env_smooth).to(dev)
    dc = R.D_COND if a.cond == "pae" else R.D_COND_MEL
    gen = R.RVoc(ch=a.ch, kernels=KK, dils=DD, d_cond=dc).to(dev)
    print("params (M)", round(sum(p.numel() for p in gen.parameters()) / 1e6, 3), "GMAC/s", round(R.macs_per_second(gen) / 1e9, 2), flush=True)
    from bigvgan.discriminators import MultiPeriodDiscriminator
    mpd = MultiPeriodDiscriminator(SimpleNamespace(mpd_reshapes=[2, 3, 5, 7, 11], use_spectral_norm=False, discriminator_channel_mult=1)).to(dev)
    mrd = MRDLog().to(dev)
    dparams = list(mpd.parameters()) + list(mrd.parameters())
    opt_g = torch.optim.AdamW(gen.parameters(), lr=a.lr, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(dparams, lr=a.lr, betas=(0.8, 0.99))
    gamma = 0.5 ** (1.0 / a.lr_half)
    sch_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma)
    sch_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma)
    slm, opt_s = None, None
    if a.slm_w > 0:
        from slm_disc import SLMDisc
        slm = SLMDisc(a.slm_model).to(dev)
        opt_s = torch.optim.AdamW(slm.head_params(), lr=a.lr, betas=(0.8, 0.99))
        print("SLM 判別器", a.slm_model, "head params (M)", round(sum(p.numel() for p in slm.head_params()) / 1e6, 3), flush=True)
    ema = {k: v.detach().clone() for k, v in gen.state_dict().items()}
    step0 = 0
    if a.resume and not (out / "last.pt").exists():
        print(f"--resume だが {out}/last.pt が無い: 起動しない(黙って最初から学習しない)", flush=True)
        return 1
    if a.resume:
        (out / "STOP.json").unlink(missing_ok=True)
        st = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        gen.load_state_dict(st["gen"]); mpd.load_state_dict(st["mpd"]); mrd.load_state_dict(st["mrd"])
        opt_g.load_state_dict(st["opt_g"]); opt_d.load_state_dict(st["opt_d"])
        ema = {k: v.to(dev) for k, v in st["ema"].items()}
        step0 = st["step"]
        if slm is not None and "slm" in st:
            slm.head.load_state_dict(st["slm"]); opt_s.load_state_dict(st["opt_s"])
        elif slm is not None:
            print("SLM 判別器は新規(再開元に無い)", flush=True)
        for o_ in (opt_g, opt_d):
            for g_ in o_.param_groups:
                g_["initial_lr"] = g_["lr"]
        sch_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma)
        sch_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma)
        if a.lr_reset:
            for o_ in (opt_g, opt_d):
                for g_ in o_.param_groups:
                    g_["lr"] = a.lr
                    g_["initial_lr"] = a.lr
            sch_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma)
            sch_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma)
        print("resume from", step0, "lr(--lr_reset なら --lr・それ以外は復元値から半減期 lr_half で減衰)", opt_g.param_groups[0]["lr"], flush=True)
    evg = R.RVoc(ch=a.ch, kernels=KK, dils=DD, d_cond=dc).to(dev)
    mels = [build_mel(nf, hm, nm).to(dev) for nf, hm, nm in MEL_SPECS]
    items = held(dev, a.held_hi)[:3 if a.smoke else None]
    if not ship_gate(gen, front, items):
        print("出荷ゲート FAIL: 起動しない", flush=True)
        return 1
    assert not (a.tab_dir and a.p_hi > 0), "表の包絡の学習では --p_hi を使わない(hi 行に話者がない)"
    loader = iter(torch.utils.data.DataLoader(DS(real, tts, 7 + step0, f0_acceptance() if a.f0_balance else None, load_hi_rows() if a.p_hi > 0 else None, a.p_hi,
                                                 spk_idx if a.tab_dir else None, CTX if a.tab_dir else 0), batch_size=a.bs, num_workers=a.workers, pin_memory=True,
                                              persistent_workers=True, prefetch_factor=4))
    log = open(out / "train.jsonl", "a")

    hist: list = []

    def run_eval(step: int) -> str | None:
        evg.load_state_dict(ema)
        r = {"step": step, **evaluate(evg, front, items, dev)}
        print("eval", json.dumps(r, ensure_ascii=False), flush=True)
        log.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.flush()
        hist.append((step, r["h"]["pesq"], r["h"]["am_db"], r["h"]["hf_db"]))
        if a.smoke or a.tag.startswith("diag_") or step == 0:
            return None
        return tripwire_v2(hist, 0.30 if a.guard == "v3" else 0.10) if a.guard in ("v2", "v3") else tripwire(hist)

    why0 = run_eval(step0)
    if why0:
        print("再開時のベースライン評価で中止条件:", why0, flush=True)
        return 2
    acc: dict = {}
    t0 = time.time()
    total = step0 + 30 if a.smoke else a.steps
    for step in range(step0 + 1, total + 1):
        bt = [t.to(dev, non_blocking=True) for t in next(loader)]
        xa, f0, env = bt[:3]
        with torch.no_grad():
            cond = front(xa, f0, env)
            if tabs is not None and a.tab_p > 0:
                xl, si = bt[3], bt[4]
                xc = torch.cat([xl, xa[:, PA.A:PA.A + SEG]], 1)
                mf = mfc(xc)
                p = c1n(mf)[..., -T:].softmax(1)
                tab = torch.einsum("bdk,bkt->bdt", tabs[0][si], p)
                c0n = C3causal(f2n(mf)[:, None], 3)[:, :, -T:]
                lf = cond[:, 25:26]
                vv = cond[:, 26:27]
                envc = tab / 10
                dsrc = f3n(mf)[:, 1:25, -T:] if f3n is not None else cond[:, 1:25]
                if a.tab_gamma != "0":
                    gs = torch.tensor([float(g) for g in a.tab_gamma.split(",")], device=dev)
                    envc = envc + gs[torch.randint(len(gs), (xa.shape[0],), device=dev)][:, None, None] * (dsrc - envc)
                p4 = f4n(mf)[..., -T:] if f4n is not None else None
                per_c = (p4 if a.tab_per == "f4" else tabs[1][si][:, :, None]).clamp(0, 1.5) * vv
                ct = torch.cat([c0n, envc, lf, vv, per_c], 1)
                if a.x_p > 0:
                    nx = max(1, int(round(xa.shape[0] * a.x_p)))
                    sa, sb = si[:nx], torch.randint(tabs[0].shape[0], (nx,), device=dev)
                    gx = torch.tensor([float(g) for g in a.x_gamma.split(",")], device=dev)[torch.randint(len(a.x_gamma.split(",")), (nx,), device=dev)][:, None, None]
                    envx = torch.einsum("bdk,bkt->bdt", tabs[0][sb], p[:nx]) / 10 + gx * (dsrc[:nx] - tab[:nx] / 10)
                    f0x = f0[:nx] * torch.exp(tabs[2][sb] - tabs[2][sa])[:, None]
                    lfx = torch.where(f0x > 0, torch.log(f0x.clamp(min=1.0) / 200.0), torch.zeros_like(f0x))[:, None]
                    perx = (p4[:nx] - tabs[1][sa][:, :, None] + tabs[1][sb][:, :, None]).clamp(0, 1.5) * vv[:nx]
                    condx = torch.cat([c0n[:nx], envx, lfx, vv[:nx], perx], 1)
                    excx = excitation(f0x, SEG)
                m = (torch.rand(xa.shape[0], device=dev) < a.tab_p)[:, None, None]
                if step == step0 + 1:
                    q = lambda t: [round(float(v), 3) for v in torch.quantile(t.flatten().float(), torch.tensor([0.05, 0.5, 0.95], device=dev))]
                    print("cond check: 本物 c0n", q(cond[:, 0]), "c1..24", q(cond[:, 1:25]), "per", q(cond[:, 27:31]), "| 表 c0n", q(ct[:, 0]), "c1..24", q(ct[:, 1:25]), "per", q(ct[:, 27:31]), flush=True)
                cond = torch.where(m, ct, cond)
            exc = excitation(f0, SEG)
            tgt = xa[:, PA.A - N.DELAY:PA.A - N.DELAY + SEG]
        y = gen(cond, exc)
        yl, tl = y[:, WARM:], tgt[:, WARM:]
        lm = logmel_l1(yl[:, None], tl[:, None], mels)
        loss = a.lm_w * lm
        rec = {"lm": lm}
        yxl = None
        if tabs is not None and a.x_p > 0:
            yxl = gen(condx, excx)[:, WARM:]
            lid = (1 - (ecx(yxl) * embx[sb]).sum(-1)).mean()
            loss = loss + a.x_id * lid
            rec["x_id"] = lid
        if tabs is not None and a.tab_p > 0 and step % 20 == 0:
            with torch.no_grad():
                mm = m[:, 0, 0]
                if mm.any() and (~mm).any():
                    rec["lm_tab"] = logmel_l1(yl[mm][:, None], tl[mm][:, None], mels)
                    rec["lm_real"] = logmel_l1(yl[~mm][:, None], tl[~mm][:, None], mels)
        if a.am_w > 0:
            pen, ams = am_penalty(yl, tl, a.am_margin, a.am_margin_bg, a.am_two_sided)
            loss = loss + a.am_w * pen
            rec.update({"am_pen": pen, **{"am_" + k: v for k, v in ams.items()}})
        if step > a.gan_from:
            off = random.randrange(0, yl.shape[-1] - 16384)
            r_, f_ = tl[:, off:off + 16384], yl[:, off:off + 16384]
            opt_d.zero_grad(set_to_none=True)
            a1, b1, _, _ = mpd(r_[:, None], f_.detach()[:, None])
            a2, b2, _, _ = mrd(r_, f_.detach())
            dl = sum(((p.float() - 1) ** 2).mean() + (q.float() ** 2).mean() for p, q in zip(a1 + a2, b1 + b2))
            if yxl is not None:
                fx_ = yxl[:, off:off + 16384]
                _, bx1, _, _ = mpd(r_[:fx_.shape[0], None], fx_.detach()[:, None])
                _, bx2, _, _ = mrd(r_[:fx_.shape[0]], fx_.detach())
                dl = dl + sum((q.float() ** 2).mean() for q in bx1 + bx2)
            if torch.isfinite(dl):
                dl.backward()
                torch.nn.utils.clip_grad_norm_(dparams, 500.0)
                opt_d.step()
            for p_ in dparams:
                p_.requires_grad_(False)
            _, b1, f1r, f1g = mpd(r_[:, None], f_[:, None])
            _, b2, f2r, f2g = mrd(r_, f_)
            if yxl is not None:
                _, bx1, _, _ = mpd(r_[:fx_.shape[0], None], fx_[:, None])
                _, bx2, _, _ = mrd(r_[:fx_.shape[0]], fx_)
                advx = sum(((q.float() - 1) ** 2).mean() for q in bx1 + bx2)
                loss = loss + advx
                rec["x_adv"] = advx
            for p_ in dparams:
                p_.requires_grad_(True)
            if slm is not None:
                with torch.no_grad():
                    hr = slm.feats(tl)
                opt_s.zero_grad(set_to_none=True)
                sl = ((slm.head(hr).float() - 1) ** 2).mean() + (slm(yl.detach()).float() ** 2).mean()
                if torch.isfinite(sl):
                    sl.backward()
                    torch.nn.utils.clip_grad_norm_(slm.head_params(), 500.0)
                    opt_s.step()
                for p_ in slm.head_params():
                    p_.requires_grad_(False)
                sadv = ((slm(yl).float() - 1) ** 2).mean()
                for p_ in slm.head_params():
                    p_.requires_grad_(True)
                loss = loss + a.slm_w * sadv
                rec.update({"slm_adv": sadv, "slm_dl": sl})
            adv = sum(((q.float() - 1) ** 2).mean() for q in b1 + b2)
            fm = sum(F.l1_loss(u.detach().float(), v.float()) for xr, xg in zip(f1r + f2r, f1g + f2g) for u, v in zip(xr, xg))
            loss = loss + adv + 2 * fm
            rec.update({"adv": adv, "fm": fm, "dl": dl})
        opt_g.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        if not torch.isfinite(torch.nn.utils.clip_grad_norm_(gen.parameters(), 500.0)):
            opt_g.zero_grad(set_to_none=True)
            continue
        opt_g.step()
        sch_g.step()
        sch_d.step()
        with torch.no_grad():
            for k, v in gen.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(0.999).add_(v.detach(), alpha=0.001)
        for k_, v_ in rec.items():
            acc.setdefault(k_, []).append(float(v_.detach()))
        if step % 200 == 0 or a.smoke:
            rr = {"step": step, "min": round((time.time() - t0) / 60, 1), **{k_: round(float(np.mean(v_)), 4) for k_, v_ in acc.items()}}
            acc = {}
            print(json.dumps(rr), flush=True)
            log.write(json.dumps(rr) + "\n")
            log.flush()
        if step % a.eval_every == 0 or (a.smoke and step == total):
            why = run_eval(step)
            (out / "snap").mkdir(exist_ok=True)
            atomic_save({"ema": ema, "step": step, "cfg": gen.cfg, "cond": a.cond, "env_smooth": a.env_smooth}, out / "snap" / f"ema_{step // 1000}k.pt")
            atomic_save({"gen": gen.state_dict(), "ema": ema, "mpd": mpd.state_dict(), "mrd": mrd.state_dict(), "opt_g": opt_g.state_dict(),
                        "opt_d": opt_d.state_dict(), "step": step, "cfg": gen.cfg, "cond": a.cond, "env_smooth": a.env_smooth}
                        | ({"slm": slm.head.state_dict(), "opt_s": opt_s.state_dict()} if slm is not None else {}), out / "last.pt")
            if why:
                (out / "STOP.json").write_text(json.dumps({"step": step, "why": why, "hist": hist}, ensure_ascii=False, indent=1))
                print("中止条件:", why, flush=True)
                return 2
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
