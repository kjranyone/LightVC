"""paper_snv rev26 §5 M20 の前段 D(段 I14): 操作者の音の左寄せの短時間スペクトル(Hann 2048・ホップ 240・フレーム t の窓は (t+1)H で終わる = 先読み 0)の対数パワー(1.5 kHz まで線形のビン 64 本 + 1.5〜24 kHz の mel 160 帯)と Hann 512 の対数 mel 32 帯(400 Hz〜24 kHz・時間の分解能)から、
きれいな音の前段の特徴 [包絡 29(y = train_snv_p1.to_y)・有声の logit・lf] を出す因果の畳み込み(過去だけ・層ごとのチャンネルの正規化・発話の統計なし)。
学習 = 段 I8 の split の学習の声の発話(e1・d・e1nv・48 kHz)をその場で背景と混ぜる(入力側だけ・当てる先 = 元の音の前段の特徴 data/snv_p_feat・足した無音は 0 ベクトル・有声 0)。
背景 = MUSAN の音楽と雑音(data/snv_p_feat_rec2/sources.json・音楽は許諾の塊 = アルバム・雑音はファイルで分ける)+ FMA の器楽(data/snv_sep/fma_sources.json・作者で分ける)を crc32 で学習 8 割 / 開発 2 割(bg_split)。
混ぜ = 背景なし 0.1 / 音楽 0.55(FMA 0.6・MUSAN 0.4・SN 比 U[−10, 20] dB)/ 雑音 0.35(U[−5, 30] dB)・SN 比 = 発話全体の RMS に対して・前と後にそれぞれ確率 0.5 で 0〜3 s の無音(フレームの整数倍)・
残響 確率 0.3(RT60 U[0.15, 0.8] s・直接音との比 U[−3, 10] dB・snv_rec_feat と同じ型)→ 背景 → 音量 U[−12, +4] dB → ±1 で切る。切り出し 4 s。
損失 = L1(包絡 29・全フレーム)+ BCE(有声)+ L1(lf・元の音の有声のフレーム)。検証 = 検証の声 × 開発の背景(固定の種)・SN 比 20 / 10 / 0 / −5 dB と 0 dB + 伴奏だけの時間帯ごとに、包絡の L1・有声の一致・lf の誤差(セント)・伴奏だけのフレームの有声の誤り。
features(D, x) = 推論(48 kHz の波形 → snv_p1_eval の発話の辞書)。
--fam i18(paper_snv rev33 段 I18)= i15 の族 + FSD50K の開発の組の本物の手の音(data/snv_sep/fsd50k_dev_sources.json・透かし付き・投稿者で学習 8 割 / 検証 2 割)を操作者の音と手拍子の組に足す。
--fam i15(paper_snv rev29 段 I15・M21)= 生成の痕跡に依らない族: 背景・床・操作者の音にも Irodori と同じ透かし(snv_sep_wm.py → data/snv_sep/wm)・入力の無音をデジタルの 0 にしない床(透かし付きの色つき雑音・発話の RMS の −70〜−30 dB)・
確率 0.5 で帯域を 4〜20 kHz(対数一様)で切る・確率 0.4 で操作者の音(MUSAN の手や物の音 data/snv_sep/operator_sources.json と合成の手拍子・半々)を 1〜6 個、長さ 0.08〜1 s・発話の RMS に対して −15〜+10 dB で声と時間を重ねて足す。当てる先は元の音の前段の特徴のまま。
    cd training && uv run --no-sync python train_snv_sep.py --tag snv_i14_sep --prereg results/snv_i14/prereg.yaml [--fam i14|i15] [--steps 20000] [--bs 16] [--ch 256] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import math
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_

import snv_p0_front as P
import train_snv_p1 as T

ROOT = Path(__file__).resolve().parent.parent
SR = P.SR
H = P.H
NFFT = 2048
N_LIN = 64
LIN_HZ = 1500.0
N_MEL = 160
NFFT2 = 512
N_MEL2 = 32
MEL2_LO = 400.0
D_IN = N_LIN + N_MEL + N_MEL2
CROP = 800
DIL = (1, 2, 4, 8, 16, 32, 64, 1)
D_OUT = 31
MUSAN = ROOT / "data/snv_p_feat_rec2/sources.json"
FMA = ROOT / "data/snv_sep/fma_sources.json"
WAVDIR = {"e1": ROOT / "data/snv_e1", "d": ROOT / "data/snv_d", "e1nv": ROOT / "data/snv_e1nv"}
SNR = {"music": (-10.0, 20.0), "noise": (-5.0, 30.0)}
P_BG = (("none", 0.10), ("music", 0.55), ("noise", 0.35))
P_FMA = 0.6
P_PAD, PAD_MAX = 0.5, 3.0
P_REV = 0.3
GAIN = (-12.0, 4.0)
WM = ROOT / "data/snv_sep/wm"
OPER = ROOT / "data/snv_sep/operator_sources.json"
FLOOR_DB = (-70.0, -30.0)
P_BAND, BAND_HZ = 0.5, (4000.0, 20000.0)
P_OP, OP_N, OP_DB, OP_DUR, P_CLAP = 0.4, (1, 6), (-15.0, 10.0), (0.08, 1.0), 0.5
FAMS = {"i14": {}, "i15": {"wm": True, "floor": True, "band": True, "op": True}, "i18": {"wm": True, "floor": True, "band": True, "op": True, "fsd": True},
        "i22": {"wm": True, "floor": True, "band": True, "op": True, "fsd": True, "floor_db": (-70.0, -15.0)}}
FSD_DEV = ROOT / "data/snv_sep/fsd50k_dev_sources.json"


def bg_split() -> dict:
    out: dict = {"train": {"music": [], "noise": []}, "dev": {"music": [], "noise": []}}
    ms = json.loads(MUSAN.read_text())
    for kind in ("music", "noise"):
        for p in ms["pools"][kind]:
            key = (ms["files"][Path(p).stem]["text"] or Path(p).stem) if kind == "music" else Path(p).stem
            out["dev" if zlib.crc32(f"musan|{key}".encode()) % 5 == 0 else "train"][kind].append(p)
    if FMA.exists():
        fm = json.loads(FMA.read_text())
        for p in fm["pools"]["music"]:
            key = fm["files"][Path(p).stem]["artist_id"]
            out["dev" if zlib.crc32(f"fma|{key}".encode()) % 5 == 0 else "train"]["music"].append(p)
    return out


def utterances(voices: list[str]) -> list[tuple[Path, Path]]:
    out = []
    for v in voices:
        for kind, pat in (("e1", "t*.npz"), ("d", "s*.npz"), ("e1nv", "*.npz")):
            for q in sorted((T.FEAT / kind / v).glob(pat)):
                w = WAVDIR[kind] / v / f"{q.stem}.wav"
                if w.exists():
                    out.append((q, w))
    return out


def read_bg(path: str, n: int, g: np.random.Generator) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly
    info = sf.info(path)
    up = SR // info.samplerate if SR % info.samplerate == 0 else 0
    m = n // up + 64 if up else int(n * info.samplerate / SR) + 64
    if info.frames > m:
        st = int(g.integers(0, info.frames - m))
        x = sf.read(path, start=st, stop=st + m, dtype="float64")[0]
    else:
        x = np.resize(sf.read(path, dtype="float64")[0], m)
    if x.ndim > 1:
        x = x.mean(1)
    if info.samplerate != SR:
        gg = math.gcd(SR, info.samplerate)
        x = resample_poly(x, SR // gg, info.samplerate // gg)
    x = np.resize(x, n) if len(x) < n else x[:n]
    return x / (np.sqrt((x ** 2).mean()) + 1e-12)


def read_event(path: str, g: np.random.Generator, dur: float) -> np.ndarray:
    x = read_bg(path, 3 * SR, g)
    fr = (x[: len(x) // 480 * 480].reshape(-1, 480) ** 2).mean(1)
    n = int(dur * SR)
    s = int(np.clip(int(fr.argmax()) * 480 + 240 - n // 3, 0, len(x) - n))
    e = x[s:s + n].copy()
    f = max(1, min(int(0.005 * SR), n // 4))
    e[:f] *= np.linspace(0.0, 1.0, f)
    e[n - f:] *= np.linspace(1.0, 0.0, f)
    return e / (np.sqrt((e ** 2).mean()) + 1e-12)


def lowpass(y: np.ndarray, fc: float) -> np.ndarray:
    Y = np.fft.rfft(y)
    f = np.fft.rfftfreq(len(y), 1 / SR)
    Y *= np.clip((1.1 * fc - f) / (0.1 * fc), 0.0, 1.0) ** 2
    return np.fft.irfft(Y, len(y))


def rir(g: np.random.Generator) -> np.ndarray:
    rt60, drr = float(g.uniform(0.15, 0.8)), float(g.uniform(-3, 10))
    L = int(rt60 * 1.5 * SR)
    tt = np.arange(L) / SR
    tail = g.standard_normal(L) * np.exp(-6.9 * tt / rt60)
    tail[: int(0.003 * SR)] = 0.0
    tail *= 10 ** (-drr / 20) / (np.sqrt((tail ** 2).sum()) + 1e-12)
    tail[0] += 1.0
    return tail


def mix_parts(x: np.ndarray, rms: float, bg: str, src: str | None, snr: float, g: np.random.Generator, rev: bool, gain_db: float, ext: dict | None = None, with_ev: bool = False) -> tuple:
    s = x.astype(np.float64)
    if rev:
        h = rir(g)
        n = len(s)
        z = np.fft.irfft(np.fft.rfft(s, n + len(h)) * np.fft.rfft(h, n + len(h)), n + len(h))[:n]
        s = z * (np.sqrt((s ** 2).mean()) / (np.sqrt((z ** 2).mean()) + 1e-12)) if (s ** 2).mean() > 0 else z
    ext = ext or {}
    nz = np.zeros_like(s)
    fl = np.zeros_like(s)
    if ext.get("floor"):
        fp, fdb = ext["floor"]
        fl = read_bg(fp, len(s), g) * rms * 10 ** (fdb / 20)
        nz = nz + fl
    bc = np.zeros_like(s)
    if bg != "none":
        bc = read_bg(src, len(s), g) * rms * 10 ** (-snr / 20)
        nz = nz + bc
    ev = np.zeros_like(s)
    for ep, dur, on, edb in ext.get("events", []):
        e = read_event(ep, g, dur)
        on = min(on, len(s) - len(e))
        nz[on:on + len(e)] += e * rms * 10 ** (edb / 20)
        ev[on:on + len(e)] += e * rms * 10 ** (edb / 20)
    k = 10 ** (gain_db / 20)
    s, nz, ev, fl, bc = s * k, nz * k, ev * k, fl * k, bc * k
    if ext.get("band"):
        s, nz, ev, fl, bc = lowpass(s, ext["band"]), lowpass(nz, ext["band"]), lowpass(ev, ext["band"]), lowpass(fl, ext["band"]), lowpass(bc, ext["band"])
    out = (np.clip(s + nz, -1.0, 1.0).astype(np.float32), s.astype(np.float32), nz.astype(np.float32))
    return out + (ev.astype(np.float32), fl.astype(np.float32), bc.astype(np.float32)) if with_ev else out


def mix(x: np.ndarray, rms: float, bg: str, src: str | None, snr: float, g: np.random.Generator, rev: bool, gain_db: float, ext: dict | None = None) -> np.ndarray:
    return mix_parts(x, rms, bg, src, snr, g, rev, gain_db, ext)[0]


def target(npz: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = T.load_utt(npz)
    return u["y"], u["v"], u["lf"]


class Mix(torch.utils.data.Dataset):
    def __init__(self, utts: list, bgs: dict, seed: int, n: int, fixed: list | None = None, fam: dict | None = None, parts: bool = False) -> None:
        self.utts, self.bgs, self.seed, self.n, self.fixed, self.fam, self.parts = utts, bgs, seed, n, fixed, fam or {}, parts

    def ext(self, g: np.random.Generator, spec: dict | None) -> dict:
        fam, out = self.fam, {}
        if fam.get("floor"):
            fdb = spec["floor_db"] if spec and spec.get("floor_db") is not None else float(g.uniform(*fam.get("floor_db", FLOOR_DB)))
            out["floor"] = (self.bgs["floor"][int(g.integers(len(self.bgs["floor"])))], fdb)
        op_db = spec.get("op_db") if spec else None
        if fam.get("op") and (op_db is not None or (spec is None and g.random() < P_OP)):
            ev = []
            for _ in range(int(g.integers(OP_N[0], OP_N[1] + 1))):
                pool = self.bgs["clap"] if g.random() < P_CLAP else self.bgs["op"]
                dur = float(g.uniform(*OP_DUR))
                ev.append((pool[int(g.integers(len(pool)))], dur, int(g.integers(0, CROP * H - int(dur * SR))), op_db if op_db is not None else float(g.uniform(*OP_DB))))
            out["events"] = ev
        band = spec.get("band_hz") if spec else None
        if fam.get("band") and (band is not None or (spec is None and g.random() < P_BAND)):
            out["band"] = band if band is not None else float(np.exp(g.uniform(np.log(BAND_HZ[0]), np.log(BAND_HZ[1]))))
        return out

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        import soundfile as sf
        spec = self.fixed[i] if self.fixed is not None else None
        g = np.random.default_rng((self.seed, i))
        npz, wav = self.utts[spec["utt"] if spec else int(g.integers(len(self.utts)))]
        x = sf.read(str(wav), dtype="float32")[0]
        y, v, lf = target(npz)
        Tn = min(y.shape[1], len(x) // H)
        x, y, v, lf = x[: Tn * H], y[:, :Tn], v[:Tn], lf[:Tn]
        rms = float(np.sqrt((x.astype(np.float64) ** 2).mean()) + 1e-12)
        if spec:
            pre, post = spec["pre"], spec["post"]
        else:
            pre = int(g.uniform(0, PAD_MAX) * SR / H) if g.random() < P_PAD else 0
            post = int(g.uniform(0, PAD_MAX) * SR / H) if g.random() < P_PAD else 0
        post = max(post, CROP - (pre + Tn))
        xf = np.concatenate([np.zeros(pre * H, np.float32), x, np.zeros(post * H, np.float32)])
        padz = lambda a: np.concatenate([np.zeros(a.shape[:-1] + (pre,), a.dtype), a, np.zeros(a.shape[:-1] + (post,), a.dtype)], -1)
        yf, vf, lff = padz(y), padz(v), padz(lf)
        Tf = yf.shape[1]
        st = spec["st"] if spec else int(g.integers(0, Tf - CROP + 1))
        st = min(st, Tf - CROP)
        if spec:
            bg, snr, rev, gain = spec["bg"], spec["snr"], False, 0.0
            src = spec["src"]
        else:
            r, acc, bg = g.random(), 0.0, "none"
            for name, p in P_BG:
                acc += p
                if r < acc:
                    bg = name
                    break
            src = None
            if bg == "music":
                pool = self.bgs["music_fma"] if (self.bgs["music_fma"] and g.random() < P_FMA) else self.bgs["music_musan"]
                src = pool[int(g.integers(len(pool)))]
            elif bg == "noise":
                src = self.bgs["noise"][int(g.integers(len(self.bgs["noise"])))]
            snr = float(g.uniform(*SNR[bg])) if bg != "none" else 0.0
            rev, gain = bool(g.random() < P_REV), float(g.uniform(*GAIN))
        xc = xf[st * H:(st + CROP) * H]
        xn, sc, nc, *ec = mix_parts(xc, rms, bg, src, snr, g, rev, gain, self.ext(g, spec), with_ev=self.parts == "ev")
        out = {"x": torch.from_numpy(xn), "y": torch.from_numpy(yf[:, st:st + CROP].copy()), "v": torch.from_numpy(vf[st:st + CROP].copy()),
               "lf": torch.from_numpy(lff[st:st + CROP].copy()), "snr": torch.tensor(snr if bg != "none" else 99.0), "pad": torch.from_numpy(np.r_[np.ones(pre), np.zeros(Tn), np.ones(post)][st:st + CROP].astype(np.float32))}
        return out | ({"s": torch.from_numpy(sc), "n": torch.from_numpy(nc)} if self.parts else {}) | ({"e": torch.from_numpy(ec[0]), "f": torch.from_numpy(ec[1]), "b": torch.from_numpy(ec[2]), "bt": torch.tensor(1.0 if bg == "music" else 0.0)} if ec else {})


class Front(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        import torchaudio
        self.register_buffer("win", torch.hann_window(NFFT, periodic=False), persistent=False)
        lin = torch.zeros(NFFT // 2 + 1, N_LIN)
        lin[torch.arange(1, N_LIN + 1), torch.arange(N_LIN)] = 1.0
        mel = torchaudio.functional.melscale_fbanks(NFFT // 2 + 1, LIN_HZ, SR / 2, N_MEL, SR, norm=None, mel_scale="htk")
        self.register_buffer("fb", torch.cat([lin, mel], 1), persistent=False)
        self.register_buffer("win2", torch.hann_window(NFFT2, periodic=False), persistent=False)
        self.register_buffer("fb2", torchaudio.functional.melscale_fbanks(NFFT2 // 2 + 1, MEL2_LO, SR / 2, N_MEL2, SR, norm=None, mel_scale="htk"), persistent=False)
        assert float(self.fb.sum(0).min()) > 0 and float(self.fb2.sum(0).min()) > 0, "空の帯がある"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Tn = x.shape[-1] // H
        out = []
        for n, w, fb in ((NFFT, self.win, self.fb), (NFFT2, self.win2, self.fb2)):
            fr = F_.pad(x, (n - H, 0)).unfold(-1, n, H)[..., :Tn, :]
            Pw = torch.fft.rfft(fr * w, n=n).abs().pow(2)
            out.append(torch.log(Pw @ fb + 1e-6) / 4.0)
        return torch.cat(out, -1).transpose(1, 2)


class Block(nn.Module):
    def __init__(self, ch: int, d: int) -> None:
        super().__init__()
        self.d = d
        self.ln = nn.LayerNorm(ch)
        self.conv = nn.Conv1d(ch, ch, 3, dilation=d)
        self.out = nn.Conv1d(ch, ch, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        z = self.ln(h.transpose(1, 2)).transpose(1, 2)
        z = self.conv(F_.pad(z, (2 * self.d, 0)))
        return h + self.out(F_.gelu(z))


class Sep(nn.Module):
    def __init__(self, ch: int = 256) -> None:
        super().__init__()
        self.front = Front()
        self.inp = nn.Conv1d(D_IN, ch, 1)
        self.blocks = nn.ModuleList([Block(ch, d) for d in DIL])
        self.ln = nn.LayerNorm(ch)
        self.head = nn.Conv1d(ch, D_OUT, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(self.front(x))
        for b in self.blocks:
            h = b(h)
        return self.head(self.ln(h.transpose(1, 2)).transpose(1, 2))


def split_out(o: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return o[:, :29], o[:, 29], o[:, 30]


def loss_fn(o: torch.Tensor, b: dict) -> tuple[torch.Tensor, dict]:
    y, vl, lf = split_out(o)
    l_y = (y - b["y"]).abs().mean()
    l_v = F_.binary_cross_entropy_with_logits(vl, b["v"])
    m = b["v"]
    l_f = ((lf - b["lf"]).abs() * m).sum() / m.sum().clamp_min(1.0)
    return l_y + l_v + l_f, {"y": float(l_y), "v": float(l_v), "lf": float(l_f)}


@torch.no_grad()
def features(D: nn.Module, x: np.ndarray, dev: str) -> dict:
    Tn = len(x) // H
    o = D(torch.from_numpy(np.asarray(x[: Tn * H], np.float32))[None].to(dev))[0].float()
    y, vl, lf = o[:29], o[29], o[30]
    v = (vl > 0).float()
    y = torch.cat([y[:25], y[25:29] * v[None]], 0)
    lf = lf * v
    f0 = 200.0 * torch.exp(lf) * v
    env = T.y_env(y[None])[0].cpu().numpy()
    vn = v.cpu().numpy()
    return {"y": y.cpu().numpy().astype(np.float32), "lf": lf.cpu().numpy().astype(np.float32), "v": vn.astype(np.float32), "f0": f0.cpu().numpy().astype(np.float32),
            "vs": env[:, vn > 0.5].sum(1).astype(np.float32), "vn": int((vn > 0.5).sum())}


def load(path: Path, dev: str) -> nn.Module:
    st = torch.load(path, map_location="cpu", weights_only=False)
    D = Sep(st["cfg"]["ch"]).to(dev).eval()
    D.load_state_dict(st["D"])
    return D


def val_specs(utts: list, bgs: dict, n_utt: int, fam: dict | None = None) -> list:
    g = np.random.default_rng(777)
    specs = []
    pick = g.choice(len(utts), size=min(n_utt, len(utts)), replace=False)
    mus = bgs["music_fma"] + bgs["music_musan"]
    for k, ui in enumerate(pick):
        src = mus[int(g.integers(len(mus)))]
        for cond, snr, pad in (("s20", 20.0, 0), ("s10", 10.0, 0), ("s0", 0.0, 0), ("s-5", -5.0, 0), ("lead0", 0.0, int(2.0 * SR / H)), ("clean", 0.0, 0)):
            specs.append({"utt": int(ui), "cond": cond, "bg": "none" if cond == "clean" else "music", "src": src, "snr": snr, "pre": pad, "post": pad, "st": 0})
        if fam and fam.get("op"):
            specs.append({"utt": int(ui), "cond": "op0", "bg": "none", "src": src, "snr": 0.0, "pre": 0, "post": 0, "st": 0, "op_db": 0.0, "floor_db": -60.0})
            specs.append({"utt": int(ui), "cond": "oplead", "bg": "none", "src": src, "snr": 0.0, "pre": int(2.0 * SR / H), "post": int(2.0 * SR / H), "st": 0, "op_db": 5.0, "floor_db": -60.0})
        if fam and fam.get("band"):
            specs.append({"utt": int(ui), "cond": "lp8_s10", "bg": "music", "src": src, "snr": 10.0, "pre": 0, "post": 0, "st": 0, "band_hz": 8000.0, "floor_db": -60.0})
        if fam and fam.get("floor"):
            specs.append({"utt": int(ui), "cond": "fl35", "bg": "none", "src": src, "snr": 0.0, "pre": 0, "post": 0, "st": 0, "floor_db": -35.0})
    return specs


def unit_tests() -> None:
    D = Sep(32)
    x = torch.randn(1, CROP * H) * 0.1
    o = D(x)
    assert o.shape == (1, D_OUT, CROP), o.shape
    x2 = x.clone()
    t = 500
    x2[:, (t + 1) * H:] = torch.randn(1, x.shape[1] - (t + 1) * H)
    o2 = D(x2)
    assert torch.equal(o[..., : t + 1], o2[..., : t + 1]) or float((o[..., : t + 1] - o2[..., : t + 1]).abs().max()) < 1e-5, "先読みがある"
    assert float((o[..., t + 1:] - o2[..., t + 1:]).abs().max()) > 1e-4, "未来の書き換えで後ろが動かない(検査が無効)"
    z = np.zeros(48000, np.float32)
    f = P.analyze(z)
    assert float(np.abs(T.to_y(f["env"], f["per"], f["f0"])).max()) < 1e-6 and float((f["f0"] > 0).mean()) == 0.0, "無音の前段の特徴が 0 でない"
    print("unit tests ok(先読み 0・無音 = 0)", flush=True)


def make_pools(bs_: dict, fam: dict) -> tuple[dict, dict]:
    fma = set(json.loads(FMA.read_text())["pools"]["music"]) if FMA.exists() else set()
    pools = lambda s: {"music_fma": [p for p in bs_[s]["music"] if p in fma], "music_musan": [p for p in bs_[s]["music"] if p not in fma], "noise": bs_[s]["noise"]}
    ptr, pdv = pools("train"), pools("dev")
    if fam.get("wm"):
        idx = json.loads((WM / "index.json").read_text())
        miss = [q for pl in (ptr, pdv) for v in pl.values() for q in v if q not in idx]
        assert not miss, f"透かしの版が無い背景 {len(miss)}: {miss[:3]}"
        for pl in (ptr, pdv):
            for k in pl:
                pl[k] = [idx[q] for q in pl[k]]
    if fam:
        op = json.loads(OPER.read_text())["pools"]
        idx = json.loads((WM / "index.json").read_text())
        fl = sorted(str(q) for q in WM.glob("floor_*.wav"))
        cl = sorted(str(q) for q in WM.glob("clap_*.wav"))
        assert fl and cl and all(q in idx for q in op["train"] + op["dev"]), "床・合成の手拍子・操作者の音の透かしの版が無い"
        ptr |= {"floor": fl, "clap": cl[: len(cl) * 3 // 4], "op": [idx[q] for q in op["train"]]}
        pdv |= {"floor": fl, "clap": cl[len(cl) * 3 // 4:], "op": [idx[q] for q in op["dev"]]}
    if fam.get("fsd"):
        fd = json.loads(FSD_DEV.read_text())
        idx = json.loads((WM / "index.json").read_text())
        part = lambda f: "dev" if zlib.crc32(f"fsd|{f.get('uploader')}".encode()) % 5 == 0 else "train"
        for f in fd["files"]:
            assert f["path"] in idx, f"透かしの版が無い FSD50K の手の音: {f['path']}"
            pl = pdv if part(f) == "dev" else ptr
            pl["op"].append(idx[f["path"]])
            if f["clap"]:
                pl["clap"].append(idx[f["path"]])
    return ptr, pdv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--ch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prereg", default="results/snv_i14/prereg.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fam", default="i14", choices=sorted(FAMS))
    a = ap.parse_args()
    fam = FAMS[a.fam]
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    if not a.smoke and not (ROOT / a.prereg).exists():
        print(f"{a.prereg} が無い: 起動しない", flush=True)
        return 1
    print("prereg", ROOT / a.prereg, "paper current/paper_snv.md", flush=True)
    torch.manual_seed(a.seed)
    unit_tests()
    split = json.loads((ROOT / "results/snv_i8_m13/split.json").read_text())
    tr, va = utterances(split["train"]), utterances(split["val"])
    bs_ = bg_split()
    (out / "split.json").write_text(json.dumps(split))
    (out / "bg_split.json").write_text(json.dumps(bs_, indent=0))
    ptr, pdv = make_pools(bs_, fam)
    print("fam", a.fam, fam, "utts train", len(tr), "val", len(va), "bg train", {k: len(v) for k, v in ptr.items()}, "dev", {k: len(v) for k, v in pdv.items()}, flush=True)
    assert ptr["music_musan"] and ptr["noise"] and pdv["music_musan"], "背景の組が空"
    dev = a.device
    D = Sep(a.ch).to(dev)
    print("D params", sum(p.numel() for p in D.parameters()), "受容野", 1 + 2 * sum(DIL), "frames", flush=True)
    steps = 200 if a.smoke else a.steps
    opt = torch.optim.AdamW(D.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    dl = torch.utils.data.DataLoader(Mix(tr, ptr, a.seed, steps * a.bs, fam=fam), batch_size=a.bs, num_workers=a.workers, worker_init_fn=lambda i: torch.set_num_threads(1),
                                     persistent_workers=a.workers > 0, prefetch_factor=4 if a.workers else None, drop_last=True)
    vspecs = val_specs(va, pdv, 8 if a.smoke else 24, fam)
    vdl = torch.utils.data.DataLoader(Mix(va, pdv, 999, len(vspecs), vspecs, fam), batch_size=6, num_workers=2)
    vbatches = list(vdl)

    @torch.no_grad()
    def evaluate() -> dict:
        D.eval()
        conds = [s["cond"] for s in vspecs]
        rows: dict = {}
        k = 0
        for vb in vbatches:
            b = {kk: x.to(dev) for kk, x in vb.items()}
            y, vl, lf = split_out(D(b["x"]))
            for i in range(b["x"].shape[0]):
                c = conds[k]
                k += 1
                v = b["v"][i] > 0.5
                vh = vl[i] > 0
                both = v & vh
                cents = (1200 / math.log(2)) * (lf[i] - b["lf"][i]).abs()[both]
                pad = b["pad"][i] > 0.5
                r = rows.setdefault(c, {"y": [], "vacc": [], "cents": [], "pad_fv": []})
                r["y"].append(float((y[i] - b["y"][i]).abs().mean()))
                r["vacc"].append(float((vh == v).float().mean()))
                if both.any():
                    r["cents"].append(float(cents.median()))
                if pad.any():
                    r["pad_fv"].append(float(vh[pad].float().mean()))
        D.train()
        return {c: {kk: (round(float(np.median(x)), 4) if x else None) for kk, x in r.items()} for c, r in rows.items()}

    hist, t0, it = [], time.time(), iter(dl)
    D.train()
    for step in range(1, steps + 1):
        vb = next(it)
        b = {k: x.to(dev, non_blocking=True) for k, x in vb.items()}
        loss, parts = loss_fn(D(b["x"]), b)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.nn.utils.clip_grad_norm_(D.parameters(), 1.0))
        opt.step()
        sched.step()
        if step % 100 == 0 or step == 1:
            print(f"step {step} loss {float(loss):.4f} y {parts['y']:.4f} v {parts['v']:.4f} lf {parts['lf']:.4f} gn {gn:.2f} {(time.time() - t0) / step:.3f}s/step", flush=True)
        if step % 2000 == 0 or step == steps:
            ev = evaluate()
            hist.append({"step": step, "val": ev})
            print("val", step, json.dumps(ev), flush=True)
            st = {"D": D.state_dict(), "cfg": {"ch": a.ch, "dil": list(DIL), "nfft": NFFT, "n_lin": N_LIN, "lin_hz": LIN_HZ, "n_mel": N_MEL, "nfft2": NFFT2, "n_mel2": N_MEL2, "mel2_lo": MEL2_LO, "hop": H, "fam": a.fam}, "step": step, "tag": a.tag, "prereg": a.prereg}
            torch.save(st, out / "last.pt.tmp")
            (out / "last.pt.tmp").replace(out / "last.pt")
            if step == 10000:
                torch.save(st, out / "step10k.pt")
            (out / "train_log.json").write_text(json.dumps(hist))
    print("done", steps, f"{time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
