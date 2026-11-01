"""段 1 の基準 rev1(current/e2e_ppg.md §7・オーナー承認 10-18): 内容(C1 の事後)・log f0・有声・c0(操作者の音量)・目標の表(内容で索引した包絡 = Σ_k p_k T_k)・目標の周期性の定数 → 波形。
操作者の包絡・周期性は入力に無い(声質の通り道を構造で断つ)。出力部 rvoc3hi の重みから始め(pre の事後の列 = 0 で始める = 0 step は表だけの条件)、
実女声の自己再構成を波形の損失 + MPD/MRD の GAN で端から端まで学習する。
評価: 保留の実女声 30 人の自己再構成(SR・diag_e2e1 と同じ組・写し合成 CP と比べる)と dev の男声 → 女声 60 組(PV)を、Rust の前段の書き出し
(.front の C1・.cond の c0 / f0 / 有声)と Rust の登録の表で鳴らし eval_v1perp.py。
    uv run python train_ppgvoc.py --tag diag_ppgvoc1 [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import pae as PA
import rvoc as R
import train_rvoc as TR
from fsync import fline
from train_nvoc import MRDLog
from train_s1_1 import MEL_SPECS, build_mel, logmel_l1

ROOT = Path(__file__).resolve().parent.parent
H = Path(os.environ.get("LVC_DEV", "/home/kojirotanaka/lightvc_scratch_20261013"))
MANIFEST = ROOT / "data/rvoc_f0hi/manifest.json"
TAB = ROOT / "data/rvoc_tables_full"
INIT = ROOT / "results/rvoc3hi"
K = 200
D_PPG = R.D_COND + K
W = 200
REF_SEC = 20.0


def dev_excluded() -> set:
    excl = set((H / "excl.txt").read_text().split())
    for f in list(H.glob("*/devset.json")) + [H / "eval_targets.json"]:
        j = json.loads(f.read_text())
        excl |= set(j["targets"])
        excl |= {p["spk"] for p in j.get("pairs", []) if isinstance(p, dict) and "spk" in p}
    excl |= {r["id"] for r in json.loads((ROOT / "results/conv_p0/d237.json").read_text())["rows"] if r["grp"] == "REAL_POOL"}
    return excl


def e2e_held() -> list:
    """diag_e2e1 の保留 30 人(d274_e2e.py と同じ手順・--extra なし)。[(話者, クリップ A, クリップ B)]。"""
    E2 = H / "e2e"
    data: dict = {}
    for m in json.loads((E2 / "meta.json").read_text())["fem"]:
        c, f = E2 / "fem" / f"{m['key']}.cond", E2 / "fem" / f"{m['key']}.front"
        if c.exists() and min(c.stat().st_size // (31 * 4), f.stat().st_size // (224 * 4)) > W + 10:
            data.setdefault(m["spk"], []).append(m["key"])
    spk = sorted(k for k, v in data.items() if len(v) >= 2)
    random.Random(274).shuffle(spk)
    return [(k, data[k][0], data[k][1]) for k in spk[:30]]


def table_files(files: list) -> tuple:
    """build_tables_rvoc.py の参照の選び方(話者の f0 のある全行を名前順・20 s まで)。(使ったファイル, 使った出典)。"""
    tot, used, srcs = 0.0, set(), set()
    for wav, src, dur in sorted(files):
        if tot >= REF_SEC:
            break
        n = int(dur * N.SR) // N.HOP
        if n < 50:
            continue
        used.add(wav)
        srcs.add(src.split("_")[0])
        tot += n * N.HOP / N.SR
    return used, srcs


def load_rows(held_spk: set, tab_real: bool = False) -> tuple:
    """tab_real = True: 表 = data/rvoc_tables_real(build_tables_rvoc.py --real_only で作り直した実音声だけの表)・表のファイルも実音声の行だけで数える。"""
    rows = json.loads(MANIFEST.read_text())["rows"]
    by: dict = {}
    for r in rows:
        if r.get("f0") and (not tab_real or r["src"].startswith("real")):
            by.setdefault(r["spk"], []).append((r["wav"], r["src"], r["dur"]))
    tabs = {p.stem for p in TAB.glob("*.npz")}
    ex_p, ex_s = TR.hi_eval_paths(), TR.held_speakers()
    dx = dev_excluded()
    real = [r for r in rows if r["keep"] and r["src"].startswith("real_female")]
    n0 = len({r["spk"] for r in real})
    real = [r for r in real if not TR.eval_excluded(r, ex_p, ex_s) and r["spk"] not in dx and r["spk"] not in held_spk]
    n1 = len({r["spk"] for r in real})
    real = [r for r in real if r["spk"] in tabs]
    n2 = len({r["spk"] for r in real})
    tf = {s: table_files(by[s]) for s in {r["spk"] for r in real}}
    real = [r for r in real if tf[r["spk"]][1] == {"real"} and r["wav"] not in tf[r["spk"]][0]]
    n3 = len({r["spk"] for r in real})
    counts = {"real_female_spk": n0, "after_eval_dev_held_excl": n1, "with_table": n2, "table_real_only_and_not_table_file": n3, "files": len(real),
              "hours": round(sum(r["dur"] for r in real) / 3600, 1)}
    spks = sorted({r["spk"] for r in real})
    return [(r["wav"], str(ROOT / r["f0"]), r["sr"], r["dur"], r["spk"]) for r in real], spks, counts


def load_sm_rows(manifest: str, tab: Path) -> tuple:
    """腕 64 の合成の男声の自己再構成の行: 表のある話者の keep な行から、表に使ったファイル(table_files・build_tables_rvoc.py と同じ選び方)を除く。"""
    rows = json.loads(Path(manifest).read_text())["rows"]
    by: dict = {}
    for r in rows:
        if r.get("f0"):
            by.setdefault(r["spk"], []).append((r["wav"], r["src"], r["dur"]))
    tabs = {q.stem for q in tab.glob("*.npz")}
    keep = [r for r in rows if r.get("keep") and r["spk"] in tabs]
    tf = {s: table_files(by[s])[0] for s in {r["spk"] for r in keep}}
    keep = [r for r in keep if r["wav"] not in tf[r["spk"]]]
    spks = sorted({r["spk"] for r in keep})
    counts = {"files_all": len(rows), "spk_with_table": len(tabs), "train_files": len(keep), "spk": len(spks), "hours": round(sum(r["dur"] for r in keep) / 3600, 2)}
    return [(r["wav"], str(ROOT / r["f0"]), r["sr"], r["dur"], r["spk"]) for r in keep], spks, counts


def cond_ppg(p: torch.Tensor, c0n: torch.Tensor, lf: torch.Tensor, vv: torch.Tensor, tab: torch.Tensor, per: torch.Tensor) -> torch.Tensor:
    """[c0n, Σ p_k T_k / 10 (24), log f0, 有声, 周期性の定数 × 有声 (4), 事後 (200)]。最初の 31 行は rvoc3hi の条件と同じ並び。"""
    env = torch.einsum("bdk,bkt->bdt", tab, p) / 10
    pr = per.clamp(0.0, 1.5) * vv if per.dim() == 3 else per[:, :, None] * vv
    return torch.cat([c0n, env, lf, vv, pr, p], 1)


CTX = {"on": False, "adapt": False, "ch": 192}
OPSLOW: dict = {"on": False, "tau": 4.0, "gain": 1.5}
C0LOW: dict = {"on": False, "v": None, "a": 0.78}
PERDYN: dict = {"on": False}
CVIN: dict = {"on": False, "path": "", "d": 64}
H1FIX: dict = {"on": False, "sigma": 0.0, "ft": 400.0, "taper": False}
C0TAB: dict = {"on": False, "tab": None, "k": 5}
C2CONST: dict = {"on": False, "v": None}


def c0tab_frames(p: torch.Tensor) -> torch.Tensor:
    """中立の c0(diag_ppgvoc65・paper_mf E-A1t): 事後の argmax の単位ごとの学習の女声の c0 の平均を因果な k フレームの平均に(先頭は最初の値の複写)[B, 1, T]。推論の流れ = c0tab_np と同じ式。"""
    v = C0TAB["tab"].to(p.device)[p.argmax(1)][:, None]
    return F.avg_pool1d(F.pad(v, (C0TAB["k"] - 1, 0), mode="replicate"), C0TAB["k"], 1)


def c0tab_np(u: np.ndarray, tab: np.ndarray, k: int = 5) -> np.ndarray:
    v = tab[u].astype(np.float64)
    return np.convolve(np.concatenate([np.full(k - 1, v[0]), v]), np.ones(k) / k, mode="valid").astype(np.float32)


F0V: dict = {"on": False, "vtab": None, "a": 1.0 - float(np.exp(-5.0 / 15.0)), "jump": 7.0 / 12.0 * float(np.log(2.0)), "hold": 3, "gap": 10, "k": 5}


def f0v_np(f0: np.ndarray, u: np.ndarray, vtab: np.ndarray) -> np.ndarray:
    """中立の f0・有声(diag_ppgvoc67・paper_mf E-A1b): 有声 = 単位ごとの有声の率の表を因果な k フレームの平均 > 0.5・f0 = 対数の一次の EMA(τ 15 ms)・7 半音を超える飛びは
    hold フレームまで前の値を保持してから乗り換え・gap フレーム以上の無声の後は入り直し・有声で入力の f0 が無いところは保持。推論の流れ(d700_streams.py)と同じ式。f0 [T] Hz(0 = 無声)→ [T]。"""
    k = F0V["k"]
    r = vtab[u].astype(np.float64)
    vm = np.convolve(np.concatenate([np.full(k - 1, r[0]), r]), np.ones(k) / k, mode="valid") > 0.5
    out = np.zeros(len(f0), np.float32)
    y, c, g = None, 0, 0
    for t in range(len(f0)):
        if f0[t] > 0:
            x = float(np.log(f0[t]))
            if y is None or g >= F0V["gap"]:
                y, c = x, 0
            elif abs(x - y) > F0V["jump"] and c < F0V["hold"]:
                c += 1
            elif abs(x - y) > F0V["jump"]:
                y, c = x, 0
            else:
                y, c = y + F0V["a"] * (x - y), 0
            g = 0
        else:
            g += 1
        if vm[t] and y is not None:
            out[t] = float(np.exp(y))
    return out


def f0v_t(f0: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """f0 [B, T] Hz・事後 p [B, K, T] → 中立の f0 [B, T](f0v_np を行ごとに)。"""
    u = p.argmax(1).cpu().numpy()
    f = f0.detach().float().cpu().numpy()
    return torch.from_numpy(np.stack([f0v_np(f[i], u[i], F0V["vtab"]) for i in range(len(f))])).to(f0.device, f0.dtype)


def h1fix_delta(lf: torch.Tensor, vv: torch.Tensor, sigma: float, ft: float, taper: bool = False) -> torch.Tensor:
    """推論だけ(D611): 有声の枠で、目標の声域 ft からいまの f0 までの mel の帯に σ dB/oct の傾き(f0 より下は平ら・taper なら f0/2 へ直線で 0・f ≥ ft は 0)を
    表の包絡の行(c1..c24 / 10)の差として返す。lf・vv [B, 1, T] → [B, 24, T]。因果(そのフレームの f0・有声と登録の定数だけ)。"""
    if "D" not in H1FIX:
        import librosa
        import zsvc as Z
        H1FIX["D"] = Z.dct_mat(N.N_MEL)[1:25].double()
        H1FIX["fc"] = torch.from_numpy(librosa.mel_frequencies(N.N_MEL + 2, fmin=0.0, fmax=N.FMAX)[1:-1].copy()).double()
    D, fc = H1FIX["D"].to(lf.device), H1FIX["fc"].to(lf.device)
    f0 = (200.0 * torch.exp(lf.double())).clamp(max=ft)
    f = torch.maximum(fc[None, :, None], f0).clamp(max=ft)
    dl = sigma * torch.log2(ft / f) * (float(np.log(10)) / 20) * (vv > 0.5).double()
    if taper:
        dl = dl * ((fc[None, :, None] - f0 / 2) / (f0 / 2)).clamp(0.0, 1.0)
    return (torch.einsum("km,bmt->bkt", D, dl) / 10).to(lf.dtype)


def cvin_setup(path: str, dev: str) -> None:
    """連続の内容の特徴の入力。.npz = 非因果の教師(diag_ppgvoc49・診断のみ: ContentVec 最終層・L2 正規化 → 主成分 64・分散で白色化・D502)。
    .pt = 因果な生徒 C2(train_c2.py・C1 と同じ骨格と入力の規約・製品の経路)。"""
    if path.endswith(".pt"):
        import c1_content as CC
        s2 = torch.load(path, map_location="cpu", weights_only=False)
        net = CC.C1(s2["cfg"]["k"], s2["cfg"]["ch"], tuple(s2["cfg"]["dils"])).to(dev)
        net.load_state_dict(s2["net"])
        CVIN.update(on=True, path=path, d=int(s2["cfg"]["k"]), src="c2", net=net.eval().requires_grad_(False), front=CC.MelFront().to(dev))
        return
    import torchaudio
    from transformers import HubertModel
    z = np.load(path)
    CVIN.update(on=True, path=path, d=int(z["comp"].shape[0]), mean=torch.from_numpy(z["mean"]).float().to(dev),
                comp=torch.from_numpy(z["comp"] / np.sqrt(z["ev"])[:, None]).float().to(dev),
                m=HubertModel.from_pretrained("lengyue233/content-vec-best").to(dev).eval().requires_grad_(False),
                rs=torchaudio.transforms.Resample(N.SR, 16000).to(dev))


@torch.no_grad()
def cvin_frames(a48: torch.Tensor, T: int) -> torch.Tensor:
    """a48 [B, L](C1 と同じ規約の音声 = 先頭 WIN − HOP は文脈)→ 最後の T フレームの連続の内容の特徴 [B, d, T]。教師の時刻の規約は train_c1.teacher_post と同じ(15 ms の遅れ・右は 0.5 s の無音)。"""
    import c1_content as CC
    if CVIN.get("src") == "c2":
        return CVIN["net"](CVIN["front"](a48))[..., -T:]
    cw = N.WIN - N.HOP
    xp = torch.cat([a48, a48.new_zeros(a48.shape[0], 24000)], 1)
    h = F.normalize(CVIN["m"](CVIN["rs"](xp)).last_hidden_state, dim=-1)
    zz = (h - CVIN["mean"]) @ CVIN["comp"].T
    return CC.teacher_targets(zz, pre48=cw, T=(a48.shape[1] - cw) // N.HOP)[..., -T:]


def harm_l1(y: torch.Tensor, t: torch.Tensor, f0: torch.Tensor, nh: int = 4) -> tuple:
    """低い倍音の対数振幅の L1(D607・diag_ppgvoc62): Hann 2048・hop 240 の STFT で h·f0(h = 1..nh・最寄りの bin ±1 の最大)を出力 y と実録音 t で比べる。
    f0 [B, T] は y のフレームに合わせたもの。前後のフレームがともに有声で f0 の変化 3% 未満のフレームだけ。返り値 (損失, 有効な枠の割合, 出力と目標の H1 − H2 の平均 dB)。"""
    win = torch.hann_window(2048, device=y.device)
    Y = torch.stft(y.float(), 2048, N.HOP, window=win, return_complex=True).abs()
    Tm = torch.stft(t.float(), 2048, N.HOP, window=win, return_complex=True).abs()
    n = min(Y.shape[-1], f0.shape[1])
    fa = f0[:, :n].float()
    fp = torch.cat([fa[:, :1], fa[:, :-1]], 1)
    ok = (fa > 0) & (fp > 0) & (torch.log(fa.clamp(min=1.0) / fp.clamp(min=1.0)).abs() < 0.03)
    fm = 0.5 * (fa + fp)
    ls, amp = [], []
    for h in range(1, nh + 1):
        idx = (h * fm / (N.SR / 2048)).round().long().clamp(1, Y.shape[1] - 2)
        cand = torch.stack([idx - 1, idx, idx + 1], 1)
        ay = torch.stack([Y[..., :n].gather(1, cand[:, j, None])[:, 0] for j in range(3)]).amax(0)
        at = torch.stack([Tm[..., :n].gather(1, cand[:, j, None])[:, 0] for j in range(3)]).amax(0)
        ls.append((torch.log(ay + 1e-4) - torch.log(at + 1e-4)).abs())
        amp.append((ay.detach(), at))
    m = ok.float()
    L = (torch.stack(ls) * m[None]).sum() / (m.sum() * nh + 1e-6)
    with torch.no_grad():
        d = lambda a_, b_: float(((20 * torch.log10(a_ + 1e-4) - 20 * torch.log10(b_ + 1e-4)) * m).sum() / (m.sum() + 1e-6))
        h12 = (d(amp[0][0], amp[1][0]), d(amp[0][1], amp[1][1]))
    return L, float(m.mean()), h12


def c0low_t(c: torch.Tensor) -> torch.Tensor:
    """音量の条件を低い帯域(100 Hz〜4 kHz)の水準に(diag_ppgvoc47): c [B, ≥25, T] → 行 0 = a·(c0n + Σ_{r≥1} v_r·行 r)。他の行はそのまま。"""
    v = torch.as_tensor(C0LOW["v"], dtype=c.dtype, device=c.device)
    out = c.clone()
    out[:, 0] = C0LOW["a"] * (c[:, 0] + torch.einsum("r,brt->bt", v[1:25], c[:, 1:25]))
    return out


def c0low_np(c: np.ndarray) -> np.ndarray:
    """Rust の書き出しの条件 c [T, 31] に同じ変換。"""
    v = np.asarray(C0LOW["v"], dtype=np.float32)
    out = c.copy()
    out[:, 0] = C0LOW["a"] * (c[:, 0] + c[:, 1:25] @ v[1:25])
    return out
SLOW: dict = {"on": False, "sigma": 1.0, "seed": 0, "causal": False}
MID: dict = {"on": False, "sigma": 1.0, "causal": False}


def minphase_fir(amp: torch.Tensor) -> torch.Tensor:
    """振幅スペクトル [k, L/2+1] → 最小位相の FIR [k, L](実ケプストラムの折り返し)= 白色雑音を因果に濾過して同じ形のスペクトルを作る(Rust と同じ作り方)。"""
    L = 2 * (amp.shape[-1] - 1)
    la = torch.log(torch.maximum(amp, amp.amax(-1, keepdim=True) * 1e-4))
    c = torch.fft.irfft(la, L)
    w = torch.zeros(L)
    w[0], w[1:L // 2], w[L // 2] = 1.0, 2.0, 1.0
    return torch.fft.irfft(torch.exp(torch.fft.rfft(c * w)), L)


def slow_load(path: str, sigma: float = 1.0, D: dict | None = None) -> None:
    """遅い音色の揺らぎ(D339 の固定の統計: 基底 B [24, k]・成分ごとの平均の振幅スペクトル amp [k, SEG/2+1]・成分の標準偏差 tgt [k])を読む。D = MID なら中間の帯(D364 の統計)。"""
    D = SLOW if D is None else D
    z = np.load(path)
    D.update(on=True, path=str(path), sigma=float(sigma), B=torch.from_numpy(z["B"]).float(), amp=torch.from_numpy(z["amp"]).float(), tgt=torch.from_numpy(z["tgt"]).float())
    h = minphase_fir(D["amp"])
    D["fir"] = h
    D["fir_gain"] = D["tgt"] / h.pow(2).sum(-1).sqrt()


GAMMA64 = 0x9E3779B97F4A7C15


def splitmix_normals(seed: int, n: int, start: int = 0) -> np.ndarray:
    """共通の乱数(Rust の rvoc.rs と同じ): 一様 q 番目 = splitmix64(seed + (q+1)·γ) の上位 53 bit + 0.5 → Box-Muller の対(偶数 = r cos・奇数 = r sin)。start = 正規の通し番号の開始(偶数)。float64。"""
    q = np.arange(start + 1, start + n + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = np.uint64(seed) + q * np.uint64(GAMMA64)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    u = ((z >> np.uint64(11)).astype(np.float64) + 0.5) * 2.0 ** -53
    r, th = np.sqrt(-2.0 * np.log(u[0::2])), 2.0 * np.pi * u[1::2]
    out = np.empty(n)
    out[0::2], out[1::2] = r * np.cos(th), r * np.sin(th)
    return out


def slow_noise_prng(T: int, seed: int, D: dict | None = None) -> np.ndarray:
    """製品の経路と同じ遅い揺らぎ [24, T](float32): 共通の乱数を時間順(フレームごとに k 成分)に引き、最初の L−1 フレームで FIR を満たしてから(定常の開始)、
    成分ごとに最小位相 FIR で因果に濾過 × 利得 → 基底 × σ。"""
    D = SLOW if D is None else D
    h = D["fir"].double().numpy()
    k, L = h.shape
    e = splitmix_normals(seed, (T + L - 1) * k).reshape(T + L - 1, k).T
    w = np.stack([np.convolve(e[j], h[j], "valid") for j in range(k)]) * D["fir_gain"].double().numpy()[:, None]
    return (D["sigma"] * (D["B"].double().numpy() @ w)).astype(np.float32)


def opslow_causal(r: torch.Tensor, tau_s: float) -> torch.Tensor:
    """操作者の遅い音色の揺らぎ(因果・D451): r = 操作者の包絡 − 操作者の表の引き [n, 24, T] → 過去 0.5 s(100 フレーム)の平均 − 時定数 tau_s の指数移動平均 → SLOW の基底へ射影。先読み 0・発話の統計なし。"""
    s = F.avg_pool1d(F.pad(r, (99, 0), mode="replicate"), 100, 1)
    a = float(np.exp(-1.0 / (tau_s * 200.0)))
    m = torch.empty_like(s)
    acc = s[..., 0]
    for t in range(s.shape[-1]):
        acc = a * acc + (1 - a) * s[..., t]
        m[..., t] = acc
    B = SLOW["B"].to(r.device)
    return torch.einsum("dk,nkt->ndt", B, torch.einsum("dk,ndt->nkt", B, s - m))


def opslow_train(r: torch.Tensor, tau_s: float) -> torch.Tensor:
    """学習用の opslow_causal(D451・diag_ppgvoc45): 過去 0.5 s の平均 s − 指数移動平均。区間が短い(2.8 s)ので移動平均の初期値を区間の s の平均にする(推論の収束した状態の近似・学習の中だけ)。行列で一度に計算。"""
    s = F.avg_pool1d(F.pad(r, (99, 0), mode="replicate"), 100, 1)
    T = s.shape[-1]
    a = float(np.exp(-1.0 / (tau_s * 200.0)))
    t = torch.arange(T, device=r.device, dtype=s.dtype)
    W = torch.where(t[:, None] >= t[None, :], (1 - a) * a ** (t[:, None] - t[None, :]), torch.zeros((), device=r.device, dtype=s.dtype))
    m = torch.einsum("ndj,tj->ndt", s, W) + (a ** (t + 1))[None, None] * s.mean(-1, keepdim=True)
    B = SLOW["B"].to(r.device)
    return torch.einsum("dk,nkt->ndt", B, torch.einsum("dk,ndt->nkt", B, s - m))


def op_tab(set_dir: Path, key: str) -> np.ndarray:
    """組の置き場と発話の名前(pPPPP_tTTT…)→ 操作者の登録の表(OPSLOW の描画に使う)。"""
    return load_enroll(set_dir / "d85_work" / f"op_{key[1:5]}")[0]


def slow_from_ckpt(st: dict) -> None:
    """ckpt の slow(統計のパス)・slow_sigma・slow_causal(製品の経路と同じ因果な共通の乱数)を SLOW に入れる。opslow_tau があれば遅い揺らぎの源を操作者の揺らぎ(OPSLOW・D451)にする。"""
    OPSLOW.update(on=bool(st.get("opslow_tau")), tau=float(st.get("opslow_tau") or 4.0), gain=float(st.get("opslow_gain", 1.5)))
    C0LOW.update(on=bool(st.get("c0low")), v=st.get("c0low_v"), a=float(st.get("c0low_a", 0.78)))
    PERDYN.update(on=bool(st.get("per_dyn")))
    if st.get("slow"):
        slow_load(st["slow"], st.get("slow_sigma", 1.0))
        SLOW["causal"] = bool(st.get("slow_causal", False))
    else:
        SLOW["on"] = False
    if st.get("mid"):
        slow_load(st["mid"], st.get("mid_sigma", 1.0), MID)
        MID["causal"] = SLOW["causal"]
    else:
        MID["on"] = False


def slow_noise(n: int, T: int, dev, g: torch.Generator | None = None, D: dict | None = None) -> torch.Tensor:
    """乱数の遅い揺らぎ [n, 24, T]: 成分ごとに白色雑音へ平均の振幅スペクトルをかけ(乱数の位相・SEG ごと)・標準偏差を学習の揺らぎにそろえ・基底で 24 次元へ・σ 倍。入力に依らない(発話の統計なし)。
    SLOW["causal"]: 白色雑音を最小位相の FIR で因果に濾過(定常の標準偏差 = tgt・製品の経路と同じ作り方・先読み 0)。"""
    D = SLOW if D is None else D
    B, amp, tgt = D["B"], D["amp"], D["tgt"]
    if D.get("causal"):
        h = D["fir"]
        L = h.shape[-1]
        e = torch.randn(n, T + L - 1, h.shape[0], generator=g).transpose(1, 2)
        w = F.conv1d(e, h.flip(-1)[:, None], groups=h.shape[0]) * D["fir_gain"][None, :, None]
        return (D["sigma"] * torch.einsum("dk,nkt->ndt", B, w)).to(dev)
    L = 2 * (amp.shape[-1] - 1)
    ns = -(-T // L)
    ph = torch.rand(n, amp.shape[0], ns, amp.shape[-1], generator=g)
    w = torch.fft.irfft(amp[None, :, None] * torch.exp(2j * torch.pi * ph), L).reshape(n, amp.shape[0], ns * L)[..., :T]
    w = w / (w.std(-1, keepdim=True) + 1e-9) * tgt[None, :, None]
    return (D["sigma"] * torch.einsum("dk,nkt->ndt", B, w)).to(dev)


def slow_real(env: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """学習の目標の本物の遅い揺らぎ [n, 24, T]: (包絡 − 表の包絡)の 0.5 s の移動平均 − 区間の平均 → 基底へ射影して戻す(D339 と同じ定義)。"""
    s = F.avg_pool1d(F.pad(env - base, (50, 49), mode="replicate"), 100, 1)
    s = s - s.mean(-1, keepdim=True)
    B = SLOW["B"].to(env.device)
    return torch.einsum("dk,nkt->ndt", B, torch.einsum("dk,ndt->nkt", B, s))


def mid_real(env: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
    """学習の目標の本物の中間の帯の動き [n, 24, T]: (包絡 − 表の包絡)の 0.15 s 平均 − 0.5 s 平均(約 1〜3 Hz・D363)→ 中間の帯の基底へ射影して戻す(d339_slownoise.BAND = (100, 30) と同じ定義)。"""
    r = env - base
    m = F.avg_pool1d(F.pad(r, (15, 14), mode="replicate"), 30, 1) - F.avg_pool1d(F.pad(r, (50, 49), mode="replicate"), 100, 1)
    B = MID["B"].to(env.device)
    return torch.einsum("dk,nkt->ndt", B, torch.einsum("dk,ndt->nkt", B, m))


class AdaptP(torch.nn.Module):
    """事後のアダプタ(因果・先読み 0): p' = softmax(log p + g(log p))・g の最後の層は 0 始め = 最初は恒等。男女の判別器に逆らわせて男声と女声の事後を区別できなくする(不変性正則化)。"""

    def __init__(self, ch: int = 128):
        super().__init__()
        self.a = R.CConv(K, ch, 3)
        self.b = torch.nn.Conv1d(ch, K, 1)
        torch.nn.init.zeros_(self.b.weight)
        torch.nn.init.zeros_(self.b.bias)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        lp = torch.log(p.clamp(min=1e-6))
        return (lp + self.b(F.leaky_relu(self.a(lp), 0.1))).softmax(1)


class GenderD(torch.nn.Module):
    """事後の系列から男声(1)・女声(0)を当てる判別器(因果な畳み込み + 時間の平均)。学習だけ。"""

    def __init__(self, ch: int = 128):
        super().__init__()
        self.c1 = R.CConv(K, ch, 3)
        self.c2 = R.CConv(ch, ch, 3, 2)
        self.o = torch.nn.Linear(ch, 1)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.c2(F.leaky_relu(self.c1(torch.log(p.clamp(min=1e-6))), 0.2)), 0.2)
        return self.o(h.mean(-1))[:, 0]


class CtxEnc(torch.nn.Module):
    """因果な文脈の符号器(左寄せ・先読み 0): [事後 K・c0n・log f0・有声・表の包絡 24] → 包絡のずれ 24(最後の層は 0 で始める)。
    膨張 1..64・カーネル 3 = 受容野 2·(1+2+…+64)+1 = 255 フレーム ≈ 1.27 秒(D286: 生成の欠けは 2〜8 秒の窓で大きい)。"""

    def __init__(self, ch: int = 192, dils: tuple = (1, 2, 4, 8, 16, 32, 64)):
        super().__init__()
        self.inp = R.CConv(K + 3 + 24, ch, 1)
        self.blocks = torch.nn.ModuleList(R.CConv(ch, ch, 3, d) for d in dils)
        self.out = torch.nn.Conv1d(ch, 24, 1)
        torch.nn.init.zeros_(self.out.weight)
        torch.nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(F.leaky_relu(h, 0.1))
        return self.out(F.leaky_relu(h, 0.1))


class TabVoc(R.RVoc):
    """話者 = 目標の登録の表(内容で索引)と周期性の定数。spk = (表 [B, 24, K], 周期性 [B, 4])。CTX["on"] なら文脈の符号器が包絡のずれを足す。"""

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        if CTX["on"]:
            self.ctx = CtxEnc(CTX["ch"])
        if CTX["adapt"]:
            self.adapt = AdaptP()

    def make(self, p: torch.Tensor, c0n: torch.Tensor, lf: torch.Tensor, vv: torch.Tensor, spk: tuple, z: torch.Tensor | None = None, zm: torch.Tensor | None = None, cvf: torch.Tensor | None = None) -> torch.Tensor:
        """cvf = 連続の内容の特徴 [n, d, T](CVIN["on"] のとき条件の末尾の行・無ければ 0)。z = 遅い音色の揺らぎ [n, 24, T](学習の自己再構成 = 本物)。SLOW["on"] で z が無ければ乱数の揺らぎ(評価・推論は長さで種を固定)。"""
        if hasattr(self, "adapt"):
            p = self.adapt(p)
        if C0TAB["on"]:
            c0n = c0tab_frames(p)
        if C2CONST["on"]:
            cvf = C2CONST["v"].to(p.device).view(1, -1, 1).expand(p.shape[0], -1, p.shape[-1])
        cond = cond_ppg(p, c0n, lf, vv, *spk)
        if H1FIX["on"]:
            cond = torch.cat([cond[:, :1], cond[:, 1:25] + h1fix_delta(lf, vv, H1FIX["sigma"], H1FIX["ft"], H1FIX["taper"]), cond[:, 25:]], 1)
        if hasattr(self, "ctx"):
            d = self.ctx(torch.cat([p, c0n, lf, vv, cond[:, 1:25]], 1))
            cond = torch.cat([cond[:, :1], cond[:, 1:25] + d, cond[:, 25:]], 1)
        if SLOW["on"]:
            if z is None and SLOW.get("causal") and not self.training:
                sd = SLOW.get("seed_fixed")
                sd = SLOW["seed"] + p.shape[-1] if sd is None else sd
                z = torch.from_numpy(slow_noise_prng(p.shape[-1], sd)).to(p.device)[None].expand(p.shape[0], -1, -1)
            elif z is None:
                g = None if self.training else torch.Generator().manual_seed(SLOW["seed"] + p.shape[-1])
                z = slow_noise(p.shape[0], p.shape[-1], p.device, g)
            cond = torch.cat([cond[:, :1], cond[:, 1:25] + z, cond[:, 25:]], 1)
        elif z is not None:
            cond = torch.cat([cond[:, :1], cond[:, 1:25] + z, cond[:, 25:]], 1)
        if MID["on"]:
            if zm is None and MID.get("causal") and not self.training:
                zm = torch.from_numpy(slow_noise_prng(p.shape[-1], SLOW["seed"] + p.shape[-1] + 7777, MID)).to(p.device)[None].expand(p.shape[0], -1, -1)
            elif zm is None:
                g = None if self.training else torch.Generator().manual_seed(SLOW["seed"] + p.shape[-1] + 7777)
                zm = slow_noise(p.shape[0], p.shape[-1], p.device, g, MID)
            cond = torch.cat([cond[:, :1], cond[:, 1:25] + zm, cond[:, 25:]], 1)
        if CVIN["on"]:
            cond = torch.cat([cond, cvf if cvf is not None else cond.new_zeros(cond.shape[0], CVIN["d"], cond.shape[-1])], 1)
        return cond

    def forward(self, cond: torch.Tensor, exc: torch.Tensor, spk=None) -> torch.Tensor:
        return super().forward(cond, exc)


class SVVoc(R.RVoc):
    """StreamVC 型の較正用(製品ではない): 話者 = 大域の話者ベクトル e(凍結 ECAPA・192)だけ。条件の包絡と周期性の行 = 線形(e) を全フレームに複写・
    各段に FiLM(1 + γ(e), β(e)・0 で始める)。spk = e [B, 192]。"""

    def __init__(self, *args, d_spk: int = 192, **kw):
        super().__init__(*args, **kw)
        self.film = torch.nn.ModuleList(torch.nn.Linear(d_spk, 2 * (self.cfg["ch"] >> (i + 1))) for i in range(len(R.UPS)))
        for f in self.film:
            torch.nn.init.zeros_(f.weight)
            torch.nn.init.zeros_(f.bias)
        self.lin = torch.nn.Linear(d_spk, 28)

    def make(self, p: torch.Tensor, c0n: torch.Tensor, lf: torch.Tensor, vv: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        s = self.lin(spk)[:, :, None]
        return torch.cat([c0n, s[:, :24].expand(-1, -1, p.shape[-1]), lf, vv, s[:, 24:] * vv, p], 1)

    def forward(self, cond: torch.Tensor, exc: torch.Tensor, spk: torch.Tensor) -> torch.Tensor:
        x = self.pre(cond)
        for i, r in enumerate(R.UPS):
            x = self.up[i](R.interp_up(F.leaky_relu(x, 0.1), r))
            x = x + self.src[i](exc)
            g, b = self.film[i](spk)[:, :, None].chunk(2, 1)
            x = x * (1 + g) + b
            x = sum(rb(x) for rb in self.res[i]) / len(self.res[i])
        return self.post(F.leaky_relu(x, 0.01)).squeeze(1)


def init_gen(dev: str, mode: str = "table", lin_init: tuple | None = None) -> R.RVoc:
    s0 = torch.load(INIT / "snap/ema_330k.pt", map_location="cpu", weights_only=False)
    c = s0["cfg"]
    gen = (TabVoc if mode == "table" else SVVoc)(ch=c["ch"], kernels=tuple(c["kernels"]), dils=tuple(c["dils"]), d_cond=D_PPG + (CVIN["d"] if CVIN["on"] else 0))
    new = gen.state_dict()
    sd = dict(new)
    for k, v in s0["ema"].items():
        if new[k].shape != v.shape:
            w = torch.zeros_like(new[k])
            w[:, :v.shape[1]] = v
            v = w
        sd[k] = v
    if lin_init is not None:
        sd["lin.weight"], sd["lin.bias"] = lin_init
    gen.load_state_dict(sd)
    return gen.to(dev)


def ridge_lin(E: np.ndarray, Y: np.ndarray, lam: float = 1.0) -> tuple:
    """線形(e) の初期値: 学習の話者の e → [表の単位平均 / 10 (24)・周期性の 75 パーセンタイル (4)] のリッジ回帰(0 step で各話者の大域の包絡と周期性を出す)。"""
    Xa = np.concatenate([E, np.ones((len(E), 1))], 1)
    Wt = np.linalg.solve(Xa.T @ Xa + lam * np.diag(np.r_[np.ones(E.shape[1]), 0.0]), Xa.T @ Y)
    return torch.from_numpy(Wt[:-1].T.copy()).float(), torch.from_numpy(Wt[-1].copy()).float()


def ppg_form(lg: torch.Tensor, mode: str) -> torch.Tensor:
    """C1 の logit [B, K, T] → 生成器の入力の事後。soft = softmax(本線)・hard = argmax の one-hot(D276 でフレームの男女の漏れ 0.93 → 0.69)・tau<x> = softmax(lg / x)。"""
    if mode == "soft":
        return lg.softmax(1)
    if mode == "hard":
        return F.one_hot(lg.argmax(1), lg.shape[1]).transpose(1, 2).to(lg.dtype)
    return (lg / float(mode[3:])).softmax(1)


PPG = "soft"


def load_dump(d: Path, k: str) -> tuple:
    c = np.fromfile(d / f"{k}.cond", "<f4").reshape(-1, 31)
    lg = np.fromfile(d / f"{k}.front", "<f4").reshape(-1, 224)[:, :K]
    n = min(len(c), len(lg))
    return c[:n], lg[:n]


def load_enroll(en: Path) -> tuple:
    j = json.loads((en / "enroll.json").read_text())
    return np.fromfile(en / "enroll.bin", "<f4").reshape(j["D"], j["K"]), np.array(j["per_p75"], np.float32)


def ref_ecapa(wavs: list, dev: str) -> list:
    """参照(20 s・48 kHz)の ECAPA(build_spk_emb.py と同じ: 先頭 20 s)。"""
    import idloss as ID
    from d6_unit_cov import load48
    ec = ID.Ecapa(dev)
    out = []
    with torch.no_grad():
        for w in wavs:
            x = torch.from_numpy(load48(w)[: int(REF_SEC * N.SR)])[None].to(dev)
            out.append(ec(x)[0].cpu().numpy().astype(np.float32))
    del ec
    return out


def eval_items(held: list, mode: str = "table", dev: str = "cuda") -> tuple:
    """表は両方とも Rust の登録: SR = d275_heldenroll.py(クリップ A を除く 20 s)・PV = dev の d85_work/en_TTT。
    mode = "sv": 話者ベクトル = 同じ参照(SR = held_en/ref_TTT.wav・PV = dev/d37/_work/ref_TTT.wav)の ECAPA。"""
    E2 = H / "e2e"
    sr_items, pv_items = [], []
    for i, (spk, ka, _) in enumerate(held):
        c, lg = load_dump(E2 / "fem", ka)
        sr_items.append([f"h{i:04d}_t{i:03d}_SR", c, lg, load_enroll(E2 / "held_en" / f"en_{i:03d}"), E2 / "held_en" / f"ref_{i:03d}.wav"])
    ed = E2 / "eval_dev"
    for f in sorted(ed.glob("p00*.cond")):
        ti = int(f.stem.split("_t")[1][:3])
        c, lg = load_dump(ed, f.stem)
        pv_items.append([f"{f.stem}_PV", c, lg, load_enroll(H / "dev" / "d85_work" / f"en_{ti:03d}"), H / "dev" / "d37" / "_work" / f"ref_{ti:03d}.wav"])
    if mode == "sv":
        refs = sorted({str(it[4]) for it in sr_items + pv_items})
        em = dict(zip(refs, ref_ecapa(refs, dev)))
        for it in sr_items + pv_items:
            it[3] = em[str(it[4])]
    return [tuple(it[:4]) for it in sr_items], [tuple(it[:4]) for it in pv_items]


def ff_items(sets: list) -> list:
    """女声 → 女声の評価(入力の床が無い・D297): <set> の Rust の書き出し(e2e/eval_<set>)を目標ごとに 1 本・目標の表 = <set>/d85_work/en_TTT。名前 f<k>PPP_tTTT_FF。"""
    out = []
    for k, s_ in enumerate(sets):
        ed = H / "e2e" / f"eval_{s_}"
        seen: set = set()
        for f in sorted(ed.glob("p0*.cond")):
            ti = int(f.stem.split("_t")[1][:3])
            if ti in seen:
                continue
            seen.add(ti)
            c, lg = load_dump(ed, f.stem)
            out.append((f"f{k}{f.stem[1:5]}_t{ti:03d}_FF", c, lg, load_enroll(H / s_ / "d85_work" / f"en_{ti:03d}")))
    return out


def spk_tensor(spk, dev: str):
    if isinstance(spk, tuple):
        return tuple(torch.from_numpy(np.asarray(v, np.float32)).to(dev)[None] for v in spk)
    return torch.from_numpy(np.asarray(spk, np.float32)).to(dev)[None]


def cond_exc(g: R.RVoc, c: np.ndarray, lg: np.ndarray, spk, dev: str, cvf: torch.Tensor | None = None) -> tuple:
    if C0LOW["on"]:
        c = c0low_np(c)
    p = ppg_form(torch.from_numpy(lg.T.copy()).float().to(dev)[None], PPG)
    ct = torch.from_numpy(c.T.copy()).float().to(dev)[None]
    st = spk_tensor(spk, dev)
    if PERDYN["on"] and isinstance(st, tuple):
        st = (st[0], ct[:, 27:31])
    if F0V["on"]:
        f0 = f0v_t(torch.where(ct[:, 26] > 0.5, 200.0 * torch.exp(ct[:, 25]), torch.zeros_like(ct[:, 25])), p)
        ct = torch.cat([ct[:, :25], torch.where(f0 > 0, torch.log(f0.clamp(min=1.0) / 200.0), torch.zeros_like(f0))[:, None], (f0 > 0).to(ct.dtype)[:, None], ct[:, 27:]], 1)
    cond = g.make(p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], st, cvf=cvf)
    f0 = torch.where(ct[:, 26] > 0.5, 200.0 * torch.exp(ct[:, 25]), torch.zeros_like(ct[:, 25]))
    return cond, TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)), st


@torch.no_grad()
def render(g: R.RVoc, c: np.ndarray, lg: np.ndarray, spk, dev: str, dyn: float = 0.0, T_op: np.ndarray | None = None, cvf: torch.Tensor | None = None) -> np.ndarray:
    """dyn > 0: 包絡の行に dyn·(操作者の包絡 − Σp T_操作者 / 10)を足す(動きの入力・T_op = 操作者の登録の表)。
    OPSLOW["on"]: 遅い揺らぎ = gain·opslow_causal(操作者の包絡 − Σp T_op / 10)(乱数の代わり・T_op 必須)。"""
    if OPSLOW["on"]:
        assert T_op is not None, "OPSLOW には操作者の表 T_op が要る(op_tab)"
        if C0LOW["on"]:
            c = c0low_np(c)
        p = ppg_form(torch.from_numpy(lg.T.copy()).float().to(dev)[None], PPG)
        ct = torch.from_numpy(c.T.copy()).float().to(dev)[None]
        st = spk_tensor(spk, dev)
        if PERDYN["on"] and isinstance(st, tuple):
            st = (st[0], ct[:, 27:31])
        zo = OPSLOW["gain"] * opslow_causal(ct[:, 1:25] - torch.einsum("dk,bkt->bdt", torch.from_numpy(T_op).float().to(dev), p) / 10, OPSLOW["tau"])
        cond = g.make(p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], st, zo, cvf=cvf)
        f0 = torch.where(ct[:, 26] > 0.5, 200.0 * torch.exp(ct[:, 25]), torch.zeros_like(ct[:, 25]))
        exc = TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0))
        y = g(cond, exc, st)[0].cpu().numpy()[N.DELAY:]
        return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)
    cond, exc, st = cond_exc(g, c, lg, spk, dev, cvf=cvf)
    if dyn > 0:
        pp = ppg_form(torch.from_numpy(lg.T.copy()).float().to(dev)[None], PPG)
        res = torch.from_numpy(c[:, 1:25].T.copy()).float().to(dev)[None] - torch.einsum("dk,bkt->bdt", torch.from_numpy(T_op).float().to(dev), pp) / 10
        cond = torch.cat([cond[:, :1], cond[:, 1:25] + dyn * res, cond[:, 25:]], 1)
    y = g(cond, exc, st)[0].cpu().numpy()[N.DELAY:]
    return (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32)


def future_invariance(g: R.RVoc, items: list, dev: str) -> int:
    """生成器の未来不変性(実音声由来の条件・励起・1 bit): フレーム k 以降を別の実音声の値に書き換え、k·H より前の出力が変わらないこと。"""
    import copy
    gc = copy.deepcopy(g).cpu().eval()

    worst = 0
    with torch.no_grad():
        c1, e1, s1 = cond_exc(gc, *items[0][1:], "cpu")
        c2, e2, _ = cond_exc(gc, *items[1][1:], "cpu")
        T_ = min(c1.shape[-1], c2.shape[-1], 600)
        c1, e1, c2, e2 = c1[..., :T_], e1[..., :T_ * N.HOP], c2[..., :T_], e2[..., :T_ * N.HOP]
        y = gc(c1, e1, s1)[0]
        for k in (97, 211, 350):
            ca, ea = c1.clone(), e1.clone()
            ca[..., k:], ea[..., k * N.HOP:] = c2[..., k:], e2[..., k * N.HOP:]
            nz = ((gc(ca, ea, s1)[0] - y).abs() > 0).nonzero()
            assert len(nz) > 0, "書き換えが出力を動かさない = INCONCLUSIVE"
            worst = max(worst, k * N.HOP - int(nz[0]))
    return max(0, worst)


class SLMPool(torch.nn.Module):
    """凍結 SSL(既定 WavLM-base-plus・物差しの WavLM-large とは別)の全層の特徴の上の判別器。フレームの枝(slm_disc.SLMDisc の頭と同じ)+ プールの枝
    (1×1 の射影の時間方向の平均と標準偏差 → 線形 = 物差しと同じ型の統計)。出力 = [フレームの logit, プールの logit]。"""

    def __init__(self, name: str = "microsoft/wavlm-base-plus", ch: int = 256):
        super().__init__()
        from slm_disc import SLMDisc
        base = SLMDisc(name, ch)
        self.ssl, self.head = base.ssl, base.head
        self.feats = base.feats
        wn = torch.nn.utils.parametrizations.weight_norm
        self.pool = torch.nn.Sequential(wn(torch.nn.Linear(2 * ch, ch)), torch.nn.LeakyReLU(0.2), wn(torch.nn.Linear(ch, 1)))
        self.proj = torch.nn.Linear(192, 2 * ch, bias=False)
        torch.nn.init.zeros_(self.proj.weight)

    def train(self, mode: bool = True):
        super().train(mode)
        self.ssl.eval()
        return self

    def score(self, h: torch.Tensor, e: torch.Tensor | None = None) -> list:
        """e [B, 192](目標の話者の埋め込み)があればプールの枝に射影の項 <W e, [平均, 標準偏差]> を足す(目標で条件づけた判別器・最初は 0)。"""
        z = self.head[1](self.head[0](h))
        fr = self.head[2:](z)
        st = torch.cat([z.mean(-1), z.std(-1)], 1)
        pl = self.pool(st)
        if e is not None:
            pl = pl + (self.proj(e) * st).sum(-1, keepdim=True)
        return [fr, pl]

    def forward(self, y48: torch.Tensor, e: torch.Tensor | None = None) -> list:
        return self.score(self.feats(y48), e)

    def head_params(self) -> list:
        return list(self.head.parameters()) + list(self.pool.parameters()) + list(self.proj.parameters())


class MalePool:
    """交差の標本の男声の操作者(Rust の書き出し・meta.json の male・評価の操作者 m0xx は除く)。話者ごとの有声の log f0 の中央値 = 操作者の登録の定数の代わり。"""

    def __init__(self, dirs: list, T: int, seed: int = 0):
        self.items, self.T, self.tab, self.p75 = [], T, {}, {}
        lfs: dict = {}
        c0s: dict = {}
        for d in dirs:
            d = Path(d)
            for m in json.loads((d / "meta.json").read_text())["male"]:
                if m["key"].startswith("m0") or m["spk"].startswith("m0"):
                    continue
                c, f = d / "male" / f"{m['key']}.cond", d / "male" / f"{m['key']}.front"
                if not c.exists():
                    continue
                cc, lg = load_dump(d / "male", m["key"])
                if len(cc) < T + 10:
                    continue
                zf = d / "c2" / f"{m['key']}.npy"
                if CVIN["on"]:
                    assert CVIN.get("src") == "c2" and zf.exists(), ("男声の交差の C2 の特徴が無い(d504_c2_male.py)", zf)
                    zc = np.load(zf)
                    assert zc.shape[1] == len(cc), ("C2 の特徴のフレーム数が書き出しと違う", zf, zc.shape, len(cc))
                z4 = None
                if PERDYN["on"]:
                    zf4 = d / "f4" / f"{m['key']}.npy"
                    assert zf4.exists(), ("男声の交差の F4 が無い(d523_f4_male.py)", zf4)
                    z4 = np.load(zf4)
                    assert z4.shape[1] == len(cc), ("F4 のフレーム数が書き出しと違う", zf4, z4.shape, len(cc))
                self.items.append((cc[:, :27].copy(), lg.astype(np.float16), m["spk"], zc if CVIN["on"] else None, z4))
                en = d / "male_en" / m["spk"]
                if m["spk"] not in self.tab and (en / "enroll.json").exists():
                    self.tab[m["spk"]] = load_enroll(en)[0]
                    self.p75[m["spk"]] = load_enroll(en)[1]
                v = cc[:, 26] > 0.5
                lfs.setdefault(m["spk"], []).append(cc[v, 25])
                c0s.setdefault(m["spk"], []).append(cc[cc[:, 0] > 0.5, 0])
        self.mu = {k: float(np.median(np.concatenate(v))) + float(np.log(200.0)) for k, v in lfs.items()}
        self.c0lv = {k: float(np.concatenate(v).mean()) for k, v in c0s.items()}
        self.rng = random.Random(seed)

    def sample(self, n: int, dev: str) -> tuple:
        """→ (条件 [n, 27, T](0 = c0n・1..24 = 包絡・25 = log f0・26 = 有声), logit [n, K, T], 声域の mu [n], 操作者の表 [n, 24, K](登録が無ければ 0))。"""
        cs, ls, mus, ts, zs, f4s, p75s, lvs = [], [], [], [], [], [], [], []
        p75_mean = np.mean(list(self.p75.values()), 0) if self.p75 else np.zeros(4, np.float32)
        for _ in range(n):
            c, lg, spk, zc, z4 = self.items[self.rng.randrange(len(self.items))]
            o = self.rng.randrange(0, len(c) - self.T)
            cs.append(c[o:o + self.T].T); ls.append(lg[o:o + self.T].astype(np.float32).T); mus.append(self.mu[spk]); lvs.append(self.c0lv[spk])
            if zc is not None:
                zs.append(zc[:, o:o + self.T].astype(np.float32))
            if z4 is not None:
                f4s.append(z4[:, o:o + self.T].astype(np.float32))
                p75s.append(self.p75.get(spk, p75_mean))
            ts.append(self.tab.get(spk, np.zeros((24, K), np.float32)))
        t = lambda z: torch.from_numpy(np.stack(z)).float().to(dev)
        self.last_cvf = t(zs) if zs else None
        self.last_f4, self.last_p75 = (t(f4s), t(p75s)) if f4s else (None, None)
        self.last_c0lv = torch.tensor(lvs, device=dev)
        return t(cs), t(ls), torch.tensor(mus, device=dev), t(ts)


class PairPool:
    """同じ文の男声 / 女声の操作者(data/pair_tts_v4l の Rust の書き出し・d330_pairdump.py)。文ごとに男声と女声のクリップを持つ。話者ごとの有声の log f0 の中央値 = 登録の定数の代わり。"""

    def __init__(self, d: str, seed: int = 0):
        d = Path(d)
        self.by: dict = {}
        lfs: dict = {}
        for m in json.loads((d / "meta.json").read_text())["pair"]:
            cc, lg = load_dump(d / "pair", m["key"])
            self.by.setdefault(m["text"], {"m": [], "f": []})[m["g"]].append((cc[:, :27].copy(), lg.astype(np.float16), m["spk"]))
            v = cc[:, 26] > 0.5
            lfs.setdefault(m["spk"], []).append(cc[v, 25])
        self.mu = {k: float(np.median(np.concatenate(v))) + float(np.log(200.0)) for k, v in lfs.items()}
        self.texts = sorted(t for t, v in self.by.items() if v["m"] and v["f"])
        self.rng = random.Random(seed)

    def sample(self, tmax: int) -> list:
        """→ [(男声の (条件 [27, T], logit [K, T], mu)), (女声の同じ文の ...)]・T ≤ tmax。"""
        t = self.rng.choice(self.texts)
        out = []
        for g in ("m", "f"):
            c, lg, spk = self.rng.choice(self.by[t][g])
            n = min(len(c), tmax)
            out.append((c[:n].T.copy(), lg[:n].astype(np.float32).T.copy(), self.mu[spk]))
        return out


def interp_rows(p: np.ndarray, pos: np.ndarray) -> np.ndarray:
    j0 = np.clip(np.floor(pos).astype(int), 0, len(p) - 1)
    j1 = np.minimum(j0 + 1, len(p) - 1)
    fr = (pos - j0)[:, None]
    return (1 - fr) * p[j0] + fr * p[j1]


def softmax_np(l: np.ndarray) -> np.ndarray:
    e = np.exp(l - l.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


class SwapPool:
    """事後の入れ替えの対(d396_swapprep.py・VCTK の全話者共通の文・diag_ppgvoc31)。目標の利得はそのまま(log-mel L1 を最小にする利得は 0 dB・RMS の差 +3.1 dB は帯域の釣り合いの違い = 1〜3 kHz で出力が +8 dB 明るい・日本語の実女声では −1〜−3 dB = VCTK の表に従い切れていない・d396b)。目標 F = 実録音の女声: 条件(音量・高さ・有声)・登録の表・実録音の区間は F のまま、
    事後だけを同じ文を読んだ源 X(男声 / 他の女声)のものを滑らかな時間写像で F の時刻へ写して入れる(D395: 自己再構成では事後が目標の声質を運ぶ)。
    確率 p_rt で源の代わりに F 自身の事後の往復(F → X → F)= 写しの痕跡をそろえた自己の対照(D395b)。目標の区間 = 実録音[t0·HOP − off : + seg]。"""

    def __init__(self, d: str, T: int, seg: int, p_rt: float = 1 / 3, p_male: float = 0.5, seed: int = 0):
        d = Path(d)
        j = json.loads((d / "pool.json").read_text())
        self.off, self.T, self.seg, self.tf, self.males = int(j["off"]), T, seg, j["tf"], j["males"]
        self.c, self.lg, self.wav, self.tab, self.maps = {}, {}, {}, {}, {}
        for s in self.tf + self.males:
            c, lg = load_dump(d / "dump", s)
            self.c[s], self.lg[s] = c[:, :27].copy(), lg.astype(np.float16)
        for f in self.tf:
            self.wav[f] = sf.read(d / "wav" / f"{f}.wav", dtype="float32")[0]
            self.tab[f] = load_enroll(d / "en" / f)
            for x in self.males + [g for g in self.tf if g != f]:
                z = np.load(d / "maps" / f"{f}__{x}.npz")
                self.maps[(f, x)] = (z["fx"], z["xf"])
        self.p_rt, self.p_male = p_rt, p_male
        self.rng = random.Random(seed)

    def one(self, f: str, kind: str, x: str, t0: int) -> np.ndarray:
        fx, xf = self.maps[(f, x)]
        pos = fx[t0:t0 + self.T].astype(np.float64)
        lo, hi = int(np.floor(pos.min())), min(int(np.ceil(pos.max())) + 2, len(xf))
        if kind == "rt":
            pf = softmax_np(self.lg[f].astype(np.float32))
            src = interp_rows(pf, xf[lo:hi].astype(np.float64))
        else:
            src = softmax_np(self.lg[x][lo:hi].astype(np.float32))
        return interp_rows(src, pos - lo)

    def pick(self) -> tuple:
        f = self.rng.choice(self.tf)
        kind = "rt" if self.rng.random() < self.p_rt else ("m" if self.rng.random() < self.p_male else "f")
        pool = self.males if (kind == "m" or (kind == "rt" and self.rng.random() < self.p_male)) else [g for g in self.tf if g != f]
        x = self.rng.choice(pool)
        lo = max(0, -(-self.off // N.HOP))
        hi = min(len(self.c[f]) - self.T, (len(self.wav[f]) + self.off - self.seg) // N.HOP)
        return f, kind, x, self.rng.randrange(lo, hi)

    def sample(self, n: int, dev: str) -> tuple:
        """→ (事後の log 確率 [n, K, T], 条件 [n, 27, T], 目標の実録音 [n, seg], 表 T [n, 24, K], 周期性 [n, ·], 種類の一覧)。"""
        ls, cs, ws, ts, ps, ks = [], [], [], [], [], []
        for _ in range(n):
            f, kind, x, t0 = self.pick()
            ls.append(np.log(self.one(f, kind, x, t0) + 1e-8).T.astype(np.float32))
            cs.append(self.c[f][t0:t0 + self.T].T)
            s0 = t0 * N.HOP - self.off
            ws.append(self.wav[f][s0:s0 + self.seg])
            ts.append(self.tab[f][0]); ps.append(self.tab[f][1]); ks.append(kind)
        t = lambda z: torch.from_numpy(np.stack(z)).float().to(dev)
        return t(ls), t(cs), t(ws), t(ts), t(ps), ks

    def check(self, n: int = 24) -> dict:
        """写しの単体検査: 窓の中で F 自身の事後と argmax が一致する割合(rt は高く・m / f は内容だけの一致で低い)。"""
        out: dict = {}
        for _ in range(n):
            f, kind, x, t0 = self.pick()
            q = self.one(f, kind, x, t0)
            pf = self.lg[f][t0:t0 + self.T].astype(np.float32)
            out.setdefault(kind, []).append(float((q.argmax(1) == pf.argmax(1)).mean()))
        return {k: round(float(np.mean(v)), 3) for k, v in out.items()}



class Ltas(torch.nn.Module):
    """長時間平均スペクトル(帯域の相対 dB・build_ltas.ltas_np と同じ定義: 2048 点・480 hop・Hann・全フレームのパワーの合計 → 帯域 → 全帯で割る → 10 log10)。"""

    def __init__(self, bands: tuple, sr: int = 48000):
        super().__init__()
        f = torch.fft.rfftfreq(2048, 1 / sr)
        self.register_buffer("M", torch.stack([((f >= lo) & (f < hi)).float() for lo, hi in zip(bands[:-1], bands[1:])]))
        self.register_buffer("w", torch.hann_window(2048, periodic=False))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        P = torch.stft(y.float(), 2048, 480, window=self.w, center=False, return_complex=True).abs().pow(2).sum(-1)
        b = P @ self.M.T
        return 10 * torch.log10(b / b.sum(-1, keepdim=True) + 1e-12)


class OrigD(torch.nn.Module):
    """同じ文・同じ目標の 2 つの変換(男声の操作者由来 / 女声の操作者由来)の SSL の全層の特徴 → 1×1 の射影の時間方向の平均と標準偏差 → 差 → 線形(偏りなし)
    = 1 つ目が男声由来である logit(反対称・内容と目標は対で打ち消す)。線形の重みは 0 から(最初は生成器へ勾配なし)。"""

    def __init__(self, d_in: int, ch: int = 128):
        super().__init__()
        self.proj = torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(d_in, ch, 1))
        self.w = torch.nn.Linear(2 * ch, 1, bias=False)
        torch.nn.init.zeros_(self.w.weight)

    def phi(self, h: torch.Tensor) -> torch.Tensor:
        z = self.proj(h.float())
        return torch.cat([z.mean(-1), z.std(-1)], -1)

    def forward(self, hm: torch.Tensor, hf: torch.Tensor) -> torch.Tensor:
        return self.w(self.phi(hm) - self.phi(hf))[:, 0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="diag_ppgvoc1")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lm_w", type=float, default=45.0)
    ap.add_argument("--fline_w", type=float, default=0.0, help="フレーム周期(200 Hz)の高域の包絡変調の線(fsync.fline・amline の微分可能な版・D435〜D437)の重み: relu(fline − fline_m) を自己再構成と交差の出力に(0 = なし)")
    ap.add_argument("--sexpert", type=float, default=0.0, help="自己再構成の入力の側の性別の摂動の確率(D434・diag_ppgvoc44): C1 の事後と c0n を WORLD で f0 ×0.5〜0.65・包絡 ÷1.08〜1.2 にした入力から(1/3 は再合成だけ)・目標は実録音のまま")
    ap.add_argument("--fline_m", type=float, default=0.5, help="fline の許容(dB・本物の [p15, p85] の上 0.54 相当)")
    ap.add_argument("--eval_every", type=int, default=2500)
    ap.add_argument("--kill_step", type=int, default=5000)
    ap.add_argument("--kill_pv", type=float, default=3.62, help="kill_step の男声 → 女声 v1⊥ の決定値の中央値がこれより大きければ止める(基準 4.02 − 0.4)")
    ap.add_argument("--kill_sr", type=float, default=2.73, help="kill_step の保留の自己再構成の v1 の決定値の中央値がこれより大きければ止める(diag_e2e1 の SR)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--spk", default="table", choices=("table", "sv"), help="table = 目標の表(本線)・sv = StreamVC 型の較正用(大域の話者ベクトル = 凍結 ECAPA + FiLM)")
    ap.add_argument("--spk_emb", default=str(ROOT / "data/rvoc_tables_full_ecapa.npz"), help="--spk sv の学習の話者ベクトル(build_spk_emb.py・表と同じ参照のファイル)")
    ap.add_argument("--tab_real", action="store_true", help="表を実音声だけで作り直した data/rvoc_tables_real にする(本番)")
    ap.add_argument("--ppg", default="soft", help="生成器の入力の事後の形: soft・hard・tau<x>(ppg_form)")
    ap.add_argument("--init_ema", default="", help="生成器の重みをこの ckpt の ema から始める(last_ema.pt)")
    ap.add_argument("--x_male", nargs="*", default=[], help="交差の標本の男声の操作者の書き出しの置き場(meta.json の male・<dir>/male/*.cond・.front)")
    ap.add_argument("--x_nm", type=int, default=0, help="1 バッチの男声の操作者 × ランダムな女声の表の交差の標本の数(敵対の損失だけ)")
    ap.add_argument("--x_nf", type=int, default=0, help="1 バッチの女声どうしの交差の標本の数(バッチの中で表をずらす・高さを表の声域へ・敵対の損失だけ)")
    ap.add_argument("--x_w", type=float, default=1.0, help="交差の標本の敵対の損失の重み")
    ap.add_argument("--slm_w", type=float, default=0.0, help="SSL 判別器(SLMPool・フレーム + プールの枝)の敵対の重み。偽物 = 自己再構成と交差の標本・本物 = バッチの実女声(0 = なし)")
    ap.add_argument("--slm_model", default="microsoft/wavlm-base-plus")
    ap.add_argument("--slm_cond", action="store_true", help="SSL 判別器のプールの枝を目標の話者の埋め込み(--x_emb)で条件づける(射影)+ 本物 × 別人の埋め込み = 偽物(組み違い)")
    ap.add_argument("--dyn_g", default="0", help="動きの入力: 包絡の行 += γ·(操作者の包絡 − Σp T_操作者 / 10)。γ を標本ごとにこの一覧から選ぶ(自己再構成 = 本人の教師の包絡・交差 = 操作者 A / 男声の包絡と登録)。0 = なし")
    ap.add_argument("--x_id", type=float, default=0.0, help="交差の標本の同一性の損失(1 − cos(ECAPA(出力), 目標 B の埋め込み))の重み(0 = なし)")
    ap.add_argument("--x_ct", type=float, default=0.0, help="交差の標本の内容の一致の損失(出力の C1 の事後と入力の事後の KL・ずれ --ct_lag フレーム・最初の 100 フレームは除く)の重み(0 = なし)")
    ap.add_argument("--ct_female_only", action="store_true", help="内容の一致を女声どうしの交差の標本だけにかける(男声の事後の形に合わせると男声らしさを押し戻す・diag_ppgvoc10〜12)")
    ap.add_argument("--ct_hard", action="store_true", help="内容の一致の目標を入力の事後の argmax(硬い単位)にする(柔らかい形は男女の情報を運ぶ・D276: フレーム 0.93 → 0.69)")
    ap.add_argument("--ct_lag", type=int, default=1, help="出力のフレーム t + lag が入力のフレーム t に対応(出力の遅延 DELAY 240 = 1 フレーム・煙テストで実測)")
    ap.add_argument("--x_emb", default=str(ROOT / "data/rvoc_tables_real_ecapa.npz"), help="--x_id の目標の埋め込み(build_spk_emb.py --real_only)")
    ap.add_argument("--x_pair", default="", help="同じ文の男声 / 女声の操作者の書き出し(d330_pairdump.py)= 同じ目標への 2 つの変換の SSL の統計(平均・標準偏差)を男声の側だけ女声の側(勾配なし)へそろえる")
    ap.add_argument("--pair_n", type=int, default=2)
    ap.add_argument("--pair_w", type=float, default=1.0)
    ap.add_argument("--pair_sec", type=float, default=5.0)
    ap.add_argument("--allow_ppg_change", action="store_true", help="init_ema と違う事後の形(--ppg)で学習を続けることを明示的に許す(diag_ppgvoc22: 柔らかい → 硬い)")
    ap.add_argument("--slow_stats", default="", help="遅い音色の揺らぎ(D339 の固定の統計 .npz)を条件に入れる: 自己再構成 = 目標の本物の揺らぎ(確率 --slow_pnoise で乱数)・交差と評価 = 乱数(D334〜D340)")
    ap.add_argument("--slow_pnoise", type=float, default=0.3)
    ap.add_argument("--opslow_train", type=float, default=0.0, help="学習でも遅い揺らぎを操作者の揺らぎ(因果・D451)で作る利得(diag_ppgvoc45): 自己再構成 = 本人の包絡 − 本人の表の引き・交差 = 操作者の包絡 − 操作者の表の引き(乱数を使わない)・評価と製品も同じ(0 = なし)")
    ap.add_argument("--opslow_tau", type=float, default=4.0)
    ap.add_argument("--per_w", type=float, default=0.0, help="周期性の一致の損失(diag_ppgvoc54): 自己再構成の出力の pae.periodicity(4 帯)と実録音の同じ量の L1(有声のフレーム・教師の f0)の重み(0 = なし・交差の標本にはかけない)")
    ap.add_argument("--f1_train", action="store_true", help="自己再構成(と女声どうしの交差)の log f0・有声・励起を因果の F1(results/f0est3・Rust の書き出しとフレーム一致)にする(diag_ppgvoc55・学習と推論の一致): 教師の f0 は息の端を有声にしない = 推論の F1 の余分な有声(息の端の約 12%)を学習で見ない(D533)。包絡・周期性の分析は教師の f0 のまま")
    ap.add_argument("--per_dyn", action="store_true", help="周期性の条件を操作者の動きに(diag_ppgvoc53・耳の所見『F0 がしっかり出ず・かすれ』= D522): 行 = clamp(目標の per_p75 + F4(操作者)(t) − 操作者の per_p75)× 有声(自己再構成 = F4(t)・推論 = Rust の --f4 の書き出しの列 27..30 と同じ式)")
    ap.add_argument("--x_c0shift", action="store_true", help="男声の交差の標本の c0n を話者の発声中の平均から 4.40 へずらす(推論の水準ずらし D558 と同じ形・無音には足さない・diag_ppgvoc57)")
    ap.add_argument("--x_noslow", action="store_true", help="男声の交差の標本の遅い音色の揺らぎを 0 に(男声 → 女声の推論 = 操作者の遅い揺らぎの利得 0・D562・diag_ppgvoc57)")
    ap.add_argument("--x_fixk", default="", help="男声の交差の標本の高さを目標の声域ではなく操作者の f0 × 2^(k/12) に(k は標本ごとにこの集合から・例 0,5,7,12)= 固定の半音の運用(オーナー判断 10-25)に学習をそろえる(D597・diag_ppgvoc60)")
    ap.add_argument("--psd_w", type=float, default=0.0, help="f0 で条件づけた細かい構造の判別器(psd_disc.PSD・倍音の番号の軸で帯の包絡を除いた z)の敵対の重み。本物 = 実女声(教師の f0)・偽物 = 自己再構成と交差(指令の f0)(0 = なし・D599・diag_ppgvoc61)")
    ap.add_argument("--psd_ch", type=int, default=32)
    ap.add_argument("--h1_w", type=float, default=0.0, help="自己再構成の低い倍音(h = 1..4)の対数振幅の L1 の重み(harm_l1・D607: 低い k の基音の弱さ = F1 の倍読みの源・diag_ppgvoc62)(0 = なし)")
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--skip_eval0", action="store_true", help="step 0 の評価を飛ばす(計時のベンチマーク専用・results/bench_cpu_swap)")
    ap.add_argument("--cv_noise", type=float, default=0.0, help="学習で C2 の特徴に足す独立な正規雑音の標準偏差(diag_ppgvoc52・情報の絞り = 細部への依存を断つ・推論は雑音なし・交差の標本にも同じ)")
    ap.add_argument("--cvin", default="", help="連続の内容の特徴(ContentVec の主成分 .npz・D502)を生成器の入力に足す(diag_ppgvoc49・非因果の教師 = 診断のみ・男声の交差・入れ替え・対の標本とは併用しない)")
    ap.add_argument("--cv_w", type=float, default=0.0, help="内容の特徴の一致(D501): 自己再構成の出力と実録音の ContentVec(lengyue233/content-vec-best・最終層・正規化)のフレームの 1 − cos の重み(0 = なし)")
    ap.add_argument("--cv_xw", type=float, default=0.0, help="女声どうしの交差の標本の出力と操作者の実録音(内容の元)の ContentVec の 1 − cos の重み(0 = なし・男声の交差は録音が無いのでかけない)")
    ap.add_argument("--c0low", default="", help="音量の条件を低い帯域の水準に(diag_ppgvoc47): 重み v の .npy(results/diag_ppgvoc47/c0low_v.npy)・c0n' = 0.78·(c0n + Σ v_r·行 r)を学習・評価・推論で")
    ap.add_argument("--mid_stats", default="", help="中間の帯(約 1〜3 Hz)の音色の動きも条件に入れる(D364 の統計 .npz・自己再構成 = 本物・確率 --slow_pnoise で乱数・交差と評価 = 乱数・diag_ppgvoc25)")
    ap.add_argument("--mid_sigma", type=float, default=0.5, help="評価・推論の中間の帯の乱数の倍率")
    ap.add_argument("--slow_sigma", type=float, default=1.0, help="評価・推論の乱数の揺らぎの倍率")
    ap.add_argument("--x_orig", type=float, default=0.0, help="対の由来の判別器(OrigD)を騙す損失の重み(男声の側だけ・0 = なし・--x_pair が要る)= 男声の分を判別の方向に集めて消す(D332: 全次元の L1 では男声の分は 5%)")
    ap.add_argument("--orig_ch", type=int, default=128)
    ap.add_argument("--orig_lr", type=float, default=2e-4)
    ap.add_argument("--orig_buf", type=int, default=48, help="由来の判別器の学習に使う過去の対(勾配なしの特徴)の数")
    ap.add_argument("--orig_nb", type=int, default=6, help="1 step で過去の対から足す数")
    ap.add_argument("--x_swap", default="", help="事後の入れ替えの対の置き場(d396_swapprep.py・VCTK の共通の文)= 目標 F の実録音を F の条件・表と源 X の事後で再構成(diag_ppgvoc31・D395)")
    ap.add_argument("--swap_n", type=int, default=2, help="1 step の入れ替えの標本の数")
    ap.add_argument("--swap_w", type=float, default=1.0, help="入れ替えの標本の再構成(log-mel・MPD / MRD の敵対・特徴の一致)の重み")
    ap.add_argument("--swap_prt", type=float, default=1 / 3, help="入れ替えの標本のうち F 自身の事後の往復(写しの痕跡をそろえた自己の対照)の割合")
    ap.add_argument("--swap_pmale", type=float, default=0.5, help="源が男声である割合")
    ap.add_argument("--x_dltas", type=float, default=0.0, help="女声どうしの交差の標本の話者差の帯域の一致(D399 の表への追従): |(LTAS(出力) − LTAS(操作者の同じ区間の実録音)) − (LTAS_参照(目標) − LTAS_参照(操作者))| の 400 Hz 以上の帯の平均(dB)の重み")
    ap.add_argument("--ltas", default=str(ROOT / "data/rvoc_tables_real_ltas.npz"), help="build_ltas.py の話者ごとの参照の LTAS")
    ap.add_argument("--static_off", action="store_true", help="自己再構成で区間の実際の包絡と表の引きの平均のずれ(静的なずれ = 録音・収録日・D408)を包絡の行に足す(推論・交差の標本 = 0 = 表そのもの)= c0n・事後から静的なずれを推す近道を不要にする(diag_ppgvoc34)")
    ap.add_argument("--pert", type=float, default=0.0, help="入力の側の録音の経路の摂動の確率(D400・diag_ppgvoc32): C1 の事後と c0n・包絡を、低域通過(確率 0.5・5〜14 kHz)と 2 kHz の高域の棚(±9 dB)をかけた入力から計算・目標は実録音のまま")
    ap.add_argument("--pert_c0_only", action="store_true", help="--pert の摂動を c0n(包絡の分析)だけにかける(C1 の事後は元の音声から・D400: 漏れは c0n だけが運ぶ・35 の衝突の切り分け)")
    ap.add_argument("--inv_w", type=float, default=0.0, help="事後のアダプタ(AdaptP)の男女の不変性の敵対の重み(男声の交差の標本の事後を判別器に女声と判定させる・0 = アダプタなし)")
    ap.add_argument("--ff_eval", nargs="*", default=[], help="女声 → 女声の評価の組(例 devff)= 群 FF(目標ごとに 1 本・v1)")
    ap.add_argument("--kill_ff", type=float, default=99.0, help="kill_step の FF の v1 の決定値の中央値がこれより大きければ止める")
    ap.add_argument("--ctx_ch", type=int, default=192, help="文脈の符号器の ch(1 フレームの畳み込みは重みの読み出しが律速 = 縮めると製品の経路が軽くなる)")
    ap.add_argument("--ctx", action="store_true", help="文脈の符号器(CtxEnc・受容野 約 1.27 秒)を足す")
    ap.add_argument("--seg_mult", type=int, default=1, help="学習の区間を SEG の何倍にするか(長い区間の判別器のため・bs はそれに合わせて下げる)")
    ap.add_argument("--dyn_self", action="store_true", help="--dyn_g を自己再構成だけに(交差には足さない・diag_ppgvoc58): 包絡の行のずれ z = (1 − γ)·z_遅い + γ·(本人の教師の包絡 − 本人の表の引き)= 行への追従を学ぶ(推論は R の話者の分を z へ)")
    ap.add_argument("--sm_manifest", default="", help="合成の男声の自己再構成の台帳(prep_male_train_f0.py の f0_manifest.json・diag_ppgvoc64)")
    ap.add_argument("--sm_tab", default="", help="その話者の表(build_tables_rvoc.py --manifest)")
    ap.add_argument("--sm_n", type=int, default=0, help="1 step の男声の自己再構成の標本の数(女声のバッチ --bs に足す・0 = なし)")
    ap.add_argument("--sm_w", type=float, default=1.0, help="男声の自己再構成の log-mel・fline の重み(女声の項とは別の項)")
    ap.add_argument("--sm_adv", action="store_true", help="男声の自己再構成を MPD / MRD の本物・偽物にも入れる(既定 = 再構成の損失だけ・SSL 判別器には常に入れない)")
    ap.add_argument("--c0_tab", default="", help="c0 を中立に(diag_ppgvoc65・paper_mf E-A1t): 事後の argmax の単位ごとの c0 の表(f32 × K・--c0_tab_build で作る)を因果な 5 フレームの平均で(学習・評価・推論とも)")
    ap.add_argument("--c0_tab_build", default="", help="学習の女声の行から c0 の表を作って書き出して終わる(--c0_tab_batches バッチ・20 フレーム未満の単位は全体の平均)")
    ap.add_argument("--c0_tab_batches", type=int, default=400)
    ap.add_argument("--vv_tab", default="", help="f0・有声を中立に(diag_ppgvoc67・paper_mf E-A1b): 単位ごとの有声の率の表(f32 × K・--vv_tab_build で作る)と f0 の因果な平滑(f0v_np)を学習・評価・交差とも")
    ap.add_argument("--vv_tab_build", default="", help="学習の女声の行から単位ごとの有声の率(教師の f0 > 0)の表を作って書き出して終わる(--c0_tab_batches バッチ)")
    ap.add_argument("--c2_const", default="", help="C2 を定数に(diag_ppgvoc65): f32 × d のファイル(推論の LIGHTVC_DIAG_C2 と同じ・学習・評価・交差とも)")
    ap.add_argument("--dyn_eval", type=float, default=0.0, help="評価の γ(PV の操作者の表 = dev の d85_work/op_PPPP・SR = 本人の登録)")
    a = ap.parse_args()
    global PPG, TAB
    PPG = a.ppg
    CTX["on"] = a.ctx
    CTX["ch"] = a.ctx_ch
    CTX["adapt"] = a.inv_w > 0
    if a.seg_mult > 1:
        TR.SEG = 33600 * a.seg_mult
        TR.T = TR.SEG // N.HOP
        print("学習の区間", TR.SEG, "標本", TR.T, "フレーム", flush=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    PERDYN.update(on=bool(a.per_dyn))
    if a.cvin:
        cvin_setup(a.cvin, "cuda")
    if a.c2_const:
        assert CVIN["on"], "--c2_const には --cvin が要る"
        v_ = np.fromfile(a.c2_const, dtype="<f4")
        assert len(v_) == CVIN["d"], f"--c2_const は {CVIN['d']} 個"
        C2CONST.update(on=True, v=torch.from_numpy(v_).float().to(dev))
        print("C2 = 定数", a.c2_const, "平均", round(float(v_.mean()), 4), flush=True)
    if a.vv_tab:
        t_ = np.fromfile(a.vv_tab, dtype="<f4")
        assert len(t_) == K, f"--vv_tab は {K} 個"
        assert not a.x_swap and not a.x_pair and not a.f1_train and a.opslow_train <= 0, "--vv_tab は入れ替え・対の標本・F1 の学習・操作者の遅い揺らぎと併用しない"
        F0V.update(on=True, vtab=t_.astype(np.float64))
        print("f0・有声 = 中立", a.vv_tab, "有声の率 > 0.5 の単位", int((t_ > 0.5).sum()), "/", K, "EMA a", round(F0V["a"], 4), flush=True)
    if a.c0_tab:
        t_ = np.fromfile(a.c0_tab, dtype="<f4")
        assert len(t_) == K, f"--c0_tab は {K} 個"
        C0TAB.update(on=True, tab=torch.from_numpy(t_).float().to(dev))
        print("c0 = 単位ごとの表", a.c0_tab, "範囲", round(float(t_.min()), 3), round(float(t_.max()), 3), flush=True)
        assert (a.x_nm == 0 or CVIN.get("src") == "c2") and not a.x_swap and not a.x_pair, "非因果の教師の特徴は男声の交差と併用しない(録音が無い)・入れ替え・対の標本とは併用しない"
    if not a.smoke and not (out / "prereg.yaml").exists():
        print(f"results/{a.tag}/prereg.yaml が無い: 起動しない", flush=True)
        return 1
    print(f"prereg: results/{a.tag}/prereg.yaml", flush=True)
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    held = e2e_held()
    if a.tab_real:
        TAB = ROOT / "data/rvoc_tables_real"
    rows, spks, counts = load_rows({h[0] for h in held}, a.tab_real)
    print("data", json.dumps(counts), flush=True)
    (out / "data_counts.json").write_text(json.dumps(counts, indent=1))
    spk_idx = {s: i for i, s in enumerate(spks)}
    zs = [np.load(TAB / f"{s}.npz") for s in spks]
    tabT = torch.from_numpy(np.stack([z["T"] for z in zs])).float().to(dev)
    tabP = torch.from_numpy(np.stack([z["per_p75"] for z in zs])).float().to(dev)
    tabMu = torch.from_numpy(np.array([float(z["mu"]) for z in zs])).float().to(dev)
    nbf, sm_idx = a.bs, {}
    if a.sm_n > 0:
        assert a.spk == "table" and a.psd_w == 0 and a.per_w == 0 and a.h1_w == 0 and a.inv_w == 0 and a.cv_w == 0 and a.cv_xw == 0 and a.x_dltas == 0 and not a.x_swap and not a.x_pair and not a.f1_train, "男声の自己再構成は P58 の構成の損失だけで検査済み"
        srows, sspk, scnt = load_sm_rows(a.sm_manifest, Path(a.sm_tab))
        assert not set(sspk) & set(spks), "男声と女声の話者 ID が重なる"
        print("男声の自己再構成", json.dumps(scnt), "1 step", a.sm_n, "重み", a.sm_w, "判別器", a.sm_adv, flush=True)
        (out / "data_counts_sm.json").write_text(json.dumps(scnt, indent=1))
        sm_idx = {s: len(spks) + i for i, s in enumerate(sspk)}
        zm = [np.load(Path(a.sm_tab) / f"{s}.npz") for s in sspk]
        tabT = torch.cat([tabT, torch.from_numpy(np.stack([z["T"] for z in zm])).float().to(dev)])
        tabP = torch.cat([tabP, torch.from_numpy(np.stack([z["per_p75"] for z in zm])).float().to(dev)])
        tabMu = torch.cat([tabMu, torch.from_numpy(np.array([float(z["mu"]) for z in zm])).float().to(dev)])
    ecx, embX = None, None
    if a.slm_cond and a.x_id <= 0:
        zx = np.load(a.x_emb)
        embX = torch.from_numpy(np.stack([zx[q] for q in spks])).float().to(dev)
        print("条件づけた SSL 判別器の埋め込み", a.x_emb, flush=True)
    if a.x_id > 0:
        import idloss as ID
        ecx = ID.Ecapa(dev)
        zx = np.load(a.x_emb)
        miss = [q for q in spks if q not in zx.files]
        assert not miss, f"同一性の埋め込みの無い話者 {len(miss)}"
        embX = torch.from_numpy(np.stack([zx[q] for q in spks])).float().to(dev)
        print("交差の同一性の損失", a.x_id, a.x_emb, flush=True)
    xpool = None
    if a.x_nm > 0:
        xpool = MalePool(a.x_male, TR.T, 285)
        print("交差の男声の操作者", len(xpool.items), "クリップ", len(xpool.mu), "人", flush=True)
    ppool = None
    if a.x_pair:
        assert a.slm_w > 0 and a.spk == "table", "--x_pair は SSL 判別器(特徴を共用)と表の経路が要る"
        ppool = PairPool(a.x_pair, 330)
        print("同じ文の男声 / 女声の対", len(ppool.texts), "文", len(ppool.mu), "人", flush=True)
    assert a.x_orig <= 0 or ppool is not None, "--x_orig には --x_pair が要る"
    ltas, tabL = None, None
    if a.x_dltas > 0:
        assert a.x_nf > 0 and a.spk == "table", "--x_dltas は女声どうしの交差の標本(--x_nf)と表の経路が要る"
        zl = np.load(a.ltas)
        miss = [q for q in spks if q not in zl.files]
        assert not miss, f"LTAS の無い話者 {len(miss)}"
        tabL = torch.from_numpy(np.stack([zl[q] for q in spks])).float().to(dev)
        ltas = Ltas(tuple(int(b) for b in zl["bands"])).to(dev)
        print("話者差の帯域の一致", a.x_dltas, a.ltas, "帯", zl["bands"].tolist(), "(損失は 400 Hz 以上 = 3 帯目から)", flush=True)
    spool = None
    if a.x_swap:
        assert a.spk == "table", "--x_swap は表の経路だけ"
        spool = SwapPool(a.x_swap, TR.T, TR.SEG, a.swap_prt, a.swap_pmale, 396)
        print("事後の入れ替えの対: 目標", len(spool.tf), "人・源の男声", len(spool.males), "人・ずれ OFF", spool.off, "・写しの単体検査(argmax の一致)", spool.check(), flush=True)
    if a.slow_stats:
        assert a.spk == "table", "--slow_stats は表の経路だけ"
        slow_load(a.slow_stats, a.slow_sigma)
        print("遅い音色の揺らぎ", a.slow_stats, "乱数の確率", a.slow_pnoise, "評価の σ", a.slow_sigma, flush=True)
    if a.mid_stats:
        assert a.slow_stats, "--mid_stats は --slow_stats と一緒に"
        slow_load(a.mid_stats, a.mid_sigma, MID)
        print("中間の帯の音色の動き", a.mid_stats, "評価の σ", a.mid_sigma, flush=True)
    lin0, embS = None, None
    if a.spk == "sv":
        ze = np.load(a.spk_emb)
        miss = [s for s in spks if s not in ze.files]
        assert not miss, f"話者ベクトルの無い話者 {len(miss)}"
        E_ = np.stack([ze[s] for s in spks]).astype(np.float64)
        Y_ = np.concatenate([np.stack([z["T"].mean(1) for z in zs]) / 10, np.stack([z["per_p75"] for z in zs])], 1).astype(np.float64)
        lin0 = ridge_lin(E_, Y_)
        r2 = 1 - ((E_ @ lin0[0].double().numpy().T + lin0[1].double().numpy() - Y_) ** 2).sum() / ((Y_ - Y_.mean(0)) ** 2).sum()
        print("線形(e) の初期値のリッジの R2(学習の話者)", round(float(r2), 3), flush=True)
        embS = torch.from_numpy(E_).float().to(dev)
    import c1_content as CC
    cvm, cvrs = None, None
    if a.cv_w > 0 or a.cv_xw > 0:
        import torchaudio
        from transformers import HubertModel
        cvm = HubertModel.from_pretrained("lengyue233/content-vec-best").to(dev).eval().requires_grad_(False)
        cvrs = torchaudio.transforms.Resample(N.SR, 16000).to(dev)
        print("内容の特徴の一致 ContentVec", a.cv_w, a.cv_xw, flush=True)

    def cvfeat(x48: torch.Tensor) -> torch.Tensor:
        return F.normalize(cvm(cvrs(x48)).last_hidden_state, dim=-1)
    import train_c1 as T1
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1n = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev); c1n.load_state_dict(st1["net"]); c1n.eval().requires_grad_(False)
    mfc = CC.MelFront().to(dev)
    f4n = None
    if a.per_dyn:
        import lvl as LV
        s4 = torch.load(ROOT / "results/f4_1/last.pt", map_location="cpu", weights_only=False)
        f4n = LV.PerEst(s4["cfg"]["ch"], tuple(s4["cfg"]["dils"])).to(dev)
        f4n.load_state_dict(s4["net"])
        f4n.eval().requires_grad_(False)
        print("周期性の動き F4", s4.get("step"), flush=True)
    f1f, f1n = None, None
    if a.f1_train:
        import f0est as FE
        s1_ = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
        f1f = FE.Front().to(dev)
        f1n = FE.F0Est(ch=s1_["cfg"]["ch"], dils=tuple(s1_["cfg"]["dils"])).to(dev)
        f1n.load_state_dict(s1_["net"])
        f1n.eval().requires_grad_(False)
        print("学習の f0・有声 = F1(f0est3)", s1_.get("step"), flush=True)
    sm = torch.load(INIT / "snap/ema_330k.pt", map_location="cpu", weights_only=False)
    front = TR.Front("pae", TR.ckpt_env_smooth(sm, None)).to(dev)
    gen = init_gen(dev, a.spk, lin0)
    if a.init_ema:
        s_i = torch.load(a.init_ema, map_location="cpu", weights_only=False)
        assert s_i.get("spk", "table") == a.spk and (s_i.get("ppg", "soft") == a.ppg or a.allow_ppg_change), "init_ema の構成が違う(事後の形を意図して変えるなら --allow_ppg_change)"
        cur = gen.state_dict()
        sd_i = {k: v for k, v in s_i["ema"].items() if k in cur and cur[k].shape == v.shape}
        if CVIN["on"]:
            for k, v in s_i["ema"].items():
                if k in cur and k not in sd_i and v.dim() >= 2 and cur[k].shape[0] == v.shape[0] and cur[k].shape[2:] == v.shape[2:] and cur[k].shape[1] == v.shape[1] + CVIN["d"]:
                    w_ = torch.zeros_like(cur[k])
                    w_[:, :v.shape[1]] = v
                    sd_i[k] = w_
                    print("入力の列を 0 で足した鍵", k, tuple(v.shape), "→", tuple(cur[k].shape), flush=True)
        miss = gen.load_state_dict(sd_i, strict=not (a.ctx or a.inv_w > 0))
        if len(sd_i) < len(s_i["ema"]):
            print("形の違う鍵を除いた(新しく 0 から)", len(s_i["ema"]) - len(sd_i), flush=True)
        print("生成器の初期値", a.init_ema, "step", s_i["step"], "足りない鍵(新しい部品)", len(getattr(miss, "missing_keys", [])), flush=True)
    print("params (M)", round(sum(p.numel() for p in gen.parameters()) / 1e6, 3), "GMAC/s", round(R.macs_per_second(gen) / 1e9, 2), flush=True)
    from bigvgan.discriminators import MultiPeriodDiscriminator
    mpd = MultiPeriodDiscriminator(SimpleNamespace(mpd_reshapes=[2, 3, 5, 7, 11], use_spectral_norm=False, discriminator_channel_mult=1)).to(dev)
    mrd = MRDLog().to(dev)
    sl = torch.load(INIT / "last.pt", map_location="cpu", weights_only=False)
    mpd.load_state_dict(sl["mpd"]); mrd.load_state_dict(sl["mrd"])
    del sl
    dparams = list(mpd.parameters()) + list(mrd.parameters())
    gd, opt_gd = None, None
    if a.inv_w > 0:
        gd = GenderD().to(dev)
        opt_gd = torch.optim.AdamW(gd.parameters(), lr=a.lr, betas=(0.8, 0.99))
        assert a.x_nm > 0, "--inv_w には男声の交差の標本(--x_nm)が要る"
    slm, opt_s = None, None
    if a.slm_w > 0:
        slm = SLMPool(a.slm_model).to(dev)
        opt_s = torch.optim.AdamW(slm.head_params(), lr=a.lr, betas=(0.8, 0.99))
        print("SSL 判別器", a.slm_model, "head params (M)", round(sum(q.numel() for q in slm.head_params()) / 1e6, 3), flush=True)
    psd, opt_p, ps_taps = None, None, None
    if a.psd_w > 0:
        import psd_disc as PS
        psd = PS.PSD(a.psd_ch).to(dev)
        opt_p = torch.optim.AdamW(psd.parameters(), lr=a.lr, betas=(0.8, 0.99))
        ps_taps = PS.lpf_taps().to(dev)
        print("f0 で条件づけた判別器 PSD params (M)", round(sum(q.numel() for q in psd.parameters()) / 1e6, 3), "重み", a.psd_w, flush=True)
    od, opt_o, obuf = None, None, []
    if a.x_orig > 0:
        od = OrigD(slm.ssl.config.hidden_size * (slm.ssl.config.num_hidden_layers + 1), a.orig_ch).to(dev)
        opt_o = torch.optim.AdamW(od.parameters(), lr=a.orig_lr, betas=(0.8, 0.99), weight_decay=1e-3)
        print("対の由来の判別器 params (M)", round(sum(q.numel() for q in od.parameters()) / 1e6, 3), "重み", a.x_orig, "過去の対", a.orig_buf, flush=True)
    opt_g = torch.optim.AdamW(gen.parameters(), lr=a.lr, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(dparams, lr=a.lr, betas=(0.8, 0.99))
    ema = {k: v.detach().clone() for k, v in gen.state_dict().items()}
    evg = init_gen(dev, a.spk, lin0)
    mels = [build_mel(nf, hm, nm).to(dev) for nf, hm, nm in MEL_SPECS]
    sr_items, pv_items = eval_items(held, a.spk, dev)
    ffi = ff_items(a.ff_eval) if a.ff_eval else []
    if ffi:
        print("eval items FF", len(ffi), flush=True)
    dyn_gs = torch.tensor([float(v) for v in a.dyn_g.split(",")], device=dev)
    dyn_on = bool((dyn_gs > 0).any())
    if dyn_on:
        assert a.spk == "table", "動きの入力は表の経路だけ"
        assert a.x_nm == 0 or len(xpool.tab) == len(xpool.mu), f"男声の登録が足りない {len(xpool.tab)} / {len(xpool.mu)}(d291_maleenroll.py)"
    op_tab = {}
    for nm, *_ in pv_items:
        pi = int(nm[1:5])
        if pi not in op_tab:
            op_tab[pi] = load_enroll(H / "dev" / "d85_work" / f"op_{pi:04d}")[0]
    t_op = lambda nm, spk: spk[0] if nm.endswith("_SR") else (load_enroll(H / a.ff_eval[int(nm[1])] / "d85_work" / f"op_{nm[2:6]}")[0] if nm.endswith("_FF") else op_tab[int(nm[1:5])])
    if a.c0low:
        C0LOW.update(on=True, v=[float(x) for x in np.load(a.c0low)], a=0.78)
        print("音量の条件 = 低い帯域の水準(c0low)・v[1:4]", [round(x, 3) for x in C0LOW["v"][1:4]], flush=True)
    if a.opslow_train > 0:
        OPSLOW.update(on=True, tau=a.opslow_tau, gain=a.opslow_train)
        print("遅い揺らぎ = 操作者の揺らぎ(学習・評価とも)利得", a.opslow_train, "時定数", a.opslow_tau, flush=True)
    print("eval items SR", len(sr_items), "PV", len(pv_items), flush=True)
    la = future_invariance(gen, sr_items, dev)
    print(f"生成器の未来不変性(実音声由来の条件・励起・1 bit): 先読み {la} sample", flush=True)
    if la != 0:
        print("未来不変性 FAIL: 起動しない", flush=True)
        return 1
    loader = iter(torch.utils.data.DataLoader(TR.DS(rows, rows, 11, spk_idx=spk_idx, ctx=T1.PRE, pert=a.pert, pert_c0_only=a.pert_c0_only, sexpert=a.sexpert), batch_size=a.bs, num_workers=a.workers, pin_memory=True,
                                              persistent_workers=True, prefetch_factor=4))
    sloader = None
    if a.sm_n > 0:
        sloader = iter(torch.utils.data.DataLoader(TR.DS(srows, srows, 12, spk_idx=sm_idx, ctx=T1.PRE, pert=a.pert, pert_c0_only=a.pert_c0_only, sexpert=a.sexpert), batch_size=a.sm_n,
                                                   num_workers=max(2, a.workers // 3), pin_memory=True, persistent_workers=True, prefetch_factor=4))
    if a.vv_tab_build:
        assert not a.vv_tab, "表を作るときは --vv_tab なし"
        n_ = torch.zeros(K, dtype=torch.float64, device=dev)
        s_ = torch.zeros(K, dtype=torch.float64, device=dev)
        for _ in range(a.c0_tab_batches):
            xa, f0, env, xl, si = [t.to(dev, non_blocking=True) for t in next(loader)][:5]
            with torch.no_grad():
                u_ = c1n(mfc(torch.cat([xl, xa[:, PA.A:PA.A + TR.SEG]], 1)))[..., -TR.T:].argmax(1).reshape(-1)
            n_ += torch.bincount(u_, minlength=K).double()
            s_ += torch.bincount(u_, weights=(f0.reshape(-1) > 0).double(), minlength=K)
        gm = float(s_.sum() / n_.sum())
        tab = torch.where(n_ >= 20, s_ / n_.clamp(min=1), torch.full_like(s_, gm)).float().cpu().numpy()
        tab.astype("<f4").tofile(a.vv_tab_build)
        rep_ = {"batches": a.c0_tab_batches, "frames": int(n_.sum()), "units_ge20": int((n_ >= 20).sum()), "global_voiced": round(gm, 4), "units_voiced_gt_0.5": int((tab > 0.5).sum())}
        Path(a.vv_tab_build).with_suffix(".json").write_text(json.dumps(rep_, indent=1))
        print("有声の表", a.vv_tab_build, json.dumps(rep_), flush=True)
        return 0
    if a.c0_tab_build:
        assert not a.c0_tab, "表を作るときは --c0_tab なし"
        n_ = torch.zeros(K, dtype=torch.float64, device=dev)
        s_ = torch.zeros(K, dtype=torch.float64, device=dev)
        for _ in range(a.c0_tab_batches):
            xa, f0, env, xl, si = [t.to(dev, non_blocking=True) for t in next(loader)][:5]
            with torch.no_grad():
                c_ = front(xa, f0, env)[:, 0].reshape(-1).double()
                u_ = c1n(mfc(torch.cat([xl, xa[:, PA.A:PA.A + TR.SEG]], 1)))[..., -TR.T:].argmax(1).reshape(-1)
            n_ += torch.bincount(u_, minlength=K).double()
            s_ += torch.bincount(u_, weights=c_, minlength=K)
        gm = float(s_.sum() / n_.sum())
        tab = torch.where(n_ >= 20, s_ / n_.clamp(min=1), torch.full_like(s_, gm)).float().cpu().numpy()
        tab.astype("<f4").tofile(a.c0_tab_build)
        rep_ = {"batches": a.c0_tab_batches, "frames": int(n_.sum()), "units_ge20": int((n_ >= 20).sum()), "global_mean": round(gm, 4), "min": round(float(tab.min()), 4), "max": round(float(tab.max()), 4)}
        Path(a.c0_tab_build).with_suffix(".json").write_text(json.dumps(rep_, indent=1))
        print("c0 の表", a.c0_tab_build, json.dumps(rep_), flush=True)
        return 0
    log = open(out / "train.jsonl", "a")
    cp_glob = str(ROOT / "results/diag_e2e1/eval/h*_CP.wav")

    def run_eval(step: int) -> str | None:
        evg.load_state_dict(ema)
        evg.eval()
        ev = out / f"ev_{step:06d}"
        ev.mkdir(exist_ok=True)
        for nm, c, lg, spk in sr_items + pv_items + ffi:
            sf.write(ev / f"{nm}.wav", render(evg, c, lg, spk, dev, a.dyn_eval, t_op(nm, spk) if (a.dyn_eval > 0 or OPSLOW["on"]) else None), N.SR)
        groups = [f"SR={ev}/h*_SR.wav", f"PV={ev}/p*_PV.wav"] + ([f"FF={ev}/f*_FF.wav"] if ffi else []) + ([f"CP={cp_glob}"] if step == 0 else [])
        r = subprocess.run([sys.executable, str(Path(__file__).parent / "eval_v1perp.py"), "--groups", *groups, "--out", str(ev / "v1perp.json")],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print("eval_v1perp 失敗", r.stderr[-2000:], flush=True)
            return "eval_v1perp 失敗"
        j = json.loads((ev / "v1perp.json").read_text())
        rec = {"step": step, **{g: {"v1": j[g]["v1"]["logit"], "v1_ci": j[g]["v1"]["logit_ci95"], "v1perp": j[g]["v1perp"]["logit"],
                                    "v1perp_ci": j[g]["v1perp"]["logit_ci95"], "auc_v1": j[g]["v1"]["auc"]} for g in j}}
        print("eval", json.dumps(rec, ensure_ascii=False), flush=True)
        log.write(json.dumps({"eval": rec}, ensure_ascii=False) + "\n")
        log.flush()
        if a.smoke or step != a.kill_step:
            return None
        why = []
        if rec["PV"]["v1perp"] > a.kill_pv:
            why.append(f"PV v1⊥ {rec['PV']['v1perp']} > {a.kill_pv}")
        if rec["SR"]["v1"] > a.kill_sr:
            why.append(f"SR v1 {rec['SR']['v1']} > {a.kill_sr}")
        if ffi and rec["FF"]["v1"] > a.kill_ff:
            why.append(f"FF v1 {rec['FF']['v1']} > {a.kill_ff}")
        return "・".join(why) or None

    if not a.skip_eval0 and run_eval(0):
        return 2
    acc: dict = {}
    t0 = time.time()
    total = 30 if a.smoke else a.steps
    SEG, T = TR.SEG, TR.T
    for step in range(1, total + 1):
        bt = [t.to(dev, non_blocking=True) for t in next(loader)]
        if sloader is not None:
            bt = [torch.cat([u, v.to(dev, non_blocking=True)]) for u, v in zip(bt, next(sloader))]
        xa, f0, env, xl, si = bt[:5]
        xp = bt[5] if len(bt) > 5 else xa
        env_o = bt[6] if len(bt) > 6 else env
        with torch.no_grad():
            ct = front(xp, f0, env)
            ct_o = ct if env_o is env else front(xa, f0, env_o)
            if C0LOW["on"]:
                ct = c0low_t(ct)
            xc = torch.cat([xl, xp[:, PA.A:PA.A + SEG]], 1)
            p = ppg_form(c1n(mfc(xc))[..., -T:], a.ppg)
            cvf = cvin_frames(xc, T) if CVIN["on"] else None
            p4 = f4n(mfc(xc))[..., -T:] if a.per_dyn else None
            if cvf is not None and a.cv_noise > 0:
                cvf = cvf + a.cv_noise * torch.randn_like(cvf)
            spk = (tabT[si], p4 if a.per_dyn else tabP[si]) if a.spk == "table" else embS[si]
            if f1n is not None:
                need = FE.CTX48 + T * N.HOP
                xf = xc[:, -need:] if xc.shape[1] >= need else F.pad(xc, (need - xc.shape[1], 0))
                v1_, p1_ = f1n(f1f(xf, T))
                f0_t = f0
                f0 = FE.decode(v1_, p1_, 0.5).to(f0.dtype)
                ct = torch.cat([ct[:, :25], torch.where(f0 > 0, torch.log(f0.clamp(min=1.0) / 200.0), torch.zeros_like(f0))[:, None], (f0 > 0).to(ct.dtype)[:, None], ct[:, 27:]], 1)
                if a.smoke and step == 1:
                    vt_, vf_ = f0_t > 0, f0 > 0
                    print("F1 の検査(step 1): 有声の一致", round(float((vt_ == vf_).float().mean()), 4), "F1 のみ有声", round(float((vf_ & ~vt_).float().mean()), 4), "教師のみ", round(float((vt_ & ~vf_).float().mean()), 4),
                          "両方有声の |Δ半音| 中央値", round(float((12 * torch.log2(f0[vt_ & vf_] / f0_t[vt_ & vf_])).abs().median()), 3), flush=True)
            if F0V["on"]:
                f0_t = f0
                f0 = f0v_t(f0, p)
                ct = torch.cat([ct[:, :25], torch.where(f0 > 0, torch.log(f0.clamp(min=1.0) / 200.0), torch.zeros_like(f0))[:, None], (f0 > 0).to(ct.dtype)[:, None], ct[:, 27:]], 1)
                if a.smoke and step == 1:
                    vt_, vf_ = f0_t > 0, f0 > 0
                    print("f0・有声の中立化の検査(step 1): 有声の一致", round(float((vt_ == vf_).float().mean()), 4), "中立のみ有声", round(float((vf_ & ~vt_).float().mean()), 4), "教師のみ", round(float((vt_ & ~vf_).float().mean()), 4),
                          "両方有声の |Δ半音| 中央値", round(float((12 * torch.log2(f0[vt_ & vf_] / f0_t[vt_ & vf_])).abs().median()), 3), flush=True)
            exc = TR.excitation(f0, SEG)
            tgt = xa[:, PA.A - N.DELAY:PA.A - N.DELAY + SEG]
        zs = None
        if SLOW["on"]:
            with torch.no_grad():
                if a.opslow_train > 0:
                    zs = a.opslow_train * opslow_train(ct_o[:, 1:25] - torch.einsum("bdk,bkt->bdt", tabT[si], p) / 10, a.opslow_tau)
                else:
                    zs = slow_real(ct_o[:, 1:25], torch.einsum("bdk,bkt->bdt", tabT[si], p) / 10)
                nz_ = torch.rand(zs.shape[0], device=dev) < (0.0 if a.opslow_train > 0 else a.slow_pnoise)
                if bool(nz_.any()):
                    zs[nz_] = slow_noise(int(nz_.sum()), zs.shape[-1], dev)
        zms = None
        if MID["on"]:
            with torch.no_grad():
                zms = mid_real(ct[:, 1:25], torch.einsum("bdk,bkt->bdt", tabT[si], p) / 10)
                nz_ = torch.rand(zms.shape[0], device=dev) < a.slow_pnoise
                if bool(nz_.any()):
                    zms[nz_] = slow_noise(int(nz_.sum()), zms.shape[-1], dev, None, MID)
        if a.static_off:
            with torch.no_grad():
                m_st = (ct_o[:, 1:25] - torch.einsum("bdk,bkt->bdt", tabT[si], p) / 10).mean(-1, keepdim=True)
            zs = m_st.expand(-1, -1, ct.shape[-1]) if zs is None else zs + m_st
            if a.smoke and step == 1:
                print("静的なずれの検査(step 1): 行ごとの |m| 平均 c1..c4", [round(float(v), 3) for v in m_st[:, :4, 0].abs().mean(0)], flush=True)
        if dyn_on and a.dyn_self:
            with torch.no_grad():
                g_sr = dyn_gs[torch.randint(len(dyn_gs), (xa.shape[0],), device=dev)][:, None, None]
                res_sr = ct_o[:, 1:25] - torch.einsum("bdk,bkt->bdt", tabT[si], p) / 10
                zs = g_sr * res_sr if zs is None else (1 - g_sr) * zs + g_sr * res_sr
                if a.smoke and step == 1:
                    print("動きの自己再構成の検査(step 1): γ", [round(float(v), 2) for v in g_sr[:, 0, 0]], "残差の RMS", round(float(res_sr.pow(2).mean().sqrt()), 4), flush=True)
        cond = gen.make(p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], spk, zs, zms, cvf=cvf)
        if dyn_on and not a.dyn_self:
            with torch.no_grad():
                g_sr = dyn_gs[torch.randint(len(dyn_gs), (xa.shape[0],), device=dev)][:, None, None]
                res_sr = ct[:, 1:25] - torch.einsum("bdk,bkt->bdt", tabT[si], p) / 10
            cond = torch.cat([cond[:, :1], cond[:, 1:25] + g_sr * res_sr, cond[:, 25:]], 1)
        y = gen(cond, exc, spk)
        yl, tl = y[:, TR.WARM:], tgt[:, TR.WARM:]
        if a.smoke and step == 1 and a.x_ct > 0:
            with torch.no_grad():
                qo = c1n(mfc(torch.cat([torch.zeros(y.shape[0], N.WIN - N.HOP, device=dev), y], 1))).argmax(1)
                pa = p.argmax(1)
                for lg_ in (0, 1, 2):
                    print("内容の一致の時刻の検査(自己再構成): lag", lg_, "argmax の一致", round(float((qo[:, 100 + lg_:] == pa[:, 100:pa.shape[1] - lg_]).float().mean()), 3), flush=True)
        yx = None
        if a.x_nm > 0 or a.x_nf > 0:
            with torch.no_grad():
                px, c0x, lfx, vx, bx, d_mu, rx, perx = [], [], [], [], [], [], [], []
                if a.x_nf > 0:
                    nf = min(a.x_nf, nbf)
                    b_ = si[torch.roll(torch.arange(nbf, device=dev), 1)][:nf]
                    px.append(p[:nf]); c0x.append(ct[:nf, 0:1]); lfx.append(ct[:nf, 25:26]); vx.append(ct[:nf, 26:27]); bx.append(b_); d_mu.append(tabMu[b_] - tabMu[si[:nf]])
                    rx.append(ct[:nf, 1:25] - torch.einsum("bdk,bkt->bdt", tabT[si[:nf]], p[:nf]) / 10)
                    if a.per_dyn:
                        perx.append(tabP[b_][:, :, None] + p4[:nf] - tabP[si[:nf]][:, :, None])
                if a.x_nm > 0:
                    cm, lgm, mum, tm = xpool.sample(a.x_nm, dev)
                    if a.x_c0shift:
                        cm = cm.clone()
                        c0_before = float(cm[:, 0][cm[:, 0] > 0.5].mean())
                        cm[:, 0] = torch.where(cm[:, 0] > 0.05, cm[:, 0] + (4.40 - xpool.last_c0lv)[:, None], cm[:, 0])
                        if a.smoke and step == 1:
                            print("男声の交差の水準ずらしの検査(step 1): 話者の水準", [round(float(v), 3) for v in xpool.last_c0lv], "発声中の c0n の平均", round(c0_before, 3), "→", round(float(cm[:, 0][cm[:, 0] > 0.5].mean()), 3), flush=True)
                    if C0LOW["on"]:
                        cm = c0low_t(cm)
                    b_ = torch.randint(len(spks), (a.x_nm,), device=dev)
                    pm_ = ppg_form(lgm, a.ppg)
                    if a.x_fixk:
                        kset = torch.tensor([float(v) for v in a.x_fixk.split(",")], device=dev)
                        dmm = kset[torch.randint(len(kset), (a.x_nm,), device=dev)] * float(np.log(2.0)) / 12.0
                        if a.smoke and step == 1:
                            print("固定の半音の交差の検査(step 1): 半音", [round(float(v) * 12 / float(np.log(2.0)), 2) for v in dmm], "目標の声域へのずれだった場合", [round(float(v) * 12 / float(np.log(2.0)), 2) for v in (tabMu[b_] - mum)], flush=True)
                    else:
                        dmm = tabMu[b_] - mum
                    px.append(pm_); c0x.append(cm[:, 0:1]); lfx.append(cm[:, 25:26]); vx.append((cm[:, 26:27] > 0.5).float()); bx.append(b_); d_mu.append(dmm)
                    rx.append(cm[:, 1:25] - torch.einsum("bdk,bkt->bdt", tm, pm_) / 10)
                    if a.per_dyn:
                        perx.append(tabP[b_][:, :, None] + xpool.last_f4 - xpool.last_p75[:, :, None])
                px, c0x, lfx, vx, bx, d_mu, rx = (torch.cat(z) for z in (px, c0x, lfx, vx, bx, d_mu, rx))
                lfx = torch.where(vx > 0.5, lfx + d_mu[:, None, None], torch.zeros_like(lfx))
                f0x = torch.where(vx[:, 0] > 0.5, 200.0 * torch.exp(lfx[:, 0]), torch.zeros_like(lfx[:, 0]))
                if F0V["on"]:
                    f0x = f0v_t(f0x, px)
                    vx = (f0x > 0).to(vx.dtype)[:, None]
                    lfx = torch.where(vx > 0.5, torch.log(f0x.clamp(min=1.0) / 200.0)[:, None], torch.zeros_like(lfx))
                excx = TR.excitation(f0x, SEG)
                spkx = (tabT[bx], torch.cat(perx) if a.per_dyn else tabP[bx]) if a.spk == "table" else embS[bx]
            cvfx = None
            if cvf is not None:
                cvfx = torch.cat(([cvf[:min(a.x_nf, xa.shape[0])]] if a.x_nf > 0 else []) + ([xpool.last_cvf + a.cv_noise * torch.randn_like(xpool.last_cvf)] if a.x_nm > 0 else []))
            zx = a.opslow_train * opslow_train(rx, a.opslow_tau) if a.opslow_train > 0 else None
            if a.x_noslow and a.x_nm > 0:
                assert zx is None and SLOW["on"], "--x_noslow は乱数の遅い揺らぎの学習(--slow_stats・--opslow_train なし)で使う"
                zx = slow_noise(px.shape[0], px.shape[-1], dev)
                zx[px.shape[0] - a.x_nm:] = 0.0
            condx = gen.make(px, c0x, lfx, vx, spkx, zx, cvf=cvfx)
            if dyn_on and not a.dyn_self:
                g_x = dyn_gs[torch.randint(len(dyn_gs), (px.shape[0],), device=dev)][:, None, None]
                condx = torch.cat([condx[:, :1], condx[:, 1:25] + g_x * rx, condx[:, 25:]], 1)
            yx_full = gen(condx, excx, spkx)
            yx = yx_full[:, TR.WARM:]
        ysw, lmsw, kinds = None, None, None
        if spool is not None:
            with torch.no_grad():
                lsw_, csw, tsw, Tsw, Psw, kinds = spool.sample(a.swap_n, dev)
                psw = ppg_form(lsw_, a.ppg)
                f0sw = torch.where(csw[:, 26] > 0.5, 200.0 * torch.exp(csw[:, 25]), torch.zeros_like(csw[:, 25]))
                exsw = TR.excitation(f0sw, SEG)
                spsw = (Tsw, Psw)
            ysw = gen(gen.make(psw, csw[:, 0:1], csw[:, 25:26], csw[:, 26:27], spsw), exsw, spsw)[:, TR.WARM:]
            tswl = tsw[:, TR.WARM:]
            lmsw = logmel_l1(ysw[:, None], tswl[:, None], mels)
        lm = logmel_l1(yl[:nbf, None], tl[:nbf, None], mels)
        lmm = logmel_l1(yl[nbf:, None], tl[nbf:, None], mels) if sloader is not None else None
        if a.smoke and step == 1 and sloader is not None:
            with torch.no_grad():
                fm_ = f0[nbf:]
                print("男声の自己再構成の検査(step 1): 行", xa.shape[0], "女声", nbf, "表の番号", [int(v) for v in si], "表の f0 の中央値 Hz", [round(float(torch.exp(tabMu[v])), 1) for v in si],
                      "男声の行の教師の f0 の中央値", round(float(fm_[fm_ > 0].median()), 1), "log-mel L1 女声 / 男声", round(float(lm), 4), round(float(lmm), 4), flush=True)
        lh1 = None
        if a.h1_w > 0:
            lh1, okh, h12 = harm_l1(yl, tl, f0[:, TR.WARM // N.HOP:])
            if a.smoke and step == 1:
                print("低い倍音の検査(step 1): 損失", round(float(lh1.detach()), 4), "有効な枠", round(okh, 3), "H1 − H2 dB 出力 / 実録音", [round(v, 2) for v in h12],
                      "| 自分どうし", round(float(harm_l1(tl, tl, f0[:, TR.WARM // N.HOP:])[0]), 6), flush=True)
        lper = None
        if a.per_w > 0:
            f0s = torch.cat([f0[:, :1], f0[:, :-1]], 1)[:, :T]
            pad = lambda z: F.pad(z, (PA.A, PA.A))
            with torch.no_grad():
                pr_ = PA.periodicity(pad(tgt), f0s)
            py_ = PA.periodicity(pad(y), f0s)
            w0 = TR.WARM // N.HOP + 2
            mv = (f0s[:, w0:-2] > 0).float()[:, None]
            lper = ((py_[..., w0:-2] - pr_[..., w0:-2]).abs() * mv).sum() / (mv.sum() * 4 + 1e-6)
            if a.smoke and step == 1:
                print("周期性の一致の検査(step 1): L1", round(float(lper), 4), "| 実録音の有声の平均", [round(float(v), 3) for v in (pr_[..., w0:-2] * mv).sum((0, 2)) / (mv.sum() + 1e-6)], flush=True)
        loss = a.lm_w * lm
        if lmm is not None:
            loss = loss + a.sm_w * a.lm_w * lmm
        if lh1 is not None:
            loss = loss + a.h1_w * lh1
        if lper is not None:
            loss = loss + a.per_w * lper
        if lmsw is not None:
            loss = loss + a.swap_w * a.lm_w * lmsw
            if a.smoke and step == 1:
                with torch.no_grad():
                    rr = lambda z: float(20 * torch.log10(z.pow(2).mean().sqrt() + 1e-9))
                    print("入れ替えの標本の検査(step 1): 種類", kinds, "log-mel L1 自己", round(float(lm), 4), "入れ替え", round(float(lmsw), 4),
                          "| RMS dB 出力 / 目標", [(round(rr(ysw[i]), 1), round(rr(tswl[i]), 1)) for i in range(ysw.shape[0])], flush=True)
        off = random.randrange(0, yl.shape[-1] - 16384)
        nd_ = xa.shape[0] if a.sm_adv else nbf
        r_, f_ = tl[:nd_, off:off + 16384], yl[:nd_, off:off + 16384]
        opt_d.zero_grad(set_to_none=True)
        a1, b1, _, _ = mpd(r_[:, None], f_.detach()[:, None])
        a2, b2, _, _ = mrd(r_, f_.detach())
        dl = sum(((p_.float() - 1) ** 2).mean() + (q.float() ** 2).mean() for p_, q in zip(a1 + a2, b1 + b2))
        if yx is not None:
            fx_ = yx[:, off:off + 16384]
            _, bx1, _, _ = mpd(r_[:fx_.shape[0], None], fx_.detach()[:, None])
            _, bx2, _, _ = mrd(r_[:fx_.shape[0]], fx_.detach())
            dl = dl + a.x_w * sum((q.float() ** 2).mean() for q in bx1 + bx2)
        if ysw is not None:
            rs_, fs_ = tswl[:, off:off + 16384], ysw[:, off:off + 16384]
            a1s, b1s, _, _ = mpd(rs_[:, None], fs_.detach()[:, None])
            a2s, b2s, _, _ = mrd(rs_, fs_.detach())
            dl = dl + a.swap_w * sum(((p_.float() - 1) ** 2).mean() + (q.float() ** 2).mean() for p_, q in zip(a1s + a2s, b1s + b2s))
        if torch.isfinite(dl):
            dl.backward()
            torch.nn.utils.clip_grad_norm_(dparams, 500.0)
            opt_d.step()
        for p_ in dparams:
            p_.requires_grad_(False)
        _, b1, f1r, f1g = mpd(r_[:, None], f_[:, None])
        _, b2, f2r, f2g = mrd(r_, f_)
        advx = None
        if yx is not None:
            _, bx1, _, _ = mpd(r_[:fx_.shape[0], None], fx_[:, None])
            _, bx2, _, _ = mrd(r_[:fx_.shape[0]], fx_)
            advx = sum(((q.float() - 1) ** 2).mean() for q in bx1 + bx2)
        advsw, fmsw = None, None
        if ysw is not None:
            _, b1s, f1rs, f1gs = mpd(rs_[:, None], fs_[:, None])
            _, b2s, f2rs, f2gs = mrd(rs_, fs_)
            advsw = sum(((q.float() - 1) ** 2).mean() for q in b1s + b2s)
            fmsw = sum(F.l1_loss(u.detach().float(), v.float()) for xr, xg in zip(f1rs + f2rs, f1gs + f2gs) for u, v in zip(xr, xg))
        for p_ in dparams:
            p_.requires_grad_(True)
        adv = sum(((q.float() - 1) ** 2).mean() for q in b1 + b2)
        sadv = None
        if slm is not None:
            with torch.no_grad():
                hr = slm.feats(tl[:nbf])
            fakes = [yl[:nbf]] + ([yx] if yx is not None else [])
            es = [embX[si[:nbf]], embX[bx] if yx is not None else None] if a.slm_cond else [None, None]
            er = embX[si[:nbf]] if a.slm_cond else None
            opt_s.zero_grad(set_to_none=True)
            sl = sum(((o.float() - 1) ** 2).mean() for o in slm.score(hr, er)) + sum((o.float() ** 2).mean() for fk, ee in zip(fakes, es) for o in slm(fk.detach(), ee))
            if a.slm_cond:
                sl = sl + (slm.score(hr, embX[si[:nbf][torch.roll(torch.arange(nbf, device=dev), 1)]])[1].float() ** 2).mean()
            if torch.isfinite(sl):
                sl.backward()
                torch.nn.utils.clip_grad_norm_(slm.head_params(), 500.0)
                opt_s.step()
            for q_ in slm.head_params():
                q_.requires_grad_(False)
            sadv = sum(((o.float() - 1) ** 2).mean() for fk, ee in zip(fakes, es) for o in slm(fk, ee))
            for q_ in slm.head_params():
                q_.requires_grad_(True)
        lpsd, dps = None, None
        if psd is not None:
            with torch.no_grad():
                zr, okr = PS.psgram(tgt, f0, TR.WARM, 48, taps=ps_taps)
                zf = [PS.psgram(u_, g_, TR.WARM, 48, taps=ps_taps) for u_, g_ in [(y.detach(), f0)] + ([(yx_full.detach(), f0x)] if yx is not None else [])]
            opt_p.zero_grad(set_to_none=True)
            dps = PS.masked_ls(psd(zr), okr, 1.0) + sum(PS.masked_ls(psd(z_), o_, 0.0) for z_, o_ in zf)
            if torch.isfinite(dps):
                dps.backward()
                torch.nn.utils.clip_grad_norm_(psd.parameters(), 500.0)
                opt_p.step()
            for q_ in psd.parameters():
                q_.requires_grad_(False)
            gz = [PS.psgram(u_, g_, TR.WARM, 48, taps=ps_taps) for u_, g_ in [(y, f0)] + ([(yx_full, f0x)] if yx is not None else [])]
            lpsd = sum(PS.masked_ls(psd(z_), o_, 1.0) for z_, o_ in gz)
            for q_ in psd.parameters():
                q_.requires_grad_(True)
            loss = loss + a.psd_w * lpsd
            if a.smoke and step == 1:
                print("f0 で条件づけた判別器の検査(step 1): 本物", PS.z_stats(zr, okr), "有効な枠", round(float(okr.float().mean()), 3), "| 自己", PS.z_stats(*zf[0]),
                      "| 交差", PS.z_stats(*zf[1]) if len(zf) > 1 else None, "有効な枠", round(float(zf[1][1].float().mean()), 3) if len(zf) > 1 else None, flush=True)
        fm = sum(F.l1_loss(u.detach().float(), v.float()) for xr, xg in zip(f1r + f2r, f1g + f2g) for u, v in zip(xr, xg))
        loss = loss + adv + 2 * fm
        if advx is not None:
            loss = loss + a.x_w * advx
        if advsw is not None:
            loss = loss + a.swap_w * (advsw + 2 * fmsw)
        linv = None
        if gd is not None and yx is not None:
            pf_ = gen.adapt(p)
            pm2 = gen.adapt(pm_)
            opt_gd.zero_grad(set_to_none=True)
            lg_ = F.binary_cross_entropy_with_logits(gd(pf_.detach()), torch.zeros(pf_.shape[0], device=dev)) + F.binary_cross_entropy_with_logits(gd(pm2.detach()), torch.ones(pm2.shape[0], device=dev))
            lg_.backward()
            opt_gd.step()
            for q_ in gd.parameters():
                q_.requires_grad_(False)
            linv = F.binary_cross_entropy_with_logits(gd(pm2), torch.zeros(pm2.shape[0], device=dev))
            for q_ in gd.parameters():
                q_.requires_grad_(True)
            loss = loss + a.inv_w * linv
        lct = None
        lcv, lcvx = None, None
        if cvm is not None:
            with torch.no_grad():
                hcv = cvfeat(tl)
            if a.cv_w > 0:
                lcv = (1 - (cvfeat(yl) * hcv).sum(-1)).mean()
                loss = loss + a.cv_w * lcv
            if a.cv_xw > 0 and yx is not None and a.x_nf > 0:
                nf_ = min(a.x_nf, xa.shape[0])
                lcvx = (1 - (cvfeat(yx[:nf_]) * hcv[:nf_]).sum(-1)).mean()
                loss = loss + a.cv_xw * lcvx
            if a.smoke and step == 1:
                with torch.no_grad():
                    sh = (1 - (hcv * torch.roll(hcv, 1, 0)).sum(-1)).mean()
                print("内容の特徴の一致の検査(step 1): 自己", None if lcv is None else round(float(lcv), 4), "交差", None if lcvx is None else round(float(lcvx), 4), "| 別の発話どうしの基準", round(float(sh), 4), flush=True)
        if a.x_ct > 0 and yx is not None:
            nct = min(a.x_nf, xa.shape[0]) if a.ct_female_only else yx_full.shape[0]
            qx = c1n(mfc(torch.cat([torch.zeros(nct, N.WIN - N.HOP, device=dev), yx_full[:nct]], 1)))
            L = min(qx.shape[-1] - a.ct_lag, px.shape[-1])
            lq = torch.log_softmax(qx[..., 100 + a.ct_lag:L + a.ct_lag], 1)
            pt = px[:nct, :, 100:L]
            if a.ct_hard:
                lct = F.nll_loss(lq, pt.argmax(1))
            else:
                lct = -(pt * lq).sum(1).mean() + (pt * torch.log(pt.clamp(min=1e-8))).sum(1).mean()
            loss = loss + a.x_ct * lct
        lpair, lorig, dorig, oacc = None, None, None, None
        if ppool is not None:
            stat = lambda h: torch.cat([h.mean(-1), h.std(-1)], -1)
            lps, hms, hfs = [], [], []
            for _ in range(a.pair_n):
                b_ = int(torch.randint(len(spks), (1,)))
                ys_ = []
                for gi, (c_, lg_p, mu_) in enumerate(ppool.sample(int(a.pair_sec * N.SR / N.HOP))):
                    with torch.no_grad():
                        cc_ = torch.from_numpy(c_).float().to(dev)[None]
                        pp_ = ppg_form(torch.from_numpy(lg_p).float().to(dev)[None], a.ppg)
                        vv_ = (cc_[:, 26:27] > 0.5).float()
                        lf_ = torch.where(vv_ > 0.5, cc_[:, 25:26] + (tabMu[b_] - mu_), torch.zeros_like(vv_))
                        f0_ = torch.where(vv_[:, 0] > 0.5, 200.0 * torch.exp(lf_[:, 0]), torch.zeros_like(lf_[:, 0]))
                        ex_ = TR.excitation(f0_, cc_.shape[-1] * N.HOP)
                        sp_ = (tabT[b_:b_ + 1], tabP[b_:b_ + 1])
                    if gi == 0:
                        y_ = gen(gen.make(pp_, cc_[:, 0:1], lf_, vv_, sp_), ex_, sp_)[:, TR.WARM:]
                        h_ = slm.feats(y_)
                        ys_.append(stat(h_))
                        hms.append(h_)
                    else:
                        with torch.no_grad():
                            y_ = gen(gen.make(pp_, cc_[:, 0:1], lf_, vv_, sp_), ex_, sp_)[:, TR.WARM:]
                            h_ = slm.feats(y_)
                            ys_.append(stat(h_))
                            hfs.append(h_)
                lps.append((ys_[0] - ys_[1]).abs().mean())
            lpair = torch.stack(lps).mean()
            if a.pair_w > 0:
                loss = loss + a.pair_w * lpair
            if od is not None:
                cur = [(hm.detach().half(), hf.half()) for hm, hf in zip(hms, hfs)]
                rep = random.sample(obuf, min(a.orig_nb, len(obuf)))
                opt_o.zero_grad(set_to_none=True)
                zc = torch.cat([od(hm, hf) for hm, hf in cur])
                dorig = F.softplus(-torch.cat([zc] + [od(hm, hf) for hm, hf in rep])).mean()
                oacc = (zc.detach() > 0).float().mean()
                dorig.backward()
                torch.nn.utils.clip_grad_norm_(od.parameters(), 100.0)
                opt_o.step()
                obuf = (obuf + cur)[-a.orig_buf:]
                for q_ in od.parameters():
                    q_.requires_grad_(False)
                z_ = torch.cat([od(hm, hf) for hm, hf in zip(hms, hfs)])
                lorig = 0.5 * (F.softplus(z_) + F.softplus(-z_)).mean() - float(np.log(2.0))
                for q_ in od.parameters():
                    q_.requires_grad_(True)
                loss = loss + a.x_orig * lorig
        ldl = None
        if ltas is not None and yx is not None:
            nf = min(a.x_nf, xa.shape[0])
            b_ = si[torch.roll(torch.arange(xa.shape[0], device=dev), 1)][:nf]
            with torch.no_grad():
                l_op = ltas(tl[:nf])
                d_ref = tabL[b_] - tabL[si[:nf]]
            d_out = ltas(yx[:nf]) - l_op
            ldl = (d_out - d_ref)[:, 2:].abs().mean()
            loss = loss + a.x_dltas * ldl
            if a.smoke and step == 1:
                print("話者差の帯域の検査(step 1): |d_ref| 平均", round(float(d_ref[:, 2:].abs().mean()), 2), "dB・|d_out| 平均", round(float(d_out[:, 2:].abs().mean()), 2), "dB・損失", round(float(ldl), 3),
                      "| 自己の検査 LTAS(実録音)− 参照 平均", round(float((l_op - tabL[si[:nf]])[:, 2:].abs().mean()), 2), "dB", flush=True)
        lid = None
        if ecx is not None and yx is not None:
            lid = (1 - (ecx(yx) * embX[bx]).sum(-1)).mean()
            loss = loss + a.x_id * lid
        lfl, flx_ = None, None
        if a.fline_w > 0:
            fl_s = fline(yl[:nbf])
            if sloader is not None:
                lflm = F.relu(fline(yl[nbf:]) - a.fline_m).mean()
                loss = loss + a.sm_w * a.fline_w * lflm
            flxv = fline(yx) if yx is not None else None
            flx_ = flxv.mean() if flxv is not None else None
            flv = torch.cat([fl_s] + ([flxv] if flxv is not None else []))
            lfl = F.relu(flv - a.fline_m).mean()
            loss = loss + a.fline_w * lfl
            fl_s = fl_s.mean()
        if sadv is not None:
            loss = loss + a.slm_w * sadv
        opt_g.zero_grad(set_to_none=True)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        if not torch.isfinite(torch.nn.utils.clip_grad_norm_(gen.parameters(), 500.0)):
            opt_g.zero_grad(set_to_none=True)
            continue
        opt_g.step()
        with torch.no_grad():
            for k, v in gen.state_dict().items():
                if v.dtype.is_floating_point:
                    ema[k].mul_(0.999).add_(v.detach(), alpha=0.001)
        for k_, v_ in ({"lm": lm, "adv": adv, "fm": fm, "dl": dl} | ({"x_adv": advx} if advx is not None else {}) | ({"slm_adv": sadv, "slm_dl": sl} if sadv is not None else {}) | ({"x_id": lid} if lid is not None else {}) | ({"x_ct": lct} if lct is not None else {}) | ({"cv": lcv} if lcv is not None else {}) | ({"per": lper} if lper is not None else {}) | ({"cv_x": lcvx} if lcvx is not None else {}) | ({"x_pair": lpair} if lpair is not None else {}) | ({"x_orig": lorig, "orig_d": dorig, "orig_acc": oacc} if lorig is not None else {}) | ({"inv": linv, "gd": lg_} if linv is not None else {}) | ({"sw_lm": lmsw, "sw_adv": advsw, "sw_fm": fmsw} if lmsw is not None else {}) | ({"x_dltas": ldl} if ldl is not None else {}) | ({"fline": fl_s, "fline_x": flx_ if flx_ is not None else fl_s, "l_fline": lfl} if lfl is not None else {}) | ({"psd": lpsd, "psd_d": dps} if lpsd is not None else {}) | ({"h1": lh1} if lh1 is not None else {}) | ({"sm_lm": lmm} if lmm is not None else {})).items():
            acc.setdefault(k_, []).append(float(v_.detach()))
        if step % a.log_every == 0 or a.smoke:
            rr = {"step": step, "min": round((time.time() - t0) / 60, 1), "sec": round(time.time() - t0, 2), **{k_: round(float(np.mean(v_)), 4) for k_, v_ in acc.items()}}
            acc = {}
            print(json.dumps(rr), flush=True)
            log.write(json.dumps(rr) + "\n")
            log.flush()
        if step % a.eval_every == 0 or (a.smoke and step == total):
            ck = {"ema": ema, "step": step, "cfg": gen.cfg, "d_ppg": D_PPG, "spk": a.spk, "ppg": a.ppg, "dyn_g": a.dyn_g, "dyn_self": a.dyn_self, "dyn_eval": a.dyn_eval, "ctx": a.ctx, "adapt": a.inv_w > 0, "ctx_ch": a.ctx_ch, "x_pair": a.x_pair, "pair_w": a.pair_w, "x_orig": a.x_orig, "slow": a.slow_stats, "slow_sigma": a.slow_sigma, "mid": a.mid_stats, "mid_sigma": a.mid_sigma, "x_swap": a.x_swap, "swap_w": a.swap_w, "pert": a.pert, "pert_c0_only": a.pert_c0_only, "x_dltas": a.x_dltas, "static_off": a.static_off, "fline_w": a.fline_w, "sexpert": a.sexpert, "opslow_tau": a.opslow_tau if a.opslow_train > 0 else 0.0, "opslow_gain": a.opslow_train, "c0low": bool(a.c0low), "c0low_v": C0LOW["v"] if a.c0low else None, "c0low_a": C0LOW["a"], "cv_w": a.cv_w, "cv_xw": a.cv_xw, "cvin": a.cvin, "cv_noise": a.cv_noise, "per_dyn": a.per_dyn, "per_w": a.per_w, "f1_train": a.f1_train, "x_c0shift": a.x_c0shift, "c0_tab": a.c0_tab, "c2_const": a.c2_const, "vv_tab": a.vv_tab, "x_noslow": a.x_noslow, "h1_w": a.h1_w, "psd_w": a.psd_w, "psd": psd.state_dict() if psd is not None else None}
            ck.update(sm_manifest=a.sm_manifest, sm_n=a.sm_n, sm_w=a.sm_w, sm_adv=a.sm_adv)
            TR.atomic_save(ck, out / "last_ema.pt")
            if not a.smoke:
                (out / "snap").mkdir(exist_ok=True)
                TR.atomic_save(ck, out / "snap" / f"ema_{step:06d}.pt")
            why = run_eval(step)
            if why:
                (out / "STOP.json").write_text(json.dumps({"step": step, "why": why}, ensure_ascii=False, indent=1))
                print("中止条件:", why, flush=True)
                return 2
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
