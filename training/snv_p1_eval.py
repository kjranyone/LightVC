"""P1 の評価(synth_nam_vc §8・運用の条件): 操作者の音の前段 v3 → E(操作者の登録の平均を引く)→ C(目標の参照の構造・高さ = 操作者の高さ + 固定の k 半音)→ rvoc3hi → 48 kHz。
test: M = 本物の男声 12 区間(外郎売・namikawa・登録 = 先頭 30 s)→ A0F の目標 24 声(構造の参照 = j 0〜2)・k ∈ {0, 5, 7, 12}・ECAPA + LDA の 24 声中の 1 位(登録 = j 0〜2 の元の音・M_alt = j 6〜7)。
      S = 目標の j 3〜5 の自己の reamp(参照 j 0〜2)の 1 位・CER = k 0 / 7 × 目標 4 声の large-v3 の仮名の CER(源の書き起こし 対 出力)・
      TEX = N0r の女声の音素の無い発声 84 本の自己の reamp(参照 = その声の j 0〜2)・UTMOS = M の出力の平均。差 = 系の対ごとに目標の声で束ねた 95% 区間。
dev: VCTK の男声 6 人 × 2 本(登録 = 別の 2 本)→ P1 の検証の女声 30 人(構造 = t00〜t02・登録 = t03〜t05 の元の音)・k = 7・ECAPA の 30 声中の 1 位。
    cd training && uv run --no-sync python snv_p1_eval.py dev|test <tag> [<tag> ...] [judge=results/snv_p1c_judge/judge.pt](dev で判定器の 30 声中の 1 位も)
test の warp=α = 操作者の源と登録の包絡を先に一律の伸縮 α で(演奏の入力だけが変わる探り・高さと有声は元のまま)。
系の名前に +sep[:<tag>[@ck]](例 snv_i10_c_pr+sep:snv_i14_sep・= は物差しの引数と衝突するので :)= paper_snv §5 M20: 操作者の側(源・操作者の登録・自己の構造 = 鍵が self| の参照)を前段 D(train_snv_sep.features・発話の辞書の wav か wav_path から)の出力に置き換える。目標の参照は元のまま。
+mask:<tag>[@ck](paper_snv §5 M23)= 同じ操作者の側を、前段 v3 に時間周波数の重みを掛けた特徴(train_snv_mask.features)に置き換える。+maskd:<tag>[@ck] = 同じで、有声と高さを重みで濾した YIN(snv_p0_front.f0_track_myin)で決める(paper_snv §9 O13)。
系の名前に +rewav = 操作者の側を、波形から前段 v3 を解析し直した特徴(重み 1 の train_snv_mask.features_with_mask = 門が閉じたままの +mask と同じ計算)に置き換える。保存の特徴と波形の作りが違う組(開発の組は 16 bit の wav)で、+mask の対照にする。
系の名前に +vtl(例 snv_i10_c_pr+vtl)= paper_snv §5 M17: 操作者の登録だけで声道の長さの伸縮 α を当てはめる(格子 0.7〜1.45・推論の E の内容の出力の有声の平均を、登録の高さの中央値で決めた性の学習の声の分布で z にしたノルムを最小に・母集団は results/<名>/vtl_pop.npz)。源と E の構造に同じ α・目標の参照はそのまま。
保存に e_play = wst があれば(paper_snv §5 M28)、推論の E は snv_encw.EncW(登録の統計で白くした演奏 → 軌跡)。
保存に e_lift があれば(paper_snv §5 M19)、推論の E が読む全ての入力(源・操作者の登録・目標の参照の鍵)の包絡の c_K〜c24 を保存の lift_mean に置き換える(C が読む包絡は元のまま)。
描画器の界面(段 P95 step e・既定): 描画器に渡す包絡の 25 行に、描画器の学習の界面どおり因果の 3 点の平滑(重み = 描画器の ckpt の env_smooth・snv_p0_front.causal_smooth と同じ・smooth_env)を掛ける(模型に渡す値は変えない・Rust の SnvStream と同じ計算)。環境変数 SNV_ESM=0 = 掛けない(段 P1〜P94 の測りの再現)。
環境変数 SNV_F0G=1(段 P95 step c3)= 描画器に渡す高さと有声だけを高さの番(snv_f0_guard.clean・med = 操作者の登録の f0 の中央値)で整える(模型に渡す値は変えない・Rust の f0_guard と同じ計算)。
環境変数 SNV_KV16=1(段 P94 step c)= p3 の型の系の、目標の構造の注意の鍵と値を f16 に丸める(Rust の f16 の表の経路と同じ値)。
環境変数 SNV_WF16=1(段 P94)= p3 の型の系の、回路と CPC のフレームごとに読む重みを f16 に丸める(half_weights・Rust の f16 の重みの経路と同じ値)。
系の名前に +hp:<Hz>(+cs: の前・+mask: の後・paper_snv 段 P88)= 前段 D に渡す操作者の波形(源と登録)に、因果の 2 次の Butterworth の高域通過を掛ける(前段つきの系だけ)。
系の名前に +cs:<a>(+aw: の前・paper_snv 段 P85 step d)= 符号に渡す入力の包絡(源と操作者の登録・c0〜c24)を因果の 1 次の平均 y[t] = a y[t−1] + (1 − a) y[t] でならす。錨・描画・目標の構造は変えない。
系の名前に +aw:<α>(+tf:・+fb: の前・paper_snv 段 P76)= 符号に渡す入力の包絡(源と操作者の登録)の周波数軸を固定の α 倍にする(warp_utt・25 次の固定の線形)。錨と描画の高さは変えない。
系の名前に +tf:src / +tf:enr(+fb: の前・paper_snv §5 M47)= 操作者ごとの座標の線形 (T, b) を、白くした符号の後ろに置く。回路の出力の渡りが錨の渡りに揃うように、変換する発話そのもの(src・上限)か操作者の登録(enr)の上で当てはめる。歩数・学習率・λ は環境変数 SNV_TF(既定 60,0.02,0.01)。
装置 = 環境変数 SNV_DEV か、無ければ GPU の空きが 4 GiB 以上なら cuda・未満なら cpu(学習と並ぶ途中信号)。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import nvoc as N
import rvoc as R
import snv_f0_guard as FG
import train_rvoc as TR
import snv_p0_front as P0
import train_snv_p1 as T

ROOT = Path(__file__).resolve().parent.parent
FEAT = T.FEAT
A0 = ROOT / "results/snv_a0"
A1 = ROOT / "results/snv_a1"
N0R = ROOT / "results/snv_n0r/wav"
CK_R = ROOT / "results/rvoc3hi/snap/ema_330k.pt"


def pick_dev() -> str:
    if os.environ.get("SNV_DEV"):
        return os.environ["SNV_DEV"]
    if not torch.cuda.is_available():
        return "cpu"
    free = torch.cuda.mem_get_info()[0]
    if free < 4 * 2**30:
        print(f"[snv_p1_eval] GPU の空き {free / 2**30:.1f} GiB < 4 GiB → cpu で評価", file=sys.stderr, flush=True)
        return "cpu"
    return "cuda"


DEV = pick_dev()
KS = (0, 5, 7, 12)
SR = 48000
VTL_DCT = 59
VTL_F0_SPLIT = 165.0
VTL_GRID = tuple(sorted(set(np.round(np.exp(np.linspace(np.log(0.7), np.log(1.45), 33)), 4).tolist()) | {1.0}))


def wav_path(rel: str) -> Path | None:
    kind, _, rest = rel.partition("/")
    base = {"e1": ROOT / "data/snv_e1", "d": ROOT / "data/snv_d", "e1nv": ROOT / "data/snv_e1nv", "a0": A0 / "wav", "n0r": N0R}.get(kind)
    for q in (FEAT / f"{rel}.wav",) + ((base / f"{rest}.wav",) if base is not None else ()):
        if q.exists():
            return q
    return None


def utt(rel: str) -> dict:
    u = T.load_utt(FEAT / f"{rel}.npz")
    w = wav_path(rel)
    return u | {"wav_path": str(w)} if w is not None else u


def warp_utt(u: dict, alpha: float) -> dict:
    import snv_p0c_anchor as A
    y = u["y"].copy()
    env = np.concatenate([y[:1] * 10 + T.PA.C0_SIL, y[1:25] * 10], 0)
    ew = A.warp_env(env, alpha)
    y[0], y[1:25] = (ew[0] - T.PA.C0_SIL) / 10, ew[1:25] / 10
    return u | {"y": y}


def half_weights(C: torch.nn.Module, cpc: torch.nn.Module) -> None:
    """段 P94: 回路と CPC の、フレームごとに読む重み(畳み込みと線形の weight・bias と登録だけの層は除く)を f16 に丸めて f32 に戻す(Rust の f16 の重みの経路と同じ値)。"""
    mods = [C.inp, C.head, C.q, C.o] + [m for b in C.blocks for m in (b.conv, b.out)] + [cpc.e1, cpc.e2, cpc.e3, cpc.ar_in] + [m for b in cpc.ar for m in (b.conv, b.out)]
    with torch.no_grad():
        for m in mods:
            m.weight.copy_(m.weight.half().float())


def smooth_utt(u: dict, a: float) -> dict:
    y = u["y"].copy()
    for t in range(1, y.shape[1]):
        y[:25, t] = a * y[:25, t - 1] + (1 - a) * y[:25, t]
    return u | {"y": y}


def struct_inputs(refs: list[dict]) -> tuple:
    ry = torch.from_numpy(np.concatenate([r["y"] for r in refs], 1))[None].to(DEV)
    rv = torch.from_numpy(np.concatenate([r["v"] for r in refs]))[None].to(DEV)
    mu = torch.from_numpy(sum(r["vs"] for r in refs) / max(1, sum(r["vn"] for r in refs)))[None].to(DEV)
    return ry, rv, torch.ones(rv.shape, dtype=torch.bool, device=DEV), mu


def resid_part(r: torch.Tensor, v: torch.Tensor, c: str) -> torch.Tensor:
    """paper_snv §5 M46: 源の包絡の残り r [1, 25, T] の成分。e = 強さ(c0)・l = 粗い形(c1〜c12)・u = 細部(c13〜c24)・x / v = 無声 / 有声のフレームの形(c1〜c24)・h / s = 形の速い / 遅い成分(因果の 1 次の平均・時定数 8 フレーム = 40 ms)。"""
    m = torch.zeros_like(r)
    if c == "e":
        m[:, :1] = r[:, :1]
    elif c == "l":
        m[:, 1:13] = r[:, 1:13]
    elif c == "u":
        m[:, 13:] = r[:, 13:]
    elif c in "xv":
        w = (v[:, None] > 0).float()
        m[:, 1:] = r[:, 1:] * (w if c == "v" else 1 - w)
    elif c in "hs":
        from scipy.signal import lfilter
        a = float(np.exp(-1.0 / 8.0))
        slow = torch.from_numpy(lfilter([1 - a], [1.0, -a], r[:, 1:].cpu().numpy().astype(np.float64), axis=-1).astype(np.float32)).to(r.device)
        m[:, 1:] = slow if c == "s" else r[:, 1:] - slow
    else:
        raise ValueError(c)
    return m


def fast_part(x: torch.Tensor, tau: float = 8.0, n: int = 64) -> torch.Tensor:
    """paper_snv §5 M47: [B, D, T] の時間の速い成分(因果の 1 次の平均(時定数 tau フレーム・長さ n で打ち切り)を引く)。微分できる。"""
    a = float(np.exp(-1.0 / tau))
    k = ((1 - a) * a ** torch.arange(n - 1, -1, -1, dtype=x.dtype, device=x.device))[None, None]
    B, D, L = x.shape
    return x - torch.nn.functional.conv1d(torch.nn.functional.pad(x.reshape(B * D, 1, L), (n - 1, 0)), k).reshape(B, D, L)


def agree(o: torch.Tensor, w: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """paper_snv §5 M47: 一致 A = マスク m [B, T] の上の、粗い形(c1〜c12)の速い成分の余弦(フレームと次元を通して 1 つ)。"""
    fo, fw = fast_part(o[:, 1:13]) * m[:, None], fast_part(w[:, 1:13]) * m[:, None]
    return (fo * fw).sum() / (fo.pow(2).sum() * fw.pow(2).sum()).sqrt().clamp(min=1e-8)


TF_STEPS, TF_LR, TF_LAM = (lambda q: (int(q[0]), float(q[1]), float(q[2])))(os.environ.get("SNV_TF", "60,0.02,0.01").split(","))


def mu_of(refs: list[dict]) -> torch.Tensor:
    return torch.from_numpy(sum(r["vs"] for r in refs) / max(1, sum(r["vn"] for r in refs)))[None].to(DEV)


class System:
    def __new__(cls, tag: str, gen: torch.nn.Module):
        return Ens(tag, gen) if tag.startswith("ens:") else super().__new__(cls)

    def __init__(self, tag: str, gen: torch.nn.Module) -> None:
        tag0, _, fb = tag.partition("+fb:")
        tag0, _, self.tf = tag0.partition("+tf:")
        tag0, _, aw = tag0.partition("+aw:")
        self.aw = float(aw) if aw else 0.0
        tag0, _, cs = tag0.partition("+cs:")
        self.cs = float(cs) if cs else 0.0
        tag0, _, hp = tag0.partition("+hp:")
        self.hp = float(hp) if hp else 0.0
        base, _, extra = tag0.partition("+")
        self.tcache: dict = {}
        self.fb = [q for q in fb.split(",") if q]
        self.pins: dict = {}
        self.rcache: dict = {}
        name, _, ck = base.partition("@")
        self.name, self.vtl, self.vtl_alpha = name, extra == "vtl", {}
        self.hg, self.hmu = (float(extra[2:]) / 100 if extra.startswith("hg") else 0.0), {}
        self.hc0: dict = {}
        self.c0cache: dict = {}
        self.ml = float(extra[2:]) / 100 if extra.startswith("ml") else 0.0
        self.at = float(extra[2:]) / 100 if extra.startswith("at") else 0.0
        self.ms = float(extra[2:]) / 100 if extra.startswith("ms") else 0.0
        self.sep, self.pcache = None, {}
        if extra.startswith("sep"):
            import train_snv_sep as SEP
            sname, _, sck = (extra.partition(":")[2] or "snv_i14_sep").partition("@")
            self.SEP, self.sep = SEP, SEP.load(ROOT / "results" / sname / f"{sck or 'last'}.pt", DEV)
        elif extra.startswith("mask"):
            import train_snv_mask as MK
            mname, _, mck = (extra.partition(":")[2] or "snv_i17_mask").partition("@")
            self.SEP, self.sep = MK, MK.load(ROOT / "results" / mname / f"{mck or 'last'}.pt", DEV)
            if extra.startswith("maskd"):
                self.sep.dec = "myin"
        elif extra == "rewav":
            import train_snv_mask as MK
            self.SEP, self.sep = MK, "rewav"
        st = torch.load(ROOT / "results" / name / f"{ck or 'last'}.pt", map_location="cpu", weights_only=True)
        self.p12 = "TH" in st
        if self.p12:
            import train_snv_p12 as P12
            self.P12 = P12
            self.M12 = P12.Circuit().to(DEV).eval()
            self.M12.load_state_dict(st["M"])
            self.C = lambda a, lf, v, th: self.M12.dec(a, lf, v, th)
        else:
            self.C = (T.CircQ() if st.get("circ") == "q" else T.Circ()).to(DEV).eval()
            self.C.load_state_dict(st["C"])
        self.p11 = "Es" in st
        self.p3 = "proj" in st
        self.envc = bool(st.get("envc"))
        self.cy = T.code_y if self.envc else (lambda u: u)
        if self.p12:
            pass
        elif self.p11:
            import train_snv_p11 as P11
            self.P11 = P11
            if st.get("e_play") == "wst":
                import snv_encw as EW
                self.Es = EW.EncW(ROOT / st["cpc"], ROOT / st["pca"]).to(DEV).eval()
            else:
                self.Es = P11.EncS().to(DEV).eval()
            self.Es.load_state_dict(st["Es"])
            self.lift = int(st.get("e_lift") or 0)
            self.lmean = torch.tensor(st["lift_mean"], dtype=torch.float32, device=DEV) if self.lift else None
        elif self.p3:
            import train_snv_p3 as P3
            self.Pl = P3.make_play(ROOT / st["cpc"], st.get("play", "c"), ROOT / st["pca"] if st.get("pca") else None, st.get("lags"), st.get("shrink"), ROOT / st["cb"] if st.get("cb") else None, bool(st.get("xc0"))).to(DEV).eval()
            self.Pl.proj.load_state_dict(st["proj"])
            if os.environ.get("SNV_WF16"):
                half_weights(self.C, self.Pl.cpc)
            if self.tf:
                for q in list(self.C.parameters()) + list(self.Pl.parameters()):
                    q.requires_grad_(False)
        else:
            self.E = T.Enc().to(DEV).eval()
            self.E.load_state_dict(st["E"])
        self.gen, self.tag, self.step = gen, tag, st["step"]
        self.cache: dict = {}
        self.scache: dict = {}

    def ex(self, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.P11.xin(self.P11.lift(y, self.lift, self.lmean), v)

    def warp(self, u: dict) -> dict:
        u = smooth_utt(u, self.cs) if self.cs else u
        return warp_utt(u, self.aw) if self.aw else u

    def prep(self, u: dict) -> dict:
        if self.sep is None:
            return u
        if id(u) not in self.pcache:
            x = u.get("wav")
            if x is None:
                assert "wav_path" in u, "前段 D は操作者の波形が要る(発話の辞書に wav か wav_path)"
                import librosa
                x = librosa.load(u["wav_path"], sr=SR, mono=True)[0]
            x = np.asarray(x, np.float32)
            if self.hp:
                from scipy.signal import butter, sosfilt
                x = sosfilt(butter(2, self.hp, "highpass", fs=SR, output="sos"), x).astype(np.float32)
            f = self.SEP.features_with_mask(x, torch.ones(self.SEP.P.n_frames(len(x)), self.SEP.NB, dtype=torch.float64)) if self.sep == "rewav" else self.SEP.features(self.sep, x, DEV)
            if self.envc:
                n = f["y"].shape[1]
                f = f | {"yc": np.concatenate([T.to_y(P0.fixed_env(x, n), np.zeros((4, n), np.float32), np.ones(n, np.float32))[:25], f["y"][25:]], 0).astype(np.float32)}
            self.pcache[id(u)] = (u, f)
        return self.pcache[id(u)][1]

    def fit(self, utts: list[dict]) -> torch.Tensor:
        tt = lambda u, k: torch.from_numpy(u[k]).to(DEV)
        return self.P12.fit_theta(self.M12, [tt(u, "y") for u in utts], [tt(u, "lf") for u in utts], [tt(u, "v") for u in utts])

    @torch.no_grad()
    def e_vmean(self, utts: list[dict]) -> np.ndarray:
        ry, rv, rm, _ = struct_inputs(utts)
        x = self.ex(ry, rv)
        a = self.Es(x, self.Es.structure(x, rm))[0][0, :VTL_DCT]
        return a[:, rv[0] > 0.5].mean(1).cpu().numpy()

    def vtl_pop(self, sex: str) -> tuple[np.ndarray, np.ndarray]:
        p = ROOT / "results" / self.name / "vtl_pop.npz"
        if not p.exists():
            split = json.loads((ROOT / "results" / self.name / "split.json").read_text())
            out = {}
            for sx in ("em", "ef"):
                ms = np.stack([self.e_vmean([T.load_utt(q) for q in sorted((FEAT / "e1" / v).glob("t*.npz"))[:4]]) for v in split["train"] if v.startswith(sx)])
                out[f"mu_{sx}"], out[f"sd_{sx}"] = ms.mean(0), ms.std(0) + 1e-4
            tmp = p.with_name(f"vtl_pop.{os.getpid()}.tmp.npz")
            np.savez(tmp, **out)
            os.replace(tmp, p)
        z = np.load(p)
        return z[f"mu_{sex}"], z[f"sd_{sex}"]

    def fit_vtl(self, utts: list[dict]) -> float:
        key = tuple(id(u) for u in utts)
        if key not in self.vtl_alpha:
            f0 = np.concatenate([u["f0"][u["v"] > 0.5] for u in utts])
            mu, sd = self.vtl_pop("em" if np.median(f0) < VTL_F0_SPLIT else "ef")
            self.vtl_alpha[key] = min(VTL_GRID, key=lambda a: float(np.linalg.norm((self.e_vmean([warp_utt(u, a) for u in utts]) - mu) / sd)))
        return self.vtl_alpha[key]

    @torch.no_grad()
    def op_stat(self, utts: list[dict]) -> torch.Tensor:
        key = tuple(id(u) for u in utts)
        if key not in self.scache:
            utts = [self.prep(u) for u in utts]
            if self.p12:
                self.scache[key] = self.fit(utts)
            elif self.p11 and self.vtl:
                al = self.fit_vtl(utts)
                ry, rv, rm, _ = struct_inputs([warp_utt(u, al) for u in utts])
                self.scache[key] = self.Es.structure(self.ex(ry, rv), rm)
            elif self.p11:
                ry, rv, rm, _ = struct_inputs(utts)
                self.scache[key] = self.Es.structure(self.ex(ry, rv), rm)
            elif self.p3:
                assert not self.envc or all("yc" in u for u in utts), "固定の窓の包絡が無い登録の発話が在る"
                ry, rv, rm, _ = struct_inputs([self.warp(self.cy(u)) for u in utts])
                self.scache[key] = self.Pl.mean_of(ry, rv, rm)
            else:
                self.scache[key] = mu_of(utts)
        return self.scache[key]

    @torch.no_grad()
    def structure(self, key: str, refs: list[dict]) -> dict:
        if key not in self.cache:
            if key.startswith("self|"):
                refs = [self.prep(u) for u in refs]
            ry, rv, rm, mu = struct_inputs(refs)
            if self.p12:
                self.cache[key] = self.fit(refs)
            elif self.p11:
                rx = self.ex(ry, rv)
                self.cache[key] = self.C.structure(ry, self.Es(rx, self.Es.structure(rx, rm))[0], rm)
            elif self.p3:
                assert not self.envc or all("yc" in u for u in refs), f"固定の窓の包絡が無い参照が在る({key})"
                ryc = struct_inputs([self.cy(u) for u in refs])[0]
                self.cache[key] = self.C.structure(ry, self.Pl(ryc, rv, self.Pl.mean_of(ryc, rv, rm)), rm)
                if os.environ.get("SNV_KV16"):
                    self.cache[key] = self.cache[key] | {q: self.cache[key][q].half().float() for q in ("k", "v")}
            else:
                self.cache[key] = T.structure_of(self.E, self.C, ry, rv, rm, mu)
            if self.at and isinstance(self.cache[key], dict) and "k" in self.cache[key]:
                self.cache[key] = self.cache[key] | {"k": self.cache[key]["k"] * self.at}
            self.hmu[id(self.cache[key])] = (ry[0, :25] * rv).sum(-1) / rv.sum().clamp(min=1.0)
            m0 = self.hmu[id(self.cache[key])][0]
            self.hc0[id(self.cache[key])] = (m0, (((ry[0, 0] - m0) ** 2 * rv[0]).sum() / rv.sum().clamp(min=1.0)).sqrt().clamp(min=1e-3))
        return self.cache[key]

    @torch.no_grad()
    def play(self, y: torch.Tensor, v: torch.Tensor, stat: torch.Tensor) -> torch.Tensor:
        if self.p12:
            return self.M12.enc(y, v, stat)[0]
        if self.p11:
            return self.Es(self.ex(y, v), stat)[0]
        return self.Pl(y, v, stat) if self.p3 else self.E(T.e_in(y, v, stat))

    @torch.no_grad()
    def classes(self, y: torch.Tensor, v: torch.Tensor, st: tuple) -> torch.Tensor:
        import snv_cmn as CM
        xe = torch.einsum("bij,bjt->bit", st[1], self.Pl.stack(self.Pl.ctx(y, v), st[0])).transpose(1, 2)[0]
        return CM.posterior(xe, self.Pl.cb, self.Pl.cb_s2)

    @torch.no_grad()
    def op_resid(self, stat: list[dict]) -> torch.Tensor:
        """paper_snv §5 M44: 操作者の登録の、発話を除いた構造での回路の残りの、級ごとの平均 [K, 25](事前の重み 20 フレームで 0 へ寄せる)。"""
        key = tuple(id(u) for u in stat)
        if key not in self.rcache:
            us = [self.prep(u) for u in stat]
            st = self.op_stat(stat)
            num, den = 0.0, 0.0
            for j, u in enumerate(us):
                ry, rv, rm, _ = struct_inputs(us[:j] + us[j + 1:])
                So = self.C.structure(ry, self.Pl(ry, rv, st), rm)
                y, v = torch.from_numpy(u["y"])[None].to(DEV), torch.from_numpy(u["v"])[None].to(DEV)
                r = y[:, :25] - self.C(self.Pl(y, v, st), torch.from_numpy(u["lf"])[None].to(DEV), v, So)[:, :25]
                g = self.classes(y, v, st)
                num, den = num + g.T @ r[0].T, den + g.sum(0)
            self.rcache[key] = (stat, num / (den[:, None] + 20.0), num.sum(0) / (den.sum() + 20.0))
        return self.rcache[key]

    def anchor(self, src: dict, stat: list[dict], S: dict, q: bool = False) -> torch.Tensor:
        """源の包絡の一律の伸縮(錨・snv_p0c_anchor)[1, 29, T]。α は操作者の登録の平均と目標の登録の平均から。q = 伸縮の後の平均の差も足す。src は前段を通した発話の辞書。"""
        import snv_p0c_anchor as AN
        yn = src["y"]
        hm = self.hmu[id(S)].cpu().numpy()
        mu_t = np.concatenate([hm[:1] * 10 + T.PA.C0_SIL, hm[1:25] * 10], 0)
        mu_o = mu_of([self.prep(u) for u in stat])[0].cpu().numpy()
        a = AN.fit_alpha(mu_o, mu_t)
        ew = AN.warp_env(np.concatenate([yn[:1] * 10 + T.PA.C0_SIL, yn[1:25] * 10], 0), a)
        if q:
            ew[1:25] += (mu_t[1:25] - AN.warp_env(mu_o, a)[1:25])[:, None]
        return torch.from_numpy(np.concatenate([(ew[:1] - T.PA.C0_SIL) / 10, ew[1:25] / 10, yn[25:29]], 0).astype(np.float32))[None].to(DEV)

    def tf_inputs(self, u: dict, stat: list[dict], S: dict, dlf: float) -> tuple:
        """当てはめと一致の入力: 前段を通した発話 u の (符号の式の特徴 [1, n + 1, T]・描く高さ・有声・錨・有声で門が閉じたフレームのマスク)。"""
        y, v = torch.from_numpy(u["y"])[None].to(DEV), torch.from_numpy(u["v"])[None].to(DEV)
        g = torch.from_numpy(u["gate"])[None].to(DEV) if "gate" in u else torch.zeros_like(v)
        with torch.no_grad():
            x = self.Pl.feat(y, v, self.op_stat(stat))
        return x, torch.from_numpy(u["lf"])[None].to(DEV) + dlf * v, v, self.anchor(u, stat, S), ((v > 0.5) & (g <= 0.5)).float()

    def tf_play(self, x: torch.Tensor, Tb: tuple | None) -> torch.Tensor:
        n = x.shape[1] - int(self.Pl.xc0)
        if Tb is not None:
            x = torch.cat([torch.einsum("ij,bjt->bit", Tb[0], x[:, :n]) + Tb[1][None, :, None], x[:, n:]], 1)
        return self.Pl.proj(x.transpose(1, 2)).transpose(1, 2)

    def tf_fit(self, src: dict, stat: list[dict], S: dict, dlf: float) -> tuple:
        """paper_snv §5 M47: 操作者ごとの座標の線形 (T, b) を、目標の声で描いた出力の渡りが錨の渡りに揃うように当てはめる。+tf:src = 変換する発話そのもの(上限)・+tf:enr = 操作者の登録の発話。"""
        assert self.p3 and self.tf in ("src", "enr"), self.tf
        key = (id(src) if self.tf == "src" else tuple(id(u) for u in stat), id(S), round(dlf, 6))
        if key not in self.tcache:
            ins = [self.tf_inputs(u, stat, S, dlf) for u in ([src] if self.tf == "src" else [self.prep(u) for u in stat])]
            n = ins[0][0].shape[1] - int(self.Pl.xc0)
            Tm, b = torch.eye(n, device=DEV, requires_grad=True), torch.zeros(n, device=DEV, requires_grad=True)
            opt = torch.optim.Adam([Tm, b], lr=TF_LR)
            log = []
            with torch.enable_grad():
                for _ in range(TF_STEPS):
                    a = torch.stack([agree(self.C(self.tf_play(x, (Tm, b)), lf, v, S), yw, m) for x, lf, v, yw, m in ins]).mean()
                    loss = 1 - a + TF_LAM * ((Tm - torch.eye(n, device=DEV)).pow(2).sum() + b.pow(2).sum())
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    log.append(float(a))
            self.tcache[key] = (src, stat, Tm.detach(), b.detach(), log)
        return self.tcache[key][2:]

    @torch.no_grad()
    def fallback(self, kind: str, src: dict, y: torch.Tensor, v: torch.Tensor, pl: torch.Tensor, yo: torch.Tensor, stat: list[dict], S: dict) -> torch.Tensor:
        """paper_snv §5 M43(系 2t): 門が開いたフレーム(swapall は全てのフレーム)で、内容を源の包絡から通す。src は前段を通した発話の辞書(gate = フレームごとの門)。swapp<成分><β> = 全てのフレームで残りの 1 つの成分だけ(§5 M46・resid_part)。rep<成分> = 出力の 1 つの成分を錨の同じ成分に差し替える診断(段 P65 step b・p = 周期性)。c0lo / c0hi = c0s の片側(回路の c0 との min / max・段 P67 step a2)。c0s = 出力の強さを、登録の統計で標準化した源の強さ e(t) を目標の参照の平均と標準偏差に戻したものにする(§5 M48)。c0j<τ> = c0s を、回路の c0 との差が目標の標準偏差の τ 倍以下のフレームだけに当てる(それより上は回路の c0 のまま・段 P77)。c0g<θ> = c0s を、門の確率(持続の条件と閾の前の値)が θ 未満のフレームだけに当てる(段 P77 step a″)。"""
        g = torch.from_numpy(src["gate"])[None, None].to(DEV) if "gate" in src else torch.zeros_like(v)[:, None]
        if kind.startswith("swap"):
            al = kind.startswith("swapall")
            lf0 = torch.from_numpy(src["lf"])[None].to(DEV)
            key = "self|" + ",".join(str(id(u)) for u in stat)
            self.pins.setdefault(key, stat)
            So = self.structure(key, stat)
            r = y[:, :25] - self.C(pl, lf0, v, So)[:, :25]
            if kind[4] == "p":
                return torch.cat([yo[:, :25] + float(kind[6:]) * resid_part(r, v, kind[5]), yo[:, 25:]], 1)
            if kind[4] in "kc":
                _, Rk, Rc = self.op_resid(stat)
                r = r - ((self.classes(y, v, self.op_stat(stat)) @ Rk).T[None] if kind[4] == "k" else Rc[None, :, None])
                return torch.cat([yo[:, :25] + float(kind[5:]) * r, yo[:, 25:]], 1)
            return torch.cat([yo[:, :25] + float(kind[7:] if al else kind[4:]) * (1.0 if al else g) * r, yo[:, 25:]], 1)
        if kind in ("c0s", "c0lo", "c0hi") or kind.startswith(("c0j", "c0g")):
            key = tuple(id(u) for u in stat)
            if key not in self.c0cache:
                c = np.concatenate([u["y"][0][u["v"] > 0.5] for u in (self.prep(u) for u in stat)])
                self.c0cache[key] = (stat, float(c.mean()), max(float(c.std()), 1e-3))
            _, m0, s0 = self.c0cache[key]
            d = self.hc0[id(S)][0] + self.hc0[id(S)][1] * (y[:, :1] - m0) / s0
            if kind.startswith("c0g"):
                gp = torch.from_numpy(src["gate_p"])[None, None].to(DEV) if "gate_p" in src else torch.zeros_like(v)[:, None]
                return torch.cat([torch.where(gp < float(kind[3:]), d, yo[:, :1]), yo[:, 1:]], 1)
            if kind.startswith("c0j"):
                return torch.cat([torch.where((d - yo[:, :1]) / self.hc0[id(S)][1] <= float(kind[3:]), d, yo[:, :1]), yo[:, 1:]], 1)
            return torch.cat([d if kind == "c0s" else torch.minimum(yo[:, :1], d) if kind == "c0lo" else torch.maximum(yo[:, :1], d), yo[:, 1:]], 1)
        if kind.startswith("rep"):
            yw = self.anchor(src, stat, S)
            if kind[3] == "p":
                return torch.cat([yo[:, :25], yw[:, 25:29], yo[:, 29:]], 1)
            return torch.cat([yo[:, :25] + resid_part(yw[:, :25] - yo[:, :25], v, kind[3]), yo[:, 25:]], 1)
        yw = self.anchor(src, stat, S, "q" in kind)
        if kind.startswith("split"):
            yw = torch.cat([yw[:, :13], yo[:, 13:29]], 1)
        elif kind.startswith("mix"):
            lam = float(kind.lstrip("mixq"))
            yw = torch.cat([lam * yw[:, :25] + (1 - lam) * yo[:, :25], yo[:, 25:29]], 1)
        return torch.cat([torch.where(g > 0.5, yw, yo[:, :29]), yo[:, 29:]], 1)

    @torch.no_grad()
    def convert(self, src: dict, stat: list[dict], S: dict, k: int, with_y: bool = False, dlf_c: float | None = None):
        if self.vtl and self.p11:
            src = warp_utt(src, self.fit_vtl(stat))
        src = self.prep(src)
        self.f0_med = FG.median_f0([self.prep(u)["f0"] for u in stat]) if os.environ.get("SNV_F0G") else 0.0
        y = torch.from_numpy(src["y"])[None].to(DEV)
        v = torch.from_numpy(src["v"])[None].to(DEV)
        lf = torch.from_numpy(src["lf"])[None].to(DEV) + k * T.LN2_12 * v
        lfc = lf if dlf_c is None else (torch.from_numpy(src["lf"])[None].to(DEV) + dlf_c) * v
        assert not self.envc or "yc" in src, "固定の窓の包絡が無い源"
        pl = self.play(torch.from_numpy(self.warp(self.cy(src))["y"])[None].to(DEV) if self.aw or self.cs or self.envc else y, v, self.op_stat(stat))
        if self.tf:
            Tm, b, self.tf_log = self.tf_fit(src, stat, S, k * T.LN2_12 if dlf_c is None else dlf_c)
            pl = self.tf_play(self.Pl.feat(y, v, self.op_stat(stat)), (Tm, b))
        yo = self.C(pl, lfc, v, S)
        for kind in self.fb:
            yo = self.fallback(kind, src, y, v, pl, yo, stat, S)
        if self.hg:
            m = self.hmu[id(S)][None, 13:25, None]
            yo = torch.cat([yo[:, :13], torch.where(v[:, None] > 0, m + self.hg * (yo[:, 13:25] - m), yo[:, 13:25]), yo[:, 25:]], 1)
        if self.ms:
            a = self.hmu[id(S)][1:25] - torch.from_numpy(np.concatenate([u["y"][1:25, u["v"] > 0] for u in stat], 1).mean(1)).to(DEV)
            sh = self.ms * torch.cat([0.15 * a[:12], 0.04 * a[12:]])
            yo = torch.cat([yo[:, :1], torch.where(v[:, None] > 0, yo[:, 1:25] + sh[None, :, None], yo[:, 1:25]), yo[:, 25:]], 1)
        if self.ml:
            mo = (yo[:, 1:25] * v[:, None]).sum(-1, keepdim=True) / v.sum().clamp(min=1.0)
            yo = torch.cat([yo[:, :1], torch.where(v[:, None] > 0, yo[:, 1:25] + self.ml * (self.hmu[id(S)][None, 1:25, None] - mo), yo[:, 1:25]), yo[:, 25:]], 1)
        w = self.render(src, yo, k)
        return (w, yo) if with_y else w

    @torch.no_grad()
    def render(self, src: dict, yo: torch.Tensor, k: int) -> np.ndarray:
        f0s, vs, lfs = src["f0"], src["v"], src["lf"]
        if getattr(self, "f0_med", 0.0) > 0:
            f0s = FG.clean(f0s, self.f0_med)
            vs = (f0s > 0).astype(np.float32)
            lfs = np.where(f0s > 0, np.log(np.maximum(f0s, 1.0) / 200.0), 0.0).astype(np.float32)
        v = torch.from_numpy(vs)[None].to(DEV)
        lf = torch.from_numpy(lfs)[None].to(DEV) + k * T.LN2_12 * v
        cond = torch.cat([smooth_env(self.gen, yo[:, :25]), lf[:, None], v[:, None], yo[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
        f0 = torch.from_numpy(f0s * 2 ** (k / 12))[None].to(DEV)
        return self.gen(cond, TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:].astype(np.float32)


class Ens:
    """ens:<系>|<系>|… = 複数の系の回路の出力(描画の前の 29 次元)の平均を描く(paper_snv 段 P74 step d)。"""

    def __init__(self, tag: str, gen: torch.nn.Module) -> None:
        self.tag, self.parts = tag, [System(t, gen) for t in tag[4:].split("|")]
        self.step, self.sep = self.parts[0].step, self.parts[0].sep

    def prep(self, u: dict) -> dict:
        return self.parts[0].prep(u)

    def structure(self, key: str, refs: list[dict]) -> list:
        return [p.structure(key, refs) for p in self.parts]

    @torch.no_grad()
    def convert(self, src: dict, stat: list[dict], S: list, k: int, with_y: bool = False, dlf_c: float | None = None):
        yo = sum(p.convert(src, stat, s, k, with_y=True, dlf_c=dlf_c)[1][:, :29] for p, s in zip(self.parts, S)) / len(self.parts)
        w = self.parts[0].render(self.parts[0].prep(src), yo, k)
        return (w, yo) if with_y else w


def smooth_env(gen: torch.nn.Module, env: torch.Tensor) -> torch.Tensor:
    w = getattr(gen, "env_smooth", 0.0)
    if w <= 0 or os.environ.get("SNV_ESM") == "0":
        return env
    q = torch.nn.functional.pad(env, (2, 0), mode="replicate")
    return w * q[..., :-2] + (1 - 2 * w) * q[..., 1:-1] + w * q[..., 2:]


def load_gen() -> torch.nn.Module:
    st = torch.load(CK_R, map_location="cpu", weights_only=False)
    cfg = st["cfg"]
    g = R.RVoc(ch=cfg["ch"], kernels=tuple(cfg["kernels"]), dils=tuple(cfg["dils"]), d_cond=cfg["d_cond"]).to(DEV)
    g.load_state_dict(st["ema"])
    g.env_smooth = float(st.get("env_smooth", 0.0))
    return g.eval().requires_grad_(False)


def main() -> int:
    mode, tags = sys.argv[1], [t for t in sys.argv[2:] if t != "smoke" and "=" not in t]
    smoke = "smoke" in sys.argv[2:]
    opt = dict(t.split("=", 1) for t in sys.argv[2:] if "=" in t)
    from snv_a0_measure import cer, kana, load, to16
    from snv_a0b_measure import proj
    from snv_a4_eval import cluster_ci
    from snv_n0_measure import trim
    from snv_n2c_tex import tex_dist, tex_stats
    from speechbrain.inference.speaker import EncoderClassifier
    gen = load_gen()
    systems = [System(t, gen) for t in tags]
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})
    Tz = np.load(ROOT / "data/snv_e1_feat/ecapa_lda_T.npz")
    Tt = (Tz["mu"], Tz["scalings"])

    def emb(w16: np.ndarray, lda: bool = True) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(w16).float()[None].to(DEV)).reshape(-1)
        e = torch.nn.functional.normalize(e, dim=0).cpu().numpy()
        return proj(e, Tt) if lda else e

    w16 = lambda w: to16(w.astype(np.float64), SR).astype(np.float32)
    lw = lambda p: to16(*load(p)).astype(np.float32)
    res: dict = {"mode": mode, "systems": {s.tag: {"step": s.step} for s in systems}, "prereg": "results/snv_p1/prereg.yaml"}
    out = ROOT / "results/snv_p1" / (("smoke_" if smoke else "") + f"{mode}_" + "_".join(t.replace("snv_p1_", "").replace("@", "-") for t in tags) + (f"_warp{opt['warp']}" if "warp" in opt else "") + ".json")
    if "out" in opt:
        out = ROOT / opt["out"]
    if mode == "dev":
        split = json.loads((ROOT / "results" / tags[0].partition("@")[0] / "split.json").read_text())
        tg = [v for v in split["val"] if v.startswith("ef")]
        spk = sorted({p.stem.split("_")[0] for p in (FEAT / "vctk").glob("*.npz")})
        srcs, enr = [], {}
        for s_ in spk:
            fs = sorted((FEAT / "vctk").glob(f"{s_}_*.npz"))
            enr[s_] = [T.load_utt(f) for f in fs[:2]]
            srcs += [(s_, T.load_utt(f)) for f in fs[2:4]]
        if smoke:
            tg, srcs = tg[:5], srcs[:2]
        Et = {v: emb(np.concatenate([lw(ROOT / f"data/snv_e1/{v}/t0{j}.wav") for j in (3, 4, 5)]), lda=False) for v in tg}
        J = None
        if "judge" in opt:
            from train_snv_p1j import Judge
            sj = torch.load(ROOT / opt["judge"], map_location="cpu", weights_only=False)
            J = Judge(len(sj["classes"]), norm=sj.get("norm", "none")).to(DEV).eval().requires_grad_(False)
            J.load_state_dict(sj["J"])
            Jt = {v: J.embed(torch.from_numpy(np.concatenate([utt(f"e1/{v}/t0{j}")["y"] for j in (3, 4, 5)], 1))[None].to(DEV))[0] for v in tg}
            Jm = torch.stack([Jt[v] for v in tg])
            jhits = {s.tag: [] for s in systems}
            res["judge"] = opt["judge"]
        hits = {s.tag: [] for s in systems}
        grp = []
        for si, (s_, src) in enumerate(srcs):
            for v in tg:
                grp.append(v)
                for s in systems:
                    S = s.structure(f"dev|{v}", [utt(f"e1/{v}/t0{j}") for j in (0, 1, 2)])
                    w, yo = s.convert(src, enr[s_], S, 7, with_y=True)
                    e = emb(trim(w16(w)), lda=False)
                    hits[s.tag].append(max(tg, key=lambda u: float(e @ Et[u])) == v)
                    if J is not None:
                        with torch.no_grad():
                            jhits[s.tag].append(tg[int((Jm @ J.embed(yo)[0]).argmax())] == v)
        res["dev_M_k7"] = {t: round(float(np.mean(h)), 4) for t, h in hits.items()}
        if J is not None:
            res["dev_judge_top1_k7"] = {t: round(float(np.mean(h)), 4) for t, h in jhits.items()}
        res["n_trials"] = len(grp)
        if len(systems) > 1:
            a_, b_ = np.array(hits[systems[-1].tag], float), np.array(hits[systems[0].tag], float)
            res["dev_dM_last_minus_first"] = {"mean": round(float((a_ - b_).mean()), 4), "ci95_target_cluster": cluster_ci(a_ - b_, np.array(grp))}
        print(json.dumps(res, ensure_ascii=False), flush=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        return 0
    meta = json.loads((A1 / "convert_meta.json").read_text())
    tgts, srcs = meta["targets"], meta["sources"]
    ks = (0, 7) if smoke else KS
    if smoke:
        tgts, srcs = tgts[:3], srcs[:2]
    Et = {v: emb(np.concatenate([lw(A0 / f"wav/{v}_{j}.wav") for j in (0, 1, 2)])) for v in tgts}
    Et2 = {v: emb(np.concatenate([lw(A0 / f"wav/{v}_{j}.wav") for j in (6, 7)])) for v in tgts}
    top = lambda e, ref: max(tgts, key=lambda u: float(e @ ref[u]))
    Es = {sp: emb(lw(FEAT / f"real/{sp}_enroll.wav")) for sp in ("uirou", "namikawa")}
    trefs = {v: [utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tgts}
    refs_of = lambda v: trefs[v] if v in trefs else [utt(f"a0/{v}_{j}") for j in (0, 1, 2)]
    src_u = {s_: utt(f"real/{s_}") for s_ in srcs}
    mu_op = {sp: [utt(f"real/{sp}_enroll")] for sp in ("uirou", "namikawa")}
    if "warp" in opt:
        src_u = {k: warp_utt(u, float(opt["warp"])) for k, u in src_u.items()}
        mu_op = {sp: [warp_utt(u, float(opt["warp"])) for u in us] for sp, us in mu_op.items()}
        res["warp_operator_input"] = float(opt["warp"])
    from faster_whisper import WhisperModel
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16" if DEV == "cuda" else "int8")

    def tr(w: np.ndarray) -> str:
        segs, _ = asr.transcribe(w, language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(x.text for x in segs))

    ref_txt = {s_: tr(lw(FEAT / f"real/{s_}.wav")) for s_ in srcs}
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(DEV).eval()
    mos = lambda w: float(utmos(torch.from_numpy(w).float()[None].to(DEV), 16000).item())
    scale = np.load(ROOT / "results/snv_n2c/tex_scale.npy")
    n0r = [it for it in json.loads((N0R / "items.json").read_text()) if it["sex"] == "F"][::4]
    n0r = n0r[:4] if smoke else n0r
    vec: dict = {}
    wd = ROOT / ("results/snv_p1/wav_smoke" if smoke else "results/snv_p1/wav")
    wd.mkdir(parents=True, exist_ok=True)
    for s in systems:
        row: dict = {}
        for k in ks:
            hit, hit2, grp, ms_, sc = [], [], [], [], []
            for s_ in srcs:
                for v in tgts:
                    w = s.convert(src_u[s_], mu_op[s_.rsplit("_", 1)[0]], s.structure(f"t|{v}", trefs[v]), k)
                    y16 = w16(w)
                    e = emb(trim(y16))
                    hit.append(top(e, Et) == v), hit2.append(top(e, Et2) == v), grp.append(v)
                    sc.append(float(e @ Es[s_.rsplit("_", 1)[0]]) > float(e @ Et[v]))
                    ms_.append(mos(y16))
                    if s_ in srcs[:1] and v in tgts[:2]:
                        sf.write(wd / f"{s.tag}__{s_}__{v}__k{k}.wav", w, SR)
            vec[(s.tag, k)] = (np.array(hit, float), np.array(grp))
            res.setdefault("trials", {}).setdefault(s.tag, {})[f"k{k}"] = {"hit": [int(h) for h in hit], "target": grp}
            row[f"M_k{k}"] = round(float(np.mean(hit)), 4)
            row[f"M_alt_k{k}"] = round(float(np.mean(hit2)), 4)
            row[f"utmos_k{k}"] = round(float(np.mean(ms_)), 3)
            row[f"src_closer_k{k}"] = round(float(np.mean(sc)), 4)
        row["M_mean_4k"] = round(float(np.mean([row[f"M_k{k}"] for k in ks])), 4)
        sh = []
        for v in tgts:
            for j in (3, 4, 5):
                u = utt(f"a0/{v}_{j}")
                w = s.convert(u, trefs[v], s.structure(f"t|{v}", trefs[v]), 0)
                sh.append(top(emb(trim(w16(w))), Et) == v)
        row["S"] = round(float(np.mean(sh)), 4)
        cs, ct = [], []
        for k in (0, 7):
            for s_ in srcs:
                for v in tgts[:4]:
                    w = s.convert(src_u[s_], mu_op[s_.rsplit("_", 1)[0]], s.structure(f"t|{v}", trefs[v]), k)
                    cs.append(cer(ref_txt[s_], tr(w16(w))))
                    ct.append({"src": s_, "tgt": v, "k": k, "cer": round(float(cs[-1]), 4)})
        res["trials"][s.tag]["cer"] = ct
        row["cer_k0_k7"] = round(float(np.mean(cs)), 4)
        tx = []
        for it in n0r:
            nme = f"{it['voice']}_{it['cat']}_{it['k']}"
            u = utt(f"n0r/{nme}")
            rf = refs_of(it["voice"])
            w = s.convert(u, rf, s.structure(f"t|{it['voice']}", rf), 0)
            tx.append(tex_dist(tex_stats(w16(w)), tex_stats(lw(N0R / f"{nme}.wav")), scale))
        row["nv_tex"] = round(float(np.mean(tx)), 4)
        res["systems"][s.tag].update(row)
        print(s.tag, json.dumps(row), flush=True)
    if len(systems) > 1:
        base = systems[0].tag
        res["diff_vs_first"] = {}
        for s in systems[1:]:
            d = {}
            for k in ks:
                a_, g_ = vec[(s.tag, k)]
                b_, _ = vec[(base, k)]
                d[f"dM_k{k}"] = {"mean": round(float((a_ - b_).mean()), 4), "ci95_target_cluster": cluster_ci(a_ - b_, g_)}
            a4 = np.mean([vec[(s.tag, k)][0] for k in ks], 0)
            b4 = np.mean([vec[(base, k)][0] for k in ks], 0)
            d["dM_mean_4k"] = {"mean": round(float((a4 - b4).mean()), 4), "ci95_target_cluster": cluster_ci(a4 - b4, vec[(base, ks[0])][1])}
            res["diff_vs_first"][s.tag] = d
        print("diff", json.dumps(res["diff_vs_first"]), flush=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
