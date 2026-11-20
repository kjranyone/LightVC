"""段 I13 の後の切り分け(学習なし・開発の組): 試験の伴奏の区間(語りの下の伴奏 + 伴奏だけの時間帯)の型を検証の Irodori の声で作り、崩れを包絡の側(推論の E が読む入力)と、前段から C と描画器へ素通りする高さ・有声・f0 の側に分ける。
条件 = clean / musN(本物の音楽・rec2 と同じ MUSAN の 135 本から発話ごとに 1 本・発話の RMS に対する SN 比 N dB)/ leadN(同じ音楽で、発話の前に 3 s・後に 2 s の伴奏だけの時間帯を足す)。conds= で選ぶ(既定 mus10,mus0,mus-5,lead0 = 族の SN 比 U[0, 20] の中・下限・外と、試験の namikawa_1・2 の語りの帯域の SN 比 約 −8〜−3 dB)。
混ぜ: _y = 包絡と周期性は乱した音から・高さ・有声・f0 は元の音(lead は元の音に同じ長さの無音を足して解析)/ _pv = 高さ・有声・f0 だけ乱した音から。登録(E の参照・自己の構造)は同じ SN 比の musN を同じ混ぜで・目標の構造は元のまま。
CER の参照 = 元の音の写しの書き起こし(全ての条件で同じ・伴奏の区間の参照の脆さを避ける)・copy_loss = 乱した音の写しの書き起こしの同じ参照に対する CER。系 = snv_p1_eval.System・自己の reamp と A0F の目標 2 声(k 0)・発話 = 検証の Irodori の声の e1(男声 n・女声 n)。
音楽は推論の E の学習(rec2)と同じ集合 = 族の中の音源(試験の伴奏は族の外の録音)。dcopy=<tag>[@ck] = 前段 D の出力をそのまま描画器で音にした写しの CER(系の名前 dcopy・E と C を通さない D の内容の上限)。pool=dev|train = train_snv_sep.bg_split の音楽(段 I14 の前段 D の開発 / 学習の組・MUSAN + FMA)。mix=0 で混ぜを省く(前段 D の系は波形から作り直すので混ぜは意味を持たない)。
段 I15(paper_snv rev29・ノイズ耐性): clapN = 本物の手拍子と手の音(FSD50K の評価の組・CC0 / CC BY・data/snv_sep/fsd50k_eval_sources.json・学習に使わない)を 4 個、発話の RMS に対して N dB で声と時間を重ねて足す・
opN = 同じく操作者の音の全ての類(叩く・擦る・紙・ファスナーなど)・clapleadN = 手拍子を伴奏だけの時間帯と同じ前 3 s・後 2 s の声の無い時間にだけ 3 個(声に化けるか)。登録(E の参照・自己の構造)にも同じ条件で足す。
enroll=clean = 登録(E の参照・自己の構造)はどの条件でもきれいな音のまま(源だけを乱す)。
oracle=1(paper_snv rev32 M23 の上限・学習なし)= 包絡の mel の帯の振幅に理想の重み min(1, きれいな振幅 / 乱れた振幅)(高さは乱れた音の前段 v3 のもの)を掛けた変種 _mo(周期性は乱れたまま)と _mop(周期性もきれいな音)を足す。
oracle=1 は _ms も足す = 2048 点の格子の理想の比の重み |S|²/(|S|²+|N|²)(S = きれいな音・N = 乱れた音 − きれいな音)を M23 と同じ掛け方(train_snv_mask.features_with_mask)で通した変種(学んだ重みの上限)。
ovar=(段 I18 の後の切り分け・学習なし)= 同じ格子の重みの変種を M23 と同じ掛け方で通した条件 {cond}_{変種} を足す: f<床>(理想の比の重みを床で止める = 深さの不足の再現・f0 は _ms と同じ)/ bin(理想の 2 値の重み = 声の勝つビンだけ残す・埋めない)/ lk<k>(lmask=<tag>[@ck] の学んだ重みの k 乗・lk1 = 学んだ重みそのもの)/
hold(声の勝つビンを理想の判定で選び、手の音の勝つビンは同じビンの直前の声の勝つフレームのきれいな声のパワーを持ち越す = 因果に埋める上限)/ holdx(同じ判定で、持ち越すのは乱れた音のパワー = 判定さえ当たれば作れる)/ lhold<θ>(判定 = 学んだ重み ≥ θ・持ち越すのは乱れた音のパワー)。
変種に +p を付けると(例 holdx+p・f0+p)、理想の判定で手の音の勝つビンが乱れた音のパワーの半分を越えるフレームでは、周期性(y の 25〜28)を直前のそうでないフレームの値で持ち越す。
+d を付けると(例 bin+d)、有声と高さ(と周期性の計算に使う高さ)をきれいな音の前段 v3 のものにする(前段の決定の飽和の分を分ける)。
+v<α> を付けると(例 lk1+v0.8)、有声と高さを snv_p0_front.f0_track_masked(高さの候補の費用に α × (1 − 候補の倍音のエネルギーのうち重みが声と読む割合)を足す・重み 1 で前段 v3 と 1 bit 同じ)で決める。
+y を付けると(例 lk1+y)、有声と高さを snv_p0_front.f0_track_myin(YIN の低い帯の区間とスペクトルに重みを掛けてから候補を作る・重みが全て 1 のフレームは元のまま = 前段 v3 と 1 bit 同じ)で決める。
+g<θ> を付けると(例 lk1+g0.1)、重みが声と読む 4 kHz より下のエネルギーの割合がθ未満のフレームを無声にする(声の無い時間の門・重み 1 で前段 v3 と同じ)。
ld<θ>(段 I19 の後・lmask が判定の網)= 学んだ判定 ≥ θ のビンを外す。lkb<θ>(lmask が比の網)= 学んだ比 ≥ θ のビンだけ残す(2 値)。+u を付けると、乱れた音の前段 v3 が無声と判定したフレームだけで外す(有声のフレームは重み 1)。変種ごとに手の音のエネルギーの残り nk_db と声の残り vk_db(格子の上の和の比)を記録する。
前と後に声の無い時間を足す条件(lead・claplead・oplead)では、出力の声の無い時間の大きさ pad_db(声の時間の出力の RMS に対する dB)と、出力を前段 v3 で解析した有声の率 pad_vfrac も記録する(雑音が声に化けたか)。
段 P48(paper_snv rev119)の訂正: 音楽は train_snv_sep.read_bg(学習の族と同じ)で読む。前の snv_rec_feat.real_bg は 48 kHz の音源(pool=dev の FMA)を発話の 1/3 の長さだけ読んで 3 回繰り返していた。前の記録を再現するときだけ music=loop。
    cd training && uv run --no-sync python snv_diag/i13_bgm_dev_probe.py snv_i13_e_l13 snv_i12_e_r2 snv_i10_c_pr [n=6] [conds=mus10,mus0,mus-5,lead0] [music=loop] [out=results/snv_i13/bgm_dev_probe.json]
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nvoc as N
import snv_p0_front as P
import snv_p1_eval as EV
import snv_rec_feat as RF
import train_rvoc as TR
import train_snv_p1 as T

ROOT = EV.ROOT
DEV = EV.DEV
SR = P.SR
LEAD, TAIL = 3.0, 2.0
MUSIC_LOOP = False


def utt_of(x: np.ndarray) -> dict:
    Tn = P.n_frames(len(x))
    f = P.analyze(x[: Tn * P.H].astype(np.float32))
    f0, env, per = f["f0"], f["env"], f["per"]
    s, n = T.voiced_mean(env, f0)
    lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
    return {"y": T.to_y(env, per, f0), "lf": lf, "v": (f0 > 0).astype(np.float32), "f0": f0.astype(np.float32), "vs": s.astype(np.float32), "vn": n, "wav": x[: Tn * P.H].astype(np.float32)}


def with_music(x: np.ndarray, src: str, seed: int, lead: bool, snr: float) -> tuple[np.ndarray, np.ndarray]:
    g = np.random.default_rng(seed)
    rms = np.sqrt((x ** 2).mean())
    a, b = (int(LEAD * SR), int(TAIL * SR)) if lead else (0, 0)
    clean = np.concatenate([np.zeros(a), x, np.zeros(b)])
    if MUSIC_LOOP:
        bg = RF.real_bg(len(clean), src, g)
    else:
        import train_snv_sep as SEP
        bg = SEP.read_bg(src, len(clean), g)
    noisy = clean + bg * rms * 10 ** (-snr / 20)
    return clean, np.clip(noisy, -1.0, 1.0)


def with_events(x: np.ndarray, srcs: list[str], seed: int, lead: bool, lvl: float, k: int = 4) -> tuple[np.ndarray, np.ndarray]:
    import train_snv_sep as SEP
    g = np.random.default_rng(seed)
    rms = np.sqrt((x ** 2).mean())
    a, b = (int(LEAD * SR), int(TAIL * SR)) if lead else (0, 0)
    clean = np.concatenate([np.zeros(a), x, np.zeros(b)])
    noisy = clean.copy()
    for j in range(k):
        e = SEP.read_event(srcs[int(g.integers(len(srcs)))], g, float(g.uniform(0.1, 0.8)))
        if lead:
            on = int(g.integers(0, max(1, a - len(e)))) if j % 2 == 0 or b <= len(e) else a + len(x) + int(g.integers(0, b - len(e)))
        else:
            on = int(g.integers(0, max(1, len(clean) - len(e))))
        on = min(on, len(noisy) - len(e))
        noisy[on:on + len(e)] += e * rms * 10 ** (lvl / 20)
    return clean, np.clip(noisy, -1.0, 1.0)


def pad_stats(w: np.ndarray, n_speech: int) -> dict:
    a, b = int(LEAD * SR), int(TAIL * SR)
    m = np.zeros(len(w), bool)
    m[:a] = True
    m[a + n_speech:] = True
    sp = w[a:a + n_speech]
    db = 20 * np.log10((np.sqrt((w[m] ** 2).mean()) + 1e-9) / (np.sqrt((sp ** 2).mean()) + 1e-9))
    Tn = P.n_frames(len(w))
    f0 = P.analyze(w[: Tn * P.H].astype(np.float32))["f0"]
    fm = np.zeros(Tn, bool)
    fm[: a // P.H] = True
    fm[(a + n_speech) // P.H:] = True
    return {"pad_db": round(float(db), 2), "pad_vfrac": round(float((f0[fm[: len(f0)]] > 0).mean()), 3)}


def mel_amp(x: np.ndarray, f0: np.ndarray) -> np.ndarray:
    import pyworld
    import pae as PA
    fb, _ = PA._np_consts()
    Tn = len(f0)
    xa = np.concatenate([np.zeros(P.PAD_L), x.astype(np.float64), np.zeros(PA.A)])
    T0 = 1.0 / np.where(f0 > 0, f0, 200.0)
    t = np.arange(Tn)
    tau = np.minimum(t * P.H, (t + 1) * P.H - 2 - np.ceil(1.5 * T0 * SR))
    sp = pyworld.cheaptrick(xa, np.where(f0 > 0, f0, 0.0).astype(np.float64), (P.PAD_L + tau) / SR, SR, fft_size=PA.NFFT)
    return np.sqrt(sp) @ fb.T


def oracle_utt(un: dict, clean: np.ndarray, noisy: np.ndarray, uc: dict, per_clean: bool) -> dict:
    import pae as PA
    _, dct = PA._np_consts()
    f0 = un["f0"].astype(np.float64)
    Tn = len(f0)
    an, ac = mel_amp(noisy[: Tn * P.H], f0), mel_amp(clean[: Tn * P.H], f0)
    m = np.clip(ac / np.maximum(an, 1e-12), 0.0, 1.0)
    env = (dct[:25] @ np.log(np.maximum(m * an, 1e-5)).T).astype(np.float32)
    y = T.to_y(env, np.zeros((4, Tn), np.float32), un["f0"])
    y[25:29] = (uc["y"][25:29] if per_clean else un["y"][25:29]) * (un["v"][None] > 0.5)
    s_, n_ = T.voiced_mean(env, un["f0"])
    return un | {"y": y, "vs": s_.astype(np.float32), "vn": n_}


def stft_oracle_utt(clean: np.ndarray, noisy: np.ndarray) -> dict:
    import train_snv_mask as MK
    Tn = P.n_frames(len(noisy))
    cs, ns = torch.from_numpy(clean[: Tn * P.H].astype(np.float32))[None], torch.from_numpy((noisy - clean)[: Tn * P.H].astype(np.float32))[None]
    ps, pn = MK.power(cs)[0], MK.power(ns)[0]
    u = MK.features_with_mask(noisy, (ps / (ps + pn + 1e-10)).sqrt().double())
    return u | {"wav": noisy[: Tn * P.H].astype(np.float32)}


def mask_variants(clean: np.ndarray, noisy: np.ndarray, spec: list[str], lmask) -> list[tuple[str, dict, dict]]:
    import train_snv_mask as MK
    Tn = P.n_frames(len(noisy))
    xn = noisy[: Tn * P.H].astype(np.float32)
    cs, ns = torch.from_numpy(clean[: Tn * P.H].astype(np.float32))[None], torch.from_numpy(xn - clean[: Tn * P.H].astype(np.float32))[None]
    ps, pn = MK.power(cs)[0].double(), MK.power(ns)[0].double()
    irm = ps / (ps + pn + 1e-10)
    lm = None
    if lmask is not None:
        with torch.no_grad():
            lm = torch.sigmoid(lmask(torch.from_numpy(xn)[None].to(DEV))[0].double()).transpose(0, 1).cpu()[:Tn]
    px = MK.power(torch.from_numpy(xn)[None])[0].double()

    def hold(src: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        out, last = torch.zeros_like(src), torch.zeros(src.shape[1], dtype=src.dtype)
        for t in range(src.shape[0]):
            last = torch.where(keep[t], src[t], last)
            out[t] = last
        return out
    det = ((pn > ps).double() * px).sum(1) > 0.5 * px.sum(1)
    f0c = P.f0_track(torch.from_numpy(clean[: Tn * P.H].astype(np.float32)))[0]
    out = []
    for vfull in spec:
        v, _, plus = vfull.partition("+")
        if v == "hold":
            r = (hold(ps, pn <= ps) / (ps + pn + 1e-10)).clamp(0.0, 1.0)
        elif v == "holdx":
            r = (hold(px, pn <= ps) / (px + 1e-10)).clamp(0.0, 1.0)
        elif v.startswith("lkb"):
            r = (lm >= float(v[3:])).double()
        elif v.startswith("ld"):
            r = (lm < float(v[2:])).double()
            if plus == "u":
                r = torch.where((P.f0_track(torch.from_numpy(xn))[0][:Tn] > 0)[:, None].cpu(), torch.ones_like(r), r)
        elif v.startswith("lhold"):
            r = (hold(px, lm >= float(v[5:])) / (px + 1e-10)).clamp(0.0, 1.0)
        else:
            r = irm.clamp(min=float(v[1:])) if v.startswith("f") else (ps > pn).double() if v == "bin" else lm ** float(v[2:])
        db = lambda e: round(float(10 * torch.log10((r * e).sum() / (e.sum() + 1e-20) + 1e-20)), 2)
        f0v = P.f0_track_masked(torch.from_numpy(xn), r.sqrt(), float(plus[1:]))[0] if plus.startswith("v") else P.f0_track_myin(torch.from_numpy(xn), r.sqrt())[0] if plus == "y" else None
        if plus.startswith("g"):
            kb = int(4000 / (P.SR / 2048)) + 1
            share = (r[:, :kb] * px[:, :kb]).sum(1) / (px[:, :kb].sum(1) + 1e-20)
            f0v = P.f0_track(torch.from_numpy(xn))[0].clone()
            f0v[: len(share)][(share < float(plus[1:])).numpy()[: len(f0v)]] = 0.0
        u = MK.features_with_mask(noisy, r.sqrt(), f0c if plus == "d" else f0v)
        if plus == "p":
            y = u["y"].copy()
            last = np.zeros(4, np.float32)
            for t in range(min(y.shape[1], len(det))):
                if bool(det[t]):
                    y[25:29, t] = last * (u["v"][t] > 0.5)
                else:
                    last = y[25:29, t].copy()
            u = u | {"y": y}
        out.append((vfull, u | {"wav": xn}, {"nk_db": db(pn), "vk_db": db(ps), "det_frac": round(float(det.double().mean()), 3)}))
    return out


def parse(c: str) -> tuple[str, str, bool, float]:
    for pre in ("claplead", "oplead", "clap", "op", "lead", "mus"):
        if c.startswith(pre):
            return c, pre, pre in ("lead", "claplead", "oplead"), float(c[len(pre):])
    raise ValueError(c)


def main() -> int:
    global MUSIC_LOOP
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    tags = [x for x in sys.argv[1:] if "=" not in x]
    MUSIC_LOOP = opt.get("music") == "loop"
    n = int(opt.get("n", 6))
    conds = [parse(c) for c in opt.get("conds", "mus10,mus0,mus-5,lead0").split(",")]
    evp = json.loads((ROOT / "data/snv_sep/fsd50k_eval_sources.json").read_text())["pools"] if any(k in ("clap", "op", "claplead", "oplead") for _, k, _, _ in conds) else {}

    def degrade(x: np.ndarray, kind: str, lead: bool, lvl: float, src: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
        if kind in ("mus", "lead"):
            return with_music(x, src, seed, lead, lvl)
        return with_events(x, evp["clap" if kind.startswith("clap") else "op"], seed, lead, lvl, 3 if lead else 4)
    split = json.loads((ROOT / "results/snv_i8_m13/split.json").read_text())
    val = split["val"]
    fem = [v for v in val if v.startswith("ef")]
    voices = [v for v in val if v.startswith("em")][:n] + fem[:: max(1, len(fem) // n)][:n]
    if "pool" in opt:
        import train_snv_sep as SEP
        pool = SEP.bg_split()[opt["pool"]]["music"]
    else:
        pool = json.loads((ROOT / "data/snv_p_feat_rec2/sources.json").read_text())["pools"]["music"]
    load = lambda p: librosa.load(str(p), sr=SR, mono=True)[0].astype(np.float64)
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, kana, to16
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16" if DEV == "cuda" else "int8")

    def tr_(w: np.ndarray) -> str:
        segs, _ = asr.transcribe(to16(w.astype(np.float64), EV.SR).astype(np.float32), language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(x.text for x in segs))

    gen = EV.load_gen()

    @torch.no_grad()
    def copy(u: dict) -> np.ndarray:
        y = torch.from_numpy(u["y"])[None].to(DEV)
        lf = torch.from_numpy(u["lf"])[None].to(DEV)
        v = torch.from_numpy(u["v"])[None].to(DEV)
        cond = torch.cat([y[:, :25], lf[:, None], v[:, None], y[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
        f0 = torch.from_numpy(u["f0"])[None].to(DEV)
        return gen(cond, TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:].astype(np.float32)

    meta = json.loads((EV.A1 / "convert_meta.json").read_text())
    tgts = meta["targets"][:2]
    trefs = {t: [EV.utt(f"a0/{t}_{j}") for j in (0, 1, 2)] for t in tgts}
    systems = [EV.System(t, gen) for t in tags]
    dsep = None
    if "dcopy" in opt:
        import train_snv_sep as SEP
        dname, _, dck = opt["dcopy"].partition("@")
        dsep = SEP.load(ROOT / "results" / dname / f"{dck or 'last'}.pt", DEV)
    pv = lambda a, b: a | {k: b[k] for k in ("lf", "v", "f0")}
    lmask = None
    if "lmask" in opt:
        import train_snv_mask as MK
        lname, _, lck = opt["lmask"].partition("@")
        lmask = MK.load(ROOT / "results" / lname / f"{lck or 'last'}.pt", DEV)
    vstats = []
    rows = []
    for vi, vv in enumerate(voices):
        wavs = sorted((ROOT / "data/snv_e1" / vv).glob("t*.wav"))
        xs, xr = load(wavs[0]), [load(p) for p in wavs[1:4]]
        src = pool[zlib.crc32(vv.encode()) % len(pool)]
        clean_u, clean_st = utt_of(xs), [utt_of(x) for x in xr]
        ref = tr_(copy(clean_u))
        st_of = {(kind if kind in ("mus", "lead") else kind.replace("lead", ""), lvl): [utt_of(degrade(x, kind, False, lvl, src, 7 + j)[1]) for j, x in enumerate(xr)]
                 for _, kind, _, lvl in conds}
        cases = [("clean", clean_u, clean_st, None)]
        for name, kind, lead, snr in conds:
            st_n = clean_st if opt.get("enroll") == "clean" else st_of[(kind if kind in ("mus", "lead") else kind.replace("lead", ""), snr)]
            c, z = degrade(xs, kind, lead, snr, src, 1 + vi)
            uc, un = (utt_of(c), utt_of(z)) if lead else (clean_u, utt_of(z))
            cases += [(name, un, st_n, un)]
            if opt.get("oracle") == "1":
                cases += [(f"{name}_mo", oracle_utt(un, c, z, uc, False), st_n, None), (f"{name}_mop", oracle_utt(un, c, z, uc, True), st_n, None), (f"{name}_ms", stft_oracle_utt(c, z), st_n, None)]
            if "ovar" in opt:
                for v, uv, stv in mask_variants(c, z, opt["ovar"].split(","), lmask):
                    cases += [(f"{name}_{v}", uv, st_n, None)]
                    vstats.append({"voice": vv, "cond": f"{name}_{v}"} | stv)
            if opt.get("mix", "1") == "1":
                cases += [(f"{name}_y", pv(un, uc), [pv(a, b) for a, b in zip(st_n, clean_st)], None), (f"{name}_pv", pv(uc, un), [pv(b, a) for a, b in zip(st_n, clean_st)], None)]
        for cond, u, st, cu in cases:
            cl = round(cer(ref, tr_(copy(cu))), 4) if cu is not None else None
            if dsep is not None and "wav" in u and not cond.endswith(("_y", "_pv")):
                dout = tr_(copy(SEP.features(dsep, u["wav"], DEV)))
                r = {"voice": vv, "cond": cond, "sys": "dcopy", "tgt": None, "cer": cer(ref, dout), "copy_loss": cl, "voiced_frac": round(float(u["v"].mean()), 3), "src": Path(src).name, "ref": ref[:40], "out": dout[:60]}
                rows.append(r)
                print(json.dumps(r, ensure_ascii=False), flush=True)
            padded = cond.startswith(("lead", "claplead", "oplead"))
            for s in systems:
                for tgt in [None] + tgts:
                    S = s.structure(f"self|{cond}|{vv}", st) if tgt is None else s.structure(f"t|{tgt}", trefs[tgt])
                    w = s.convert(u, st, S, 0)
                    out = tr_(w)
                    r = {"voice": vv, "cond": cond, "sys": s.tag, "tgt": tgt, "cer": cer(ref, out), "copy_loss": cl, "voiced_frac": round(float(u["v"].mean()), 3),
                         "src": Path(src).name, "ref": ref[:40], "out": out[:60]}
                    if padded:
                        r |= pad_stats(np.asarray(w, np.float64), len(xs))
                    rows.append(r)
                    print(json.dumps(r, ensure_ascii=False), flush=True)
    res: dict = {"music_read": "loop" if MUSIC_LOOP else "full", "tags": tags, "voices": voices, "conds": [c for c, _, _, _ in conds], "lead_s": LEAD, "tail_s": TAIL, "summary": {}}
    for cond in dict.fromkeys(r["cond"] for r in rows):
        res["summary"][cond] = {}
        for s in tags + (["dcopy"] if dsep is not None else []):
            for kind, sel in (("self", lambda r: r["tgt"] is None), ("F", lambda r: r["tgt"] is not None)):
                R = [r["cer"] for r in rows if r["cond"] == cond and r["sys"] == s and sel(r)]
                if R:
                    res["summary"][cond][f"{s}|{kind}"] = {"cer": round(float(np.mean(R)), 4), "cer_median": round(float(np.median(R)), 4)}
        for s in tags:
            pdb = [r["pad_db"] for r in rows if r["cond"] == cond and r["sys"] == s and "pad_db" in r]
            if pdb:
                res["summary"][cond][f"{s}|pad"] = {"pad_db_median": round(float(np.median(pdb)), 2), "pad_vfrac_median": round(float(np.median([r["pad_vfrac"] for r in rows if r["cond"] == cond and r["sys"] == s and "pad_vfrac" in r])), 3)}
        C = [r["copy_loss"] for r in rows if r["cond"] == cond and r["copy_loss"] is not None]
        res["summary"][cond]["copy_loss"] = round(float(np.mean(C)), 4) if C else None
        res["summary"][cond]["voiced_frac"] = round(float(np.mean([r["voiced_frac"] for r in rows if r["cond"] == cond])), 3)
    if vstats:
        res["vstats"] = {c: {k: round(float(np.median([r[k] for r in vstats if r["cond"] == c])), 2) for k in ("nk_db", "vk_db", "det_frac")} for c in dict.fromkeys(r["cond"] for r in vstats)}
        res["vstats_rows"] = vstats
    res["rows"] = rows
    print(json.dumps(res["summary"], indent=1), flush=True)
    if vstats:
        print(json.dumps(res["vstats"], indent=1), flush=True)
    (ROOT / opt.get("out", "results/snv_i13/bgm_dev_probe.json")).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
