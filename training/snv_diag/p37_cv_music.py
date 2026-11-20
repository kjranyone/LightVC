"""段 P37 step c(results/snv_p37/prereg.yaml の step_c・学習なし・評価の入力だけ): 本物の男声(開発の組 devA の有効な源)に本物の音楽を足した物差し。
開発の足された音の物差し(i13_bgm_dev_probe)は検証の Irodori の声で、前段の重みの学習の族の中の声。重みと門の利得が本物の男声に移るかを、devA の録音 + 伴奏で測る。
源と登録(e0〜e7)の 16 bit の wav に、門と重みの学習に使っていない音楽(train_snv_sep.bg_split の dev・話者ごとに 1 本)を発話の RMS に対する SN 比 N dB で足す(i13_bgm_dev_probe.with_music)。条件 = clean / musN。
段 P39: clapN = 本物の手拍子(FSD50K の評価の組・data/snv_sep/fsd50k_eval_sources.json・学習に使わない)4 個を、発話の RMS に対して N dB で声と重ねる(i13_bgm_dev_probe.with_events)。set=devB = 確認の組(results/snv_d5/dev_rows_devB.json の ref)。段 P48(rev119): 音楽は既定で train_snv_sep.read_bg で読む(i13_bgm_dev_probe.with_music の訂正・前の記録は 48 kHz の音源を 1/3 の長さで 3 回繰り返していた・music=loop で再現)。
enroll=clean = 登録はきれいなまま(源だけを乱す)。leadN = 源の前 3 s・後 2 s に伴奏だけの時間帯(登録は musN と同じ)。prodN = 制作の混ぜ方(語りの間だけ伴奏を 8 dB 下げる・立ち上がり 30 ms・戻り 400 ms → 全体の圧縮 3 : 1 → 尖頭値 −1 dBFS → AAC 128 kb/s・SN 比は下げる前の伴奏で決める・理想の重みの行は作らない)。
門つきの系の行には、源の門の開く率 gate_open を書く。
段 P44: oracle=1 = 理想の比の重み(2048 点の格子・録音を S・足した音を N として |S|² /(|S|² + |N|²))を源と登録に掛けた特徴(i13_bgm_dev_probe.stft_oracle_utt・高さと有声は足した入力の前段 v3 のまま)を、先頭の系(前段なしの系)に通した行 <系>+ideal を足す(評価の中の計算で、学習の目標にしない)。
hybrid=<重みの tag> = 学んだ重み m_L と理想の重み m_I の混ぜを、同じ経路(全てのフレームに掛ける・高さは前段 v3 のまま)で通した行を足す: +learned = m_L・+hmax = max(m_L, m_I)(声を落とす誤りだけ直す)・+hmin = min(m_L, m_I)(足された音を残す誤りだけ直す)。
段 P42: skip=32 n=64 ref=<snv_dev_eval.py の出力> = devB の 33〜64 人目(参照の書き起こしはその出力の ref)。
段 P53: comb=<門の tag> combw=0,1 combsrc=myin,clean [combideal=1] = 門つきの重み(研究の形と同じ計算)の、門が開いたフレームで 1.5 kHz より下の h·f0 に最も近い格子の点と ±w 点を max(m, 1) にした行 <先頭の系>+mask:<tag>+comb:<src>:w<w> を足す。
高さの源 myin = 系の濾した YIN(足した入力)・clean = きれいな録音の前段 v3 の高さ(評価の中だけ)。特徴の高さは myin のまま。combideal=1 = clean・w 1 で 1 の代わりに理想の比の重みにする行(+combideal:clean:w1)。
行に重みの声の残り vk・足された音の残り nk(事前の前は vk0・nk0・全フレームのエネルギーの割合)と、事前が上げた格子の割合 kf を書く。only_extra=1 = 系そのものの行を作らない(足す行だけ)。
段 P54: rate=r = 源と登録を i15_rate_probe.wsola で r 倍に速めてから音を足す(高さは保つ)。srcfloor=1 = 足さない条件で、速めた源そのものの書き起こしの CER の行 src を足す。
系 = snv_p1_eval.System(門なしの系は、同じ波形を前段 v3 で解析した特徴 = +rewav と同じ計算)。目標は成績表と同じ割り当て(A0F の先頭 4 声から 2 声)・k 0 と 7。CER の参照 = 元の音の書き起こし(成績表と同じ)。
出力の行は数値だけ(書き起こしは保存しない)。Common Voice の録音は評価の入力だけに使う(学習・目標にしない・話者の特定を試みない)。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/p37_cv_music.py snv_p30_c snv_p30_c+mask:snv_i24_gate_hyst snv_p30_c+mask:snv_p37_gate [n=24] [conds=mus10,mus0,clap0] [set=devA] [skip=0] [ref=results/snv_d5/dev_rows_devA.json] [out=results/snv_p37/cv_music.json]
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i13_bgm_dev_probe as B
import snv_p1_eval as EV
import train_snv_p1 as T
import train_snv_sep as SEP

ROOT = EV.ROOT
DEV = EV.DEV
KS = (0, 7)


def aac(x: np.ndarray) -> np.ndarray:
    import subprocess
    import tempfile
    import soundfile as sf
    n = len(x)
    with tempfile.TemporaryDirectory() as d:
        sf.write(f"{d}/a.wav", np.clip(x, -1.0, 1.0), B.SR, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{d}/a.wav", "-ac", "1", "-c:a", "aac", "-b:a", "128k", f"{d}/a.m4a"], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{d}/a.m4a", "-ar", str(B.SR), "-ac", "1", "-c:a", "pcm_f32le", f"{d}/b.wav"], check=True)
        y = sf.read(f"{d}/b.wav", dtype="float64")[0]
    N = 1 << int(np.ceil(np.log2(n + len(y))))
    c = np.fft.irfft(np.fft.rfft(y, N) * np.conj(np.fft.rfft(x.astype(np.float64), N)), N)
    lags = np.concatenate([np.arange(0, 4801), np.arange(-4800, 0)])
    lag = int(lags[np.argmax(c[lags])])
    y = y[lag:] if lag >= 0 else np.concatenate([np.zeros(-lag), y])
    return np.concatenate([y, np.zeros(max(0, n - len(y)))])[:n]


def smooth(target: np.ndarray, down: float, up: float, hop_s: float) -> np.ndarray:
    a_dn, a_up = np.exp(-hop_s / down), np.exp(-hop_s / up)
    out, v = np.empty(len(target)), float(target[0])
    for t, g_ in enumerate(target):
        a = a_dn if g_ < v else a_up
        v = a * v + (1 - a) * g_
        out[t] = v
    return out


def produced(x: np.ndarray, src: str, seed: int, snr: float) -> tuple[np.ndarray, np.ndarray]:
    import train_snv_sep as SEP
    g = np.random.default_rng(seed)
    n, hop = len(x), B.SR // 100
    rms = np.sqrt((x ** 2).mean())
    m = SEP.read_bg(src, n, g) * rms * 10 ** (-snr / 20)
    nf = n // hop + 1
    fr = lambda y: np.sqrt(np.array([np.mean(y[t * hop:(t + 1) * hop] ** 2) if t * hop < n else 0.0 for t in range(nf)]) + 1e-12)
    ev = fr(x)
    duck = smooth(np.where(ev > np.percentile(ev, 95) * 10 ** (-20 / 20), 10 ** (-8 / 20), 1.0), 0.030, 0.400, 0.01)
    up = lambda gf: np.interp(np.arange(n) / hop, np.arange(nf), gf)
    y = x + m * up(duck)
    L = 20 * np.log10(fr(y))
    th = np.percentile(L, 90) - 6.0
    gr = smooth(10 ** (-np.maximum(L - th, 0.0) * (1 - 1 / 3) / 20), 0.010, 0.200, 0.01)
    y = y * up(gr)
    y = y * 10 ** (-1 / 20) / (np.abs(y).max() + 1e-12)
    return x, np.clip(aac(y), -1.0, 1.0)


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    tags = [x for x in sys.argv[1:] if "=" not in x]
    n = int(opt.get("n", 24))
    conds = opt.get("conds", "mus10,mus0").split(",")
    split = opt.get("set", "devA")
    man = json.loads((ROOT / "data/cv22_ja/dev_manifest.json").read_text())
    dv = json.loads((ROOT / opt.get("ref", f"results/snv_d5/dev_rows_{split}.json")).read_text())["ref"]
    skip = int(opt.get("skip", 0))
    oracle = opt.get("oracle") == "1"
    B.MUSIC_LOOP = opt.get("music") == "loop"
    enroll_clean = opt.get("enroll") == "clean"
    hyb = None
    if opt.get("hybrid"):
        import torch
        import train_snv_mask as MK
        hyb = MK.load(ROOT / "results" / opt["hybrid"] / "last.pt", DEV)

    def hybrid_utt(clean: np.ndarray, noisy: np.ndarray, kind: str) -> dict:
        Tn = B.P.n_frames(len(noisy))
        cs, zs = torch.from_numpy(clean[: Tn * B.P.H].astype(np.float32)), torch.from_numpy(noisy[: Tn * B.P.H].astype(np.float32))
        ps, pn = MK.power(cs[None])[0], MK.power((zs - cs)[None])[0]
        mi = (ps / (ps + pn + 1e-10)).sqrt().double()[:Tn]
        with torch.no_grad():
            ml = MK.weight(hyb, hyb(zs[None].to(DEV))[0].double()).transpose(0, 1).cpu()[:Tn]
        m = ml if kind == "learned" else torch.maximum(ml, mi) if kind == "hmax" else torch.minimum(ml, mi)
        return MK.features_with_mask(noisy, m) | {"wav": noisy[: Tn * B.P.H].astype(np.float32)}
    cmb, cvars = None, []
    if opt.get("comb"):
        import torch
        import train_snv_mask as MK
        cmb = MK.load(ROOT / "results" / opt["comb"] / "last.pt", DEV)
        cvars = [("one", s, int(w)) for s in opt.get("combsrc", "myin,clean").split(",") for w in opt.get("combw", "0,1").split(",")] + ([("ideal", "clean", 1)] if opt.get("combideal") == "1" else [])
    only_extra = opt.get("only_extra") == "1"

    def comb_utt(clean: np.ndarray, noisy: np.ndarray, level: str, src: str, w: int) -> dict:
        Tn = B.P.n_frames(len(noisy))
        xt = torch.from_numpy(noisy[: Tn * B.P.H].astype(np.float32))
        cs = torch.from_numpy(clean[: Tn * B.P.H].astype(np.float32))
        with torch.no_grad():
            xin = xt[None].to(DEV)
            p = torch.sigmoid(cmb.gate(xin)[0, 0].double()).cpu()[:Tn]
            mm = torch.sigmoid(cmb.mask(xin)[0].double()).transpose(0, 1).cpu()[:Tn].sqrt()
        op = MK.hysteresis(p, *cmb.hyst, cmb.thr) if cmb.hyst else p >= cmb.thr
        m = torch.where(op[:, None], mm, torch.ones_like(mm))
        f0m = B.P.f0_track_myin(xt, m)[0]
        f0 = (f0m if src == "myin" else B.P.f0_track(cs)[0]).double().cpu()[:Tn]
        nb = m.shape[1]
        hz = B.SR / (2 * (nb - 1))
        keep = torch.zeros(Tn, nb, dtype=torch.bool)
        rows_, h = torch.arange(Tn), 1
        while True:
            ok = (f0 > 0) & (h * f0 <= 1500.0)
            if not bool(ok.any()):
                break
            k0 = torch.round(h * f0 / hz).long()
            for d in range(-w, w + 1):
                keep[rows_[ok], (k0[ok] + d).clamp(0, nb - 1)] = True
            h += 1
        keep &= op[:, None]
        ps, pn = MK.power(cs[None])[0].double()[:Tn], MK.power((xt - cs)[None])[0].double()[:Tn]
        hi = torch.ones_like(m) if level == "one" else (ps / (ps + pn + 1e-10)).sqrt()
        m2 = torch.where(keep, torch.maximum(m, hi), m)
        frac = lambda mk, pw: float((mk.pow(2) * pw).sum() / (pw.sum() + 1e-20))
        lo = torch.arange(nb) * hz <= 1500.0
        sel = op & (f0 > 0)
        kf = float(keep[sel][:, lo].double().mean()) if bool(sel.any()) else 0.0
        st = {"vk": round(frac(m2, ps), 4), "nk": round(frac(m2, pn), 4), "vk0": round(frac(m, ps), 4), "nk0": round(frac(m, pn), 4), "kf": round(kf, 4)}
        return MK.features_with_mask(noisy, m2, f0m) | {"wav": noisy[: Tn * B.P.H].astype(np.float32), "st": st}
    pool = SEP.bg_split()["dev"]["music"]
    claps = json.loads((ROOT / "data/snv_sep/fsd50k_eval_sources.json").read_text())["pools"]["clap"] if any(c.startswith("clap") for c in conds) else []
    rate = float(opt.get("rate", 1.0))
    if rate != 1.0:
        import i15_rate_probe as RP
        load = lambda name: RP.wsola(librosa.load(str(T.FEAT / f"cv/{name}.wav"), sr=B.SR, mono=True)[0].astype(np.float64), rate)
    else:
        load = lambda name: librosa.load(str(T.FEAT / f"cv/{name}.wav"), sr=B.SR, mono=True)[0].astype(np.float64)
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, to16, kana
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16" if DEV == "cuda" else "int8")

    def tr(w: np.ndarray) -> str:
        segs, _ = asr.transcribe(to16(w.astype(np.float64), EV.SR).astype(np.float32), language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(x.text for x in segs))

    gen = EV.load_gen()
    tgts = json.loads((EV.A1 / "convert_meta.json").read_text())["targets"][:4]
    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tgts}
    systems = [EV.System(t, gen) for t in tags]
    rows = []
    for i, s in enumerate([x for x in man["speakers"] if x["split"] == split][skip:n], start=skip):
        sp = s["speaker"]
        srcs = [f"{sp}_{c['name']}" for c in s["src"][:2] if f"{sp}_{c['name']}" in dv and dv[f"{sp}_{c['name']}"]["text_vs_orig"] <= 0.1]
        if not srcs:
            continue
        xe = [load(f"{sp}_{c['name']}") for c in s["enroll"]]
        mus = pool[zlib.crc32(sp.encode()) % len(pool)]
        tg = [tgts[i % 4], tgts[(i + 2) % 4]]
        for cond in ["clean"] + conds:
            if cond == "clean":
                mixp = lambda x, seed, src_: (x, x)
            elif cond.startswith("clap"):
                mixp = lambda x, seed, src_: B.with_events(x, claps, seed, False, float(cond[4:]))
            elif cond.startswith("lead"):
                mixp = lambda x, seed, src_: B.with_music(x, mus, seed, src_, float(cond[4:]))
            elif cond.startswith("prod"):
                mixp = lambda x, seed, src_: produced(x, mus, seed, float(cond[4:]))
            else:
                mixp = lambda x, seed, src_: B.with_music(x, mus, seed, False, float(cond[3:]))
            mix = lambda x, seed: mixp(x, seed, True)[1]
            enr = [B.utt_of(x if enroll_clean else mixp(x, 7 + j, False)[1]) for j, x in enumerate(xe)]
            enr_i = ([B.utt_of(x) for x in xe] if enroll_clean else [B.stft_oracle_utt(*mixp(x, 7 + j, False)) for j, x in enumerate(xe)]) if oracle and cond != "clean" and not cond.startswith("prod") else None
            for name in srcs:
                xs = load(name)
                if cond == "clean" and opt.get("srcfloor") == "1":
                    rows.append({"spk": sp, "idx": i, "src": name, "cond": cond, "sys": "src", "tgt": "", "k": 0, "cer": round(cer(dv[name]["orig"], tr(xs)), 4)})
                u = B.utt_of(mix(xs, 1 + i))
                for sy in ([] if only_extra else systems):
                    go = sy.prep(u).get("gate_open") if sy.sep is not None and sy.sep != "rewav" else None
                    for v in tg:
                        for k in KS:
                            w = sy.convert(u, enr, sy.structure(f"t|{v}", trefs[v]), k)
                            rows.append({"spk": sp, "idx": i, "src": name, "cond": cond, "sys": sy.tag, "tgt": v, "k": k, "cer": round(cer(dv[name]["orig"], tr(w)), 4)} | ({"gate_open": round(float(go), 3)} if go is not None else {}))
                if hyb is not None and cond != "clean":
                    for hk in ("learned", "hmax", "hmin"):
                        eh = [hybrid_utt(x, mix(x, 7 + j), hk) for j, x in enumerate(xe)]
                        uh = hybrid_utt(xs, mix(xs, 1 + i), hk)
                        for v in tg:
                            for k in KS:
                                w = systems[0].convert(uh, eh, systems[0].structure(f"t|{v}", trefs[v]), k)
                                rows.append({"spk": sp, "idx": i, "src": name, "cond": cond, "sys": f"{systems[0].tag}+{hk}", "tgt": v, "k": k, "cer": round(cer(dv[name]["orig"], tr(w)), 4)})
                if cmb is not None and cond != "clean":
                    for level, csrc, cw in cvars:
                        ec = [comb_utt(x, mix(x, 7 + j), level, csrc, cw) for j, x in enumerate(xe)]
                        uc = comb_utt(xs, mix(xs, 1 + i), level, csrc, cw)
                        tag = f"{systems[0].tag}+mask:{opt['comb']}+comb{'' if level == 'one' else 'ideal'}:{csrc}:w{cw}"
                        for v in tg:
                            for k in KS:
                                w = systems[0].convert(uc, ec, systems[0].structure(f"t|{v}", trefs[v]), k)
                                rows.append({"spk": sp, "idx": i, "src": name, "cond": cond, "sys": tag, "tgt": v, "k": k, "cer": round(cer(dv[name]["orig"], tr(w)), 4)} | uc["st"])
                if enr_i is not None:
                    ui = B.stft_oracle_utt(*mixp(xs, 1 + i, True))
                    for v in tg:
                        for k in KS:
                            w = systems[0].convert(ui, enr_i, systems[0].structure(f"t|{v}", trefs[v]), k)
                            rows.append({"spk": sp, "idx": i, "src": name, "cond": cond, "sys": f"{systems[0].tag}+ideal", "tgt": v, "k": k, "cer": round(cer(dv[name]["orig"], tr(w)), 4)})
        seen = list(dict.fromkeys(r["sys"] for r in rows if r["idx"] == i))
        print("speaker", i, {c: {t: round(float(np.mean([min(r["cer"], 1.0) for r in rows if r["idx"] == i and r["cond"] == c and r["sys"] == t])), 3) for t in seen if any(r["idx"] == i and r["cond"] == c and r["sys"] == t for r in rows)} for c in ["clean"] + conds}, flush=True)
    res = {"prereg": opt.get("prereg", "results/snv_p37/prereg.yaml(step_c)"), "music_read": "loop" if B.MUSIC_LOOP else "full", "enroll": "clean" if enroll_clean else "same", "rate": rate, "set": split, "tags": tags, "conds": ["clean"] + conds, "n_speakers": len({r["spk"] for r in rows}), "rows": rows}
    (ROOT / opt.get("out", "results/snv_p37/cv_music.json")).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
