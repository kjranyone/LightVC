"""段 P57 step b の族の入力(paper_snv rev134 §5 M42): 読み手(符号と回路 C)の学習に入れる「前段を通った入力」の特徴を作る。
E1 と E1-NV の全ての発話について、版 k ごとに、前段の重みの学習の族(train_snv_sep の p38 = Irodori の声かその擬似の操作者の版 + 本物の非音声の音源・床・帯域・符号化・操作者の音)と同じ混ぜを発話の全長で作り、
研究の形の前段そのもの(results/<front>/last.pt = 門 + 重み + 濾した高さ + 持続・凍結)に通した特徴を保存する(feat=v3 は同じ混ぜを前段 v3 だけで解析した特徴 = 前段の門と重みを使わない読み手の族)。
feat=both(既定)= 同じ混ぜから 2 つを 1 回で作る(data/snv_p_feat_<out> と data/snv_p_feat_<out_v3>)。門が 1 フレームも開かない項目は 2 つが同じ特徴になる(重み 1 の濾した高さは前段 v3 と同じ)。
版 k0 = Irodori の声そのまま・k1 = 擬似の操作者の版(data/snv_p_wav_x16w・無ければ k0 と同じ声)。
録音の場 = (声, k) で決まる(snv_rec_feat と同じ考え方: 同じ場の発話は、背景の種類と音源のファイルと SN 比・残響の応答・音量・床・帯域の切り・符号化・操作者の音の有無が同じ = 登録と源が同じ場に在る運用)。
発話ごとに変わるのは、背景の切り出しの位置・前後の背景だけの時間・操作者の音の数と位置と大きさ。乱数は (声, k) と (発話, k) で決まる(feat を替えても同じ混ぜ)。
背景 = なし 0.10・音楽 0.55(SN 比 U[−5, 20] dB)・雑音 0.35(U[0, 30] dB)。背景が在るとき確率 0.7 で発話の前に 0.5〜2 s・確率 0.5 で後に 0〜1 s の背景だけの時間を足す(門の持続が開いた状態で声が始まる = 運用と同じ)。
残響 0.3・音量 U[−12, +4] dB・床(色つき雑音 −70〜−30 dB)・帯域の切り 0.5・操作者の音 0.4・符号化(MP3 32 kHz 48 kb/s)0.3 は p38 の族と同じ率。
音源は学習の声には学習の側の組(train_snv_sep.make_pools の train)、検証の 36 声には開発の側の組(dev・勾配に使わない)。入力側だけの加工(当てる先は data/snv_p_feat のきれいな特徴のまま)。本物の音声は使わない。
前段を通した特徴には envu = 同じ高さの決定の下で重みを掛けない包絡(前段が何を外したかを読み手に渡す形のため)も保存する。
出力 = data/snv_p_feat_<out>/{e1,e1nv}/<声>/<発話>_k<k>.npz: f0・env・per = 発話の区間(きれいな特徴とフレームが揃う)・f0_pre / env_pre / per_pre と f0_post / env_post / per_post = 前後の背景だけの時間(当てる先は無音)。台帳 = manifest.json(項目ごとの混ぜの値と門の開く率)。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_p57_feat.py [front=snv_p45_gate_t80] [feat=both|front|v3] [out=fr57] [out_v3=v357] [workers=16] [limit=N] [selftest]
"""
from __future__ import annotations

import json
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import pae as PA
import snv_p0_front as P
import train_snv_mask as MK
import train_snv_p1 as T
import train_snv_sep as SEP

ROOT = Path(__file__).resolve().parent.parent
H = P.H
FAM = SEP.FAMS["p38"]
P_BG = (("none", 0.10), ("music", 0.55), ("noise", 0.35))
SNR = {"music": (-5.0, 20.0), "noise": (0.0, 30.0)}
P_PRE, PRE_S, P_POST, POST_S = 0.7, (0.5, 2.0), 0.5, (0.0, 1.0)
_G: dict = {}


def front_raw(G, x: np.ndarray) -> tuple[dict, float]:
    Tn = P.n_frames(len(x))
    x = np.asarray(x[: Tn * H], np.float32)
    xt = torch.from_numpy(x)
    if G is None:
        m, opn = torch.ones(Tn, MK.NB, dtype=torch.float64), 0.0
        f0 = P.f0_track(xt)[0]
    else:
        with torch.no_grad():
            p = torch.sigmoid(G.gate(xt[None])[0, 0].double())[:Tn]
            op = MK.hysteresis(p, *G.hyst, G.thr) if G.hyst else p >= G.thr
            mm = torch.sigmoid(G.mask(xt[None])[0].double()).transpose(0, 1)[:Tn].sqrt()
        if G.gamma != 1.0:
            mm = mm.pow(G.gamma)
        m, opn = torch.where(op[:, None], mm, torch.ones_like(mm)), float(op.double().mean())
        f0 = P.f0_track_myin(xt, m)[0] if G.dec == "myin" else P.f0_track(xt)[0]
    f0n = f0.cpu().numpy().astype(np.float64)
    pw = P.power_p(xt, Tn)
    pm = pw * m.pow(2)
    per = P.periodicity_from_power(pm, f0).cpu().numpy()
    fb, _ = PA._np_consts()
    num, den = pm.numpy() @ fb.T, pw.numpy() @ fb.T
    g = np.where(den > 0, np.sqrt(np.divide(num, den, out=np.ones_like(num), where=den > 0)), 1.0)
    amp = P.envelope_amp(x, f0n)
    env = P.envelope_from_amp(amp * g)
    out = {"f0": f0n.astype(np.float32), "env": env.astype(np.float32), "per": per.astype(np.float32)}
    return (out | {"envu": P.envelope_from_amp(amp).astype(np.float32)} if G is not None else out), opn


def scene_of(voice: str, k: int, pools: dict) -> dict:
    g = np.random.default_rng([zlib.crc32(voice.encode()), k, 5757])
    r, acc, bg = g.random(), 0.0, "none"
    for name, pr in P_BG:
        acc += pr
        if r < acc:
            bg = name
            break
    src = None
    if bg == "music":
        pool = pools["music_fma"] if (pools["music_fma"] and g.random() < SEP.P_FMA) else pools["music_musan"]
        src = pool[int(g.integers(len(pool)))]
    elif bg == "noise":
        src = pools["noise"][int(g.integers(len(pools["noise"])))]
    sc = {"bg": bg, "src": src, "snr": float(g.uniform(*SNR[bg])) if bg != "none" else 0.0, "gain": float(g.uniform(*SEP.GAIN)),
          "floor": (pools["floor"][int(g.integers(len(pools["floor"])))], float(g.uniform(*SEP.FLOOR_DB))), "op": bool(g.random() < SEP.P_OP),
          "band": float(np.exp(g.uniform(np.log(SEP.BAND_HZ[0]), np.log(SEP.BAND_HZ[1])))) if g.random() < SEP.P_BAND else None, "codec": bool(g.random() < FAM["codec"])}
    return sc | {"rir": SEP.rir(g) if g.random() < SEP.P_REV else None}


def reverb(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    import scipy.fft as sfft
    s = x.astype(np.float64)
    n = len(s)
    nf = sfft.next_fast_len(n + len(h), real=True)
    z = sfft.irfft(sfft.rfft(s, nf) * sfft.rfft(h, nf), nf)[:n]
    return (z * (np.sqrt((s ** 2).mean()) / (np.sqrt((z ** 2).mean()) + 1e-12)) if (s ** 2).mean() > 0 else z).astype(np.float32)


def mix_item(rel: str, k: int, pools: dict) -> tuple[np.ndarray, int, int, int, dict]:
    kind, voice, stem = rel.split("/")
    sc = scene_of(voice, k, pools)
    g = np.random.default_rng([zlib.crc32(rel.encode()), k, 57])
    wav = SEP.WAVDIR[kind] / voice / f"{stem}.wav"
    xop = k == 1 and SEP.xop_wav(wav).exists()
    x = sf.read(str(SEP.xop_wav(wav) if xop else wav), dtype="float32")[0]
    Tc = int(np.load(T.FEAT / f"{rel}.npz")["f0"].shape[0])
    Tn = min(Tc, len(x) // H)
    x = x[: Tn * H]
    rms = float(np.sqrt((x.astype(np.float64) ** 2).mean()) + 1e-12)
    bg = sc["bg"]
    pre = int(g.uniform(*PRE_S) * P.SR / H) if (bg != "none" and g.random() < P_PRE) else 0
    post = int(g.uniform(*POST_S) * P.SR / H) if (bg != "none" and g.random() < P_POST) else 0
    xf = np.concatenate([np.zeros(pre * H, np.float32), x, np.zeros(post * H, np.float32)])
    if sc["rir"] is not None:
        xf = reverb(xf, sc["rir"])
    ext: dict = {"floor": sc["floor"]}
    if sc["op"]:
        ev = []
        for _ in range(int(g.integers(SEP.OP_N[0], SEP.OP_N[1] + 1))):
            pl = pools["clap"] if g.random() < SEP.P_CLAP else pools["op"]
            dur = float(g.uniform(*SEP.OP_DUR))
            ev.append((pl[int(g.integers(len(pl)))], dur, int(g.integers(0, max(1, len(xf) - int(dur * P.SR)))), float(g.uniform(*SEP.OP_DB))))
        ext["events"] = ev
    if sc["band"] is not None:
        ext["band"] = sc["band"]
    xn = SEP.mix(xf, rms, bg, sc["src"], sc["snr"], g, False, sc["gain"], ext)
    if sc["codec"]:
        xn = SEP.codec_mp3(xn)
    meta = {"xop": bool(xop), "bg": bg, "snr": round(sc["snr"], 2), "src": Path(sc["src"]).name if sc["src"] else None, "pre": pre, "post": post, "rev": sc["rir"] is not None, "gain_db": round(sc["gain"], 2),
            "floor_db": round(sc["floor"][1], 1), "events": len(ext.get("events", [])), "band_hz": round(sc["band"]) if sc["band"] is not None else None, "codec": sc["codec"]}
    return np.asarray(xn, np.float32), pre, Tn, post, meta


def save(dst: Path, f: dict, pre: int, Tn: int, post: int, meta: dict, opn: float) -> dict:
    cut = lambda a, s, e: a[..., s:e]
    d = {key: cut(v, pre, pre + Tn) for key, v in f.items()} | {f"{key}_pre": cut(v, 0, pre) for key, v in f.items()} | {f"{key}_post": cut(v, pre + Tn, pre + Tn + post) for key, v in f.items()}
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp.npz")
    np.savez(tmp, **d)
    tmp.replace(dst)
    m = meta | {"gate_open": round(opn, 3), "voiced_pre": round(float((d["f0_pre"] > 0).mean()), 3) if pre else None, "envu": "envu" in f}
    dst.with_suffix(".json").write_text(json.dumps(m))
    return m


def work(job: tuple) -> tuple:
    rel, k, val, front, outs = job
    torch.set_num_threads(1)
    dsts = {kind: ROOT / f"data/snv_p_feat_{o}/{rel}_k{k}.npz" for kind, o in outs.items()}
    have = {kind: json.loads(q.with_suffix(".json").read_text()) for kind, q in dsts.items() if q.exists() and q.with_suffix(".json").exists()}
    if "front" in have and not have["front"].get("envu"):
        del have["front"]
    if len(have) == len(dsts):
        return rel, k, have
    if "pools" not in _G:
        ptr, pdv = SEP.make_pools(SEP.bg_split(), FAM)
        _G["pools"] = {False: ptr, True: pdv}
        _G["front"] = MK.load(ROOT / "results" / front / "last.pt", "cpu") if "front" in outs else None
    xn, pre, Tn, post, meta = mix_item(rel, k, _G["pools"][val])
    opn, ff = None, None
    if "front" in dsts:
        if "front" in have:
            opn = have["front"]["gate_open"]
        else:
            ff, opn = front_raw(_G["front"], xn)
            assert ff["f0"].shape[0] == pre + Tn + post, (rel, ff["f0"].shape, pre, Tn, post)
            have["front"] = save(dsts["front"], ff, pre, Tn, post, meta, opn)
    if "v3" in dsts and "v3" not in have:
        f3 = {q: v for q, v in ff.items() if q != "envu"} if (ff is not None and opn == 0.0) else front_raw(None, xn)[0]
        have["v3"] = save(dsts["v3"], f3, pre, Tn, post, meta, 0.0)
    return rel, k, have


def selftest(front: str) -> None:
    import librosa
    G = MK.load(ROOT / "results" / front / "last.pt", "cpu")
    x = librosa.load(str(ROOT / "data/snv_e1/ef000/t00.wav"), sr=P.SR, mono=True)[0]
    g = np.random.default_rng(0)
    pools = SEP.make_pools(SEP.bg_split(), FAM)[0]
    rms = float(np.sqrt((x.astype(np.float64) ** 2).mean()))
    for name, xx in (("clean", x), ("music 5 dB", SEP.mix(x, rms, "music", pools["music_musan"][0], 5.0, g, False, 0.0, {}))):
        xx = np.asarray(xx, np.float32)
        f, opn = front_raw(G, xx)
        ref = MK.features(G, xx, "cpu")
        assert float(np.abs(T.to_y(f["env"], f["per"], f["f0"]) - ref["y"]).max()) == 0.0 and float(np.abs(f["f0"] - ref["f0"]).max()) == 0.0, f"{name}: train_snv_mask.features と違う"
        assert abs(opn - ref["gate_open"]) < 1e-12
        print("selftest", name, "門の開く率", round(opn, 3), flush=True)
    f, _ = front_raw(None, x)
    a = P.analyze(x[: P.n_frames(len(x)) * H])
    assert float(np.abs(T.to_y(f["env"], f["per"], f["f0"]) - T.to_y(a["env"], a["per"], a["f0"])).max()) == 0.0, "feat=v3 が前段 v3 と違う"
    xa, pre, Tn, post, meta = mix_item("e1/ef000/t01", 1, pools)
    xb = mix_item("e1/ef000/t01", 1, pools)[0]
    assert len(xa) == (pre + Tn + post) * H and float(np.abs(xa - xb).max()) == 0.0, "混ぜが (発話, k) で決まらない"
    m2 = mix_item("e1/ef000/t02", 1, pools)[4]
    same = ("bg", "snr", "src", "rev", "gain_db", "floor_db", "band_hz", "codec")
    assert all(meta[q] == m2[q] for q in same), "同じ (声, k) の発話の場が違う"
    y = reverb(x, SEP.rir(np.random.default_rng(1)))
    assert abs(float(np.sqrt((y.astype(np.float64) ** 2).mean())) / rms - 1.0) < 1e-3, "残響が音量を変える"
    print("selftest ok(train_snv_mask.features と 1 bit 同じ・feat=v3 は前段 v3 と同じ・混ぜは (発話, k) で決まる・場は (声, k) で決まる)", meta, flush=True)


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    front, feat = opt.get("front", "snv_p45_gate_t80"), opt.get("feat", "both")
    if "selftest" in sys.argv[1:]:
        selftest(front)
        return 0
    outs = ({"front": opt.get("out", "fr57")} if feat in ("both", "front") else {}) | ({"v3": opt.get("out_v3", "v357")} if feat in ("both", "v3") else {})
    split = json.loads((ROOT / "results/snv_p30_c/split.json").read_text())
    val = set(split["val"])
    jobs = []
    for kind, pat in (("e1", "t*.npz"), ("e1nv", "*.npz")):
        for q in sorted((T.FEAT / kind).glob(f"*/{pat}")):
            if (SEP.WAVDIR[kind] / q.parent.name / f"{q.stem}.wav").exists():
                for k in (0, 1):
                    jobs.append((f"{kind}/{q.parent.name}/{q.stem}", k, q.parent.name in val, front, outs))
    first = set(split["train"][::8]) | set(split["train"][4::8]) | val
    jobs.sort(key=lambda j: not (j[0].startswith("e1/") and j[0].split("/")[1] in first))
    if "limit" in opt:
        jobs = jobs[:: max(1, len(jobs) // int(opt["limit"]))][: int(opt["limit"])]
    print("items", len(jobs), "front", front, "feat", feat, "out", {kd: f"data/snv_p_feat_{o}" for kd, o in outs.items()}, flush=True)
    man: dict = {kd: {} for kd in outs}
    with ProcessPoolExecutor(int(opt.get("workers", 16))) as ex:
        for i, (rel, k, have) in enumerate(ex.map(work, jobs, chunksize=4), 1):
            for kd in outs:
                man[kd][f"{rel}_k{k}"] = have[kd]
            if i % 1000 == 0:
                print(i, "/", len(jobs), flush=True)
    for kd, o in outs.items():
        (ROOT / f"data/snv_p_feat_{o}/manifest.json").write_text(json.dumps({"paper": "current/paper_snv.md §5 M42", "front": front if kd == "front" else None, "feat": kd,
                                                                             "family": "p38 と同じ率の混ぜ(発話の全長・場は (声, k) で決まる・音楽 U[−5, 20] dB・雑音 U[0, 30] dB・前後の背景だけの時間)",
                                                                             "sources": "train_snv_sep.make_pools(bg_split, FAMS['p38'])(学習の声 = train・検証の声 = dev)・帰属は data/snv_p_feat_rec2/sources.json・data/snv_sep/*.json", "n": len(man[kd]), "items": man[kd]}, ensure_ascii=False))
    kd0 = "front" if "front" in outs else "v3"
    print("done", len(man[kd0]), "門の開く率の平均", round(float(np.mean([m["gate_open"] for m in man[kd0].values()])), 3), {b: sum(m["bg"] == b for m in man[kd0].values()) for b in ("none", "music", "noise")}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
