"""段 P57 step a(results/snv_p57/prereg.yaml・学習なし・評価の入力だけ): 伴奏の上の本物の男声で、内容が「符号の入力(前段の特徴)」に残っているかと、今の読み手(符号 + 回路 C)が前段の残りにどれだけ弱いかを測る。
物差しは p37_cv_music.py と同じ(devA の有効な源・同じ音楽・同じ種・正しい読み込み)。行は 2 種類:
rows=copy  = 前段の特徴をそのまま描画器で音にした写し(変換なし・k 0)の CER。特徴 = none(足した入力の前段 v3)・front(研究の形の門 + 重み + 濾した高さ)・ideal(理想の比の重み)。足さない条件は none だけ。
             写しの CER が低く変換の CER が高ければ、内容は符号の入力に在り、読み手の側で失われている。
rows=floor = 理想の比の重みを床 f(パワーの比の下限・0.1 = −10 dB・0.01 = −20 dB)で止めた特徴を源と登録に掛け、先頭の系(前段なし)で変換した行 <系>+idealf<f>。読み手が要る深さ。
rows=delay = 本当の伴奏のパワーを D 秒だけ遅らせて今の伴奏の代わりに使う重み(系 2s の q3)。N̂(t) = 伴奏のパワーの、t − D で終わる 0.1 s の平均。重み = max(P_x − N̂, 0) /(max(P_x − N̂, 0) + N̂)(P_x = 混ざった音のパワー)。
             D = 0x は平均なしの今の伴奏そのもの(この式の天井)。行 <系>+delay<D>。伴奏が D だけ前から推せるなら戻り、推せないなら戻らない。
rows=judge = 判定と値の切り分け(prereg の step_a_judge)。ibin = 理想の 2 値(格子ごとに声が勝つ所だけ残す)・lbin = 学んだ重みの 2 値(研究の形の実効の重み > 0.5 の所だけ残す・門が閉じたフレームは全て残す)・
             lji = 学んだ判定 + 理想の値(学んだ重み > 0.5 の所に理想の比・他は 0)・ijl = 理想の判定 + 学んだ値(声が勝つ所に学んだ重み・他は 0)。高さと有声は理想の行と同じ(足した入力の前段 v3)。行 <系>+judge<名>。
floor・delay・judge の行は目標 1 声・k 0 だけ(同じ試行の基準の行と対にする)。行に重みの声の残り vk・足された音の残り nk(エネルギーの割合)を書く。
出力の行は数値だけ(書き起こしは保存しない)。Common Voice の録音は評価の入力だけに使う(学習・目標にしない・話者の特定を試みない)。理想の重みは評価の中の計算で、学習の目標にしない。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/p57_info.py snv_p30_c rows=copy front=snv_p45_gate_t80 n=64 out=results/snv_p57/copy.json
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/p57_info.py snv_p30_c rows=floor,delay floors=0.1,0.01 delays=0x,0,0.1,0.3,1.0 n=64 out=results/snv_p57/oracle.json
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i13_bgm_dev_probe as B
import nvoc as N
import pae as PA
import snv_p1_eval as EV
import train_rvoc as TR
import train_snv_mask as MK
import train_snv_p1 as T
import train_snv_sep as SEP

ROOT = EV.ROOT
DEV = EV.DEV
KS = (0, 7)


AVG = 20


FRONT = None


def base_of(clean: np.ndarray, noisy: np.ndarray) -> dict:
    Tn = B.P.n_frames(len(noisy))
    xn = noisy[: Tn * B.P.H].astype(np.float32)
    xt, cs = torch.from_numpy(xn), torch.from_numpy(clean[: Tn * B.P.H].astype(np.float32))[None]
    f0 = B.P.f0_track(xt)[0]
    f0n = f0.cpu().numpy().astype(np.float64)
    b = {"xn": xn, "f0": f0, "f0n": f0n, "amp": B.P.envelope_amp(xn, f0n), "pw": B.P.power_p(xt, Tn),
         "ps": MK.power(cs)[0].double(), "pn": MK.power(xt[None] - cs)[0].double(), "px": MK.power(xt[None])[0].double()}
    if FRONT is not None:
        with torch.no_grad():
            p = torch.sigmoid(FRONT.gate(xt[None].to(DEV))[0, 0].double()).cpu()[:Tn]
            mm = torch.sigmoid(FRONT.mask(xt[None].to(DEV))[0].double()).transpose(0, 1).cpu()[:Tn]
        op = MK.hysteresis(p, *FRONT.hyst, FRONT.thr) if FRONT.hyst else p >= FRONT.thr
        b["rl"] = torch.where(op[:, None], mm, torch.ones_like(mm))
    return b


def feats_of(b: dict, m: torch.Tensor) -> dict:
    f0n = b["f0n"]
    pm = b["pw"] * m.double().pow(2)
    per = B.P.periodicity_from_power(pm, b["f0"]).cpu().numpy()
    fb, _ = PA._np_consts()
    num, den = pm.numpy() @ fb.T, b["pw"].numpy() @ fb.T
    g = np.where(den > 0, np.sqrt(np.divide(num, den, out=np.ones_like(num), where=den > 0)), 1.0)
    env = B.P.envelope_from_amp(b["amp"] * g)
    s, n = T.voiced_mean(env, f0n)
    lf = np.where(f0n > 0, np.log(np.maximum(f0n, 1.0) / 200.0), 0.0).astype(np.float32)
    return {"y": T.to_y(env, per, f0n), "lf": lf, "v": (f0n > 0).astype(np.float32), "f0": f0n.astype(np.float32), "vs": s.astype(np.float32), "vn": n}


def oracle_utt(b: dict, kind: str, par: str) -> dict:
    xn, ps, pn, px = b["xn"], b["ps"], b["pn"], b["px"]
    if kind == "floor":
        r = (ps / (ps + pn + 1e-10)).clamp(min=float(par))
    elif kind == "judge":
        ri, rl = ps / (ps + pn + 1e-10), b["rl"]
        z = torch.zeros_like(ri)
        r = {"ibin": (ps > pn).double(), "lbin": (rl > 0.5).double(), "lji": torch.where(rl > 0.5, ri, z), "ijl": torch.where(ps > pn, rl, z)}[par]
    else:
        if par == "0x":
            nh = pn
        else:
            D = int(round(float(par) * B.SR / B.P.H))
            c = torch.cat([torch.zeros(1, pn.shape[1], dtype=pn.dtype), pn.cumsum(0)])
            hi = (torch.arange(pn.shape[0]) - D).clamp(min=0) + 1
            lo = (hi - AVG).clamp(min=0)
            nh = (c[hi] - c[lo]) / (hi - lo)[:, None]
        e = (px - nh).clamp(min=0.0)
        r = e / (e + nh + 1e-10)
    kept = lambda p: round(float((r * p).sum() / (p.sum() + 1e-20)), 4)
    return feats_of(b, r.sqrt()) | {"wav": xn, "st": {"vk": kept(ps), "nk": kept(pn)}}


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    tags = [x for x in sys.argv[1:] if "=" not in x]
    n, skip = int(opt.get("n", 64)), int(opt.get("skip", 0))
    conds = opt.get("conds", "mus10,mus0").split(",")
    kinds = opt.get("rows", "copy").split(",")
    variants = ([("floor", v) for v in opt.get("floors", "0.1,0.01").split(",")] if "floor" in kinds else []) + ([("delay", v) for v in opt.get("delays", "0x,0,0.1,0.3,1.0").split(",")] if "delay" in kinds else [])
    variants += [("judge", v) for v in opt.get("judges", "ibin,lbin,lji,ijl").split(",")] if "judge" in kinds else []
    if "judge" in kinds:
        global FRONT
        FRONT = MK.load(ROOT / "results" / opt.get("front", "snv_p45_gate_t80") / "last.pt", DEV)
    split = opt.get("set", "devA")
    man = json.loads((ROOT / "data/cv22_ja/dev_manifest.json").read_text())
    dv = json.loads((ROOT / opt.get("ref", f"results/snv_d5/dev_rows_{split}.json")).read_text())["ref"]
    pool = SEP.bg_split()["dev"]["music"]
    load = lambda name: librosa.load(str(T.FEAT / f"cv/{name}.wav"), sr=B.SR, mono=True)[0].astype(np.float64)
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, to16, kana
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16" if DEV == "cuda" else "int8")

    def tr(w: np.ndarray) -> str:
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

    front = MK.load(ROOT / "results" / opt["front"] / "last.pt", DEV) if "copy" in kinds and opt.get("front") else None
    sy = EV.System(tags[0], gen) if variants else None
    tgts = json.loads((EV.A1 / "convert_meta.json").read_text())["targets"][:4]
    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tgts} if sy is not None else {}
    outp = ROOT / opt.get("out", "results/snv_p57/info.json")
    part = outp.with_suffix(".part.json")
    rows = json.loads(part.read_text())["rows"] if part.exists() else []
    done = {r["idx"] for r in rows}
    first = True
    for i, s in enumerate([x for x in man["speakers"] if x["split"] == split][skip:n], start=skip):
        sp = s["speaker"]
        srcs = [f"{sp}_{c['name']}" for c in s["src"][:2] if f"{sp}_{c['name']}" in dv and dv[f"{sp}_{c['name']}"]["text_vs_orig"] <= 0.1]
        if not srcs or i in done:
            continue
        mus = pool[zlib.crc32(sp.encode()) % len(pool)]
        tg = [tgts[i % 4], tgts[(i + 2) % 4]]
        xe = [load(f"{sp}_{c['name']}") for c in s["enroll"]] if sy is not None else []
        for cond in ["clean"] + conds:
            mixp = (lambda x, seed: (x, x)) if cond == "clean" else (lambda x, seed: B.with_music(x, mus, seed, False, float(cond[3:])))
            eb = [base_of(*mixp(x, 7 + j)) for j, x in enumerate(xe)] if sy is not None and cond != "clean" else []
            enr = {vr: [oracle_utt(b, *vr) for b in eb] for vr in variants}
            if eb and first:
                first = False
                chk = MK.features_with_mask(eb[0]["xn"], (eb[0]["ps"] / (eb[0]["ps"] + eb[0]["pn"] + 1e-10)).clamp(min=0.1).sqrt())
                assert float(np.abs(chk["y"] - oracle_utt(eb[0], "floor", "0.1")["y"]).max()) == 0.0, "feats_of が train_snv_mask.features_with_mask と違う"
            for name in srcs:
                xs = load(name)
                cl, nz = mixp(xs, 1 + i)
                ref = dv[name]["orig"]
                base = {"spk": sp, "idx": i, "src": name, "cond": cond}
                if "copy" in kinds:
                    feats = {"none": B.utt_of(nz)}
                    if cond != "clean":
                        if front is not None:
                            feats["front"] = MK.features(front, nz.astype(np.float32), DEV)
                        feats["ideal"] = B.stft_oracle_utt(cl, nz)
                    for kname, u in feats.items():
                        rows.append(base | {"sys": f"copy:{kname}", "tgt": "", "k": 0, "cer": round(cer(ref, tr(copy(u))), 4), "voiced": round(float(u["v"].mean()), 3)} | ({"gate_open": round(float(u["gate_open"]), 3)} if "gate_open" in u else {}))
                if sy is not None and cond != "clean":
                    sb = base_of(cl, nz)
                    for vr in variants:
                        uf = oracle_utt(sb, *vr)
                        w = sy.convert(uf, enr[vr], sy.structure(f"t|{tg[0]}", trefs[tg[0]]), 0)
                        rows.append(base | {"sys": f"{sy.tag}+{'idealf' if vr[0] == 'floor' else vr[0]}{vr[1]}", "tgt": tg[0], "k": 0, "cer": round(cer(ref, tr(w)), 4)} | uf["st"])
        seen = list(dict.fromkeys(r["sys"] for r in rows if r["idx"] == i))
        print("speaker", i, {c: {t: round(float(np.mean([min(r["cer"], 1.0) for r in rows if r["idx"] == i and r["cond"] == c and r["sys"] == t])), 3) for t in seen if any(r["idx"] == i and r["cond"] == c and r["sys"] == t for r in rows)} for c in ["clean"] + conds}, flush=True)
        part.write_text(json.dumps({"rows": rows}, ensure_ascii=False))
    res = {"prereg": "results/snv_p57/prereg.yaml(step_a)", "set": split, "tags": tags, "rows_kind": kinds, "front": opt.get("front"), "variants": [f"{a}{b}" for a, b in variants], "conds": ["clean"] + conds, "n_speakers": len({r["spk"] for r in rows}), "rows": rows}
    outp.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    part.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
