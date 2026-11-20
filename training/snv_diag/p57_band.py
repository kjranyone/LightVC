"""段 P57 step a の補足(results/snv_p57/prereg.yaml の step_a_band・学習なし・評価の入力だけ): 学んだ重みの誤りが、周波数のどこに在り、どこの誤りが内容を落とすか。
重みの網の入力(train_snv_sep.Front)は、1.5 kHz より下が格子そのまま(64 本)、上は mel の帯 160 本。出力は格子の 1025 本。= 1.5 kHz より上では、当てる先(格子ごとの比)より入力が粗い。
物差しは p37_cv_music.py と同じ(devA の有効な源・同じ音楽・同じ種・正しい読み込み)。
mode=profile = 包絡の mel の帯ごとに、学んだ重み(研究の形の実効の重み = 門が開いたフレームは重み・閉じたフレームは 1)と理想の比の重みの、帯の利得の対数の差(絶対値と符号つき)を、声の在るフレーム(きれいな録音の前段 v3 で有声)と無いフレームに分けて出す。
mode=split   = 重みを周波数 fc で継いだ行: LL = 学んだ重み(研究の形と同じ)・IL = fc より下が理想で上が学んだ重み・LI = 下が学んだ重みで上が理想・II = 理想。高さと有声は 4 行とも研究の形の濾した高さ(学んだ重みで決めたもの)= 違いは包絡と周期性の重みだけ。
               先頭の系(前段なし)で変換する(目標 1 声・k 0)。行に声の残り vk・足された音の残り nk を書く。
出力は数値だけ(書き起こしは保存しない)。Common Voice の録音は評価の入力だけに使う(学習・目標にしない・話者の特定を試みない)。理想の重みは評価の中の計算で、学習の目標にしない。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/p57_band.py mode=profile front=snv_p45_gate_t80 n=64 out=results/snv_p57/band_profile.json
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/p57_band.py snv_p30_c mode=split front=snv_p45_gate_t80 fc=1500 n=32 out=results/snv_p57/band_split.json
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
import pae as PA
import snv_p1_eval as EV
import train_snv_mask as MK
import train_snv_p1 as T
import train_snv_sep as SEP

ROOT = EV.ROOT
DEV = EV.DEV
P = B.P


def masks_of(G, clean: np.ndarray, noisy: np.ndarray) -> dict:
    Tn = P.n_frames(len(noisy))
    xn = noisy[: Tn * P.H].astype(np.float32)
    xt, cs = torch.from_numpy(xn), torch.from_numpy(clean[: Tn * P.H].astype(np.float32))[None]
    ps, pn, px = MK.power(cs)[0].double(), MK.power(xt[None] - cs)[0].double(), MK.power(xt[None])[0].double()
    with torch.no_grad():
        p = torch.sigmoid(G.gate(xt[None].to(DEV))[0, 0].double()).cpu()[:Tn]
        mm = torch.sigmoid(G.mask(xt[None].to(DEV))[0].double()).transpose(0, 1).cpu()[:Tn]
    op = MK.hysteresis(p, *G.hyst, G.thr) if G.hyst else p >= G.thr
    return {"xn": xn, "xt": xt, "ps": ps, "pn": pn, "px": px, "rl": torch.where(op[:, None], mm, torch.ones_like(mm)), "ri": ps / (ps + pn + 1e-10), "open": float(op.double().mean())}


def base_of(G, clean: np.ndarray, noisy: np.ndarray) -> dict:
    b = masks_of(G, clean, noisy)
    f0 = P.f0_track_myin(b["xt"], b["rl"].sqrt())[0] if G.dec == "myin" else P.f0_track(b["xt"])[0]
    f0n = f0.cpu().numpy().astype(np.float64)
    return b | {"f0": f0, "f0n": f0n, "amp": P.envelope_amp(b["xn"], f0n), "pw": P.power_p(b["xt"], len(f0n))}


def feats_of(b: dict, r: torch.Tensor) -> dict:
    f0n = b["f0n"]
    pm = b["pw"] * r.double()
    per = P.periodicity_from_power(pm, b["f0"]).cpu().numpy()
    fb, _ = PA._np_consts()
    num, den = pm.numpy() @ fb.T, b["pw"].numpy() @ fb.T
    g = np.where(den > 0, np.sqrt(np.divide(num, den, out=np.ones_like(num), where=den > 0)), 1.0)
    env = P.envelope_from_amp(b["amp"] * g)
    s, n = T.voiced_mean(env, f0n)
    lf = np.where(f0n > 0, np.log(np.maximum(f0n, 1.0) / 200.0), 0.0).astype(np.float32)
    kept = lambda p: round(float((r * p).sum() / (p.sum() + 1e-20)), 4)
    return {"y": T.to_y(env, per, f0n), "lf": lf, "v": (f0n > 0).astype(np.float32), "f0": f0n.astype(np.float32), "vs": s.astype(np.float32), "vn": n, "wav": b["xn"], "st": {"vk": kept(b["ps"]), "nk": kept(b["pn"])}}


def variant(b: dict, kind: str, kc: int) -> torch.Tensor:
    lo = torch.arange(b["rl"].shape[1]) < kc
    a, c = (b["ri"] if kind[0] == "I" else b["rl"]), (b["ri"] if kind[1] == "I" else b["rl"])
    return torch.where(lo[None], a, c)


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    tags = [x for x in sys.argv[1:] if "=" not in x]
    n, skip, mode = int(opt.get("n", 32)), int(opt.get("skip", 0)), opt.get("mode", "profile")
    conds = opt.get("conds", "mus10,mus0").split(",")
    man = json.loads((ROOT / "data/cv22_ja/dev_manifest.json").read_text())
    dv = json.loads((ROOT / "results/snv_d5/dev_rows_devA.json").read_text())["ref"]
    pool = SEP.bg_split()["dev"]["music"]
    load = lambda name: librosa.load(str(T.FEAT / f"cv/{name}.wav"), sr=B.SR, mono=True)[0].astype(np.float64)
    G = MK.load(ROOT / "results" / opt.get("front", "snv_p45_gate_t80") / "last.pt", DEV)
    fb = torch.from_numpy(PA._np_consts()[0]).double()
    cf = ((fb * torch.arange(fb.shape[1])[None]).sum(1) / fb.sum(1) * B.SR / 2048).numpy()
    spk = [x for x in man["speakers"] if x["split"] == "devA"][skip:n]
    if mode == "profile":
        acc = {c: {k: np.zeros(fb.shape[0]) for k in ("abs_v", "sgn_v", "n_v", "abs_u", "sgn_u", "n_u", "gi_v", "gl_v", "gi_u", "gl_u")} for c in conds}
        for i, s in enumerate(spk, start=skip):
            sp = s["speaker"]
            srcs = [f"{sp}_{c['name']}" for c in s["src"][:2] if f"{sp}_{c['name']}" in dv and dv[f"{sp}_{c['name']}"]["text_vs_orig"] <= 0.1]
            mus = pool[zlib.crc32(sp.encode()) % len(pool)]
            for cond in conds:
                for name in srcs:
                    cl, nz = B.with_music(load(name), mus, 1 + i, False, float(cond[3:]))
                    b = masks_of(G, cl, nz)
                    den = b["px"] @ fb.T + 1e-12
                    gl, gi = (b["rl"] * b["px"]) @ fb.T / den, (b["ri"] * b["px"]) @ fb.T / den
                    d = (torch.log(gl + MK.BAND_EPS) - torch.log(gi + MK.BAND_EPS)).numpy()
                    v = (P.f0_track(torch.from_numpy(cl[: len(b["xn"])].astype(np.float32)))[0] > 0).cpu().numpy()[: d.shape[0]]
                    a = acc[cond]
                    for key, sel in (("v", v), ("u", ~v)):
                        a[f"abs_{key}"] += np.abs(d[sel]).sum(0)
                        a[f"sgn_{key}"] += d[sel].sum(0)
                        a[f"n_{key}"] += sel.sum()
                        a[f"gi_{key}"] += torch.log(gi + MK.BAND_EPS).numpy()[sel].sum(0)
                        a[f"gl_{key}"] += torch.log(gl + MK.BAND_EPS).numpy()[sel].sum(0)
            print("speaker", i, flush=True)
        res = {"prereg": "results/snv_p57/prereg.yaml(step_a_band)", "front": opt.get("front", "snv_p45_gate_t80"), "band_center_hz": np.round(cf, 1).tolist(), "n_speakers": len(spk), "conds": {}}
        for c, a in acc.items():
            res["conds"][c] = {f"{q}_{key}": np.round(a[f"{q}_{key}"] / np.maximum(a[f"n_{key}"], 1), 4).tolist() for key in ("v", "u") for q in ("abs", "sgn", "gi", "gl")} | {"frames_v": float(a["n_v"][0]), "frames_u": float(a["n_u"][0])}
        (ROOT / opt.get("out", "results/snv_p57/band_profile.json")).write_text(json.dumps(res, indent=1))
        return 0
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, to16, kana
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16" if DEV == "cuda" else "int8")

    def tr(w: np.ndarray) -> str:
        segs, _ = asr.transcribe(to16(w.astype(np.float64), EV.SR).astype(np.float32), language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(x.text for x in segs))

    gen = EV.load_gen()
    sy = EV.System(tags[0], gen)
    kc = int(round(float(opt.get("fc", 1500)) / (B.SR / 2048)))
    kinds = opt.get("kinds", "LL,IL,LI,II").split(",")
    tgts = json.loads((EV.A1 / "convert_meta.json").read_text())["targets"][:4]
    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tgts}
    rows = []
    for i, s in enumerate(spk, start=skip):
        sp = s["speaker"]
        srcs = [f"{sp}_{c['name']}" for c in s["src"][:2] if f"{sp}_{c['name']}" in dv and dv[f"{sp}_{c['name']}"]["text_vs_orig"] <= 0.1]
        if not srcs:
            continue
        mus = pool[zlib.crc32(sp.encode()) % len(pool)]
        tg = tgts[i % 4]
        xe = [load(f"{sp}_{c['name']}") for c in s["enroll"]]
        for cond in conds:
            mixp = lambda x, seed: B.with_music(x, mus, seed, False, float(cond[3:]))
            eb = [base_of(G, *mixp(x, 7 + j)) for j, x in enumerate(xe)]
            if not rows:
                chk = MK.features(G, eb[0]["xn"], DEV)
                assert float(np.abs(chk["y"] - feats_of(eb[0], variant(eb[0], "LL", kc))["y"]).max()) < 1e-5, "LL が研究の形の特徴と違う"
            enr = {kd: [feats_of(b, variant(b, kd, kc)) for b in eb] for kd in kinds}
            for name in srcs:
                sb = base_of(G, *mixp(load(name), 1 + i))
                for kd in kinds:
                    uf = feats_of(sb, variant(sb, kd, kc))
                    w = sy.convert(uf, enr[kd], sy.structure(f"t|{tg}", trefs[tg]), 0)
                    rows.append({"spk": sp, "idx": i, "src": name, "cond": cond, "sys": f"{sy.tag}+split{kc}:{kd}", "tgt": tg, "k": 0, "cer": round(cer(dv[name]["orig"], tr(w)), 4), "gate_open": round(sb["open"], 3)} | uf["st"])
        print("speaker", i, {c: {kd: round(float(np.mean([min(r["cer"], 1.0) for r in rows if r["idx"] == i and r["cond"] == c and r["sys"].endswith(kd)])), 3) for kd in kinds} for c in conds}, flush=True)
    res = {"prereg": "results/snv_p57/prereg.yaml(step_a_band)", "front": opt.get("front", "snv_p45_gate_t80"), "fc_hz": kc * B.SR / 2048, "tags": tags, "conds": conds, "n_speakers": len({r["spk"] for r in rows}), "rows": rows}
    (ROOT / opt.get("out", "results/snv_p57/band_split.json")).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
