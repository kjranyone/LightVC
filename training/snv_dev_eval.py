"""paper_snv §7 rev41 の開発の物差し(学習なし): 本物の男声 → A0F の女声の CER の中央値(k 0・k 7)。試験の 12 区間は使わない。
組 = cv の devA / devB(training/snv_dev_set.py・先頭 n 人 × 源 s0・s1・登録 e0〜e7)と JVNV(M1・M2 の regular_01 の 6 感情・登録 = regular_02 の 6 本)。
目標 = A0F の先頭 4 声から話者の番号で 2 声を回す(構造の参照 = j 0〜2)。CER = large-v3 の仮名で、源の元の音の書き起こし 対 出力(cer)。副 = 写しの描画の書き起こし 対 出力(cer_c・段 I8〜I24 の探りの参照)・出力の UTMOS の平均。
元の音の書き起こしを参照にする理由: 利用者の機器の録音では写しの描画そのものが内容を落とす源がある(写しを参照にすると前段と描画器の落ちが見えない)。
系 = snv_p1_eval.System の名前・orig(元の音そのもの)・copy(写しの描画 = 物差しの床)・anchor(P0c の一律のずらしの錨 W = 包絡の伸縮 α を登録の平均で目標へ当てはめる)。
本人らしさ m = 出力の ECAPA + LDA が A0F の 24 声の中で目標を 1 位に当てるか(snv_p1_eval の test の M と同じ物差し・登録 = 目標の j 0〜2 の元の音)。g =(M − 0.061)/(0.635 − 0.061)(試験の床と天井)。
skip=K = 組の K+1 人目から n 人目まで(段 P45 以降の確認の組 = devA の 65〜158 人目: skip=64 n=158・オーナーの選択 2026-11-13)。話者の番号(目標の割り当て)は組の中の位置のまま。
描画(系の出力・写し・錨)は snv_p1_eval.smooth_env(描画器の界面の包絡の平滑・段 P95 step e・既定・SNV_ESM=0 で切る)を通す。
環境変数 SNV_GUARD=1(段 P95 step c)= 系の出力の波形に出力の番(snv_out_guard・Rust の out_guard と同じ計算・P_in = 操作者の登録の水準・P_out = 目標の参照の水準)を掛けてから測る。
出力 = results/snv_d5/dev_rows_{set}.json(系ごとの行と試行)。既に測った系は測り直さない(redo=1 で測り直す)。mode=m = 既にある行に m だけを足す(書き起こしと UTMOS は測り直さない)。
    cd training && uv run --no-sync python snv_dev_eval.py copy anchor snv_p3_x snv_p8_wst snv_i10_c_pr [set=devA] [n=64] [out=results/snv_d5/dev_rows_devA.json]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

import nvoc as N
import snv_out_guard as OG
import snv_p0c_anchor as AN
import snv_p1_eval as EV
import train_rvoc as TR
import train_snv_p1 as T

ROOT = EV.ROOT
DEV = EV.DEV
KS = tuple(int(v) for v in os.environ.get("SNV_KS", "0,7").split(","))
EMOS = ("anger", "disgust", "fear", "happy", "sad", "surprise")


def cv_items(split: str, n: int, skip: int = 0) -> list[dict]:
    man = json.loads((ROOT / "data/cv22_ja/dev_manifest.json").read_text())
    out = []
    for i, s in enumerate([x for x in man["speakers"] if x["split"] == split][skip:n], start=skip):
        ld = lambda c: T.load_utt(T.FEAT / f"cv/{s['speaker']}_{c['name']}.npz") | {"wav_path": str(T.FEAT / f"cv/{s['speaker']}_{c['name']}.wav")}
        enr = [ld(c) for c in s["enroll"]]
        for c in s["src"][:2]:
            out.append({"group": split, "spk": s["speaker"], "idx": i, "src": f"{s['speaker']}_{c['name']}", "u": ld(c), "enr": enr, "text": c["sentence"]})
    return out


def jvnv_items() -> list[dict]:
    sys.path.insert(0, str(Path(__file__).resolve().parent / "snv_diag"))
    import i10_jvnv_probe as JP
    text = {ln.split("|")[0]: ln.split("|")[2].strip() for ln in (JP.JV / "transcription.csv").read_text().splitlines() if ln.count("|") >= 2}
    out = []
    for i, spk in enumerate(("M1", "M2")):
        enr = [JP.feats(f"{spk}_{e}_regular_02", "jvnv", None) for e in EMOS]
        for e in EMOS:
            name = f"{spk}_{e}_regular_01"
            out.append({"group": "jvnv", "spk": spk, "idx": i, "src": name, "u": JP.feats(name, "jvnv", None), "enr": enr, "text": text[name.split("_", 1)[1]]})
    return out


def summarize(rows: list[dict]) -> dict:
    out: dict = {}
    for g in dict.fromkeys(r["group"] for r in rows):
        R = [r for r in rows if r["group"] == g]
        d: dict = {"trials": len(R), "cer_median": round(float(np.median([r["cer"] for r in R])), 4), "cer_mean": round(float(np.mean([r["cer"] for r in R])), 4),
                   "cer_c_median": round(float(np.median([r["cer_c"] for r in R])), 4), "utmos": round(float(np.mean([r["utmos"] for r in R])), 3)}
        for k in dict.fromkeys(r["k"] for r in R):
            d[f"cer_median_k{k}"] = round(float(np.median([r["cer"] for r in R if r["k"] == k])), 4)
        spk = sorted({r["spk"] for r in R})
        per = {s: [r["cer"] for r in R if r["spk"] == s] for s in spk}
        rng = np.random.default_rng(0)
        bs = [float(np.median(np.concatenate([per[s] for s in rng.choice(spk, len(spk))]))) for _ in range(1000)]
        d["cer_median_ci95_spk"] = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]
        d["frac_le_0.2"] = round(float(np.mean([r["cer"] <= 0.2 for r in R])), 4)
        if "m" in R[0]:
            M = float(np.mean([r["m"] for r in R]))
            d["M"], d["g"] = round(M, 4), round((M - 0.061) / (0.635 - 0.061), 4)
        out[g] = d
    return out


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    tags = [x for x in sys.argv[1:] if "=" not in x]
    split, n, skip = opt.get("set", "devA"), int(opt.get("n", 64)), int(opt.get("skip", 0))
    out = ROOT / opt.get("out", f"results/snv_d5/dev_rows_{split}.json")
    res = json.loads(out.read_text()) if out.exists() else {"set": split, "n": n, "skip": skip, "prereg": opt.get("prereg", "results/snv_d5/prereg.yaml"), "systems": {}, "rows": {}, "ref": {}}
    assert res.get("skip", 0) == skip and res["n"] == n, "既にある出力と話者の範囲が違う"
    only_m = opt.get("mode") == "m"
    tags = [t for t in res["rows"] if t not in ("orig", "copy") and "m" not in res["rows"][t][0]] if only_m else [t for t in tags if t not in res["systems"] or opt.get("redo") == "1"]
    if not tags:
        print(json.dumps(res["systems"], ensure_ascii=False, indent=1))
        return 0
    from snv_a0_measure import cer, kana, load, to16
    from snv_a0b_measure import proj
    from snv_n0_measure import trim
    from speechbrain.inference.speaker import EncoderClassifier
    w16 = lambda w: to16(w.astype(np.float64), EV.SR).astype(np.float32)
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})
    Tz = np.load(ROOT / "data/snv_e1_feat/ecapa_lda_T.npz")

    def emb(y16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(y16).float()[None].to(DEV)).reshape(-1)
        return proj(torch.nn.functional.normalize(e, dim=0).cpu().numpy(), (Tz["mu"], Tz["scalings"]))

    all_t = json.loads((EV.A1 / "convert_meta.json").read_text())["targets"]
    Et = {v: emb(np.concatenate([to16(*load(EV.A0 / f"wav/{v}_{j}.wav")).astype(np.float32) for j in (0, 1, 2)])) for v in all_t}
    hit = lambda y16, v: int(max(all_t, key=lambda u: float(emb(trim(y16)) @ Et[u])) == v)
    if only_m:
        gen = EV.load_gen()
        tg4 = all_t[:4]
        trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tg4}
        mu_t = {v: EV.mu_of(trefs[v])[0].cpu().numpy() for v in tg4}
        items = cv_items(split, n, skip) + jvnv_items()
        for tag in tags:
            s = None if tag == "anchor" else EV.System(tag, gen)
            i = 0
            for it in items:
                u, enr = it["u"], it["enr"]
                for k in KS:
                    for v in (tg4[it["idx"] % 4], tg4[(it["idx"] + 2) % 4]):
                        r = res["rows"][tag][i]
                        assert (r["src"], r["tgt"], r["k"]) == (it["src"], v, k), (r["src"], it["src"])
                        if tag == "anchor":
                            env = np.concatenate([u["y"][:1] * 10 + T.PA.C0_SIL, u["y"][1:25] * 10], 0)
                            ew = AN.warp_env(env, AN.fit_alpha(EV.mu_of(enr)[0].cpu().numpy(), mu_t[v]))
                            y = np.concatenate([(ew[:1] - T.PA.C0_SIL) / 10, ew[1:25] / 10, u["y"][25:29]], 0).astype(np.float32)
                            f0 = u["f0"] * 2 ** (k / 12)
                            vv = (f0 > 0).astype(np.float32)
                            lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
                            cond = torch.from_numpy(np.concatenate([y[:25], lf[None], vv[None], np.clip(y[25:29], 0, 1.5) * vv[None]], 0))[None].to(DEV)
                            cond = torch.cat([EV.smooth_env(gen, cond[:, :25]), cond[:, 25:]], 1)
                            with torch.no_grad():
                                w = gen(cond, TR.excitation(torch.from_numpy(f0.astype(np.float32))[None].to(DEV), cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[gen.delay:].astype(np.float32)
                        else:
                            w = s.convert(u, enr, s.structure(f"t|{v}", trefs[v]), k)
                        r["m"] = hit(w16(w), v)
                        i += 1
            for g, d in res["systems"][tag].items():
                M = float(np.mean([r["m"] for r in res["rows"][tag] if r["group"] == g]))
                d["M"], d["g"] = round(M, 4), round((M - 0.061) / (0.635 - 0.061), 4)
            print(tag, {g: (d["M"], d["g"]) for g, d in res["systems"][tag].items()}, flush=True)
            out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
            del s
            torch.cuda.empty_cache()
        return 0
    from faster_whisper import WhisperModel
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16" if DEV == "cuda" else "int8")
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(DEV).eval()
    mos = lambda y16: float(utmos(torch.from_numpy(y16).float()[None].to(DEV), 16000).item())

    def tr(y16: np.ndarray) -> str:
        segs, _ = asr.transcribe(y16, language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(x.text for x in segs))

    gen = EV.load_gen()

    @torch.no_grad()
    def render(y: np.ndarray, f0: np.ndarray, wav: np.ndarray | None = None) -> np.ndarray:
        v = (f0 > 0).astype(np.float32)
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
        cond = torch.from_numpy(np.concatenate([y[:25], lf[None], v[None], np.clip(y[25:29], 0, 1.5) * v[None]], 0))[None].to(DEV)
        cond = torch.cat([EV.smooth_env(gen, cond[:, :25]), cond[:, 25:]], 1)
        return EV.draw(gen, cond, torch.from_numpy(f0.astype(np.float32))[None].to(DEV), wav)

    wav48 = lambda q: librosa.load(q["wav_path"], sr=EV.SR, mono=True)[0].astype(np.float32)
    lv = {}

    def level(key: tuple, us: list[dict]) -> float:
        if key not in lv:
            lv[key] = OG.speech_level([wav48(q) for q in us])
        return lv[key]

    items = cv_items(split, n, skip) + jvnv_items()
    meta = json.loads((EV.A1 / "convert_meta.json").read_text())
    tgts = meta["targets"][:4]
    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tgts}
    mu_t = {v: EV.mu_of(trefs[v])[0].cpu().numpy() for v in tgts}
    for it in items:
        it["tg"] = [tgts[it["idx"] % 4], tgts[(it["idx"] + 2) % 4]]
        it["ref_o"] = tr(w16(librosa.load(it["u"]["wav_path"], sr=EV.SR, mono=True)[0]))
        it["ref_c"] = tr(w16(render(it["u"]["y"], it["u"]["f0"], wav48(it["u"]) if getattr(gen, "fast", 0) else None)))
        res["ref"][it["src"]] = {"orig": it["ref_o"], "copy": it["ref_c"], "text_vs_orig": round(cer(kana(it["text"]), it["ref_o"]), 4)}
    print("items", len(items), "文 対 元の音の書き起こし の CER の中央値", {g: round(float(np.median([res["ref"][i["src"]]["text_vs_orig"] for i in items if i["group"] == g])), 4) for g in dict.fromkeys(i["group"] for i in items)}, flush=True)
    for tag in tags:
        rows = []
        s = None if tag in ("orig", "copy", "anchor") else EV.System(tag, gen)
        for it in items:
            u, enr = it["u"], it["enr"]
            if tag in ("orig", "copy"):
                plan = [(None, 0)]
            else:
                plan = [(v, k) for k in KS for v in it["tg"]]
            for v, k in plan:
                if tag == "orig":
                    w = librosa.load(u["wav_path"], sr=EV.SR, mono=True)[0]
                elif tag == "copy":
                    w = render(u["y"], u["f0"], wav48(u) if getattr(gen, "fast", 0) else None)
                elif tag == "anchor":
                    env = np.concatenate([u["y"][:1] * 10 + T.PA.C0_SIL, u["y"][1:25] * 10], 0)
                    al = AN.fit_alpha(EV.mu_of(enr)[0].cpu().numpy(), mu_t[v])
                    ew = AN.warp_env(env, al)
                    y = np.concatenate([(ew[:1] - T.PA.C0_SIL) / 10, ew[1:25] / 10, u["y"][25:29]], 0).astype(np.float32)
                    w = render(y, u["f0"] * 2 ** (k / 12), wav48(u) if getattr(gen, "fast", 0) else None)
                else:
                    w = s.convert(u, enr, s.structure(f"t|{v}", trefs[v]), k)
                    if os.environ.get("SNV_GUARD"):
                        w = OG.apply(wav48(u), w, level(("op", it["spk"]), enr), level(("tg", v), trefs[v]))
                y16 = w16(w)
                o = tr(y16)
                rows.append({"group": it["group"], "spk": it["spk"], "src": it["src"], "tgt": v, "k": k, "cer": round(cer(it["ref_o"], o), 4), "cer_c": round(cer(it["ref_c"], o), 4), "utmos": round(mos(y16), 3), "out": o}
                            | ({"m": hit(y16, v)} if v is not None else {}))
        res["systems"][tag] = summarize(rows)
        res["rows"][tag] = rows
        print(tag, json.dumps(res["systems"][tag], ensure_ascii=False), flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        del s
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
