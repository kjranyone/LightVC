"""段 P15(results/snv_p15/prereg.yaml・学習なし): P8 の基準の行の誤差の分析。話者 × 目標の配置と、差し替えの条件ごとの CER・M・誤りの内訳(置換 / 脱落 / 挿入)。
条件:
  base4 = devA の先頭 n 人 × s0・s1 × A0F の先頭 4 声 × k 0・7(半分は results/snv_d5/dev_rows_devA.json の測り直し)/ self = 構造を自分の登録に(k 0)
  tref8・tref14 = 目標の構造の参照を j 0〜7 / j 0〜13 に / enr4・enr2・enr8s・src1 = 操作者の登録を e0〜e3 / e0〜e1 / e0〜e7 + 源 / 源だけ に
  wcopy・whigh = 源と登録を WORLD の分析合成に通す(k 7)/ 高さ 1.4 倍(k 1・出力の高さは wcopy の k 7 とほぼ同じ)
  fam_g0・fam_w = data/snv_d5 の対照の声 g0_c1 / 低い声 w_c2(源 t01・t02・登録 = 残りの 2 発話)
補足 2:
  enr10 = 登録を e0〜e7 + s2・s3 に / jv6・jv12・jv24・jv48 = JVNV の男声 2 人(源 regular_01 の 6 感情)の登録を regular_02 から感情ごとに 1 / 2 / 4 / 8 本に
  org_g0・org_w・org_g0_ch・org_w_ch = 生成した声(対照 / 低い声)× 録音の鎖 v2 の有無(源と登録に同じ部屋 0)。自己(k 0・tgt なし)と他の声の構造(k 7・目標 2 声)
  vspace = 話者の登録の ECAPA(素の埋め込み)から E1 の 600 声への最も近い余弦(rows.json の vspace)
補足 3:
  org_w_ch_wcopy・org_w_ch_whigh = 低い声 + 鎖 の生成した声を WORLD の写し(k 7)/ 高さ 1.4 倍(k 1)に
  jv_wcopy・jv_wlow・jv_ch・jv_wlow_ch = JVNV の男声 2 人(源 regular_01・登録 regular_02 の 6 感情)に WORLD の写し / 高さ 0.55 倍 / 鎖 v2 / 高さ 0.55 倍 + 鎖 v2(k 7)
差し替えの条件は 割り当ての目標 2 声 × k 7。CER = large-v3 の仮名で、源の元の音の書き起こし 対 出力。M = 出力の ECAPA + LDA が A0F の 24 声の中で目標を 1 位に当てるか。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/p15_probe.py [conds=base4,self,...] [n=64] [sys=snv_p8_wst] [out=results/snv_p15/rows.json] [redo=1]
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import snv_d5_chain as CH
import snv_d5_world as W
import snv_p0_front as P
import train_snv_p1 as T

ROOT = Path(__file__).resolve().parents[2]
SR = P.SR
CONDS = ("base4", "self", "tref8", "tref14", "enr4", "enr2", "enr8s", "src1", "wcopy", "whigh", "fam_g0", "fam_w", "enr10", "jv6", "jv12", "jv24", "jv48", "org_g0", "org_w", "org_g0_ch", "org_w_ch", "vspace",
         "org_w_ch_wcopy", "org_w_ch_whigh", "jv_wcopy", "jv_wlow", "jv_ch", "jv_wlow_ch")
FAM = {"fam_g0": ("g0", "c1"), "fam_w": ("w", "c2"), "org_g0": ("g0", "c1"), "org_w": ("w", "c2"), "org_g0_ch": ("g0", "c1"), "org_w_ch": ("w", "c2")}
JV_N = {"jv6": 1, "jv12": 2, "jv24": 4, "jv48": 8}
EMOS = ("anger", "disgust", "fear", "happy", "sad", "surprise")
RATIO = {"wcopy": 1.0, "whigh": 1.4}
ORGT = {"org_w_ch_wcopy": 1.0, "org_w_ch_whigh": 1.4}
JVB = {"jv_wcopy": (1.0, False), "jv_wlow": (0.55, False), "jv_ch": (None, True), "jv_wlow_ch": (0.55, True)}
K_OF = {"whigh": 1, "org_w_ch_whigh": 1}


def ops(ref: str, hyp: str) -> tuple[int, int, int]:
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), np.int32)
    d[:, 0], d[0, :] = np.arange(n + 1), np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, s, dl, ins = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            s += int(ref[i - 1] != hyp[j - 1])
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            dl, i = dl + 1, i - 1
        else:
            ins, j = ins + 1, j - 1
    return s, dl, ins


def feat(x: np.ndarray, keep: bool = False) -> dict:
    x = x[: P.n_frames(len(x)) * P.H].astype(np.float32)
    f = P.analyze(x)
    f0, env, per = f["f0"], f["env"], f["per"]
    s, n = T.voiced_mean(env, f0)
    lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
    u = {"y": T.to_y(env, per, f0), "lf": lf, "v": (f0 > 0).astype(np.float32), "f0": f0.astype(np.float32), "vs": s.astype(np.float32), "vn": n}
    return u | {"wav": x} if keep else u


def world_job(job: tuple) -> tuple:
    import librosa
    import pyworld
    key, path = job
    torch.set_num_threads(1)
    x = librosa.load(path, sr=SR, mono=True)[0].astype(np.float32)
    f0, sp, ap = W.analyze(x, SR)
    return key, {c: feat(pyworld.synthesize(f0 * r, sp, ap, SR, 5.0)[: len(x)].astype(np.float32)) for c, r in RATIO.items()}


def wav_job(job: tuple) -> tuple:
    import librosa
    key, path, voice = job
    torch.set_num_threads(1)
    x = librosa.load(path, sr=SR, mono=True)[0].astype(np.float32)
    if voice is not None:
        x = CH.apply(x, CH.room(voice, 0), np.random.default_rng(CH.seed_of(f"p15|{key}")))
    return key, feat(x)


def proc_job(job: tuple) -> tuple:
    import librosa
    key, path, steps = job
    torch.set_num_threads(1)
    x = librosa.load(path, sr=SR, mono=True)[0].astype(np.float32)
    for kind, arg in steps:
        x = CH.apply(x, CH.room(arg, 0), np.random.default_rng(CH.seed_of(f"p15|{key}"))) if kind == "ch" else W.shift(x, SR, arg)
    return key, feat(x)


def summarize(rows: list[dict]) -> dict:
    out: dict = {}
    for name, R in [("all", rows)] + [(f"k{k}", [r for r in rows if r["k"] == k]) for k in sorted({r["k"] for r in rows})]:
        c = np.array([r["cer"] for r in R])
        d = {"n": len(R), "cer_median": round(float(np.median(c)), 4), "cer_mean": round(float(np.mean(c)), 4), "frac_le_0.2": round(float(np.mean(c <= 0.2)), 3)}
        if all("m" in r for r in R):
            d["M"] = round(float(np.mean([r["m"] for r in R])), 3)
        out[name] = d
    return out


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    n, tag = int(opt.get("n", 64)), opt.get("sys", "snv_p8_wst")
    out = ROOT / opt.get("out", "results/snv_p15/rows.json")
    res = json.loads(out.read_text()) if out.exists() else {"prereg": "results/snv_p15/prereg.yaml", "sys": tag, "n": n, "rows": {}, "summary": {}}
    conds = [c for c in opt.get("conds", ",".join(CONDS)).split(",") if c not in res["rows"] or opt.get("redo") == "1"]
    assert all(c in CONDS for c in conds), conds
    if not conds:
        print(json.dumps(res["summary"], ensure_ascii=False, indent=1))
        return 0
    dv = json.loads((ROOT / "results/snv_d5/dev_rows_devA.json").read_text())
    man = json.loads((ROOT / "data/cv22_ja/dev_manifest.json").read_text())
    ld = lambda sp, c: T.load_utt(T.FEAT / f"cv/{sp}_{c['name']}.npz")
    items = []
    for i, s in enumerate([x for x in man["speakers"] if x["split"] == "devA"][:n]):
        enr = [ld(s["speaker"], c) for c in s["enroll"]]
        for c in s["src"][:2]:
            src = f"{s['speaker']}_{c['name']}"
            items.append({"set": "cv", "spk": s["speaker"], "idx": i, "src": src, "u": ld(s["speaker"], c), "enr": enr, "ref": dv["ref"][src]["orig"], "extra": [ld(s["speaker"], x) for x in s["src"][2:4]],
                          "names": [src] + [f"{s['speaker']}_{e['name']}" for e in s["enroll"]]})
    Wf: dict = {}
    if any(c in RATIO for c in conds):
        jobs = sorted({(k, str(T.FEAT / f"cv/{k}.wav")) for it in items for k in it["names"]})
        with ProcessPoolExecutor(11) as ex:
            for key, d in ex.map(world_job, jobs, chunksize=2):
                Wf[key] = d
        print("WORLD の特徴", len(Wf), flush=True)
    fam: dict = {}
    jvb: dict = {}
    jpath = lambda nm: str(ROOT / f"data/jvnv/jvnv_v1/{nm.split('_')[0]}/{nm.split('_')[1]}/regular/{nm}.wav")
    for c in [c for c in conds if c in JVB]:
        ratio, ch = JVB[c]
        jobs = [(nm, jpath(nm), tuple(([("w", ratio)] if ratio is not None else []) + ([("ch", sp)] if ch else []))) for sp in ("M1", "M2") for e in EMOS for nm in (f"{sp}_{e}_regular_01", f"{sp}_{e}_regular_02")]
        with ProcessPoolExecutor(11) as ex:
            F = dict(ex.map(proc_job, jobs, chunksize=2))
        jvb[c] = [{"set": c, "spk": sp, "idx": i, "src": f"{sp}_{e}_regular_01", "u": F[f"{sp}_{e}_regular_01"], "enr": [F[f"{sp}_{x}_regular_02"] for x in EMOS], "path": jpath(f"{sp}_{e}_regular_01")} for i, sp in enumerate(("M1", "M2")) for e in EMOS]
        print(c, "源", len(jvb[c]), "高さの中央値", round(float(np.median(np.concatenate([it["u"]["f0"][it["u"]["v"] > 0.5] for it in jvb[c]]))), 1), flush=True)
    if any(c in FAM or c in ORGT for c in conds):
        dm = json.loads((ROOT / "data/snv_d5/manifest.json").read_text())
        for c in [c for c in conds if c in ORGT]:
            vs = [v for v in dm["voices"] if (v["type"], v["cap"]) == ("w", "c2")]
            jobs = [(f"{v['voice']}|{t}", str(ROOT / f"data/snv_d5/{v['voice']}/{t}.wav"), (("ch", v["voice"]), ("w", ORGT[c]))) for v in vs for t in ("t00", "t01", "t02")]
            with ProcessPoolExecutor(11) as ex:
                F = dict(ex.map(proc_job, jobs, chunksize=2))
            fam[c] = [{"set": c, "spk": v["voice"], "idx": i, "src": f"{v['voice']}|{t}", "u": F[f"{v['voice']}|{t}"], "enr": [F[f"{v['voice']}|{o}"] for o in ("t00", "t01", "t02") if o != t], "path": str(ROOT / f"data/snv_d5/{v['voice']}/{t}.wav")}
                      for i, v in enumerate(vs) for t in ("t01", "t02")]
            print(c, "声", len(vs), "源", len(fam[c]), "高さの中央値", round(float(np.median(np.concatenate([it["u"]["f0"][it["u"]["v"] > 0.5] for it in fam[c]]))), 1), flush=True)
        for c in [c for c in conds if c in FAM]:
            vs = [v for v in dm["voices"] if (v["type"], v["cap"]) == FAM[c]]
            jobs = [(f"{v['voice']}|{t}", str(ROOT / f"data/snv_d5/{v['voice']}/{t}.wav"), v["voice"] if c.endswith("_ch") else None) for v in vs for t in ("t00", "t01", "t02")]
            with ProcessPoolExecutor(11) as ex:
                F = dict(ex.map(wav_job, jobs, chunksize=2))
            fam[c] = [{"set": c, "spk": v["voice"], "idx": i, "src": f"{v['voice']}|{t}", "u": F[f"{v['voice']}|{t}"], "enr": [F[f"{v['voice']}|{o}"] for o in ("t00", "t01", "t02") if o != t], "path": str(ROOT / f"data/snv_d5/{v['voice']}/{t}.wav")}
                      for i, v in enumerate(vs) for t in ("t01", "t02")]
            print(c, "声", len(vs), "源", len(fam[c]), "高さの中央値", round(float(np.median(np.concatenate([it["u"]["f0"][it["u"]["v"] > 0.5] for it in fam[c]]))), 1), flush=True)

    import snv_p1_eval as EV
    from faster_whisper import WhisperModel
    from snv_a0_measure import kana, load, to16
    from snv_a0b_measure import proj
    from snv_n0_measure import trim
    from speechbrain.inference.speaker import EncoderClassifier
    DEV = EV.DEV
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16" if DEV == "cuda" else "int8")
    w16 = lambda w: to16(w.astype(np.float64), SR).astype(np.float32)

    def tr(y16: np.ndarray) -> str:
        segs, _ = asr.transcribe(y16, language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(x.text for x in segs))

    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})
    Tz = np.load(ROOT / "data/snv_e1_feat/ecapa_lda_T.npz")

    def emb(y16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(y16).float()[None].to(DEV)).reshape(-1)
        return proj(torch.nn.functional.normalize(e, dim=0).cpu().numpy(), (Tz["mu"], Tz["scalings"]))

    all_t = json.loads((EV.A1 / "convert_meta.json").read_text())["targets"]
    tg4 = all_t[:4]
    Et = {v: emb(np.concatenate([to16(*load(EV.A0 / f"wav/{v}_{j}.wav")).astype(np.float32) for j in (0, 1, 2)])) for v in all_t}
    hit = lambda y16, v: int(max(all_t, key=lambda u: float(emb(trim(y16)) @ Et[u])) == v)
    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in range(14)] for v in tg4}
    gen = EV.load_gen()
    s = EV.System(tag, gen)
    import librosa
    rcache: dict = {}

    def ref_of(path: str) -> str:
        if path not in rcache:
            rcache[path] = tr(w16(librosa.load(path, sr=SR, mono=True)[0]))
        return rcache[path]

    for its in list(fam.values()) + list(jvb.values()):
        for it in its:
            it["ref"] = ref_of(it["path"])
    jv: list = []
    if any(c in JV_N for c in conds):
        import i10_jvnv_probe as JP
        for i, sp in enumerate(("M1", "M2")):
            pool = {n: [JP.feats(f"{sp}_{e}_regular_{j:02d}", "jvnv", None) for e in EMOS for j in range(2, 2 + n)] for n in sorted(set(JV_N.values()))}
            for e in EMOS:
                u = JP.feats(f"{sp}_{e}_regular_01", "jvnv", None)
                jv.append({"set": "jv", "spk": sp, "idx": i, "src": f"{sp}_{e}_regular_01", "u": u, "pool": pool, "ref": ref_of(u["wav_path"])})
    if "vspace" in conds:
        def raw(paths: list) -> np.ndarray:
            y = np.concatenate([w16(librosa.load(q, sr=SR, mono=True)[0]) for q in paths])
            with torch.no_grad():
                return torch.nn.functional.normalize(ec.encode_batch(torch.from_numpy(y).float()[None].to(DEV)).reshape(-1), dim=0).cpu().numpy()
        vs = sorted(q.name for q in (ROOT / "data/snv_e1").iterdir() if q.is_dir() and (q / "t02.wav").exists())
        E1 = np.stack([raw([str(ROOT / f"data/snv_e1/{v}/{t}.wav") for t in ("t01", "t02")]) for v in vs])
        male = np.array([v.startswith("em") for v in vs])
        res["vspace"] = {}
        for sp in dict.fromkeys(it["spk"] for it in items):
            c = E1 @ raw([str(T.FEAT / f"cv/{sp}_e{j}.wav") for j in range(8)])
            res["vspace"][sp] = {"nn_e1": round(float(c.max()), 4), "nn_e1_male": round(float(c[male].max()), 4), "top5_male": round(float(np.sort(c[male])[-5:].mean()), 4)}
        print("vspace", len(vs), "声", "最も近い余弦の中央値", round(float(np.median([d["nn_e1"] for d in res["vspace"].values()])), 3), flush=True)

    def trial(cond: str, it: dict, u: dict, enr: list, v: str | None, S: dict, k: int) -> dict:
        y16 = w16(s.convert(u, enr, S, k))
        o = tr(y16)
        sb, dl, ins = ops(it["ref"], o)
        r = {"cond": cond, "set": it["set"], "spk": it["spk"], "src": it["src"], "tgt": v, "k": k, "cer": round((sb + dl + ins) / max(len(it["ref"]), 1), 4), "sub": sb, "del": dl, "ins": ins, "n_ref": len(it["ref"])}
        if v is not None:
            r["m"] = hit(y16, v)
        return r | {"out": o} if it["set"] == "cv" else r

    for cond in conds:
        rows = []
        nref = {"tref8": 8, "tref14": 14}.get(cond, 3)
        for it in [] if cond == "vspace" else jv if cond in JV_N else jvb[cond] if cond in JVB else fam[cond] if cond in FAM or cond in ORGT else items:
            u, enr = it["u"], it["pool"][JV_N[cond]] if cond in JV_N else it["enr"]
            if cond in RATIO:
                u, enr = Wf[it["names"][0]][cond], [Wf[k][cond] for k in it["names"][1:]]
            stat = {"enr4": enr[:4], "enr2": enr[:2], "enr8s": enr + [u], "src1": [u]}.get(cond, enr)
            if cond == "enr10":
                stat = enr + it["extra"]
            if cond == "self" or cond.startswith("org_"):
                rows.append(trial(cond, it, u, enr, None, s.structure(f"self|{cond}|{it['spk']}|{it['src']}", enr), 0))
            if cond == "self":
                continue
            tgs = tg4 if cond == "base4" else (tg4[it["idx"] % 4], tg4[(it["idx"] + 2) % 4])
            for k in (0, 7) if cond == "base4" else (K_OF.get(cond, 7),):
                for v in tgs:
                    rows.append(trial(cond, it, u, stat, v, s.structure(f"t|{v}|{nref}", trefs[v][:nref]), k))
        res["rows"][cond] = rows
        if cond == "vspace":
            out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
            continue
        res["summary"][cond] = summarize([r for r in rows if r["tgt"] is not None] or rows) | ({"self": summarize([r for r in rows if r["tgt"] is None])["all"]} if cond.startswith("org_") else {})
        print(cond, json.dumps(res["summary"][cond], ensure_ascii=False), flush=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        if cond == "base4" and tag in dv["rows"]:
            old = {(r["src"], r["tgt"], r["k"]): r["cer"] for r in dv["rows"][tag] if r["group"] == "devA"}
            df = np.array([abs(r["cer"] - old[(r["src"], r["tgt"], r["k"])]) for r in rows if (r["src"], r["tgt"], r["k"]) in old])
            res["summary"]["base4"]["repro"] = {"n": int(len(df)), "abs_diff_median": round(float(np.median(df)), 4), "abs_diff_p90": round(float(np.percentile(df, 90)), 4), "abs_diff_max": round(float(df.max()), 4)}
            print("測り直しの差", res["summary"]["base4"]["repro"], flush=True)
            out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
            if np.median(df) > 0.02:
                print("STOP: 測り直しが既にある行と合わない", flush=True)
                return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
