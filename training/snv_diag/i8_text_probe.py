"""段 I8(snv_i8_m13・paper_snv rev19 §5 M13)の試験の CER 1.18 の切り分け(学習なし): 試験と同じ変換の経路(snv_p1_eval.System)で、文(学習の束に出た文 / 学習の声のどれも話さない文)× 音(Irodori / 本物の男声)× 声(自己の reamp / A0F の目標の女声へ)を並べ、large-v3 の仮名の CER(写しの描画の書き起こし 対 出力)を測る。
条件: seen_self = 検証の声の学習の内容の文・k 0 / unseen_self = 検証の声の学習に無い文・k 0 / real_self = 本物の男声 12 区間(構造と登録 = 自分の先頭 30 s)・k 0 /
real_F_k0・real_F_k7 = 本物の男声 → A0F の目標 2 声(試験と同じ構造と登録)/ unseen_mF_k0・unseen_fF_k0 = 検証の男声・女声の学習に無い文 → A0F の目標 2 声 /
real_M_k0 = 本物の男声 → 検証の Irodori の男声 2 声(構造 = その声の E1 の発話 3 本)。
明示したときだけ: trainv_mF_k0 = 学習の男声(推論の E が学習で見た声)の e1 の発話 → A0F の目標 2 声 / trainv_mself = 同じ発話の自己の reamp / val_mself = unseen_mF_k0 と同じ発話の自己の reamp。自己の reamp の構造と登録 = 同じ声の別の発話 3 本。conds= で条件を選ぶ(既定 = 全部)。warp=α = 操作者の源と登録の包絡を一律の伸縮 α で(snv_p1_eval の warp と同じ・目標の参照は元のまま)。
realdir=<data/snv_p_feat の下の名前>(例 real_codec)= 本物の男声の源と登録だけをその場所の特徴(Irodori の生成の後段に通した音・snv_diag/i15_codec_pass.py)に替える。CER の参照は元の音の写しのまま。
学習の文 = split.json の学習の声の 2 声以上が話す文(train_snv_i.Corpus の共有の内容・束はこれだけから作る)。
    cd training && uv run --no-sync python snv_diag/i8_text_probe.py [snv_p3_x snv_i8_m13] [n=12] [ckpt_dir=results/snv_i8_m13] [out=results/snv_i8/text_probe.json]
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nvoc as N
import snv_p1_eval as EV
import train_rvoc as TR
import train_snv_p1 as T

ROOT = EV.ROOT
DEV = EV.DEV


def texts_by_voice() -> dict:
    e1 = json.loads((ROOT / "data/snv_e1/manifest.json").read_text())
    dm = json.loads((ROOT / "data/snv_d/manifest.json").read_text())
    out: dict = {}
    for v in e1["voices"]:
        for f in v["files"]:
            out.setdefault(v["voice"], []).append((f"e1/{v['voice']}/{f['t']}", f["text"]))
    for v in dm["voices"]:
        for f in v["files"]:
            rel = f"d/{v['voice']}/{f['s']}"
            if (T.FEAT / f"{rel}.npz").exists():
                out.setdefault(v["voice"], []).append((rel, f["text"]))
    return out


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    tags = [x for x in sys.argv[1:] if "=" not in x] or ["snv_p3_x", "snv_i8_m13"]
    n = int(opt.get("n", 12))
    split = json.loads((ROOT / opt.get("ckpt_dir", "results/snv_i8_m13") / "split.json").read_text())
    tv = texts_by_voice()
    spk_of: dict = {}
    for v in split["train"]:
        for _, t in tv.get(v, []):
            spk_of.setdefault(t, set()).add(v)
    shared = {t for t, vs in spk_of.items() if len(vs) >= 2}
    rng = random.Random(0)
    val = split["val"][:]
    rng.shuffle(val)
    items: dict = {"seen_self": [], "unseen_self": [], "unseen_mF_k0": [], "unseen_fF_k0": []}
    for v in val:
        us = tv.get(v, [])
        seen = [u for u in us if u[1] in shared]
        unseen = [u for u in us if u[1] not in spk_of]
        if len(us) < 5:
            continue
        if seen and len(items["seen_self"]) < n:
            items["seen_self"].append((v, rng.choice(seen)))
        if unseen and len(items["unseen_self"]) < n:
            items["unseen_self"].append((v, rng.choice(unseen)))
    males = [v for v in split["val"] if v.startswith("em")]
    pools = {v: [u for u in tv.get(v, []) if u[1] not in spk_of] for v in males}
    for v in males:
        rng.shuffle(pools[v])
    for i in range(n):
        v = males[i % len(males)]
        if pools[v]:
            items["unseen_mF_k0"].append((v, pools[v].pop()))
    used_self = {v for v, _ in items["unseen_self"]}
    for v in val:
        if v.startswith("ef") and v not in used_self and len(items["unseen_fF_k0"]) < n:
            unseen = [u for u in tv.get(v, []) if u[1] not in spk_of]
            if unseen:
                items["unseen_fF_k0"].append((v, rng.choice(unseen)))
    conds = set(opt["conds"].split(",")) if "conds" in opt else None
    if conds is not None and conds & {"trainv_mF_k0", "trainv_mself", "val_mself"}:
        rng2 = random.Random(1)
        trm = [v for v in split["train"] if v.startswith("em") and sum(r.startswith("e1/") for r, _ in tv.get(v, [])) >= 4]
        rng2.shuffle(trm)
        items["trainv_mF_k0"] = [(v, rng2.choice([u for u in tv[v] if u[0].startswith("e1/")])) for v in trm[:n]]
        items["trainv_mself"] = list(items["trainv_mF_k0"])
        items["val_mself"] = list(items["unseen_mF_k0"])
    items = {c: its for c, its in items.items() if conds is None or c in conds}
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, kana, load, to16
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16" if DEV == "cuda" else "int8")

    def tr16(w16: np.ndarray) -> str:
        segs, _ = asr.transcribe(w16, language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(x.text for x in segs))

    tr = lambda w: tr16(to16(w.astype(np.float64), EV.SR).astype(np.float32))

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
    tgts, srcs = meta["targets"][:2], meta["sources"]
    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tgts}
    mtgts = males[:2]
    trefs |= {v: [EV.utt(r) for r, _ in tv[v] if r.startswith("e1/")][:3] for v in mtgts}
    enroll = {sp: [EV.utt(f"real/{sp}_enroll")] for sp in ("uirou", "namikawa")}
    wp = (lambda u: EV.warp_utt(u, float(opt["warp"]))) if "warp" in opt else (lambda u: u)
    trials: list = []
    used: dict = {}
    for its in items.values():
        for v, (rel, _) in its:
            used.setdefault(v, set()).add(rel)
    stats = {v: [wp(EV.utt(r)) for r, _ in tv[v] if r.startswith("e1/") and r not in rels][:3] for v, rels in used.items()}
    for cond, its in items.items():
        for v, (rel, text) in its:
            trials.append((cond, v, rel, wp(EV.utt(rel)), stats[v], text, 0))
    rd = opt.get("realdir", "real")
    real = {s_: wp(EV.utt(f"{rd}/{s_}")) for s_ in srcs}
    real_ref = {s_: EV.utt(f"real/{s_}") for s_ in srcs}
    enroll = {sp: [wp(EV.utt(f"{rd}/{sp}_enroll"))] for sp in enroll}
    for cond, k in (("real_self", 0), ("real_F_k0", 0), ("real_F_k7", 7), ("real_M_k0", 0)):
        if conds is None or cond in conds:
            for s_ in srcs:
                trials.append((cond, s_, f"real/{s_}", real[s_], enroll[s_.rsplit("_", 1)[0]], None, k))
    systems = [EV.System(t, gen) for t in tags]
    rows: list = []
    ref_cache: dict = {}
    for cond, who, rel, u, stat, text, k in trials:
        if rel not in ref_cache:
            ref = tr(copy(real_ref[who] if rel.startswith("real/") else u))
            orig = kana(text) if text is not None else tr16(to16(*load(T.FEAT / f"{rel}.wav")).astype(np.float32))
            ref_cache[rel] = (ref, cer(orig, ref))
        ref, c_copy = ref_cache[rel]
        tg = mtgts if cond == "real_M_k0" else tgts if cond in ("real_F_k0", "real_F_k7", "unseen_mF_k0", "unseen_fF_k0", "trainv_mF_k0") else [None]
        for s in systems:
            for tgt in tg:
                S = s.structure(f"self|{who}", stat) if tgt is None else s.structure(f"t|{tgt}", trefs[tgt])
                out = tr(s.convert(u, stat, S, k))
                r = {"cond": cond, "sys": s.tag, "src": who, "rel": rel, "tgt": tgt, "cer": cer(ref, out), "cer_copy": c_copy, "ref": ref[:60], "out": out[:60], "len_ratio": round(len(out) / max(1, len(ref)), 3)}
                rows.append(r)
                print(json.dumps(r, ensure_ascii=False), flush=True)
    res: dict = {"tags": tags, "n": n, "warp": opt.get("warp"), "shared_train_texts": len(shared), "train_texts": len(spk_of), "conditions": {}}
    for cond in dict.fromkeys(r["cond"] for r in rows):
        res["conditions"][cond] = {}
        for s in tags:
            R = [r for r in rows if r["cond"] == cond and r["sys"] == s]
            res["conditions"][cond][s] = {"cer": round(float(np.mean([r["cer"] for r in R])), 4), "cer_median": round(float(np.median([r["cer"] for r in R])), 4),
                                          "len_ratio": round(float(np.mean([r["len_ratio"] for r in R])), 3), "cer_copy": round(float(np.mean([r["cer_copy"] for r in R])), 4), "trials": len(R)}
    res["rows"] = rows
    print(json.dumps(res["conditions"], ensure_ascii=False, indent=1), flush=True)
    (ROOT / opt.get("out", "results/snv_i8/text_probe.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
