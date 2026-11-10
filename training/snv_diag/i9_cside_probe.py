"""段 I8 / I9 の声を跨ぐ内容の崩れの場所(学習なし): 学習の男声・女声の学習の内容の発話を、C の入力(推論の E の出力 / 学習の軌跡 μ_w[τ] + 真の演技 / その混ぜ)× 構造(自分 / A0F の目標の女声 2 声・k 0)で描き、large-v3 の仮名の CER(写しの書き起こし 対 出力)を並べる。
軌跡の対応 τ = 学習と同じ G(保存の Gs)の出力と型の帯の中の単調な対応・自分の構造 = 同じ声の別の普通の発話 3 本 × 最大 10 s(学習と同じ)・目標の構造 = 試験と同じ(snv_p1_eval)。
軌跡では目標の女声へも自分と同じだけ読め、E の出力でだけ崩れる = 崩れは推論の E の出力の残りを C が目標の構造と合わせて読み違えること / 軌跡でも崩れる = C の描画そのものが目標の構造で内容を落とす。
Ect+act = [E の内容 59・真の演技 5] / traj+Eact = [軌跡・E の演技 5] = 残りが内容の次元と演技の次元のどちらにあるか(k 0 だけ)。
ks= = 源の高さを k 半音ずらして(C の lf と描画の f0・snv_p1_eval.convert と同じ)自分の構造と目標の構造の両方で描く = 構造と高さの組が学習の範囲の外で内容を落とすか。sexes= で源の性別を選ぶ。
    cd training && uv run --no-sync python snv_diag/i9_cside_probe.py [ckpt=results/snv_i8_m13/last.pt] [n=12] [ks=0] [sexes=em,ef] [out=results/snv_i9/cside_probe.json]
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
import train_snv_i as I
import train_snv_p1 as T
import train_snv_p11 as P11

ROOT = I.ROOT
DEV = EV.DEV
PATHS_T = ("E", "traj", "Ect+act", "traj+Eact")


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    ck, n = opt.get("ckpt", "results/snv_i8_m13/last.pt"), int(opt.get("n", 12))
    ks = [int(k) for k in opt.get("ks", "0").split(",")]
    sexes = opt.get("sexes", "em,ef").split(",")
    st = torch.load(ROOT / ck, map_location="cpu", weights_only=False)
    assert st.get("student"), "M13 の保存でない"
    E, G, C = P11.EncS().to(DEV).eval(), P11.EncS().to(DEV).eval(), T.Circ().to(DEV).eval()
    E.load_state_dict(st["Es"]), G.load_state_dict(st["Gs"]), C.load_state_dict(st["C"])
    tpl = st["tpl"].float().to(DEV)
    tr = json.loads(((ROOT / ck).parent / "split.json").read_text())["train"]
    ctr = I.Corpus(tr)
    tlen = torch.from_numpy(ctr.tlen).to(DEV)
    band = float(st.get("band", 0.1)) if st.get("align") == "band" else 0.0
    rng = random.Random(0)
    items = []
    for sex in sexes:
        vs = [v for v in tr if v.startswith(sex)]
        rng.shuffle(vs)
        for v in vs:
            us = [u for u in ctr.u[v]["norm"] if u["key"][0] == "txt" and u["key"] in ctr.cid]
            if us and len(ctr.u[v]["norm"]) >= 4:
                items.append((sex, v, rng.choice(us)))
            if sum(s == sex for s, *_ in items) >= n:
                break
    meta = json.loads((EV.A1 / "convert_meta.json").read_text())
    tgts = meta["targets"][:2]
    tS = {}
    with torch.no_grad():
        for t in tgts:
            ry, rv, rm, _ = EV.struct_inputs([EV.utt(f"a0/{t}_{j}") for j in (0, 1, 2)])
            tS[t] = P11.structures(E, C, ry, rv, rm)[1]
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, kana, to16
    gen = EV.load_gen()
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16" if DEV == "cuda" else "int8")

    def tr_(w: np.ndarray) -> str:
        segs, _ = asr.transcribe(to16(w.astype(np.float64), EV.SR).astype(np.float32), language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(x.text for x in segs))

    @torch.no_grad()
    def render(yo: torch.Tensor, u: dict, k: int = 0) -> np.ndarray:
        v = torch.from_numpy(u["v"])[None].to(DEV)
        lf = torch.from_numpy(u["lf"])[None].to(DEV) + k * T.LN2_12 * v
        cond = torch.cat([yo[:, :25], lf[:, None], v[:, None], yo[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
        f0 = torch.from_numpy(u["f0"] * 2 ** (k / 12))[None].to(DEV)
        return gen(cond, TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:].astype(np.float32)

    rows = []
    for sex, v, u in items:
        rs = [x for x in ctr.u[v]["norm"] if x is not u][:I.REF_N]
        ry = torch.from_numpy(np.concatenate([x["y"][:, :I.REF_LEN] for x in rs], 1))[None].to(DEV)
        rv = torch.from_numpy(np.concatenate([x["v"][:I.REF_LEN] for x in rs]))[None].to(DEV)
        rm = torch.ones(rv.shape, dtype=torch.bool, device=DEV)
        y = torch.from_numpy(u["y"])[None].to(DEV)
        vv = torch.from_numpy(u["v"])[None].to(DEV)
        lf = torch.from_numpy(u["lf"])[None].to(DEV)
        nf = y.shape[-1]
        cid = ctr.cid[u["key"]]
        with torch.no_grad():
            sE, sC = P11.structures(E, C, ry, rv, rm)
            mu = E(P11.xin(y, vv), sE)[0]
            muG = G(P11.xin(y, vv), G.structure(P11.xin(ry, rv), rm))[0]
            m = torch.ones(1, nf, device=DEV)
            m[:, :T.WARM_SELF] = 0
            pm, _ = I.share_path(muG[:, :I.D_CT], m, torch.tensor([cid], device=DEV), tpl, tlen, torch.tensor([0], device=DEV), torch.tensor([nf], device=DEV), False, band=band)
            act = I.acting_feats(y, vv, ry, rv, rm)
            ins = {"E": mu, "traj": torch.cat([pm, act], 1), "Ect+act": torch.cat([mu[:, :I.D_CT], act], 1), "traj+Eact": torch.cat([pm, mu[:, I.D_CT:]], 1)}
            sl = slice(T.WARM_SELF, nf)
            res_ct = float(((mu[0][:I.D_CT, sl] - pm[0][:, sl]) ** 2).sum(0).mean())
        ref = tr_(render(y, u))
        r = {"sex": sex, "voice": v, "text": u["key"][1][:40], "cer_copy_vs_text": cer(kana(u["key"][1]), ref), "res_ct": res_ct}
        for k in ks:
            lfk = lf + k * T.LN2_12 * vv
            sfx = "" if k == 0 else f"_k{k}"
            with torch.no_grad():
                for p in ("E", "traj"):
                    r[f"self_{p}{sfx}"] = cer(ref, tr_(render(C(ins[p], lfk, vv, sC), u, k)))
                for p in (PATHS_T if k == 0 else ("E", "traj")):
                    r[f"F_{p}{sfx}"] = float(np.mean([cer(ref, tr_(render(C(ins[p], lfk, vv, tS[t]), u, k))) for t in tgts]))
        rows.append(r)
        print(len(rows), json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items()}, ensure_ascii=False), flush=True)
    res: dict = {"ckpt": ck, "step": st["step"], "n_per_sex": n, "targets": tgts, "ks": ks}
    keys = [k for k in rows[0] if k not in ("sex", "voice", "text")]
    for sex in sexes:
        R = [r for r in rows if r["sex"] == sex]
        res[sex] = {k: round(float(np.mean([r[k] for r in R])), 4) for k in keys} | {f"{k}_median": round(float(np.median([r[k] for r in R])), 4) for k in keys if k.startswith(("self_", "F_"))}
    res["rows"] = rows
    print(json.dumps({k: res[k] for k in sexes}, indent=1), flush=True)
    (ROOT / opt.get("out", "results/snv_i9/cside_probe.json")).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
