"""段 I8 / I9 の声を跨ぐ内容の崩れの通り道(学習なし・差し替え): 学習の男声と女声が同じ文を話した組で、源の発話を C で女声の目標(A0F 2 声・k 0)へ描くとき、
C の入力の 1 つを差し替えて CER(写しの書き起こし 対 出力)が戻るかを測る。
ずれ r(j) = 型の位置 j に対応づけたフレームの推論の E の内容 59 次元の平均 − μ_w[j](対応 τ = 学習と同じ G の出力と型の帯の中の単調な対応)。
orig = E の出力のまま / rm = 位置ごとのずれを除く(E − r_src(τ(t)))/ swap = 組の相手のずれに差し替える(E − r_src + r_other・相手の通らない位置は除くだけ)/
dlf = C の高さだけ組の相手の音域へ(lf + 相手と源の有声の lf の中央値の差・描画器の f0 は源のまま)。源 = 男声(相手 = 女声)と女声(相手 = 男声)。
    cd training && uv run --no-sync python snv_diag/i9_resid_probe.py [ckpt=results/snv_i8_m13/last.pt] [n=12] [out=results/snv_i9/resid_probe.json]
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


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    ck, n = opt.get("ckpt", "results/snv_i8_m13/last.pt"), int(opt.get("n", 12))
    st = torch.load(ROOT / ck, map_location="cpu", weights_only=False)
    assert st.get("student"), "M13 の保存でない"
    E, G, C = P11.EncS().to(DEV).eval(), P11.EncS().to(DEV).eval(), T.Circ().to(DEV).eval()
    E.load_state_dict(st["Es"]), G.load_state_dict(st["Gs"]), C.load_state_dict(st["C"])
    tpl = st["tpl"].float().to(DEV)
    tr = json.loads(((ROOT / ck).parent / "split.json").read_text())["train"]
    ctr = I.Corpus(tr)
    tlen = torch.from_numpy(ctr.tlen).to(DEV)
    band = float(st.get("band", 0.1)) if st.get("align") == "band" else 0.0
    cmap: dict = {}
    for v in tr:
        if len(ctr.u[v]["norm"]) < 4:
            continue
        for u in ctr.u[v]["norm"]:
            if u["key"][0] == "txt" and u["key"] in ctr.cid:
                cmap.setdefault(u["key"], []).append((v, u))
    rng = random.Random(0)
    keys = sorted(cmap, key=lambda k: k[1])
    rng.shuffle(keys)
    pairs, used = [], set()
    for key in keys:
        ms = [x for x in cmap[key] if x[0].startswith("em") and x[0] not in used]
        fs = [x for x in cmap[key] if x[0].startswith("ef") and x[0] not in used]
        if ms and fs:
            m_, f_ = rng.choice(ms), rng.choice(fs)
            pairs.append((key, m_, f_))
            used |= {m_[0], f_[0]}
        if len(pairs) >= n:
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
    def render(yo: torch.Tensor, u: dict) -> np.ndarray:
        lf = torch.from_numpy(u["lf"])[None].to(DEV)
        v = torch.from_numpy(u["v"])[None].to(DEV)
        cond = torch.cat([yo[:, :25], lf[:, None], v[:, None], yo[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
        f0 = torch.from_numpy(u["f0"])[None].to(DEV)
        return gen(cond, TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:].astype(np.float32)

    @torch.no_grad()
    def prep(v: str, u: dict) -> dict:
        rs = [x for x in ctr.u[v]["norm"] if x is not u][:I.REF_N]
        ry = torch.from_numpy(np.concatenate([x["y"][:, :I.REF_LEN] for x in rs], 1))[None].to(DEV)
        rv = torch.from_numpy(np.concatenate([x["v"][:I.REF_LEN] for x in rs]))[None].to(DEV)
        rm = torch.ones(rv.shape, dtype=torch.bool, device=DEV)
        y = torch.from_numpy(u["y"])[None].to(DEV)
        vv = torch.from_numpy(u["v"])[None].to(DEV)
        lf = torch.from_numpy(u["lf"])[None].to(DEV)
        nf = y.shape[-1]
        cid = ctr.cid[u["key"]]
        sE, sC = P11.structures(E, C, ry, rv, rm)
        mu = E(P11.xin(y, vv), sE)[0]
        muG = G(P11.xin(y, vv), G.structure(P11.xin(ry, rv), rm))[0]
        m = torch.ones(1, nf, device=DEV)
        m[:, :T.WARM_SELF] = 0
        pm, _, tau = I.share_path(muG[:, :I.D_CT], m, torch.tensor([cid], device=DEV), tpl, tlen, torch.tensor([0], device=DEV), torch.tensor([nf], device=DEV), False, band=band, ret_tau=True)
        L = int(tlen[cid])
        ok = (m[0] > 0) & (tau[0] >= 0)
        d = (mu[0, :I.D_CT] - pm[0])[:, ok].T
        j = tau[0][ok]
        cnt = torch.zeros(L, device=DEV).index_add_(0, j, torch.ones(len(j), device=DEV))
        r = torch.zeros(L, I.D_CT, device=DEV).index_add_(0, j, d) / cnt.clamp(min=1.0)[:, None]
        own = r[tau[0].clamp(min=0)] * ok[:, None]
        pos_share = float((own[ok] ** 2).sum() / (d ** 2).sum().clamp(min=1e-9))
        lfv = lf[0][vv[0] > 0]
        return {"u": u, "v": v, "mu": mu, "sC": sC, "lf": lf, "vv": vv, "tau": tau[0], "ok": ok, "r": r, "cnt": cnt, "own": own, "pos_share": pos_share, "med": float(lfv.median()) if len(lfv) else 0.0}

    def swapped(a: dict, b: dict) -> torch.Tensor:
        t0 = a["tau"].clamp(min=0)
        rb = b["r"][t0] * ((b["cnt"][t0] > 0) & a["ok"])[:, None]
        x = a["mu"].clone()
        x[0, :I.D_CT] += (rb - a["own"]).T
        return x

    def removed(a: dict) -> torch.Tensor:
        x = a["mu"].clone()
        x[0, :I.D_CT] -= a["own"].T
        return x

    rows = []
    for key, (vm, um), (vf, uf) in pairs:
        A, B = prep(vm, um), prep(vf, uf)
        for src, oth, sex in ((A, B, "em"), (B, A, "ef")):
            u = src["u"]
            ref = tr_(render(torch.from_numpy(u["y"])[None].to(DEV), u))
            dl = (oth["med"] - src["med"]) * src["vv"]
            ins = {"orig": src["mu"], "rm": removed(src), "swap": swapped(src, oth)}
            r = {"sex": sex, "voice": src["v"], "pair": oth["v"], "text": key[1][:40], "cer_copy_vs_text": cer(kana(key[1]), ref), "pos_share": src["pos_share"],
                 "r_norm": float((src["own"][src["ok"]] ** 2).sum(1).mean()), "dlf_semitones": (oth["med"] - src["med"]) / T.LN2_12}
            with torch.no_grad():
                r["self_orig"] = cer(ref, tr_(render(C(src["mu"], src["lf"], src["vv"], src["sC"]), u)))
                for k, x in ins.items():
                    r[f"F_{k}"] = float(np.mean([cer(ref, tr_(render(C(x, src["lf"], src["vv"], tS[t]), u))) for t in tgts]))
                for k in ("orig", "rm"):
                    r[f"F_{k}_dlf"] = float(np.mean([cer(ref, tr_(render(C(ins[k], src["lf"] + dl, src["vv"], tS[t]), u))) for t in tgts]))
            rows.append(r)
            print(len(rows), json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items()}, ensure_ascii=False), flush=True)
    res: dict = {"ckpt": ck, "step": st["step"], "pairs": len(pairs), "targets": tgts}
    num = [k for k in rows[0] if k not in ("sex", "voice", "pair", "text")]
    for sex in ("em", "ef"):
        R = [r for r in rows if r["sex"] == sex]
        res[sex] = {k: round(float(np.mean([r[k] for r in R])), 4) for k in num} | {f"{k}_median": round(float(np.median([r[k] for r in R])), 4) for k in num if k.startswith(("self_", "F_"))}
    res["rows"] = rows
    print(json.dumps({k: res[k] for k in ("em", "ef")}, indent=1), flush=True)
    (ROOT / opt.get("out", "results/snv_i9/resid_probe.json")).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
