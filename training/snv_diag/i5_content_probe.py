"""段 I5(snv_i5_m8・paper_snv rev11 §5 M8)の内容の崩れ(test の CER 1.11・D_CER 0.77)の切り分け(学習なし): Irodori の声の発話を自己の reamp で、C の入力を替えて描き、large-v3 の仮名の CER(写しの書き起こし 対 出力)と L1 を並べる。
経路: copy = y をそのまま描画器へ / E = C(E の出力 64)/ oracle = C([学習の軌跡 μ_w[τ](τ = E の出力と型の単調な対応・学習と同じ = 重みの align が band なら帯の中)・真の演技 5])/
E_ct+act = C([E の内容の 59・真の演技])/ ct+E_act = C([学習の軌跡・E の演技 5])。構造 = 同じ声の別の普通の発話 3 本 × 最大 10 s(学習と同じ)・k 0。
oracle が崩れる = 型(共有の軌跡)そのものが内容を運ばない / oracle は読めて E が崩れる = E が軌跡を当てられない。
E の当ての誤差は内容の 59 と演技の 5 に分け、軌跡の分散(発話の中)と比べる。内容の識別 = E の 4 フレームの平均と自分の型の単調な対応の尤度が、無作為な他の型 20 本より高い率(長さで割る)。
発話 = 学習の声の共有の内容の文 n 本(声ごとに 1 本)と、検証の声で学習の内容と同じ文 n 本。
保存に e_lift があれば(paper_snv §5 M19)推論の E の入力の包絡の c_K〜c24 を保存の lift_mean に置き換える(C の入力は元のまま)。
    cd training && uv run --no-sync python snv_diag/i5_content_probe.py [ckpt=results/snv_i5_m8/last.pt] [n=20] [out=results/snv_i5/content_probe.json]
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
    ck, n = opt.get("ckpt", "results/snv_i5_m8/last.pt"), int(opt.get("n", 20))
    st = torch.load(ROOT / ck, map_location="cpu", weights_only=False)
    E, C = P11.EncS().to(DEV).eval(), T.Circ().to(DEV).eval()
    E.load_state_dict(st["Es"]), C.load_state_dict(st["C"])
    lk, lm = int(st.get("e_lift") or 0), torch.tensor(st["lift_mean"]).to(DEV) if st.get("e_lift") else None
    ex = lambda y_, v_: P11.xin(P11.lift(y_, lk, lm), v_)
    tpl = st["tpl"].float().to(DEV)
    names = sorted(json.loads((ROOT / "data/snv_e1/manifest.json").read_text())["voices"], key=lambda v: v["voice"])
    tr, va = T.split([v["voice"] for v in names])
    sj = (ROOT / ck).parent / "split.json"
    tr = json.loads(sj.read_text())["train"] if sj.exists() else tr
    ctr, cva = I.Corpus(tr), I.Corpus(va)
    tlen = torch.from_numpy(ctr.tlen).to(DEV)
    rng = random.Random(0)
    items = []
    for tag, c, pool in (("train", ctr, tr), ("val", cva, va)):
        vs = pool[:]
        rng.shuffle(vs)
        for v in vs:
            us = [u for u in c.u[v]["norm"] if u["key"][0] == "txt" and u["key"] in ctr.cid]
            if us:
                items.append((tag, v, c, rng.choice(us)))
            if sum(t == tag for t, *_ in items) >= n:
                break
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

    def mas_score(mp: torch.Tensor, cid: int) -> float:
        L = int(tlen[cid])
        tp = tpl[cid, :L]
        lp = -0.5 * ((mp.T[:, None, :] - tp[None]) ** 2).sum(-1)
        path = I.mas_free(lp[None], torch.tensor([mp.shape[1]], device=DEV), torch.tensor([L], device=DEV))[0]
        return float(lp[torch.arange(mp.shape[1], device=DEV), path].mean())

    rows = []
    paths = ("copy", "E", "oracle", "E_ct+act", "ct+E_act")
    for tag, v, c, u in items:
        rs = [x for x in c.u[v]["norm"] if x is not u][:I.REF_N]
        ry = torch.from_numpy(np.concatenate([x["y"][:, :I.REF_LEN] for x in rs], 1))[None].to(DEV)
        rv = torch.from_numpy(np.concatenate([x["v"][:I.REF_LEN] for x in rs]))[None].to(DEV)
        rm = torch.ones(rv.shape, dtype=torch.bool, device=DEV)
        y = torch.from_numpy(u["y"])[None].to(DEV)
        vv = torch.from_numpy(u["v"])[None].to(DEV)
        lf = torch.from_numpy(u["lf"])[None].to(DEV)
        nf = y.shape[-1]
        cid = ctr.cid[u["key"]]
        with torch.no_grad():
            rx = ex(ry, rv)
            sE = E.structure(rx, rm)
            sC = C.structure(ry, E(rx, sE)[0], rm)
            mu = E(ex(y, vv), sE)[0]
            m = torch.ones(1, nf, device=DEV)
            m[:, :T.WARM_SELF] = 0
            pm, _ = I.share_path(mu[:, :I.D_CT], m, torch.tensor([cid], device=DEV), tpl, tlen, torch.tensor([0], device=DEV), torch.tensor([nf], device=DEV), False,
                                 band=float(st.get("band", 0.1)) if st.get("align") == "band" else 0.0)
            act = I.acting_feats(y, vv, ry, rv, rm)
            ins = {"E": mu, "oracle": torch.cat([pm, act], 1), "E_ct+act": torch.cat([mu[:, :I.D_CT], act], 1), "ct+E_act": torch.cat([pm, mu[:, I.D_CT:]], 1)}
            outs = {k: C(x, lf, vv, sC) for k, x in ins.items()}
            ms = m[0].bool()
            sl = slice(T.WARM_SELF, nf)
            pv = pm[0][:, sl]
            av = act[0][:, sl]
            mp_n = (nf - T.WARM_SELF) // I.POOL
            mp = mu[0][:I.D_CT, sl][:, :mp_n * I.POOL].reshape(I.D_CT, mp_n, I.POOL).mean(-1)
            own = mas_score(mp, cid)
            others = rng.sample([k for k in range(len(ctr.tlen)) if k != cid], 20)
            cid_top1 = own > max(mas_score(mp, k) for k in others)
            r = {"set": tag, "voice": v, "frames": nf,
                 "err_ct": float(((mu[0][:I.D_CT, sl] - pv) ** 2).sum(0).mean()), "var_ct": float(((pv - pv.mean(1, keepdim=True)) ** 2).sum(0).mean()),
                 "err_act": float(((mu[0][I.D_CT:, sl] - av) ** 2).sum(0).mean()), "var_act": float(((av - av.mean(1, keepdim=True)) ** 2).sum(0).mean()),
                 "tpl_norm": float((pv ** 2).sum(0).mean()), "content_top1_vs20": bool(cid_top1)}
            for k, yo in outs.items():
                r[f"l1_{k}"] = float((yo[0][:, ms] - y[0][:, ms]).abs().mean())
        ref = tr_(render(y, u))
        r["cer_copy_vs_text"] = cer(kana(u["key"][1]), ref)
        for k in paths[1:]:
            r[f"cer_{k}"] = cer(ref, tr_(render(outs[k], u)))
        rows.append(r)
        print(len(rows), tag, json.dumps({k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items() if k not in ("voice",)}), flush=True)
    res: dict = {"ckpt": ck, "step": st["step"], "n_per_set": n}
    for tag in ("train", "val"):
        R = [r for r in rows if r["set"] == tag]
        res[tag] = {k: round(float(np.mean([r[k] for r in R])), 4) for k in R[0] if k not in ("set", "voice")}
    print(json.dumps(res), flush=True)
    (ROOT / opt.get("out", "results/snv_i5/content_probe.json")).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
