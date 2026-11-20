"""段 P57 step c(results/snv_p57/prereg.yaml の step_c・学習なし): 門が開いたフレームで、内容を符号の番地だけに通さず、源の包絡の残りを通す形の、内容と着地の測り。
物差し = p37_cv_music と同じ(devA の本物の男声 + 開発の音楽・伴奏 10 dB / 0 dB / 足さない・目標 2 声・k 0 と 7・CER の参照 = 元の音の書き起こし)に、着地 m(snv_dev_eval と同じ ECAPA + LDA の 1 位)を足す。
系 = 研究の形(snv_p30_c+mask:snv_p45_gate_t80)。行:
  base       = 今の形(C(符号, 目標の登録))。System.convert と同じ値(最初の発話で確かめる)。
  swap<β>    = 門が開いたフレームだけ C(符号, 目標) + β ×(源の包絡 − C(符号, 操作者の登録))。包絡の 25 次(c0n・c1..c24)。周期性は C の出力のまま。
  swapall<β> = 同じ式を全てのフレームで。
  anch       = 門が開いたフレームだけ、源の包絡の一律の伸縮(snv_p0c_anchor・α は操作者と目標の登録の平均から)と源の周期性に替える。
  anchq      = anch に、目標の登録の有声の平均 − 伸縮した操作者の登録の有声の平均(c1..c24)を足す(step c2)。
  split12 / splitq12 = 門が開いたフレームで c0..c12 を anch / anchq から、c13..c24 と周期性を C(符号, 目標) から(step c2)。
  mix<λ> / mixq<λ> = 門が開いたフレームで λ × anch(anchq) + (1 − λ) × C(符号, 目標)(包絡の 25 次・step c2)。
源の包絡と操作者の登録は、どちらも研究の形の前段(門 + 重み)を通した特徴。操作者の登録は物差しと同じ条件で足した音を含む。
    cd training && OMP_NUM_THREADS=2 uv run --no-sync python snv_diag/p57_swap.py [kinds=base,swap1,swap0.5,anch,swapall1] [n=64] [skip=0] [conds=mus10,mus0] [out=results/snv_p57/swap.json]
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
import snv_p0c_anchor as AN
import snv_p1_eval as EV
import train_rvoc as TR
import train_snv_mask as MK
import train_snv_p1 as T
import train_snv_sep as SEP

ROOT = EV.ROOT
DEV = EV.DEV
KS = (0, 7)
TAG = "snv_p30_c+mask:snv_p45_gate_t80"


@torch.no_grad()
def gate_frames(sy: EV.System, x: np.ndarray) -> torch.Tensor:
    Tn = MK.P.n_frames(len(x))
    xt = torch.from_numpy(np.asarray(x[: Tn * MK.H], np.float32))
    p = torch.sigmoid(sy.sep.gate(xt[None].to(DEV))[0, 0].double()).cpu()[:Tn]
    return MK.hysteresis(p, *sy.sep.hyst, sy.sep.thr) if sy.sep.hyst else p >= sy.sep.thr


@torch.no_grad()
def convert(sy: EV.System, u: dict, enr: list[dict], St: dict, So: dict, k: int, kind: str, mu_t: np.ndarray) -> np.ndarray:
    src = sy.prep(u)
    y = torch.from_numpy(src["y"])[None].to(DEV)
    v = torch.from_numpy(src["v"])[None].to(DEV)
    lf0 = torch.from_numpy(src["lf"])[None].to(DEV)
    lf = lf0 + k * T.LN2_12 * v
    pl = sy.play(y, v, sy.op_stat(enr))
    yo = sy.C(pl, lf, v, St)
    if kind != "base":
        g = gate_frames(sy, u["wav"]).to(DEV)[None, None].float()
        assert g.shape[-1] == y.shape[-1], (g.shape, y.shape)
        if kind.startswith("swap"):
            beta = float(kind[7:] if kind.startswith("swapall") else kind[4:])
            r = y[:, :25] - sy.C(pl, lf0, v, So)[:, :25]
            yo = torch.cat([yo[:, :25] + beta * (1.0 if kind.startswith("swapall") else g) * r, yo[:, 25:]], 1)
        else:
            yn = src["y"]
            env = np.concatenate([yn[:1] * 10 + T.PA.C0_SIL, yn[1:25] * 10], 0)
            mu_o = EV.mu_of([sy.prep(e) for e in enr])[0].cpu().numpy()
            al = AN.fit_alpha(mu_o, mu_t)
            ew = AN.warp_env(env, al)
            if "q" in kind:
                ew[1:25] += (mu_t[1:25] - AN.warp_env(mu_o, al)[1:25])[:, None]
            yw = torch.from_numpy(np.concatenate([(ew[:1] - T.PA.C0_SIL) / 10, ew[1:25] / 10, yn[25:29]], 0).astype(np.float32))[None].to(DEV)
            if kind.startswith("split"):
                yw = torch.cat([yw[:, :13], yo[:, 13:29]], 1)
            elif kind.startswith("mix"):
                lam = float(kind.lstrip("mixq"))
                yw = torch.cat([lam * yw[:, :25] + (1 - lam) * yo[:, :25], yo[:, 25:29]], 1)
            yo = torch.where(g > 0.5, yw, yo[:, :29])
    cond = torch.cat([yo[:, :25], lf[:, None], v[:, None], yo[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
    f0 = torch.from_numpy(src["f0"] * 2 ** (k / 12))[None].to(DEV)
    return sy.gen(cond, TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:].astype(np.float32)


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    kinds = opt.get("kinds", "base,swap1,swap0.5,anch,swapall1").split(",")
    n, skip = int(opt.get("n", 64)), int(opt.get("skip", 0))
    conds = opt.get("conds", "mus10,mus0").split(",")
    split = "devA"
    man = json.loads((ROOT / "data/cv22_ja/dev_manifest.json").read_text())
    dv = json.loads((ROOT / f"results/snv_d5/dev_rows_{split}.json").read_text())["ref"]
    outp = ROOT / opt.get("out", "results/snv_p57/swap.json")
    part = outp.with_suffix(".part.json")
    rows = json.loads(part.read_text())["rows"] if part.exists() else []
    done = {r["idx"] for r in rows}
    pool = SEP.bg_split()["dev"]["music"]
    load = lambda name: librosa.load(str(T.FEAT / f"cv/{name}.wav"), sr=B.SR, mono=True)[0].astype(np.float64)
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, kana, to16
    from snv_a0_measure import load as load16
    from snv_a0b_measure import proj
    from snv_n0_measure import trim
    from speechbrain.inference.speaker import EncoderClassifier
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16" if DEV == "cuda" else "int8")
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})
    Tz = np.load(ROOT / "data/snv_e1_feat/ecapa_lda_T.npz")
    w16 = lambda w: to16(w.astype(np.float64), EV.SR).astype(np.float32)

    def tr(y16: np.ndarray) -> str:
        segs, _ = asr.transcribe(y16, language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(x.text for x in segs))

    def emb(y16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(y16).float()[None].to(DEV)).reshape(-1)
        return proj(torch.nn.functional.normalize(e, dim=0).cpu().numpy(), (Tz["mu"], Tz["scalings"]))

    all_t = json.loads((EV.A1 / "convert_meta.json").read_text())["targets"]
    Et = {v: emb(np.concatenate([to16(*load16(EV.A0 / f"wav/{v}_{j}.wav")).astype(np.float32) for j in (0, 1, 2)])) for v in all_t}
    hit = lambda y16, v: int(max(all_t, key=lambda q: float(emb(trim(y16)) @ Et[q])) == v)
    gen = EV.load_gen()
    tgts = all_t[:4]
    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tgts}
    mu_t = {v: EV.mu_of(trefs[v])[0].cpu().numpy() for v in tgts}
    sy = EV.System(TAG, gen)
    first = True
    for i, s in enumerate([x for x in man["speakers"] if x["split"] == split][skip:n], start=skip):
        sp = s["speaker"]
        srcs = [f"{sp}_{c['name']}" for c in s["src"][:2] if f"{sp}_{c['name']}" in dv and dv[f"{sp}_{c['name']}"]["text_vs_orig"] <= 0.1]
        if not srcs or i in done:
            continue
        xe = [load(f"{sp}_{c['name']}") for c in s["enroll"]]
        mus = pool[zlib.crc32(sp.encode()) % len(pool)]
        tg = [tgts[i % 4], tgts[(i + 2) % 4]]
        for cond in ["clean"] + conds:
            mixp = (lambda x, seed, src_: (x, x)) if cond == "clean" else (lambda x, seed, src_: B.with_music(x, mus, seed, False, float(cond[3:])))
            enr = [B.utt_of(mixp(x, 7 + j, False)[1]) for j, x in enumerate(xe)]
            So = sy.structure(f"self|{sp}|{cond}", enr)
            for name in srcs:
                u = B.utt_of(mixp(load(name), 1 + i, True)[1])
                go = float(gate_frames(sy, u["wav"]).double().mean())
                for v in tg:
                    St = sy.structure(f"t|{v}", trefs[v])
                    for k in KS:
                        for kind in kinds:
                            w = convert(sy, u, enr, St, So, k, kind, mu_t[v])
                            if first and kind == "base":
                                first = False
                                d = float(np.abs(w - sy.convert(u, enr, St, k)).max())
                                assert d < 5e-3, d
                                print("base の行は System.convert と同じ値", d, flush=True)
                            y16 = w16(w)
                            rows.append({"spk": sp, "idx": i, "src": name, "cond": cond, "sys": kind, "tgt": v, "k": k, "cer": round(cer(dv[name]["orig"], tr(y16)), 4), "m": hit(y16, v), "go": round(go, 3)})
        sy.cache = {q: c for q, c in sy.cache.items() if not q.startswith("self|")}
        sy.pcache.clear()
        sy.scache.clear()
        mean = lambda c, kd, q: round(float(np.mean([min(r[q], 1.0) for r in rows if r["idx"] == i and r["cond"] == c and r["sys"] == kd])), 3)
        print("speaker", i, {c: {kd: [mean(c, kd, "cer"), mean(c, kd, "m")] for kd in kinds} for c in ["clean"] + conds}, flush=True)
        part.write_text(json.dumps({"rows": rows}, ensure_ascii=False))
    res = {"prereg": "results/snv_p57/prereg.yaml(step_c)", "sys": TAG, "kinds": kinds, "conds": ["clean"] + conds, "n_speakers": len({r["spk"] for r in rows}), "rows": rows}
    outp.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    part.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
