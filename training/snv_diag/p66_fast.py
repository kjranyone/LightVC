"""段 P66 step a(results/snv_p66/prereg.yaml・学習なし): 読み手(符号と回路)は、族の中の声の速い語りで崩れるか。
E1 の学習の男声(速く話す版 data/snv_e1f が在る声の先頭 n 声)の t01〜t04 を、普通の速さ(data/snv_e1)と Irodori が速く読み直した版(data/snv_e1f・duration_scale 0.8 / 0.65 / 0.55)で、同じ (声, 文) の対にして測る。
登録 = その声の普通の速さの t05〜t07。系 = snv_p1_eval.System。行 = 写し(描くだけ)・自己(自分の構造・k 0)・女声へ(A0F の先頭 2 声・k 7)。CER の参照 = その音そのものの書き起こし(1 で切る)。
語りの速さ = 書き起こしの仮名の数 ÷ 声の区間の秒(devA と同じ数え方)。書き起こしの文字は保存も表示もしない(数値だけ)。
wsola=1.3,1.6(step a2)= 速い版の代わりに、普通の速さの音を i15_rate_probe.wsola でその倍率に速めた音を源にする(行の scale は "w<倍率>")。
    cd training && SNV_DEV=cuda OMP_NUM_THREADS=2 uv run --no-sync python snv_diag/p66_fast.py [<tag>] [n=24] [wsola=1.3,1.6] [out=results/snv_p66/fast.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import i13_bgm_dev_probe as BP
import nvoc as N
import snv_p0_front as P
import snv_p1_eval as EV
import train_rvoc as TR

ROOT = EV.ROOT
DEV = EV.DEV
TAG = "snv_p30_c+mask:snv_p45_gate_t80+fb:anch"


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    tag = ([x for x in sys.argv[1:] if "=" not in x] or [TAG])[0]
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, kana, to16
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16" if DEV == "cuda" else "int8")
    tr = lambda w: kana("".join(s.text for s in asr.transcribe(to16(w.astype(np.float64), EV.SR).astype(np.float32), language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)[0]))
    gen = EV.load_gen()

    @torch.no_grad()
    def copy(u: dict) -> np.ndarray:
        y, lf, v = (torch.from_numpy(u[k])[None].to(DEV) for k in ("y", "lf", "v"))
        cond = torch.cat([y[:, :25], lf[:, None], v[:, None], y[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
        f0 = torch.from_numpy(u["f0"])[None].to(DEV)
        return gen(cond, TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:].astype(np.float32)

    man = {(f["voice"], f["t"]): f["scale"] for f in json.loads((ROOT / "data/snv_e1f/manifest.json").read_text())["files"]}
    voices = sorted({v for v, _ in man if v.startswith("em")})[:int(opt.get("n", 24))]
    tgts = json.loads((EV.A1 / "convert_meta.json").read_text())["targets"][:2]
    trefs = {t: [EV.utt(f"a0/{t}_{j}") for j in (0, 1, 2)] for t in tgts}
    sy = EV.System(tag, gen)
    load = lambda p: librosa.load(str(p), sr=P.SR, mono=True)[0].astype(np.float64)
    c1 = lambda a, b: round(min(cer(a, b), 1.0), 4)
    rows = []
    ws = [float(q) for q in opt["wsola"].split(",")] if "wsola" in opt else []
    if ws:
        import i15_rate_probe as RP
    for vi, voice in enumerate(voices):
        enr = [BP.utt_of(load(ROOT / f"data/snv_e1/{voice}/t0{j}.wav")) for j in (5, 6, 7)]
        So = sy.structure(f"self|{voice}", enr)
        for t in ("t01", "t02", "t03", "t04"):
            plan = [(1.0, ROOT / f"data/snv_e1/{voice}/{t}.wav", None)] + ([(f"w{r}", ROOT / f"data/snv_e1/{voice}/{t}.wav", r) for r in ws] if ws else [(man[(voice, t)], ROOT / f"data/snv_e1f/{voice}/{t}.wav", None)])
            for scale, p, r_ in plan:
                x = load(p) if r_ is None else RP.wsola(load(p), r_)
                u = BP.utt_of(x)
                ref = tr(x)
                c0 = u["y"][0]
                act = c0 > (np.percentile(c0, 5) + 0.25 * (np.percentile(c0, 95) - np.percentile(c0, 5)))
                r = {"voice": voice, "t": t, "scale": scale, "kana_s": round(len(ref) / max(float(act.sum()) * 0.005, 0.1), 3), "copy": c1(ref, tr(copy(u))), "self": c1(ref, tr(sy.convert(u, enr, So, 0)))}
                r["x"] = [c1(ref, tr(sy.convert(u, enr, sy.structure(f"t|{tg}", trefs[tg]), 7))) for tg in tgts]
                rows.append(r)
        sy.cache = {q: c for q, c in sy.cache.items() if not q.startswith("self|")}
        sy.pcache.clear()
        sy.scache.clear()
        print("voice", vi, {s: [round(float(np.mean([np.mean(r[k]) for r in rows if r["voice"] == voice and r["scale"] == s])), 3) for k in ("copy", "self", "x")] for s in sorted({r["scale"] for r in rows if r["voice"] == voice}, key=str)}, flush=True)
    summ = {}
    for s in sorted({r["scale"] for r in rows}, key=str, reverse=True):
        R = [r for r in rows if r["scale"] == s]
        summ[str(s)] = {"n": len(R), "仮名 / 秒": round(float(np.median([r["kana_s"] for r in R])), 2), "写し": round(float(np.mean([r["copy"] for r in R])), 4), "自己": round(float(np.mean([r["self"] for r in R])), 4), "女声へ": round(float(np.mean([np.mean(r["x"]) for r in R])), 4)}
    print(json.dumps(summ, ensure_ascii=False, indent=1), flush=True)
    outp = ROOT / opt.get("out", "results/snv_p66/fast.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps({"prereg": "results/snv_p66/prereg.yaml(step_a)", "sys": tag, "summary": summ, "rows": rows}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
