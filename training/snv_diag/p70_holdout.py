"""段 P70 step b(results/snv_p70/prereg.yaml・診断): 男声の域を学習から除いた系(符号・PCA・符号帳・回路)は、学習に無い族の中の男声で内容を崩すか。
源 = E1 の検証の男声(6 声)× t03〜t09。登録 = その声の t00〜t02。系 = snv_p1_eval.System(門と錨は足さない = 回路そのもの)。
行 = 写し(描くだけ・系に依らない)・自己(自分の構造・k 0)・女声へ(A0F の先頭 2 声・k 7)の CER(参照 = 源の音そのものの書き起こし・1 で切る)と、女声への着地(snv_dev_eval と同じ 1 位の判定)。
p_max = 有声のフレームの、白くした符号(級の正規化の前)の級の事後確率の最大の平均(p59_rel と同じ量)。検証の女声(1 つおき・t05)でも取る(系ごとに符号帳が違うので、男声 − 女声の差で比べる)。
書き起こしの文字は保存も表示もしない(数値だけ)。
    cd training && SNV_DEV=cuda OMP_NUM_THREADS=2 uv run --no-sync python snv_diag/p70_holdout.py diag_snv_p70_fc snv_p30_c [out=results/snv_p70/holdout.json]
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
TS = ("t03", "t04", "t05", "t06", "t07", "t08", "t09")


@torch.no_grad()
def p_max(sy: EV.System, u: dict, enr: list[dict]) -> float:
    src = sy.prep(u)
    y = torch.from_numpy(src["y"])[None].to(DEV)
    v = torch.from_numpy(src["v"])[None].to(DEV)
    st = sy.op_stat(enr)
    Pl = sy.Pl
    xe = torch.einsum("bij,bjt->bit", st[1], Pl.stack(Pl.ctx(y, v), st[0])).transpose(1, 2)[0]
    c = Pl.cb
    d = (xe * xe).sum(-1, keepdim=True) - 2 * xe @ c.T + (c * c).sum(-1)
    p = torch.softmax(-d / (2 * Pl.cb_s2), -1).max(-1).values
    return round(float(p[v[0] > 0.5].mean()), 4)


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    tags = [x for x in sys.argv[1:] if "=" not in x]
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, kana, load as load16, to16
    from snv_a0b_measure import proj
    from snv_n0_measure import trim
    from speechbrain.inference.speaker import EncoderClassifier
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16" if DEV == "cuda" else "int8")
    w16 = lambda w: to16(w.astype(np.float64), EV.SR).astype(np.float32)
    tr = lambda w: kana("".join(s.text for s in asr.transcribe(w16(w), language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)[0]))
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})
    Tz = np.load(ROOT / "data/snv_e1_feat/ecapa_lda_T.npz")

    def emb(y16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(y16).float()[None].to(DEV)).reshape(-1)
        return proj(torch.nn.functional.normalize(e, dim=0).cpu().numpy(), (Tz["mu"], Tz["scalings"]))

    all_t = json.loads((EV.A1 / "convert_meta.json").read_text())["targets"]
    Et = {v: emb(np.concatenate([to16(*load16(EV.A0 / f"wav/{v}_{j}.wav")).astype(np.float32) for j in (0, 1, 2)])) for v in all_t}
    hit = lambda w, v: int(max(all_t, key=lambda q: float(emb(trim(w16(w))) @ Et[q])) == v)
    gen = EV.load_gen()

    @torch.no_grad()
    def copy(u: dict) -> np.ndarray:
        y, lf, v = (torch.from_numpy(u[k])[None].to(DEV) for k in ("y", "lf", "v"))
        cond = torch.cat([y[:, :25], lf[:, None], v[:, None], y[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
        f0 = torch.from_numpy(u["f0"])[None].to(DEV)
        return gen(cond, TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:].astype(np.float32)

    val = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())["val"]
    vm, vf = [v for v in val if v.startswith("em")], [v for v in val if v.startswith("ef")][::2]
    tgts = all_t[:2]
    trefs = {t: [EV.utt(f"a0/{t}_{j}") for j in (0, 1, 2)] for t in tgts}
    load = lambda p: librosa.load(str(p), sr=P.SR, mono=True)[0].astype(np.float64)
    utt = lambda v, t: BP.utt_of(load(ROOT / f"data/snv_e1/{v}/{t}.wav"))
    c1 = lambda a, b: round(min(cer(a, b), 1.0), 4)
    src = {(v, t): utt(v, t) for v in vm for t in TS}
    enr = {v: [utt(v, t) for t in ("t00", "t01", "t02")] for v in vm + vf}
    ref = {k: tr(load(ROOT / f"data/snv_e1/{k[0]}/{k[1]}.wav")) for k in src}
    base = {k: c1(ref[k], tr(copy(u))) for k, u in src.items()}
    res = {"prereg": "results/snv_p70/prereg.yaml(step_b)", "voices": len(vm), "rows": {}, "p_max_valf": {}, "summary": {}}
    for tag in tags:
        sy = EV.System(tag, gen)
        rows = []
        for (v, t), u in src.items():
            So = sy.structure(f"self|{v}", enr[v])
            r = {"voice": v, "t": t, "copy": base[(v, t)], "self": c1(ref[(v, t)], tr(sy.convert(u, enr[v], So, 0))), "p_max": p_max(sy, u, enr[v]), "x": [], "hit": []}
            for tg in tgts:
                w = sy.convert(u, enr[v], sy.structure(f"t|{tg}", trefs[tg]), 7)
                r["x"].append(c1(ref[(v, t)], tr(w)))
                r["hit"].append(hit(w, tg))
            rows.append(r)
        pf = [p_max(sy, utt(v, "t05"), enr[v]) for v in vf]
        res["rows"][tag], res["p_max_valf"][tag] = rows, pf
        res["summary"][tag] = {"写し": round(float(np.mean([r["copy"] for r in rows])), 4), "自己": round(float(np.mean([r["self"] for r in rows])), 4), "女声へ": round(float(np.mean([np.mean(r["x"]) for r in rows])), 4),
                               "着地": round(float(np.mean([np.mean(r["hit"]) for r in rows])), 4), "p_max 男": round(float(np.mean([r["p_max"] for r in rows])), 4), "p_max 女": round(float(np.mean(pf)), 4)}
        print(tag, json.dumps(res["summary"][tag], ensure_ascii=False), flush=True)
    outp = ROOT / opt.get("out", "results/snv_p70/holdout.json")
    outp.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
