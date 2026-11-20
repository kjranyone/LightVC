"""段 D5 の補足 5(results/snv_d5/prereg.yaml supplement_5・学習なし): 本物の声の推論の E の出力のずれは、操作者の登録の統計(一次と二次)で外せるか。
推論の E の内容の出力 59 次元 u を写す: none / mean(u − μ_op + μ_fam)/ diag((u − μ_op)σ_fam / σ_op + μ_fam)/ full(Σ_fam^{1/2} Σ_op^{−1/2}(u − μ_op)+ μ_fam)。
μ_op・σ_op・Σ_op = 操作者の登録の E の出力の有声のフレーム・μ_fam・σ_fam・Σ_fam = 学習の声 48 声(男女 24 ずつ・t03・t04・登録 t05〜t08)の声ごとの統計の平均。目標の参照の鍵・明示の演技・高さ・C はそのまま。
組: cv(devA の先頭 n 人 × s0・登録 e0〜e7)・iro(検証の Irodori の男声 6 声 × t03・t04)・jv(JVNV の regular_01 12 本)→ A0F の 2 声・k 0。CER = 元の音の書き起こし 対 出力。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/d5_moment_probe.py snv_i10_c_pr snv_i12_combo [n=24] [out=results/snv_d5/moment_probe.json]
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
import nvoc as N
import snv_p1_eval as EV
import train_rvoc as TR
import train_snv_p1 as T

ROOT = EV.ROOT
DEV = EV.DEV
D = 59
EMOS = ("anger", "disgust", "fear", "happy", "sad", "surprise")
MODES = ("none", "mean", "diag", "full")


def msqrt(S: torch.Tensor, inv: bool = False) -> torch.Tensor:
    w, V = torch.linalg.eigh(S.double())
    w = w.clamp(min=1e-6)
    return ((V * (w.rsqrt() if inv else w.sqrt())) @ V.T).float()


@torch.no_grad()
def e_out(s: EV.System, us: list[dict], enr: list[dict]) -> torch.Tensor:
    st = s.op_stat(enr)
    out = []
    for u in us:
        y, v = torch.from_numpy(u["y"])[None].to(DEV), torch.from_numpy(u["v"])[None].to(DEV)
        out.append(s.play(y, v, st)[0, :D][:, v[0] > 0.5])
    return torch.cat(out, 1)


def stats(a: torch.Tensor) -> tuple:
    mu = a.mean(1)
    c = torch.cov(a)
    return mu, a.std(1) + 1e-4, 0.9 * c + 0.1 * torch.diag(torch.diag(c))


@torch.no_grad()
def convert(s: EV.System, src: dict, enr: list[dict], S: dict, mode: str, op: tuple, fam: tuple) -> np.ndarray:
    y, v = torch.from_numpy(src["y"])[None].to(DEV), torch.from_numpy(src["v"])[None].to(DEV)
    lf = torch.from_numpy(src["lf"])[None].to(DEV)
    a = s.play(y, v, s.op_stat(enr)).clone()
    u = a[0, :D]
    if mode == "mean":
        u = u - op[0][:, None] + fam[0][:, None]
    elif mode == "diag":
        u = (u - op[0][:, None]) * (fam[1] / op[1])[:, None] + fam[0][:, None]
    elif mode == "full":
        u = msqrt(fam[2]) @ msqrt(op[2], inv=True) @ (u - op[0][:, None]) + fam[0][:, None]
    a[0, :D] = u
    yo = s.C(a, lf, v, S)
    cond = torch.cat([yo[:, :25], lf[:, None], v[:, None], yo[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
    f0 = torch.from_numpy(src["f0"])[None].to(DEV)
    return s.gen(cond, TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:].astype(np.float32)


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    tags = [a for a in sys.argv[1:] if "=" not in a] or ["snv_i10_c_pr", "snv_i12_combo"]
    n = int(opt.get("n", 24))
    import i10_jvnv_probe as JP
    split = json.loads((ROOT / "results/snv_i8_m13/split.json").read_text())
    e1 = lambda v, t: EV.utt(f"e1/{v}/{t}")
    items = [{"set": "iro", "spk": v, "src": [e1(v, t) for t in ("t03", "t04")], "enr": [e1(v, t) for t in ("t05", "t06", "t07", "t08")]} for v in split["val"] if v.startswith("em")]
    man = json.loads((ROOT / "data/cv22_ja/dev_manifest.json").read_text())
    cvu = lambda s_, nm: T.load_utt(T.FEAT / f"cv/{s_}_{nm}.npz") | {"wav_path": str(T.FEAT / f"cv/{s_}_{nm}.wav")}
    items += [{"set": "cv", "spk": s_["speaker"], "src": [cvu(s_["speaker"], "s0")], "enr": [cvu(s_["speaker"], f"e{i}") for i in range(8)]} for s_ in [x for x in man["speakers"] if x["split"] == "devA"][:n]]
    items += [{"set": "jv", "spk": sp, "src": [JP.feats(f"{sp}_{e}_regular_01", "jvnv", None) for e in EMOS], "enr": [JP.feats(f"{sp}_{e}_regular_02", "jvnv", None) for e in EMOS]} for sp in ("M1", "M2")]
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, kana, to16
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16" if DEV == "cuda" else "int8")
    w16 = lambda w: to16(w.astype(np.float64), EV.SR).astype(np.float32)

    def tr(y16: np.ndarray) -> str:
        segs, _ = asr.transcribe(y16, language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(x.text for x in segs))

    gen = EV.load_gen()
    tgts = json.loads((EV.A1 / "convert_meta.json").read_text())["targets"][:2]
    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tgts}
    for it in items:
        it["ref"] = [tr(w16(librosa.load(u["wav_path"], sr=EV.SR, mono=True)[0])) for u in it["src"]]
    fam_v = [v for v in split["train"] if v.startswith("ef")][:24] + [v for v in split["train"] if v.startswith("em")][:24]
    res: dict = {"prereg": "results/snv_d5/prereg.yaml supplement_5", "n_cv": n, "systems": {}, "rows": [], "z": {}}
    for tag in tags:
        s = EV.System(tag, gen)
        fs = [stats(e_out(s, [e1(v, t) for t in ("t03", "t04")], [e1(v, t) for t in ("t05", "t06", "t07", "t08")])) for v in fam_v]
        fam = tuple(torch.stack([f[i] for f in fs]).mean(0) for i in range(3))
        sd_mu = torch.stack([f[0] for f in fs]).std(0) + 1e-4
        zs: dict = {}
        for it in items:
            op = stats(e_out(s, it["enr"], it["enr"]))
            zs.setdefault(it["set"], []).append(float(((op[0] - fam[0]) / sd_mu).norm()))
            zs.setdefault(it["set"] + "_logsd", []).append(float((op[1] / fam[1]).log().abs().mean()))
            for mode in MODES:
                for u, rf in zip(it["src"], it["ref"]):
                    for tg in tgts:
                        e = cer(rf, tr(w16(convert(s, u, it["enr"], s.structure(f"t|{tg}", trefs[tg]), mode, op, fam))))
                        res["rows"].append({"sys": tag, "set": it["set"], "mode": mode, "spk": it["spk"], "tgt": tg, "cer": round(e, 4)})
        o = {f"{st}/{m}": round(float(np.median([r["cer"] for r in res["rows"] if r["sys"] == tag and r["set"] == st and r["mode"] == m])), 4) for st in ("iro", "cv", "jv") for m in MODES}
        res["systems"][tag] = o
        res["z"][tag] = {k: round(float(np.median(v)), 3) for k, v in zs.items()}
        print(tag, json.dumps(o, ensure_ascii=False), "\n   平均の外れ z(声ごとの平均の標準偏差で)と 標準偏差の比の対数の絶対値", res["z"][tag], flush=True)
        del s
        torch.cuda.empty_cache()
    (ROOT / opt.get("out", "results/snv_d5/moment_probe.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
