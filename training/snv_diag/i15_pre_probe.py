"""段 I15 の起動の前の検査(paper_snv rev29・学習なし)。
mode=codec = 前段 D(段 I14)が Irodori の生成の痕跡(Semantic-DACVAE の符号化と復号・SilentCipher の透かし)を声の手がかりにしているか。
本物の声(VCTK の男声・JVNV・試験の伴奏の無い区間)と検証の Irodori を、元の音 / 符号器に通した音 / 符号器と透かしに通した音(snv_diag/i15_codec_pass.py → data/snv_i15_codec)で D に入れ、
同じ元の音の前段 v3 と比べる(包絡の L1・有声の一致・取りこぼし・偽の有声・高さの誤差)。前段 v3 自身の動き(元の音 対 通した音)も並べる。
mode=mask = 検証の Irodori の声(透かしと符号器の質感を持つ)に白色雑音を SN 比 30 / 40 / 50 dB で足し(透かしは −47 dB)、D 対 元の音の前段 v3 の外れが本物の声の型(有声の取りこぼし)に近づくか。
mode=feat = 試験の 12 区間と登録を通した音(i15_codec_pass.py groups real_all)の前段 v3 の特徴を data/snv_p_feat/real_codec・real_codecwm に書く(i8_text_probe realdir= で使う)。
mode=bgwm = 声の無い背景(開発の組の音楽 24・雑音 24)に SilentCipher の透かしだけを足すと D が声と読むか(D の有声の率と c0 の平均・元の背景 対 透かしを足した背景・i15_codec_pass.py bg)。
    cd training && SNV_DEV=cpu uv run --no-sync python snv_diag/i15_pre_probe.py mode=codec|bgwm|mask|feat [sep=snv_i14_sep] [out=results/snv_i15/pre_<mode>.json]
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
import i14_real_probe as RP
import i15_codec_pass as CP
import snv_p0_front as P
import train_snv_sep as SEP

ROOT = Path(__file__).resolve().parents[2]
KEYS = ("y_l1", "y_c0", "y_c1_12", "y_c13_24", "y_per", "vacc", "miss", "false", "cents_med", "cents_signed")


def codec(D, dev: str) -> dict:
    res: dict = {}
    for g, its in CP.items().items():
        rows: dict = {"orig": [], "codec": [], "codec_wm": [], "v3_codec": []}
        for nm, w in its:
            x = librosa.load(str(w), sr=P.SR, mono=True)[0]
            n = P.n_frames(len(x))
            x = x[: n * P.H]
            ref = RP.utt_of(x)
            rows["orig"].append(RP.compare(ref, SEP.features(D, x, dev)))
            for sub in ("codec", "codec_wm"):
                z = librosa.load(str(CP.OUT / sub / g / f"{nm}.wav"), sr=P.SR, mono=True)[0][: n * P.H]
                z = np.pad(z, (0, n * P.H - len(z)))
                rows[sub].append(RP.compare(ref, SEP.features(D, z, dev)))
                if sub == "codec":
                    rows["v3_codec"].append(RP.compare(ref, RP.utt_of(z)))
        res[g] = {k: RP.agg(v) | {"n": len(v)} for k, v in rows.items()}
        for k in rows:
            print(g, k.ljust(9), {kk: res[g][k].get(kk) for kk in KEYS}, flush=True)
    return res


def mask(D, dev: str) -> dict:
    res: dict = {}
    its = CP.items()["irodori_val"]
    for snr in (None, 50, 40, 30):
        rows = []
        for nm, w in its:
            x = librosa.load(str(w), sr=P.SR, mono=True)[0]
            n = P.n_frames(len(x))
            x = x[: n * P.H]
            ref = RP.utt_of(x)
            z = x
            if snr is not None:
                g = np.random.default_rng(abs(hash(nm)) % 2**32)
                z = (x + g.standard_normal(len(x)).astype(np.float32) * np.sqrt((x ** 2).mean()) * 10 ** (-snr / 20)).astype(np.float32)
            rows.append(RP.compare(ref, SEP.features(D, z, dev)))
        res[f"snr{snr}"] = RP.agg(rows) | {"n": len(rows)}
        print(f"snr{snr}", {k: res[f'snr{snr}'].get(k) for k in KEYS}, flush=True)
    return res


def feat(D, dev: str) -> dict:
    import shutil
    out: dict = {}
    for sub, dst in (("codec", "real_codec"), ("codec_wm", "real_codecwm")):
        d = RP.FEAT / dst
        d.mkdir(parents=True, exist_ok=True)
        for w in sorted((CP.OUT / sub / "real_all").glob("*.wav")):
            x = librosa.load(str(w), sr=P.SR, mono=True)[0]
            n = P.n_frames(len(x))
            f = P.analyze(x[: n * P.H])
            np.savez(d / f"{w.stem}.npz", **{k: v.astype(np.float32) for k, v in f.items()})
            shutil.copyfile(w, d / f"{w.stem}.wav")
        out[dst] = len(list(d.glob("*.npz")))
    print(out, flush=True)
    return out


def bgwm(D, dev: str) -> dict:
    res: dict = {}
    for kind in ("music", "noise"):
        rows: dict = {"bg": [], "bg_wm": []}
        for nm, _ in CP.bg_items():
            if not nm.startswith(kind):
                continue
            for sub in rows:
                x = librosa.load(str(CP.OUT / sub / f"{nm}.wav"), sr=P.SR, mono=True)[0]
                f = SEP.features(D, x, dev)
                rows[sub].append({"vfrac": float(f["v"].mean()), "c0": float(f["y"][0].mean()), "y_abs": float(np.abs(f["y"]).mean())})
        res[kind] = {sub: {k: round(float(np.median([r[k] for r in v])), 4) for k in v[0]} | {"vfrac_p90": round(float(np.percentile([r["vfrac"] for r in v], 90)), 4), "n": len(v)} for sub, v in rows.items()}
        print(kind, json.dumps(res[kind], ensure_ascii=False), flush=True)
    return res


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    torch.set_num_threads(int(opt.get("threads", 8)))
    dev = "cpu"
    sname, _, sck = opt.get("sep", "snv_i14_sep").partition("@")
    D = SEP.load(ROOT / "results" / sname / f"{sck or 'last'}.pt", dev)
    mode = opt.get("mode", "codec")
    res = {"codec": codec, "bgwm": bgwm, "mask": mask, "feat": feat}[mode](D, dev)
    out = ROOT / opt.get("out", f"results/snv_i15/pre_{mode}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"sep": opt.get("sep", "snv_i14_sep"), "mode": mode, "result": res}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
