"""段 I10 の後の切り分け(学習なし): 本物の男声の崩れは録音(MP3・背景の音・床の雑音)か、声と話し方か。きれいな録音の本物の男声 = JVNV v1 の M1・M2(48 kHz・スタジオ・感情の演技・評価だけに使う)。
条件: jv_self = 自己の reamp(構造と登録 = 同じ話者の別の発話 6 本)・k 0 / jv_F_k0 = A0F の目標 2 声へ k 0 /
jvmp3_self・jvmp3_F_k0 = 同じ発話と登録を MP3(br= kbps・既定 64)に通して前段からやり直したもの。
extra=1 で元の JVNV に: jv_M_k0 = 検証の Irodori の男声 2 声へ(構造 = その声の E1 の発話 3 本)/ jv_F_k0_dlf = A0F の目標へ k 0・C の高さだけを目標の参照の有声の中央値へずらす(描画器の高さは元のまま)/ jv_F_k7 = A0F の目標へ k 7。chans= で組を選ぶ(jv・jvmp3)。
CER = large-v3 の仮名で、写しの描画の書き起こし 対 出力(i8_text_probe と同じ)。特徴は data/snv_p_feat/jvnv・jvnv_mp3<br>/ に置く。
chans の jvcodec / jvcwm = Irodori の生成の後段(Semantic-DACVAE の符号化と復号 / + SilentCipher の透かし)に通した JVNV(snv_diag/i15_codec_pass.py groups jvnv_all → data/snv_i15_codec・特徴 data/snv_p_feat/jvnv_codec・jvnv_codecwm)= 段 I15 の起動の前の検査。
    cd training && uv run --no-sync python snv_diag/i10_jvnv_probe.py [snv_p3_x snv_i8_m13 snv_i10_combo] [br=64] [extra=1] [chans=jv,jvmp3] [out=results/snv_i10/jvnv_probe.json]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nvoc as N
import snv_p0_front as P
import snv_p1_eval as EV
import train_rvoc as TR
import train_snv_p1 as T

ROOT = Path(__file__).resolve().parents[2]
JV = ROOT / "data/jvnv/jvnv_v1"
EMOS = ("anger", "disgust", "fear", "happy", "sad", "surprise")
CODEC = {"jvnv_codec": "codec", "jvnv_codecwm": "codec_wm"}
DEV = EV.DEV


def mp3(x: np.ndarray, br: int) -> np.ndarray:
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
    enc = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", str(P.SR), "-ac", "1", "-i", "pipe:0", "-c:a", "libmp3lame", "-b:a", f"{br}k", "-f", "mp3", "pipe:1"], input=pcm, capture_output=True, check=True).stdout
    dec = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "s16le", "-ar", str(P.SR), "-ac", "1", "pipe:1"], input=enc, capture_output=True, check=True).stdout
    return np.frombuffer(dec, "<i2").astype(np.float32) / 32767


def wav_of(name: str, sub: str) -> Path:
    spk, emo = name.split("_")[:2]
    if sub in CODEC:
        return ROOT / "data/snv_i15_codec" / CODEC[sub] / "jvnv_all" / f"{name}.wav"
    return JV / spk / emo / "regular" / f"{name}.wav"


def feats(name: str, sub: str, br: int | None) -> dict:
    dst = T.FEAT / sub / f"{name}.npz"
    if not dst.exists():
        x = librosa.load(str(wav_of(name, sub)), sr=P.SR, mono=True)[0].astype(np.float32)
        if br is not None:
            x = mp3(x, br)
        n = P.n_frames(len(x))
        f = P.analyze(x[: n * P.H])
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez(dst, **{k: v.astype(np.float32) for k, v in f.items()})
    return T.load_utt(dst) | ({"wav_path": str(wav_of(name, sub))} if br is None else {})


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    tags = [x for x in sys.argv[1:] if "=" not in x] or ["snv_p3_x", "snv_i8_m13", "snv_i10_combo"]
    br = int(opt.get("br", 64))
    text = {ln.split("|")[0]: ln.split("|")[2].strip() for ln in (JV / "transcription.csv").read_text().splitlines() if ln.count("|") >= 2}
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, kana, to16
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16" if DEV == "cuda" else "int8")

    def tr(w: np.ndarray) -> str:
        segs, _ = asr.transcribe(to16(w.astype(np.float64), EV.SR).astype(np.float32), language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(x.text for x in segs))

    gen = EV.load_gen()

    @torch.no_grad()
    def copy(u: dict) -> np.ndarray:
        y, lf, v = (torch.from_numpy(u[k])[None].to(DEV) for k in ("y", "lf", "v"))
        cond = torch.cat([y[:, :25], lf[:, None], v[:, None], y[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
        f0 = torch.from_numpy(u["f0"])[None].to(DEV)
        return gen(cond, TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:].astype(np.float32)

    meta = json.loads((EV.A1 / "convert_meta.json").read_text())
    tgts = meta["targets"][:2]
    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tgts}
    split = json.loads((ROOT / "results/snv_i8_m13/split.json").read_text())
    mtgts = [v for v in split["val"] if v.startswith("em")][:2]
    trefs |= {v: [EV.utt(f"e1/{v}/{p.stem}") for p in sorted((T.FEAT / "e1" / v).glob("t*.npz"))[:3]] for v in mtgts}
    medlf = lambda us: float(np.median(np.concatenate([u["lf"][u["v"] > 0.5] for u in us])))
    tmed = {v: medlf(rs) for v, rs in trefs.items()}
    extra = opt.get("extra") == "1"
    chans = {c: x for c, x in {"jv": ("jvnv", None), "jvmp3": (f"jvnv_mp3{br}", br), "jvcodec": ("jvnv_codec", None), "jvcwm": ("jvnv_codecwm", None)}.items() if c in opt.get("chans", "jv,jvmp3").split(",")}
    src, enr = {}, {}
    for ch, (sub, b) in chans.items():
        for spk in ("M1", "M2"):
            enr[(ch, spk)] = [feats(f"{spk}_{e}_regular_02", sub, b) for e in EMOS]
            for e in EMOS:
                src[(ch, f"{spk}_{e}_regular_01")] = feats(f"{spk}_{e}_regular_01", sub, b)
    systems = [EV.System(t, gen) for t in tags]
    rows: list = []
    for (ch, name), u in src.items():
        spk = name.split("_")[0]
        ref = tr(copy(u))
        c_copy = cer(kana(text[name.split("_", 1)[1]]), ref)
        stat = enr[(ch, spk)]
        plan = [(f"{ch}_self", [None], 0, False), (f"{ch}_F_k0", tgts, 0, False)]
        if extra and ch == "jv":
            plan += [("jv_M_k0", mtgts, 0, False), ("jv_F_k0_dlf", tgts, 0, True), ("jv_F_k7", tgts, 7, False)]
        for cond, tg, k, sh in plan:
            for s in systems:
                for tgt in tg:
                    S = s.structure(f"self|{ch}|{spk}", stat) if tgt is None else s.structure(f"t|{tgt}", trefs[tgt])
                    dl = tmed[tgt] - medlf([u]) if sh else None
                    out = tr(s.convert(u, stat, S, k, dlf_c=dl))
                    r = {"cond": cond, "sys": s.tag, "src": name, "tgt": tgt, "cer": cer(ref, out), "cer_copy": c_copy, "voiced": round(float(u["v"].mean()), 3), "dlf_semitones": None if dl is None else round(dl / T.LN2_12, 2), "ref": ref[:60], "out": out[:60]}
                    rows.append(r)
                    print(json.dumps(r, ensure_ascii=False), flush=True)
    res: dict = {"tags": tags, "br": br, "conditions": {}}
    for cond in dict.fromkeys(r["cond"] for r in rows):
        res["conditions"][cond] = {}
        for t in tags:
            R = [r for r in rows if r["cond"] == cond and r["sys"] == t]
            res["conditions"][cond][t] = {"cer": round(float(np.mean([r["cer"] for r in R])), 4), "cer_median": round(float(np.median([r["cer"] for r in R])), 4),
                                          "cer_copy": round(float(np.mean([r["cer_copy"] for r in R])), 4), "trials": len(R)}
    res["rows"] = rows
    print(json.dumps(res["conditions"], ensure_ascii=False, indent=1), flush=True)
    (ROOT / opt.get("out", "results/snv_i10/jvnv_probe.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
