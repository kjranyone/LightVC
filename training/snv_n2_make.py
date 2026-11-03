"""N2(学習なし): 音素の無い発声の自然さの物差しの較正の素材。事前登録 = results/snv_n2/prereg.yaml。
清音 = JNV(本物の日本語の非言語の発声・48 kHz・F1 / M1 = 開発・F2 / M2 = 検査)・JVNV の非言語の区間(同じ分け方)・N0r(Irodori の演技・声の番号の前半 = 開発)。
劣化 = BigVGAN の写し(V)・log-mel のぼかし B1〜B3(V と対で読む)・Griffin-Lim・白色雑音 10 / 20 / 40 dB・振幅の変調・低域通過 4 / 8 kHz。
生態の対照 = N0r の男 168 本の A2 の L1 回帰の描画(Sg・Sx = 本人の登録の構造)と同じ mel の写し(Vf)。前後の無音は Vf で決めた同じ区間で切る。
出力 = <out>/<pop>_<split>.npz(16 kHz・鍵 = 変種|名前)・results/snv_n2/items.json・hf.json(3〜12 kHz の割合)。
    cd training && uv run --no-sync python snv_n2_make.py <out>
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from snv_a0_measure import DEV, ROOT, buzz, griffinlim, load, lowpass, match_rms, noise, to16
from snv_a2_eval import vocoder
from snv_a2_mel import logmel
from snv_n0_measure import hf_share, trim
from train_snv_a2 import FEAT, G, playing

OUT = ROOT / "results/snv_n2"
JNV = ROOT / "data/jnv/JNV"
JVNV = ROOT / "data/jvnv/jvnv_v1"
N0R = ROOT / "results/snv_n0r/wav"
BLUR = {"B1": 1.0, "B2": 2.0, "B3": 4.0}
SR44 = 44100
MIN_SEC = 0.3


def jvnv_items() -> list[dict]:
    its = []
    for lab in sorted((JVNV / "nv_label").glob("*/*.txt")):
        spk, emo, ses, _ = lab.stem.split("_")
        w = JVNV / spk / emo / ses / f"{lab.stem}.wav"
        for j, ln in enumerate(lab.read_text(encoding="utf-8").splitlines()):
            f = ln.split()
            if len(f) < 2 or not w.exists():
                continue
            its.append({"pop": f"real_{spk[0]}", "split": "dev" if spk.endswith("1") else "test", "spk": f"jvnv_{spk}", "cat": emo,
                        "name": f"jvnv_{lab.stem}_{j}", "path": str(w), "t0": float(f[0]), "t1": float(f[1])})
    return its


def clean_items() -> list[dict]:
    its = []
    for spk in ("F1", "F2", "M1", "M2"):
        for p in sorted((JNV / spk).glob("*.wav")):
            its.append({"pop": f"real_{spk[0]}", "split": "dev" if spk.endswith("1") else "test", "spk": f"jnv_{spk}", "cat": p.stem.split("_")[1].replace("angry", "anger"),
                        "name": f"jnv_{p.stem}", "path": str(p)})
    its += jvnv_items()
    for it in json.loads((N0R / "items.json").read_text()):
        n, half = int(it["voice"][2:]), 12 if it["sex"] == "F" else 6
        its.append({"pop": f"iro_{it['sex']}", "split": "dev" if n < half else "test", "spk": it["voice"], "cat": it["cat"],
                    "name": f"n0r_{it['voice']}_{it['cat']}_{it['k']}", "path": str(N0R / f"{it['voice']}_{it['cat']}_{it['k']}.wav")})
    return its


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    its = clean_items()
    voc = vocoder()

    def vocode(mel: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return voc(torch.from_numpy(mel.astype(np.float32))[None].to(DEV)).squeeze().cpu().numpy().astype(np.float64)

    gs = {}
    for m in ("g", "x"):
        d = ROOT / f"results/diag_snv_a2_{m}"
        ck = torch.load(d / "last.pt", map_location="cpu", weights_only=True)
        net = G(ck["mode"]).to(DEV).eval()
        net.load_state_dict(ck["g"])
        nz = np.load(d / "norm.npz")
        gs[m] = (net, nz["mu"], nz["sd"], {})

    def own_render(m: str, voice: str, stem: str) -> np.ndarray:
        net, mu, sd, cache = gs[m]
        with torch.no_grad():
            if voice not in cache:
                rmel = np.load(FEAT / f"mel/a0enroll/{voice}.npy").astype(np.float32)
                rp = (playing(FEAT / f"sparc/a0enroll/{voice}.npz", rmel.shape[1]) - mu) / sd
                cache[voice] = net.structure(torch.from_numpy(rmel)[None].to(DEV), torch.from_numpy(rp)[None].to(DEV), torch.ones(1, rmel.shape[1], dtype=torch.bool, device=DEV))
            n_mel = np.load(FEAT / f"mel/n0r/{stem}.npy").shape[1]
            p = (playing(FEAT / f"sparc/n0r/{stem}.npz", n_mel) - mu) / sd
            y = net(torch.from_numpy(p)[None].to(DEV), cache[voice])
            return vocode(y[0].cpu().numpy())

    store: dict = defaultdict(dict)
    hf: dict = {}
    kept = []
    for i, it in enumerate(its):
        y, sr = load(Path(it["path"]))
        if "t0" in it:
            y = y[int(it["t0"] * sr): int(it["t1"] * sr)]
        y = trim(y)
        if len(y) < MIN_SEC * sr:
            continue
        it["dur"] = round(len(y) / sr, 3)
        kept.append(it)
        y44 = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SR44).astype(np.float64) if sr != SR44 else y
        mel = logmel(y44)
        var = {"clean": (y, sr), "V": (match_rms(vocode(mel)[: len(y44)], y44), SR44)}
        for b, s in BLUR.items():
            var[b] = (match_rms(vocode(gaussian_filter(mel, sigma=(s, s), mode="nearest"))[: len(y44)], y44), SR44)
        var["GL"] = (griffinlim(y), sr)
        for snr in (10, 20, 40):
            var[f"N{snr}"] = (noise(y, snr, i), sr)
        var["BZ"] = (buzz(y, sr), sr)
        var["LP4"] = (lowpass(y, sr, 4000), sr)
        var["LP8"] = (lowpass(y, sr, 8000), sr)
        if it["pop"] == "iro_M":
            stem = it["name"][4:]
            vf = vocode(np.load(FEAT / f"mel/n0r/{stem}.npy").astype(np.float32))
            _, (a, b) = librosa.effects.trim(vf, top_db=40)
            var["Vf"] = (vf[a:b], SR44)
            for m in ("g", "x"):
                sg = own_render(m, it["spk"], stem)
                var[f"S{m}"] = (match_rms(sg[a:b], vf[a:b]), SR44)
        key = f"{it['pop']}_{it['split']}"
        for v, (w, s) in var.items():
            store[key][f"{v}|{it['name']}"] = to16(w, s).astype(np.float32)
        hf[it["name"]] = {v: round(hf_share(w, s), 3) for v, (w, s) in var.items()}
        if i % 100 == 0:
            print(i, len(its), it["name"], flush=True)
    for key, d in store.items():
        np.savez(out / f"{key}.npz", **d)
    (OUT / "items.json").write_text(json.dumps(kept, ensure_ascii=False, indent=0))
    (OUT / "hf.json").write_text(json.dumps(hf, ensure_ascii=False))
    print("done", len(kept), {k: len(d) for k, d in store.items()}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
