"""V2-3a: VCTK same-text M-F ペアの選択と fine DTW アライメント教師の作成。

選択（V2-0a 凍結）: 発話ID+テキスト完全一致 / len_ratio∈[0.7,1.4] / cost≤0.40 /
5 秒以上。fine DTW = ContentVec 50fps フルグリッド（粗視化なし）。
target mel/f0 を source タイムラインへ warp し .pt として保存
（content(T_content)は学習時に E で抽出するのでここでは warp index のみ保存）。

    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 uv run python v23a_pairs.py [--cap 200]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import ship_front as SF
from extract_content import load_contentvec, content_of

ROOT = Path(__file__).resolve().parent.parent
VCTK = ROOT / "data/vctk/VCTK-Corpus/VCTK-Corpus"
OUT = ROOT / "data/vctk_pairs"
FPS = 50


def dtw_full(C: np.ndarray) -> tuple[np.ndarray, float]:
    n, m = C.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    P = np.zeros((n + 1, m + 1), dtype=np.int8)
    for i in range(1, n + 1):
        Ci = C[i - 1]
        row = D[i]
        prev = D[i - 1]
        for j in range(1, m + 1):
            c = Ci[j - 1]
            best, bk = np.inf, 0
            v = prev[j - 1]
            if v < best:
                best, bk = v, 1
            v = prev[j]
            if v < best:
                best, bk = v, 2
            v = row[j - 1]
            if v < best:
                best, bk = v, 3
            row[j] = best + c
            P[i, j] = bk
    path = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        tag = P[i, j]
        if tag == 1:
            i, j = i - 1, j - 1
        elif tag == 2:
            i -= 1
        elif tag == 3:
            j -= 1
        else:
            break
    path.reverse()
    return np.array(path), float(D[n, m] / max(n, m))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=200, help="最大ペア数(試験)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    genders = {}
    for line in open(VCTK / "speaker-info.txt"):
        p = line.split()
        if len(p) >= 3 and p[0].isdigit():
            genders["p" + p[0]] = p[2]
    utt = defaultdict(list)
    for f in (VCTK / "txt").rglob("*.txt"):
        sid, uid = f.stem.split("_")
        utt[(uid, f.read_text().strip())].append(sid)
    cross = [(k, v) for k, v in utt.items()
             if len({genders.get(s, "?") for s in v}) > 1]

    model = load_contentvec()
    import librosa

    feat_cache: dict[str, np.ndarray] = {}

    def cv_of(spk: str, uid: str) -> tuple[np.ndarray, np.ndarray, int]:
        k = f"{spk}_{uid}"
        if k not in feat_cache:
            w, _ = librosa.load(VCTK / "wav48" / spk / f"{k}.wav", sr=16000, mono=True)
            feat_cache[k] = content_of(model, w).float().numpy()
        w44, _ = librosa.load(VCTK / "wav48" / spk / f"{k}.wav", sr=44100, mono=True)
        return feat_cache[k], w44, len(w44)

    pairs = []
    for (uid, text), spks in cross:
        M = [s for s in spks if genders.get(s) == "M"]
        F = [s for s in spks if genders.get(s) == "F"]
        for m, f in product(M, F):
            pairs.append((uid, text, m, f))
    pairs.sort(key=lambda x: (x[0], x[2], x[3]))
    print(f"  candidate {len(pairs)} pairs (cap {a.cap})", flush=True)

    rows = []
    n_ok = 0
    for uid, text, m, f in pairs:
        if n_ok >= a.cap:
            break
        try:
            cm, wm, nm = cv_of(m, uid)
            cf, wf, nf = cv_of(f, uid)
        except FileNotFoundError:
            continue
        dur = min(nm, nf) / 44100
        if dur < 5.0:
            continue
        cmn = cm / (np.linalg.norm(cm, axis=-1, keepdims=True) + 1e-6)
        cfn = cf / (np.linalg.norm(cf, axis=-1, keepdims=True) + 1e-6)
        path, cost = dtw_full(1.0 - cmn @ cfn.T)
        len_ratio = len(path) / max(cm.shape[0], cf.shape[0])
        if not (0.7 <= len_ratio <= 1.4) or cost > 0.40:
            continue
        src_idx = path[:, 0]
        tgt_idx = path[:, 1]
        x = torch.from_numpy(wm[: nm]) * 32768.0
        mel_m = SF.mel(x)
        x2 = torch.from_numpy(wf[: nf]) * 32768.0
        mel_f = SF.mel(x2)
        f0m, _ = SF.causal_f0(x)
        f0f, _ = SF.causal_f0(x2)
        # melグリッド(172fps)への warp: content frame src_idx -> tgt_idx を
        # mel フレーム比でスケール
        mm = mel_m.shape[-1]
        scale = mm / max(cm.shape[0], 1)
        si_mel = np.minimum((src_idx * scale).astype(int), mel_f.shape[-1] - 1)
        ti_mel = np.minimum((tgt_idx * scale).astype(int), mel_f.shape[-1] - 1)
        mel_warp = torch.zeros(80, mm)
        seen = np.zeros(mm, dtype=bool)
        mel_warp[:, si_mel] = mel_f[:, ti_mel]
        seen[si_mel] = True
        f0f_f = f0f[: mel_f.shape[-1]]
        f0_warp = torch.zeros(mm)
        f0_warp[np.minimum(si_mel, mm - 1)] = f0f_f[np.minimum(ti_mel, f0f_f.shape[-1] - 1)]
        for i in range(1, mm):                       # 因果 fill（ホール埋め）
            if not seen[i]:
                mel_warp[:, i] = mel_warp[:, i - 1]
            if f0_warp[i] == 0:
                f0_warp[i] = f0_warp[i - 1] if f0_warp[i - 1] > 0 else 0.0
        sid_out = OUT / f"{m}_{f}_{uid}.pt"
        torch.save({
            "uid": uid, "male": m, "female": f,
            "wav_m": str(VCTK / "wav48" / m / f"{m}_{uid}.wav"),
            "wav_f": str(VCTK / "wav48" / f / f"{f}_{uid}.wav"),
            "mel_src": mel_m[:, :mm], "mel_tgt_warp": mel_warp,
            "f0_src": f0m[:mm], "f0_tgt": f0f, "f0_tgt_warp": f0_warp,
            "cost": cost, "len_ratio": len_ratio, "dur": dur,
        }, sid_out)
        rows.append(dict(male=m, female=f, uid=uid, cost=round(cost, 4),
                         len_ratio=round(len_ratio, 3), dur=round(dur, 2)))
        n_ok += 1
        if n_ok % 20 == 0:
            print(f"    {n_ok} ok", flush=True)
    json.dump(rows, open(OUT / "pairs.json", "w"), indent=1)
    print(f"  saved {n_ok} pairs -> {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
