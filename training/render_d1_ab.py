"""D1 F6耳ゲート用の盲検A/B素材(bug7再評価後・2026-09-23)。

問い: AR(履歴条件)は並列CFMよりコーラス/フェーザー感が少ないか(D1の唯一の主張軸=F6)。
試行セット:
  ovf_*  : d1_g0(AR) vs d1_g0par(並列) — 容量・データ・step同一、差は履歴入力のみ(F6の最も清潔な対照)
  full_* : d1_g1full(AR) vs s7_cfm_itp(並列・mel80なし=条件最近傍) vs s11_cfm_melin K16(現行最良)
  fullm_*: d1_g1full(AR) vs diag_d1_g1par(同条件並列・差は履歴のみ=フレーム独立サンプラー)
  fullc_*: d1_g1full(AR) vs diag_cfm_small_nospk(同予算CFMYS=速度場畳み込みの系列同時結合)
既存試行は作り直さない(--setsで追加分だけ生成・鍵はマージ・answers.mdは行追記のみ)。
各試行に GT decode(codec往復=耳でコーラスなし) を錨として混ぜる。
レベル: 試行内の全系をGT decodeにRMS整合→共通減衰(render_z0c.norm_trialと同規約・系ごとのピーク正規化禁止)。
出力: results/earbattery/d1_ab/<trial>/{A,B,..}.wav・README.md・answers.md・_key_聴取後に開く.json

    CUDA_VISIBLE_DEVICES=0 uv run python render_d1_ab.py --sets ovf,full
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import soundfile
import torch

sys.path.insert(0, str(Path(__file__).parent))
from d1_model import sample_frame_ar
from train_d1 import build_index, cond_of
from train_cfmys import LAT, F0FIX, sample_k
from eval_d1_g0 import load_arm, assert_ref_matches_source

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/earbattery/d1_ab"
SR = 48000
T_MAX = 600

SETS = {
    "ovf": [("d1_g0", "last", 8), ("d1_g0par", "last", 8)],
    "full": [("d1_g1full", "last", 8), ("s7_cfm_itp", "best", 8), ("s11_cfm_melin", "best", 16)],
    "fullm": [("d1_g1full", "last", 8), ("diag_d1_g1par", "last", 8)],
    "fullc": [("d1_g1full", "last", 8), ("diag_cfm_small_nospk", "best", 8)],
}
SET_SEED = {"ovf": 0, "full": 0, "fullm": 1, "fullc": 2}


def rms_match(y: np.ndarray, g: np.ndarray) -> np.ndarray:
    n = min(len(y), len(g))
    y, g = np.asarray(y[:n], np.float64), np.asarray(g[:n], np.float64)
    return y * math.sqrt((g ** 2).mean() / max((y ** 2).mean(), 1e-20))


def norm_trial(clips: dict, g: np.ndarray) -> dict:
    ys = {k: rms_match(v, g) for k, v in clips.items()}
    pk = max(float(np.abs(v).max()) for v in ys.values())
    a = 0.95 / pk if pk > 0.95 else 1.0
    return {k: (v * a).astype(np.float32) for k, v in ys.items()}


def pick_voiced(pairs, lats, held_spk, n: int) -> list:
    hset = set(held_spk)
    spk_all = sorted({f.parent.name for f in pairs})
    cands = []
    for s in [s for s in spk_all if s not in hset][:10]:
        best = None
        for f in sorted(f for f in pairs if f.parent.name == s):
            T = torch.load(lats[f.stem], map_location="cpu", weights_only=False)["z"].shape[0]
            if T < 500:
                continue
            f0 = torch.load(F0FIX / f.parent.name / f.name, map_location="cpu",
                            weights_only=False)["f0"].float()
            v = float((f0 > 0).float().mean())
            if best is None or v > best[0]:
                best = (v, f)
        if best is not None:
            cands.append(best)
    cands.sort(key=lambda x: -x[0])
    return [f for _, f in cands[:n]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="ovf,full")
    a = ap.parse_args()
    sets = {k: SETS[k] for k in a.sets.split(",")}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from causal_codec import CausalCodec
    ckc = torch.load(ROOT / "results/s1_3_c32/s1_3_c32_last.pt", map_location=dev,
                     weights_only=False)
    codec = CausalCodec(latent_dim=32, channels=(32, 64, 128, 256, 512)).to(dev)
    codec.load_state_dict(ckc.get("ema") or ckc["net"])
    codec.eval()
    abi = torch.load(LAT / "abi.pt", map_location=dev, weights_only=False)
    MU, SD = abi["mu"].to(dev), abi["sd"].to(dev)
    spk_emb = torch.load(ROOT / "data/ecapa_spk_mean_full.pt", map_location="cpu",
                         weights_only=False)

    def dec(z_raw: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return codec.decode(z_raw[None].to(dev))[0, 0].cpu().numpy()

    pairs, lats, held_spk = build_index(0)
    utts = pick_voiced(pairs, lats, held_spk, 3)
    arms = {}
    for st in sets.values():
        for tag, which, _ in st:
            if tag not in arms:
                arms[tag] = load_arm(tag, dev, which)

    mel_cache: dict = {}

    def mel_of(f, d):
        if f not in mel_cache:
            import librosa
            from causal_mel import causal_mel
            wv, _ = librosa.load(d["path"], sr=44100, mono=True)
            mel_cache[f] = causal_mel(torch.from_numpy(wv) * 32768.0, n_fft=1024, hop=256,
                                      num_mels=80, sr=44100)[0].half()
        return mel_cache[f]

    OUT.mkdir(parents=True, exist_ok=True)
    kp = OUT / "_key_聴取後に開く.json"
    key: dict = json.loads(kp.read_text()) if kp.exists() else {}
    for f in utts:
        d = torch.load(f, map_location="cpu", weights_only=False)
        d = {**d, "f0": torch.load(F0FIX / f.parent.name / f.name, map_location="cpu",
                                   weights_only=False)["f0"]}
        z = torch.load(lats[f.stem], map_location="cpu", weights_only=False)["z"].float().T
        T = min(z.shape[1], T_MAX)
        g = dec(z[:, :T].to(dev))
        chk = assert_ref_matches_source(g, d["path"], len(g))
        for set_name, members in sets.items():
            trial = f"{set_name}_{f.stem}"
            if trial in key:
                print(trial, "exists-skip", flush=True)
                continue
            rng = random.Random(trial)
            clips = {"gt_decode": g}
            for tag, which, K in members:
                kind, net, ck = arms[tag]
                cli = ck["cli"]
                use_mel = cli.get("mel80") if kind == "d1" else cli.get("mel_in")
                cond = cond_of(d, T, mel_of(f, d) if use_mel else None)[None].to(dev)
                with torch.no_grad():
                    if kind == "d1":
                        zh = sample_frame_ar(net, cond, K=K, seed=SET_SEED[set_name],
                                             z0_rho=float(cli.get("z0_rho", 0)))
                        zr = (zh[0] * SD[:, None] + MU[:, None]).clamp(-8, 8)
                    else:
                        s_ = None if cli.get("no_spk") else spk_emb.get(d.get("speaker"))
                        s_ = s_[None].to(dev) if s_ is not None else None
                        gen = torch.Generator(device=dev).manual_seed(SET_SEED[set_name])
                        mu, sd = ck["abi"]["mu"].to(dev), ck["abi"]["sd"].to(dev)
                        zh = sample_k(net, T, cli.get("rho", 0.9), gen, dev, cond, s_, K)
                        zr = (zh[0] * sd[:, None] + mu[:, None]).clamp(-8, 8)
                clips[f"{tag}[{which}@{int(ck['step'])},K{K}]"] = dec(zr)
            normed = norm_trial(clips, g)
            rms = [20 * np.log10(np.sqrt((v.astype(np.float64) ** 2).mean())) for v in normed.values()]
            assert max(rms) - min(rms) < 0.01, f"試行内RMS差 {max(rms)-min(rms):.3f}dB"
            names = list(normed)
            rng.shuffle(names)
            td = OUT / trial
            td.mkdir(parents=True, exist_ok=True)
            key[trial] = {"utt": f.stem, "seed": SET_SEED[set_name], "ref_check": chk, "rms_spread_db":
                          round(max(rms) - min(rms), 5), "map": {}}
            for i, nm in enumerate(names):
                letter = "ABCDEFG"[i]
                soundfile.write(td / f"{letter}.wav", normed[nm], SR)
                key[trial]["map"][letter] = nm
            print(trial, "ok", flush=True)
    kp.write_text(json.dumps(key, indent=1, ensure_ascii=False))
    trials = sorted(key)
    readme = ["# D1 F6耳ゲート 盲検A/B(2026-09-23)", "",
              "問い: **コーラス/フェーザー感(高域が多重化・うなる感じ)が少ないのはどれか。**",
              "各フォルダの A/B/C… を聴いて、下の answers.md に記入してください。",
              "`_key_聴取後に開く.json` は記入が終わるまで開かないでください。", "",
              "- 各試行には正解の錨(codec往復=これまでの耳でコーラスなし)が1本混ざっています。",
              "- 音量は試行内で揃えてあります(RMS整合→共通減衰・差<0.01dB)。音量で判別できません。",
              "- `ovf_*`・`fullm_*`・`fullc_*` は2系+錨、`full_*` は3系+錨です。",
              "- 時間が限られる場合の優先順: `fullc_*` → `full_*` → `ovf_*` → `fullm_*`。", "",
              "試行: " + ", ".join(trials)]
    (OUT / "README.md").write_text("\n".join(readme) + "\n")
    ap_ = OUT / "answers.md"
    if ap_.exists():
        cur = ap_.read_text()
        add = [f"| {t} |  |  |  |" for t in trials if f"| {t} |" not in cur]
        if add:
            ap_.write_text(cur.rstrip("\n") + "\n" + "\n".join(add) + "\n")
    else:
        ans = ["# 回答シート", "",
               "| 試行 | コーラス感が少ない順(例 C>A>B) | 総合で好ましい順 | メモ |",
               "|---|---|---|---|"] + [f"| {t} |  |  |  |" for t in trials]
        ap_.write_text("\n".join(ans) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
