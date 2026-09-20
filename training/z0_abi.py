"""G0-1: Z0-A の ABI をコードの定数から導出して `results/z0/abi.json` に凍結する。

手書きの仕様書は必ずコードとずれる。`causal_mel.abi_spec()` の既定は
`n_fft=2048 / hop=128 / num_mels=128 / 合成 256-256-128` で、本計画の実体
（`1024 / 256 / 80` と `512-512-128`）と**全項目が違う**。だから流用せず、
実際に学習が使う定数を `ship_front` / `rddsp` から読んで書き出す。

`interpretable_vc.md` §7.2 が「別欄で固定」と要求する 3 面に加え、本計画の V は
F0 依存なので **f0 検出の全定義**と **prior の全定義**を第 4・第 5 の欄として足す。
どちらも「後から変えると V も G も再学習」になる量（`TRAINING_PLAN.md` §4）。

未確定の項目は `null` と `decided_by` を書く。**null が 1 つでも残っている間は
R3 を起動しない**（同 §3.4）。
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rddsp as R
import ship_front as SF

OUT = Path("/home/kojirotanaka/kjranyone/LightVC/results/z0/abi.json")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).parent), text=True).strip()
    except Exception:
        return "unknown"


def build() -> dict:
    here = Path(__file__).parent
    return {
        "_note": "Z0-A ABI。_undecided/decided_by が残る間は R3 を起動しない（TRAINING_PLAN §4, §6.4）",
        "git": git_rev(),
        "source_hashes": {f: sha(here / f)
                          for f in ("ship_front.py", "causal_mel.py", "rddsp.py",
                                    "train_gvoc.py", "ship_check.py")},

        # 面1: mel 解析 ABI（G の出力 mel と同一仕様でなければならない）
        "mel_analysis": {
            "sr": R.SR,
            "n_fft": SF.NFFT_A,
            "win": SF.NFFT_A,
            "hop": SF.HOP_A,
            "n_mels": SF.N_MEL,
            "fmin": 0,
            "fmax": None,
            "window": "hann",
            "filterbank": "librosa slaney (causal_mel._mel_fb)",
            "magnitude": "sqrt(|STFT|^2 + 1e-9)",
            "log": "log(clamp(mel, min=1e-5))",
            "framing": "left-aligned: pad (n_fft-hop) LEFT reflect, 0 RIGHT, center=False",
            "lookahead_samples": 0,
            "_caveat": "先頭フレームは reflect パディングにより担当範囲より先を読む"
                       "（NFFT_A=1024/hop=256 で約 11.6ms）。定常部は 0。R3 前に"
                       "「先頭 N フレームは規約外」と明記するか replicate/zero に変える"
                       "（TRAINING_PLAN §6.2）",
            "_deviation": "正典は hop=128。本計画は 256（TRAINING_PLAN §1.1 逸脱B）",
        },

        # 面2: 合成 ABI
        "synthesis": {
            "n_fft": SF.NFFT_S,
            "win": SF.NFFT_S,
            "hop": SF.HOP_S,
            "causal": True,
            "ola": "manual fold, normalise by sum(w^2), drop first (nfft-hop)",
            "inherent_delay_samples": SF.NFFT_S - SF.HOP_S,
            "inherent_delay_ms": 1000.0 * (SF.NFFT_S - SF.HOP_S) / R.SR,
            "_undecided": "n_fft 512 vs 256 未決。256 は耳が『中音域の濁りの真因』と"
                          "判定した格子（RESEARCH 2026-07-22）。賭けは調波 prior が"
                          "解像度を買い戻すこと。G0-B（0 GPU 耳ゲート）で決める",
            "decided_by": "G0-B + FLOP 予算表（TRAINING_PLAN §1, §2）",
            "_deviation": "正典は 256-256-128。本計画は 512-512-128（逸脱C）",
        },

        # 面3: causal frame 時刻
        "frame_time": {
            "definition": "frame t depends on input samples [t*hop-(n_fft-hop), t*hop+hop)",
            "rightmost_sample": "t*hop + hop - 1",
            "lookahead": 0,
            "analysis_to_synthesis_index":
                "j = (t*HOP_S + HOP_S - HOP_A) // HOP_A, clamped >= 0",
            "frame_to_sample_upsample":
                "past-2-frames interpolation: t = (n+1)/hop - 2 (ship_front.frame_upsample_causal)",
        },

        # 面4: f0 検出（F0 依存 V ゆえ ABI の一部）
        "f0": {
            "analysis_stft": "cstft(NFFT_A, HOP_A) left-aligned",
            "fmin": SF.F0_MIN,
            "fmax": SF.F0_MAX,
            "cents_per_candidate": 10.0,
            "kmax_harmonics_in_score": 20,
            "score": "two-lobe harmonic sum: +1/sqrt(k) at k*f0, -0.5/sqrt(k) at (k+0.5)*f0",
            "voicing_threshold_abs": SF.VOI_ABS,
            "voicing_calibration": "female-dataset 60 spk / 41,400 frames, 2026-08-08",
            "median_filter": {"length": 5, "causal": True},
            "_warning": "VOI_ABS は NFFT_A=1024 上の値。窓や fmax を変えたら再較正が必須",
        },

        # 面5: NHV prior（同上）
        "prior": {
            "type": "NHV source-filter, spectral domain (no waveform round trip)",
            "kmax": SF.KMAX,
            "kmax_formula": "(SR/2) // F0_MIN  ← F0_MAX とは無関係",
            "harmonic_masking": "per-sample k*f0[n] < Nyquist, normalise by sqrt(effective count)",
            "noise_mix": 0.3,
            "noise_mix_status": "定数。正典 §2『breath は mel 条件チャネル・V 内部非介入』が既定。"
                                "Z5 の frozen-V OOD check で落ちたら H_v/H_n 分離へ",
            "mel_ref": SF.MEL_REF,
            "mel_ref_calibration": "causal log-mel(n_fft1024) max=1.490 / p99.9=0.578",
            "filter": "H = exp(mel_lin - MEL_REF); 絶対レベルを H が運ぶ（発話 RMS 不使用）",
        },

        # 学習側で凍結すべきもの（ABI ではないが R3 の再現に必要）
        "training_frozen": {
            "held_out_speakers": None,
            "held_out_status": "12 名ハードコード・話者あたり1発話。≥24 話者へ拡張して凍結（G0-3）",
            "decided_by": "G0-3",
            "tts_ratio": None,
            "tts_status": "現状 irodori 669 話者が学習に混入。ROADMAP §2.2『E2 以降の recon GT は"
                          "実音声のみ』と衝突。R3 前に可否を宣言",
            "gain_augmentation": None,
            "gain_aug_status": "未実装。ROADMAP §2.3『E2-B から即』。R3 から入れる",
            "loss": {"logfloor": -60.0, "consist": 1.0, "gan": 1.0, "fm": 2.0, "lmos": 100.0},
            "loss_note": "床のみは耳が否決（非整合性0.264）。床＋整合性は Zbig で耳が選好し ADOPTED",
            "discriminator": "MSSubBandCQTDisc (cqt_disc.py)",
            "trainer_fixes_required": ["best_checkpoint", "resume", "loss_spike_skip"],
        },
    }


def main() -> None:
    d = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n")

    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                yield from walk(v, f"{path}.{k}" if path else k)
        elif o is None and not path.endswith("fmax"):
            yield path
    pend = list(walk(d))
    print(f"  wrote {OUT}")
    print(f"  未確定 {len(pend)} 件:")
    for p in pend:
        print(f"    - {p}")
    print("\n  未確定が 0 になるまで R3 を起動しない。" if pend else "\n  R3 起動可。")


if __name__ == "__main__":
    main()
