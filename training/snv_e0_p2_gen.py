"""E0 P2(潜在の結合・synth_nam_vc §3): 目標役 Y を通常に生成し(潜在 z_Y を記録)、操作者役 X を z_Y から t* まで雑音を足して男声の条件(caption + 参照)で 0 まで解く。
時間の格子は Y と共有。t* が大きいほど X は男声に寄り、時刻の結合は弱まる見込み。雑音は種 s + 1000(Y の初期雑音と独立)。
項目は E0 P1 と同じ(results/snv_e0/wav/items.json)。出力 = <out>/iNNN_Y.wav・iNNN_XP{50,70,85}.wav。
    cd ../../Irodori-TTS && uv run --no-sync python ../LightVC/training/snv_e0_p2_gen.py --out ../LightVC/results/snv_e0/wav_p2
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
IRODORI = ROOT.parent / "Irodori-TTS"
PAIR = ROOT / "data/pair_tts_v4l"
TSTARS = (0.5, 0.7, 0.85)


def partial_sampler(RF):
    src = inspect.getsource(RF.sample_euler_rf_cfg)
    reps = [("def sample_euler_rf_cfg(", "def sample_euler_rf_partial("),
            ("    sway_coeff: float = -1.0,\n) -> torch.Tensor:", "    sway_coeff: float = -1.0,\n    z0: torch.Tensor | None = None,\n    t_star: float = 1.0,\n) -> torch.Tensor:"),
            ("    t_schedule = (1.0 - u) * init_scale\n", "    t_schedule = (1.0 - u) * init_scale * t_star\n    if z0 is not None:\n        x_t = (1.0 - t_schedule[0]) * z0.to(x_t) + t_schedule[0] * x_t\n")]
    for a, b in reps:
        assert src.count(a) == 1, a
        src = src.replace(a, b)
    ns = dict(vars(RF))
    exec(src, ns)
    return ns["sample_euler_rf_partial"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--items", default=str(ROOT / "results/snv_e0/wav/items.json"))
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt", default="Aratako/Irodori-TTS-v4.1-Small")
    a = ap.parse_args()
    sys.path.insert(0, str(IRODORI))
    from huggingface_hub import snapshot_download
    import irodori_tts.inference_runtime as IR
    import irodori_tts.rf as RF
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
    part = partial_sampler(RF)
    orig = IR.sample_euler_rf_cfg
    state = {"mode": "rec", "z": None, "t": 1.0}

    def wrap(**kw):
        if state["mode"] == "rec":
            z = orig(**kw)
            state["z"] = z.detach().clone()
            return z
        assert state["z"] is not None and state["z"].shape[1] == kw["sequence_length"], (None if state["z"] is None else state["z"].shape, kw["sequence_length"])
        return part(**kw, z0=state["z"], t_star=state["t"])

    IR.sample_euler_rf_cfg = wrap
    ck = Path(snapshot_download(a.ckpt)) / "model.safetensors"
    rt = InferenceRuntime.from_key(RuntimeKey(checkpoint=str(ck), model_device=a.device, codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
                                              model_precision="fp32", codec_device=a.device, codec_precision="fp32",
                                              codec_deterministic_encode=True, codec_deterministic_decode=True, compile_model=False, compile_dynamic=False))
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    its = json.loads(Path(a.items).read_text())
    (od / "items.json").write_text(json.dumps(its, ensure_ascii=False, indent=1))

    def gen(it: dict, who: str, seed: int, seconds: float | None) -> tuple:
        r = rt.synthesize(SamplingRequest(text=it["text"], caption=it[f"{who}_cap"], ref_wav=str(PAIR / it[who] / "t00.wav"), num_steps=a.steps,
                                          seed=seed, seconds=seconds, trim_tail=False), log_fn=None)
        return r.audio.detach().float().cpu().numpy().reshape(-1).astype(np.float32), r.sample_rate

    for it in its:
        p = od / f"i{it['item']:03d}"
        if Path(f"{p}_XP{int(TSTARS[-1] * 100)}.wav").exists():
            continue
        state["mode"] = "rec"
        y, sr = gen(it, "f", it["seed"], None)
        sec = (len(y) + 0.5) / sr
        sf.write(f"{p}_Y.wav", y, sr)
        state["mode"] = "part"
        for ts in TSTARS:
            state["t"] = ts
            x, _ = gen(it, "m", it["seed"] + 1000, sec)
            assert len(x) == len(y), (len(x), len(y))
            sf.write(f"{p}_XP{int(ts * 100)}.wav", x, sr)
        print(it["item"], it["m"], it["f"], it["style"] or "-", f"{sec:.2f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
