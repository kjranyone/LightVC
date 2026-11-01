"""diag_rr9c のデータ(オーナー判断 10-14: Irodori のクローンを学習の目標に使ってよい・登録 / 推論には使わない): data/realx/items.json の各項目について、
目標 A の参照(A の rr_feat/real_fem のクリップのうち項目のクリップを除いた先頭から 20 秒以上 = 学習で R の記憶に置くものと同じ)を手本に、
項目の文(男声 TTS が読んだのと同じ文)を Irodori-TTS-v4-Large に読ませる(caption なし)。出力 = data/realx/clone/p{item:04d}_t{ti:03d}_CLN.wav・
手本の stem の一覧 = data/realx/items_clone.json(train_rr の --p_clonex が記憶に同じ stem を使う)。
    cd ../../Irodori-TTS && uv run --no-sync python ../LightVC/training/gen_realx_clone.py
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_male_corpus_v4l import IRODORI

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=str(ROOT / "data/realx/items.json"))
    ap.add_argument("--out", default=str(ROOT / "data/realx/clone"))
    ap.add_argument("--ref_sec", type=float, default=20.0)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--precision", default="fp32", help="Irodori のモデルの精度(diag_rr13c の学習用の既知の話者だけ bf16・評価の項目は fp32)")
    ap.add_argument("--groups", default="", help="--premade の group で絞る(カンマ区切り・空 = 全部)")
    ap.add_argument("--premade", action="store_true", help="--items が ref_stems・clone を持つ(make_clone_multi.py の出力)= そのまま合成し items_clone.json は書かない(diag_rr12c)")
    a = ap.parse_args()
    man = json.loads((ROOT / "data/rr_feat/real_fem/manifest.json").read_text())["items"]
    by: dict = {}
    for it in man:
        by.setdefault(it["spk"], []).append(it)
    items = json.loads(Path(a.items).read_text())
    od = Path(a.out); od.mkdir(parents=True, exist_ok=True)
    meta = [m for m in items if not a.groups or m.get("group") in a.groups.split(",")] if a.premade else []
    for it in ([] if a.premade else items):
        refs, tot = [], 0.0
        for r in by[it["target"]]:
            if r["stem"] == it["stem"]:
                continue
            if tot >= a.ref_sec:
                break
            refs.append(r); tot += sf.info(str(ROOT / r["wav"])).duration
        meta.append(it | {"ref_stems": [r["stem"] for r in refs], "ref_sec": round(tot, 1), "clone": f"p{it['item']:04d}_t{it['ti']:03d}_CLN"})
    si, sk = (int(z) for z in a.shard.split("/"))
    if si == 0 and not a.premade:
        (ROOT / "data/realx/items_clone.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    sys.path.insert(0, str(IRODORI))
    from huggingface_hub import snapshot_download
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
    ck = Path(snapshot_download("Aratako/Irodori-TTS-v4-Large")) / "model.safetensors"
    rt = InferenceRuntime.from_key(RuntimeKey(checkpoint=str(ck), model_device=a.device, codec_repo="Aratako/Semantic-DACVAE-Japanese-32dim",
                                              model_precision=a.precision, codec_device=a.device, codec_precision="fp32",
                                              codec_deterministic_encode=True, codec_deterministic_decode=True, compile_model=False, compile_dynamic=False))
    stem2wav = {r["stem"]: str(ROOT / r["wav"]) for r in man}
    for m in meta:
        if (meta.index(m) if a.premade else m["item"]) % sk != si:
            continue
        wp = od / f"{m['clone']}.wav"
        if wp.exists():
            continue
        refs = [str(ROOT / w) for w in m["ref_wavs"]] if "ref_wavs" in m else [stem2wav[s] for s in m["ref_stems"]]
        r = rt.synthesize(SamplingRequest(text=m["text"], ref_wav=refs[0] if len(refs) == 1 else None, ref_wavs=refs if len(refs) > 1 else None,
                                          num_steps=a.steps, seed=a.seed + m["item"] + (zlib.crc32(m["clone"].encode()) % 100000 if a.premade else 0)), log_fn=None)
        sf.write(wp, r.audio.detach().float().cpu().numpy().reshape(-1).astype(np.float32), r.sample_rate)
        print(m["item"], len(refs), m["ref_sec"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
