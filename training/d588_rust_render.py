"""D588(推論だけ): Rust の製品経路(lightvc-app artic・P58 の書き出し + R の S 形 = R − G・固定の半音)で開発の組を鳴らし、Python の描画(d579 OURS_kRY58G)と本人らしさ・明瞭さを並べる。
入力 = <組>/d467/<key>.wav(操作者の参照の末尾 12 s + SRC 8 s)・出力 = 12 s + DELAY の後ろ 8 s(ピーク 0.9)→ <組>/d570/RUST<k><sfx>/<key>_RUST<k><sfx>.wav。
既定 = R を別のスレッドで 1 フレーム遅れ・記憶の間引き 2(製品の構成)。--sync --stride 1 で Python と同じ計算。
    uv run python d588_rust_render.py --sets dev2 dev3 --k 7
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
H = Path("/home/kojirotanaka/lightvc_scratch_20261013")
BIN = ROOT / "target/release/lightvc-app"


def one(job: tuple) -> str:
    S, key, pi, ti, k, sfx, extra, gate, esub, offline, envs, rvoc = job
    o = (S / "gate" if gate else S / "d570") / f"RUST{k}{sfx}" / f"{key}_RUST{k}{sfx}.wav"
    if o.exists():
        return "skip"
    opw = S / "d56" / "_work" / f"op_{pi:04d}.wav"
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "y.wav"
        cmd = [str(BIN), "artic", "--enroll", str(S / esub / f"en_{ti:03d}"), "--operator-enroll", str(S / "d85_work" / f"op_{pi:04d}"),
               "--mix", "1", "--causal-ct", "--per-p75", "--f4", "results/f4_1/export", "--rvoc", rvoc, "--c2", "results/c2_1/export",
               "--opslow", "4,0", "--c0-ppg-ref", str(opw), "--semitones", str(k),
               ] + ([] if "--no-rr" in extra else ["--rr", "results/d587_rbar/export", "--rr-mem", str((S / "rr_en" / f"en_{ti:03d}" / "rr_mem") if gate else (S / "rr_mem" / f"en_{ti:03d}")), "--rr-op-ref", str(opw)]) + [e for e in extra if e != "--no-rr"] + ["-i", str((S / "d56/_work" / f"pre_{key}.wav") if gate else (S / "d467" / f"{key}.wav")), "-o", str(tmp)]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"} | ({"LVC_SPIN_US": "0", "LVC_NO_PIN": "1"} if offline else {}) | {e.split("=", 1)[0]: e.split("=", 1)[1].format(pi=pi, ti=ti) for e in envs})
        if r.returncode != 0:
            return f"FAIL {key} {r.stderr[-300:]}"
        y, sr = sf.read(tmp, dtype="float32")
    n0 = 12 * sr + 240
    y = y[n0:n0 + 8 * sr]
    o.parent.mkdir(parents=True, exist_ok=True)
    sf.write(o, (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), sr)
    return "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--h1eq", type=float, default=0.0, help="生成器の後ろの適応 H1 EQ の目標 dB(0 = なし・D622 / D624)")
    ap.add_argument("--h1eq_fmax", type=float, default=0.0, help="EQ を働かせる f0 の上限 Hz(0 = なし)")
    ap.add_argument("--render", default="", help="構音の描画の書き出し(--render-artic・例 results/artic_render2/export)。登録の render_curve.f32 は --enroll_sub の側に置く")
    ap.add_argument("--enroll_sub", default="d85_work", help="目標の登録 en_TTT を取る組の下のディレクトリ")
    ap.add_argument("--no_rr", action="store_true", help="R の参照読みを外す(診断)")
    ap.add_argument("--offline_env", action="store_true", help="回し待ちなし・コア固定なし(出力はビット一致・並列の描画を速く)")
    ap.add_argument("--env", nargs="*", default=[], help="診断の環境変数 KEY=VAL({pi}・{ti} を組の番号で置換)")
    ap.add_argument("--sfx", default="", help="出力の版の名前の追加の接尾辞")
    ap.add_argument("--rvoc", default="results/diag_ppgvoc58/export", help="生成器の書き出し(リポジトリの根からの相対)")
    ap.add_argument("--gate", action="store_true", help="関門の組の構成(入力 = d56/_work/pre_<key>.wav・R の記憶 = rr_en/en_TTT/rr_mem・出力 = gate/RUST<k>)")
    ap.add_argument("--kind", default="", help="組の種類だけを描く(devset.json の kind・例 ME)")
    a = ap.parse_args()
    extra = (["--no-rr"] if a.no_rr else ["--rr-stride", str(a.stride)] + (["--rr-sync"] if a.sync else []))
    sfx = ("S" if a.sync else "") + ("" if a.stride == 2 else f"m{a.stride}")
    if a.h1eq > 0:
        extra += ["--h1eq", f"{a.h1eq:g}"] + (["--h1eq-fmax", f"{a.h1eq_fmax:g}"] if a.h1eq_fmax > 0 else [])
        sfx += f"Q{a.h1eq:g}" + ("L" if a.h1eq_fmax > 0 else "")
    sfx += a.sfx
    if a.render:
        assert a.h1eq <= 0, "--render と --h1eq は併用しない"
        extra += ["--render-artic", str(Path(a.render).resolve())]
        sfx += "R"
    jobs = []
    for s in a.sets:
        S = H / s
        for r in json.loads((S / "devset.json").read_text())["pairs"]:
            if a.kind and r.get("kind") != a.kind:
                continue
            key = f"p{r['pi']:04d}_t{r['ti']:03d}"
            if a.gate or (S / "d570" / "OURS7" / f"{key}_OURS7.wav").exists():
                jobs.append((S, key, r["pi"], r["ti"], a.k, sfx, extra, a.gate, a.enroll_sub, a.offline_env, a.env, a.rvoc))
    with Pool(a.workers) as pl:
        res = list(pl.imap_unordered(one, jobs))
    tag = [r_.split()[0] for r_ in res]
    print({x: tag.count(x) for x in set(tag)}, [r_ for r_ in res if r_.startswith("FAIL")][:3], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
