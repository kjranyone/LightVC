"""出力部の追加学習(表の包絡を条件に・オーナー許可 2026-10-10『低レイテンシ推論を守れるなら OK』)用に、学習データの話者ごとの表を前計算する。
製品の登録と同じ作り方: 話者のファイル(名前順)から 20s 以上を参照に・f0 = 前計算の教師(台帳の f0 npy)・包絡 = CheapTrick → [w,1−2w,w] 平滑(rvoc3hi の env_smooth)・
単位 = C1 の硬い単位(ファイル先頭に prime)・表 = conv_c0.table(未観測の単位は近いコードブックから借用)・周期性 = 有声フレームの 4 帯の 75 パーセンタイル・mu = 有声の log f0 の中央値。
出力 = <out>/<spk>.npz(T [24, K] float32・per_p75 [4]・mu・sec)。
    uv run python build_tables_rvoc.py --out ../data/rvoc_tables --workers 10 [--spk_list <file>]
"""
from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data/rvoc_f0hi/manifest.json"
REF_SEC = 20.0
G: dict = {}


def init() -> None:
    torch.set_num_threads(1)
    import c1_content as CC
    import train_rvoc as TR
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])); c1.load_state_dict(st1["net"]); c1.eval()
    G["c1"], G["mf"] = c1, CC.MelFront()
    G["Cb"] = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()
    sr_ = torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False)
    G["sm_w"] = TR.ckpt_env_smooth(sr_, None)


def one(job: tuple) -> str:
    spk, files, out = job
    op = Path(out) / f"{spk}.npz"
    if op.exists():
        return "skip"
    import conv_c0 as C0
    import nvoc as N
    import pae as PA
    import train_c1 as T1
    from train_rvoc import read_span
    try:
        envs, us, f0s, pers = [], [], [], []
        tot = 0.0
        for wav, f0p, sr, dur in files:
            if tot >= REF_SEC:
                break
            n = int(dur * N.SR) // N.HOP
            if n < 50:
                continue
            x = read_span(wav, sr, 0, n * N.HOP).astype(np.float32)
            f0 = np.load(f0p).astype(np.float64)[:n]
            f0 = np.pad(f0, (0, n - len(f0)))
            xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
            e = np.pad(PA.envelope(xa, f0), ((0, 0), (1, 1)), mode="edge")
            w = G["sm_w"]
            env = (w * e[:, :-2] + (1 - 2 * w) * e[:, 1:-1] + w * e[:, 2:])[:, :n]
            per = PA.periodicity(torch.from_numpy(xa)[None], torch.from_numpy(f0.astype(np.float32))[None])[0].numpy()[:, :n]
            with torch.no_grad():
                u = G["c1"](G["mf"](torch.from_numpy(T1.prime(x))[None]))[..., -n:].argmax(1)[0].numpy()
            u = np.pad(u, (0, max(0, n - len(u))), mode="edge")[:n]
            envs.append(env[1:25]); us.append(u); f0s.append(f0); pers.append(per)
            tot += n * N.HOP / N.SR
        if tot < 10.0:
            return f"short {tot:.1f}"
        E, U, F0, P = np.concatenate(envs, 1), np.concatenate(us), np.concatenate(f0s), np.concatenate(pers, 1)
        T = C0.table(E, U, G["Cb"].shape[0], G["Cb"]).astype(np.float32)
        v = F0 > 0
        np.savez(op, T=T, per_p75=np.percentile(P[:, v], 75, axis=1).astype(np.float32), mu=float(np.median(np.log(F0[v]))), sec=tot)
        return "ok"
    except Exception as ex:
        return f"err {ex}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--spk_list", default="")
    ap.add_argument("--manifest", default=str(MANIFEST), help="台帳(既定 = rvoc_f0hi・腕 64 の合成の男声 = data/male_tts_v4l_train/f0_manifest.json)")
    ap.add_argument("--real_only", action="store_true", help="参照を実音声の行だけから作る(話者 ID が TTS のコーパスと重なると名前順で TTS が先に来る・10-18 発見)")
    a = ap.parse_args()
    rows = json.loads(Path(a.manifest).read_text())["rows"]
    by: dict = {}
    for r in rows:
        if r.get("f0") and (not a.real_only or r["src"].startswith("real")):
            by.setdefault(r["spk"], []).append((r["wav"], str(ROOT / r["f0"]), r["sr"], r["dur"]))
    spks = sorted(by)
    if a.spk_list:
        keep = set(Path(a.spk_list).read_text().split())
        spks = [s for s in spks if s in keep]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    jobs = [(s, sorted(by[s]), a.out) for s in spks]
    stats: dict = {}
    with Pool(a.workers, initializer=init) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=2)):
            k = r.split()[0]
            stats[k] = stats.get(k, 0) + 1
            if i % 100 == 0:
                print(i, len(jobs), stats, flush=True)
    print("done", stats, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
