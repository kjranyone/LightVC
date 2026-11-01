"""参照を読む実現生成器 R(current/ref_reader.md)の特徴の前計算。ファイルごとに <out>/<corpus>/<spk>/<stem>.pt:
  入力(製品の因果な推定器・200fps): pk / pv = C1 の単位の事後の上位 16(添字 int16・値 fp16)[T, 16]・c0f = F2 + 因果 3 平均 [T]・e3 = F3 の包絡 c1..c24(生の単位)[24, T]・
  p4 = F4 の周期性 [4, T]・f1 = F1 の f0(確信度 0.5 で無声化 = 製品の --f0-conf 0.5)[T]・key = 因果 log-mel の DCT 1..20(DTW の整列専用)[20, T]
  正解(教師・中心窓): env = CheapTrick 包絡 c0..c24(出力部の学習と同じ [w, 1−2w, w] 平滑・生の単位)[25, T]・ft = 教師 f0(harvest)[T]
ネットは GPU(主プロセス)・CheapTrick と harvest は CPU の並列(--workers)。台帳 = <out>/<corpus>/manifest.json。
    uv run python prep_rr.py --corpus fem_neutral --out ../data/rr_feat --workers 10
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
TOPK = 16


def corpus_files(name: str) -> list:
    if name.startswith("glob:"):
        _, d, pat = name.split(":", 2)
        return sorted(Path(d).glob(pat))
    if name == "real_fem":
        import soundfile as sf
        import diag_pool as P
        import train_rvoc as TR
        ex = set(TR.held_speakers()) | {t["spk"] for t in P.load_pool(150, 3, "cpu")}
        out = []
        for d in sorted((ROOT / "female-dataset").iterdir()):
            if not d.is_dir() or d.name in ex:
                continue
            ws = sorted(d.glob("*.wav"))
            if len(ws) < 8:
                continue
            ws = ws[:12]
            if sum(sf.info(str(w)).duration for w in ws) < 40.0:
                continue
            out += ws
        return out
    if name == "real_fem2":
        its = json.loads((ROOT / "data/realx/items_clone_full2.json").read_text())
        return sorted({ROOT / w for it in its for w in it["ref_wavs"]})
    if name == "fem_neutral":
        return sorted((ROOT / "data/female_tts_corpus").glob("*/t0[0-9]_neutral.wav"))
    if name == "fem_soft":
        return sorted((ROOT / "data/female_tts_corpus").glob("*/t0[0-9]_soft.wav"))
    if name == "male_par":
        return sorted((ROOT / "data/male_tts_v4l_par").glob("*/f0[0-9].wav"))
    if name == "male_ref":
        return sorted(f for d in sorted((ROOT / "data/male_tts_v4l").iterdir()) if d.is_dir() for f in sorted(d.glob("t*.wav"))[:8])
    raise ValueError(name)


def teacher(path: str) -> tuple:
    import f0hi as H
    import nvoc as N
    import pae as PA
    import train_rvoc as TR
    from d6_unit_cov import load48
    torch.set_num_threads(1)
    x = load48(path).astype(np.float32)
    n = len(x) // N.HOP
    x = x[: n * N.HOP]
    ft, _ = H.teacher_f0(x, n)
    ft = ft[:n].astype(np.float32)
    xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
    e = np.pad(PA.envelope(xa, ft.astype(np.float64)), ((0, 0), (1, 1)), mode="edge")
    w = SM_W
    env = (w * e[:, :-2] + (1 - 2 * w) * e[:, 1:-1] + w * e[:, 2:])[:, :n].astype(np.float32)
    return path, env, ft


SM_W = 0.25


def main() -> int:
    global SM_W
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--name", default="", help="glob:<dir>:<pattern> のときの出力の名前")
    a = ap.parse_args()
    import c1_content as CC
    import f0est as FE
    import lvl as LV
    import nvoc as N
    import train_c1 as T1
    import train_c3 as C3
    import train_rvoc as TR
    from d6_unit_cov import load48
    SM_W = float(TR.ckpt_env_smooth(torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False), None))
    dev = "cuda"
    ld = lambda p: torch.load(ROOT / p, map_location="cpu", weights_only=False)
    s1 = ld("results/f0est3/last.pt"); f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev); f1.load_state_dict(s1["net"]); f1.eval()
    f1front = FE.Front().to(dev)
    s2 = ld("results/f2_2/last.pt"); f2 = LV.Lvl(s2["cfg"]["ch"], tuple(s2["cfg"]["dils"])).to(dev); f2.load_state_dict(s2["net"]); f2.eval()
    s3 = ld("results/f3_1/last.pt"); f3 = LV.EnvEst(s3["cfg"]["ch"], tuple(s3["cfg"]["dils"])).to(dev); f3.load_state_dict(s3["net"]); f3.eval()
    s4 = ld("results/f4_1/last.pt"); f4 = LV.PerEst(s4["cfg"]["ch"], tuple(s4["cfg"]["dils"])).to(dev); f4.load_state_dict(s4["net"]); f4.eval()
    sc = ld("results/c1_1/last.pt"); c1 = CC.C1(sc["cfg"]["k"], sc["cfg"]["ch"], tuple(sc["cfg"]["dils"])).to(dev); c1.load_state_dict(sc["net"]); c1.eval()
    mfront = CC.MelFront().to(dev)
    from scipy.fft import dct
    lhz = torch.log(FE.bin_hz()).float().to(dev)

    @torch.no_grad()
    def nets(x: np.ndarray) -> dict:
        n = len(x) // N.HOP
        x = x[: n * N.HOP]
        xt = torch.from_numpy(x)[None].to(dev)
        m0 = mfront(torch.cat([torch.zeros(1, N.WIN - N.HOP, device=dev), xt], 1))
        c0f = C3.causal_avg(f2(m0)[:, None], 3)[0, 0][:n]
        e3 = 10.0 * f3(m0)[0, 1:25, :n]
        p4 = f4(m0)[0, :, :n]
        v_, p_ = f1(f1front(torch.cat([torch.zeros(1, FE.CTX48, device=dev), xt], 1), n))
        pr = p_.softmax(1)
        k = pr.argmax(1, keepdim=True)
        idx = (k + torch.arange(-4, 5, device=dev)[None, :, None]).clamp(0, FE.N_BIN - 1)
        w = torch.gather(pr, 1, idx)
        c = w.sum(1)
        f0 = torch.where((torch.sigmoid(v_) > 0.5) & (c >= 0.5), ((w * lhz[idx]).sum(1) / c.clamp(min=1e-9)).exp(), torch.zeros_like(c))[0, :n]
        p = c1(mfront(torch.from_numpy(T1.prime(x))[None].to(dev)))[..., -n:].softmax(1)[0]
        pv, pk = p.T.topk(TOPK, 1)
        lm = m0[0, :, :n].cpu().numpy()
        key = dct(lm, axis=0, norm="ortho")[1:21]

        def fit(z: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.pad(z, (0, n - z.shape[-1]), mode="replicate") if z.shape[-1] < n else z[..., :n]

        return {"pk": torch.nn.functional.pad(pk, (0, 0, 0, n - pk.shape[0])).short().cpu() if pk.shape[0] < n else pk[:n].short().cpu(),
                "pv": torch.nn.functional.pad(pv, (0, 0, 0, n - pv.shape[0])).half().cpu() if pv.shape[0] < n else pv[:n].half().cpu(),
                "c0f": fit(c0f[None, None])[0, 0].float().cpu(), "e3": fit(e3[None])[0].half().cpu(), "p4": fit(p4[None])[0].half().cpu(),
                "f1": fit(f0[None, None])[0, 0].float().cpu(), "key": torch.from_numpy(key.astype(np.float16))}

    for name in a.corpus:
        files = corpus_files(name)
        if name.startswith("glob:"):
            name = a.name
        od = Path(a.out) / name
        todo = [f for f in files if not (od / f.parent.name / (f.stem + ".pt")).exists()]
        print(name, len(files), "todo", len(todo), flush=True)
        man = []
        with Pool(a.workers) as pool:
            for i, (path, env, ft) in enumerate(pool.imap_unordered(teacher, [str(f) for f in todo], chunksize=4)):
                f = Path(path)
                x = load48(f).astype(np.float32)
                d = nets(x)
                n = min(d["c0f"].shape[0], env.shape[1])
                d = {k: (v[..., :n] if v.dim() > 1 and k in ("e3", "p4", "key") else v[:n]) for k, v in d.items()}
                d["env"] = torch.from_numpy(env[:, :n]).half(); d["ft"] = torch.from_numpy(ft[:n])
                o = od / f.parent.name / (f.stem + ".pt")
                o.parent.mkdir(parents=True, exist_ok=True)
                torch.save(d, o)
                if i % 200 == 0:
                    print(name, i, len(todo), flush=True)
        for f in files:
            o = od / f.parent.name / (f.stem + ".pt")
            if o.exists():
                man.append({"spk": f.parent.name, "stem": f.stem, "wav": str(f.relative_to(ROOT)) if f.is_relative_to(ROOT) else str(f), "pt": str(o.relative_to(Path(a.out)))})
        (od / "manifest.json").write_text(json.dumps({"corpus": name, "sm_w": SM_W, "topk": TOPK, "items": man}, indent=0))
        print(name, "done", len(man), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
