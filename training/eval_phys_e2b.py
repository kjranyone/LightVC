"""phys_e2b: ゼロショット(包絡の物理パラメータ+f0 レジスタ)vs 母集団定数を、変換音声の目標話者 ECAPA 類似度で比べる。事前登録 results/phys_e2b/prereg.yaml。

    CUDA_VISIBLE_DEVICES= uv run python eval_phys_e2b.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile
import torch

sys.path.insert(0, str(Path(__file__).parent))
import artic_dsp as D
import physvc as PV
import train_phys_e1c as TM
from a1_vtl_warp import VC
from a2_dsp_vc import cer, ecapa, norm_text
from eval_phys_e2 import reference
from train_dec2 import load48

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results/phys_e2b"
D_MS = 10.0
POP = np.array([np.log(1.139), 0, 0, 0, 0, 0])
POP_ST = 10.72


def f0_stats(spk: str) -> tuple[float, float]:
    lf = []
    for u in range(25, 80):
        p = VC / f"wav48/{spk}/{spk}_{u:03d}.wav"
        if not p.exists():
            continue
        f0, _ = D.causal_yin(load48(str(p)).astype(np.float64), voi_max=0.25)
        lf.append(np.log(f0[f0 > 0]))
        if sum(len(v) for v in lf) >= 600:
            break
    v = np.concatenate(lf)
    return float(np.median(v)), float(v.std())


def convert(x: np.ndarray, dphi: np.ndarray, m_grid: np.ndarray, reg: tuple | None) -> np.ndarray:
    Dn = int(D_MS * D.SR / 1000)
    lar, a_sub, e = D.analyze(x, 24)
    f0p, _ = D.causal_yin(x, voi_max=0.45)
    ratio = 2 ** (POP_ST / 12) if reg is None else D.register_ratio(f0p, len(x), Dn, *reg)
    e1, _ = D.rrps(e, f0p, ratio, Dn, voiced=D.voiced_known(len(x), f0p, Dn, D.F0_HOP))
    y = D.synthesize(e1, D.coef_schedule(PV.phys_lar(lar, m_grid, dphi), len(x)))
    pk = float(np.abs(y).max())
    return y * (0.95 / pk if pk > 0.95 else 1.0)


def one_pair(pr: list) -> dict:
    import whisper
    m, f = pr
    emb = ecapa()
    asr = whisper.load_model("base", device="cpu")
    enc = TM.Enc()
    enc.load_state_dict(torch.load(ROOT / "results/phys_e1c/last.pt", map_location="cpu")["enc"])
    enc.eval()
    rm, rf = torch.from_numpy(reference(m)), torch.from_numpy(reference(f))
    Em, Ef = TM.env(rm[None]), TM.env(rf[None])
    with torch.no_grad():
        dzs = (enc(TM.einp(Ef), torch.ones(1, Ef.shape[1])) - enc(TM.einp(Em), torch.ones(1, Em.shape[1])))[0].numpy().astype(np.float64)
    m_grid = PV.env_grid(rm.double(), PV.GRID_W).mean(0).numpy()
    mu_s, sd_s = f0_stats(m)
    mu_t, sd_t = f0_stats(f)
    tu = [u for u in range(41, 80) if (VC / f"wav48/{f}/{f}_{u:03d}.wav").is_file()][:3]
    su = next((u for u in range(40, 80) if (VC / f"wav48/{m}/{m}_{u:03d}.wav").is_file()
               and (VC / f"txt/{m}/{m}_{u:03d}.txt").is_file()), None)
    if len(tu) < 3 or su is None:
        print(json.dumps({"pair": [m, f], "skipped": "音声ファイル欠け"}), flush=True)
        return None
    tgt = np.mean([emb(load48(str(VC / f"wav48/{f}/{f}_{u:03d}.wav")).astype(np.float64)) for u in tu], 0)
    tgt /= np.linalg.norm(tgt)
    x = load48(str(VC / f"wav48/{m}/{m}_{su:03d}.wav")).astype(np.float64)[: 8 * D.SR]
    ref_txt = norm_text((VC / f"txt/{m}/{m}_{su:03d}.txt").read_text())
    reg = (mu_s, sd_s, mu_t, sd_t)
    out = {"pair": [m, f], "src_utt": su, "tgt_utts": tu, "dphi_zs": [round(float(v), 3) for v in dzs], "reg": [round(v, 3) for v in reg]}
    for c, (dphi, rg) in {"source": (None, None), "pop": (POP, None), "zs_f0": (POP, reg), "zs_full": (dzs, reg)}.items():
        y = x if dphi is None else convert(x, dphi, m_grid, rg)
        tmp = OUT / f"tmp_{m}_{f}.wav"
        soundfile.write(tmp, y.astype(np.float32), D.SR)
        hyp = norm_text(asr.transcribe(str(tmp), language="en", fp16=False)["text"])
        out[c] = {"secs_tgt": round(float(emb(y) @ tgt), 4), "cer": round(cer(ref_txt, hyp), 3)}
        if c in ("pop", "zs_full"):
            (OUT / "wav").mkdir(exist_ok=True)
            soundfile.write(OUT / "wav" / f"{m}_to_{f}_{c}.wav", y.astype(np.float32), D.SR)
        tmp.unlink(missing_ok=True)
    print(json.dumps(out), flush=True)
    return out


def main() -> int:
    from multiprocessing import Pool
    OUT.mkdir(parents=True, exist_ok=True)
    a1 = json.loads((ROOT / "results/artic_a1/vtl_warp.json").read_text())
    with Pool(6) as pool:
        rows = [r for r in pool.map(one_pair, a1["pairs"]) if r is not None]
    mean = {c: round(float(np.mean([r[c]["secs_tgt"] for r in rows])), 4) for c in ("source", "pop", "zs_f0", "zs_full")}
    dcer = {c: round(float(np.median([r[c]["cer"] - r["source"]["cer"] for r in rows])), 3) for c in ("pop", "zs_f0", "zs_full")}
    win = round(float(np.mean([r["zs_full"]["secs_tgt"] > r["pop"]["secs_tgt"] for r in rows])), 3)
    win_f0 = round(float(np.mean([r["zs_f0"]["secs_tgt"] > r["pop"]["secs_tgt"] for r in rows])), 3)
    ok = mean["zs_full"] > mean["pop"] and win >= 0.6 and dcer["zs_full"] <= dcer["pop"] + 0.05
    rep = {"prereg": "results/phys_e2b/prereg.yaml", "n_pairs": len(rows), "mean_secs_tgt": mean, "cer_increase_median": dcer,
           "frac_zs_full_better": win, "frac_zs_f0_better": win_f0, "verdict": "PASS" if ok else "FAIL", "rows": rows}
    (OUT / "result.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: rep[k] for k in ("mean_secs_tgt", "cer_increase_median", "frac_zs_full_better", "frac_zs_f0_better", "verdict")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
