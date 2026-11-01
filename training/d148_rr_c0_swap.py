"""D148(学習なし・開発用の組): 自然さと明瞭さを持つ構成(Rust の製品経路の G05S10 = γ0.5・鋭い引き・色づけ・較正)の条件のうち c0 だけを R(diag_rr4)の c0 に替えて鳴らす。
D145: 表 + R4 の c0 は同一性を上げた(0.333 → 0.383)が明瞭さを損なった(CER 0.142)。動きのある包絡の上で R の c0 が同一性を足し、明瞭さを保てるか。
手順: Rust の artic を LIGHTVC_DUMP_COND つきで走らせ条件 [T][31] を得る(較正 b = <dev>/d114/_cal_G05S10)→ R4 の c0 = 操作者の F2 c0 + R の Δc0(d139 と同じ入力・男声の時間)→
条件の行 0 を置き換え、製品の出力部 rvtab_x(Python)で鳴らす(G05C = 置き換え・G05R = 置き換えなし = 同じ Python の経路での基準)→ d141 の半分の等化を掛ける(<cfg>H)。
出力 = <dev>/d114/pNNNN_tNNN_{G05C, G05R}B.wav(等化前)・{G05C, G05R}H.wav。
    uv run python d148_rr_c0_swap.py --s <dev> --tag diag_rr4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import train_rr as RRm

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "target/release/lightvc-app"
EXTRA = ["--rvoc", "results/rvtab_x/export", "--f4", "results/f4_1/export", "--f0-conf", "0.5", "--resid", "0.5", "--color", "--soft", "1"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--s", required=True)
    ap.add_argument("--tag", default="diag_rr4")
    ap.add_argument("--rvoc", default="results/rvtab_x/last.pt")
    a = ap.parse_args()
    import conv_c0 as C0
    import nvoc as N
    import rvoc as R
    import train_rvoc as TR
    S = Path(a.s)
    dev = "cuda"
    od = S / "d114"; dd = S / "d148_dump"; dd.mkdir(exist_ok=True)
    FR = S / "rr_real"

    def ld(corpus: str, stem: str) -> dict:
        man = {it["stem"]: it for it in json.loads((FR / corpus / "manifest.json").read_text())["items"]}
        d = torch.load(FR / man[stem]["pt"], weights_only=False)
        return {"p": RRm.dense(d["pk"], d["pv"]), "u": d["pk"][:, 0].long().numpy(), "e3": d["e3"].float().numpy(), "c0f": d["c0f"].numpy(), "f1": d["f1"].numpy(),
                "env": d["env"].float().numpy(), "ft": d["ft"].numpy()}

    ck = torch.load(ROOT / "results" / a.tag / "last.pt", map_location="cpu", weights_only=False)
    net = RRm.RR(**{k: ck["cfg"][k] for k in ("nq", "ch", "heads", "dk")}, **{k_: tuple(ck["cfg"][k_]) for k_ in ("qd", "kd") if k_ in ck["cfg"]}).to(dev).eval(); net.load_state_dict(ck["net"])
    sr_ = torch.load(ROOT / a.rvoc, map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=sr_["cfg"]["ch"], kernels=tuple(sr_["cfg"]["kernels"]), dils=tuple(sr_["cfg"]["dils"]), d_cond=sr_["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(sr_["ema"]); gen.eval().requires_grad_(False)
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()
    tgt = {}
    for t in range(30):
        r = ld("real_ref", f"ref_{t:03d}")
        n = min(r["env"].shape[1], 5000)
        rc0 = (r["env"][0, :n] - RRm.C0_SIL) / 10
        tgt[t] = {"rp": r["p"][:, :n].astype(np.float32), "rv": np.concatenate([r["env"][1:25, :n] / 10, (rc0 - rc0.mean())[None]], 0).astype(np.float32),
                  "T": C0.table(r["env"][1:25], r["u"], RRm.K, Cb).astype(np.float32)}
    with torch.no_grad():
        for f in sorted((S / "d56").glob("p*_SRC.wav")):
            pi, ti = int(f.name[1:5]), int(f.name[7:10])
            k = f.stem[:-4]
            if (od / f"{k}_G05CB.wav").exists():
                continue
            dump = dd / f"{k}.f32"
            if not dump.exists():
                subprocess.run([str(c) for c in [BIN, "artic", "-i", f, "-o", dd / f"{k}.wav", "--enroll", S / "d85_work" / f"en_{ti:03d}", "--operator-enroll", S / "d85_work" / f"op_{pi:04d}",
                                "--mix", "0", "--per-p75", "--med-from-operator", "--warp", "--alpha", "1.3", "--env-bias", S / "d114/_cal_G05S10" / f"bias_{pi:04d}_t{ti:03d}.f32"] + EXTRA],
                               cwd=ROOT, env={"PATH": "/usr/bin:/bin", "LIGHTVC_DUMP_COND": str(dump)}, check=True, capture_output=True)
            cond = np.fromfile(dump, "<f4").reshape(-1, 31).T.copy()
            b = ld("d56_src", f.stem); o = ld("d56_op", f"op_{pi:04d}")
            vo = o["f1"] > 0
            sM = {"e3": o["e3"][:, vo].mean(1), "c0": float(o["c0f"][vo].mean()), "lf": float(np.median(np.log(o["f1"][vo])))}
            vM = b["f1"] > 0
            q = np.concatenate([b["p"], (b["e3"] - sM["e3"][:, None]) / 10, (b["c0f"] - sM["c0"])[None],
                                np.where(vM, np.log(np.maximum(b["f1"], 1.0)) - sM["lf"], 0.0)[None], vM[None].astype(np.float32)], 0).astype(np.float32)
            tg = tgt[ti]
            tab = (tg["T"] @ b["p"]).astype(np.float32)
            mem = net.memory(torch.from_numpy(tg["rp"])[None].to(dev), torch.from_numpy(tg["rv"])[None].to(dev))
            _, dc0, _ = net(torch.from_numpy(q)[None].to(dev), torch.from_numpy(tab)[None].to(dev), mem, torch.ones(1, tg["rp"].shape[1], dtype=torch.bool, device=dev))
            dc0 = dc0[0].cpu().numpy()
            n = min(cond.shape[1], len(dc0))
            cond = cond[:, :n]
            f0 = np.where(cond[26] > 0.5, 200.0 * np.exp(cond[25]), 0.0).astype(np.float32)
            for nm, c0row in {"G05R": cond[0], "G05C": cond[0] + dc0[:n]}.items():
                cc = cond.copy(); cc[0] = c0row
                y = gen(torch.from_numpy(cc)[None].to(dev), TR.excitation(torch.from_numpy(f0)[None].to(dev), n * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
                sf.write(od / f"{k}_{nm}B.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
            print(k, flush=True)
    for nm in ("G05R", "G05C"):
        cal = S / "d114" / f"_cal_{nm}"
        if not cal.exists():
            cal.symlink_to(S / "d114/_cal_G05S10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
