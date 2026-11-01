"""D361: 男声の分の最大の運び手 = 事後の柔らかい形(D327b・局所の声質 D360)を、最大の単位ごとの平均の形の差で女声へ寄せる推論の補正(固定の定数・発話の統計なし)。
差 Δ_k = 女声の平均の形 − 男声の平均の形(argmax = k のフレームの softmax の平均・同じ文の TTS の対 data/pair_tts_v4l の書き出し xpair = 内容と TTS をそろえた性別の差・どちらかが 50 フレーム未満の単位は 0)。
男声入力(dev・MT)の事後に p' = 正規化(max(p + α Δ_argmax(p), 0))・候補(ppgvoc24 + 因果な揺らぎ σ 1.5)で描画し、MT の v1⊥・FT(d352m の描画)との 2 標本の判別・CER を測る。診断。
    uv run python d361_unitshape.py --ckpt ../results/diag_ppgvoc24/product_s15_007500.pt --xpair <xpair> --ft ../results/conv_p0/d352m --out ../results/conv_p0/d361
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
import nvoc as N
import train_ppgvoc as P


def unit_means(xpair: Path) -> tuple:
    S = {g: np.zeros((P.K, P.K)) for g in "mf"}
    C = {g: np.zeros(P.K) for g in "mf"}
    for m in json.loads((xpair / "meta.json").read_text())["pair"]:
        _, lg = P.load_dump(xpair / "pair", m["key"])
        p = torch.from_numpy(lg).float().softmax(-1).numpy()
        a = p.argmax(1)
        np.add.at(S[m["g"]], a, p)
        np.add.at(C[m["g"]], a, 1)
    ok = (C["m"] >= 50) & (C["f"] >= 50)
    D = np.zeros((P.K, P.K))
    D[ok] = S["f"][ok] / C["f"][ok, None] - S["m"][ok] / C["m"][ok, None]
    return D.astype(np.float32), int(ok.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--xpair", required=True)
    ap.add_argument("--ft", required=True, help="FT の描画(d328_eval の出力: conv/p01*_CV.wav・spk.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    D, nok = unit_means(Path(a.xpair))
    print("units with both genders", nok, "mean |Δ| per unit row", float(np.abs(D).sum(1)[np.abs(D).sum(1) > 0].mean()), flush=True)
    np.save(O.parent / f"{O.name}_delta.npy", D) if O.parent.exists() else None
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    P.slow_from_ckpt(st)
    g = P.init_gen(dev, "table")
    g.load_state_dict(st["ema"])
    g.eval()
    Dt = torch.from_numpy(D).to(dev)
    rows = json.loads((P.H / "dev" / "devset.json").read_text())["pairs"]
    ed = P.H / "e2e" / "eval_dev"
    groups, pairs = [], []
    for al in a.alphas:
        od = O / f"A{al}"
        od.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for r in rows:
                k = f"p{r['pi']:04d}_t{r['ti']:03d}"
                c, lg = P.load_dump(ed, k)
                spk = P.load_enroll(P.H / "dev" / "d85_work" / f"en_{r['ti']:03d}")
                p = torch.from_numpy(lg.T.copy()).float().to(dev)[None].softmax(1)
                if al > 0:
                    p = (p + al * Dt[p.argmax(1)[0]].T[None]).clamp(min=0)
                    p = p / p.sum(1, keepdim=True)
                ct = torch.from_numpy(c.T.copy()).float().to(dev)[None]
                s2 = P.spk_tensor(spk, dev)
                cond = g.make(p, ct[:, 0:1], ct[:, 25:26], ct[:, 26:27], s2)
                f0 = torch.where(ct[:, 26] > 0.5, 200.0 * torch.exp(ct[:, 25]), torch.zeros_like(ct[:, 25]))
                exc = P.TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0))
                y = g(cond, exc, s2)[0].cpu().numpy()[N.DELAY:]
                sf.write(od / f"{k}_CV.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
        groups.append(f"MT_A{al}={od}/*_CV.wav")
        pairs.append(f"A{al}={Path(a.ft)}/conv/p01*_CV.wav|{od}/p00*_CV.wav|{Path(a.ft) / 'spk.json'}")
        print("rendered", al, flush=True)
    T = Path(__file__).parent
    r1 = subprocess.run([sys.executable, str(T / "eval_v1perp.py"), "--groups", *groups, "--out", str(O / "v1perp.json")], capture_output=True, text=True)
    print(r1.stdout[-3000:], r1.stderr[-1000:] if r1.returncode else "", flush=True)
    r2 = subprocess.run([sys.executable, str(T / "d326_pair.py"), "--perm", "100", "--pairs", *pairs, "--out", str(O / "pair.json")], capture_output=True, text=True)
    print(r2.stdout[-3000:], r2.stderr[-1000:] if r2.returncode else "", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
