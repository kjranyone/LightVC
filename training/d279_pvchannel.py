"""D279: diag_ppgvoc1 の男声 → 女声(PV)の通り道の切り分け(学習なし・推論の入力だけ差し替え)。
base = 学習の評価と同じ / c0lvl = c0 を目標の水準へ(登録の c0 の表で内容ごとに Σ_k p_k (t0_目標 − t0_操作者)・登録の定数で因果)/
c0glob = 一定のずらし(登録の単位の出現数で重みづけた t0 の平均の差)/ tau05 = 事後を τ 0.5 で鋭く(学習と食い違い・参考)/
c0comp / f0comp = c0 / log f0(有声)を因果な走行平均(1 フレーム 0.01・約 0.5 s)のまわりで 0.6 倍に縮める /
pbias<a> = 事後の話者ごとの偏りの補正: logit += a·(log 平均事後_目標の参照 − log 平均事後_操作者の参照)(登録の定数・D316)/ c0op = 操作者の登録の参照の有声の c0 の中央値を目標の参照の中央値へ移す(登録の定数の較正・D312)/ c0p<N> = c0 に一定の N/100 を足す(男声の入力の有声の c0 の中央値 3.6 対 女声 5.3 = 録音の水準の差 約 13 dB・診断の定数)。
    uv run python d279_pvchannel.py --ckpt <last_ema.pt のコピー> --out <dir>
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", default="base,c0lvl,c0glob,tau05")
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.CTX["on"] = bool(st.get("ctx", False))
    P.CTX["ch"] = int(st.get("ctx_ch", 192))
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    print("ckpt step", st["step"], flush=True)
    ed = P.H / "e2e" / "eval_dev"
    pairs = {p["pi"]: p for p in json.loads((P.H / "dev" / "devset.json").read_text())["pairs"]}
    enr = lambda d: json.loads((d / "enroll.json").read_text())
    groups = []
    for v in a.variants.split(","):
        od = O / v
        od.mkdir(parents=True, exist_ok=True)
        for f in sorted(ed.glob("p00*.cond")):
            pi, ti = int(f.stem[1:5]), int(f.stem.split("_t")[1][:3])
            c, lg = P.load_dump(ed, f.stem)
            en_t, en_o = P.H / "dev" / "d85_work" / f"en_{ti:03d}", P.H / "dev" / "d85_work" / f"op_{pi:04d}"
            spk = P.load_enroll(en_t)
            jt, jo = enr(en_t), enr(en_o)
            t0t, t0o = np.array(jt["t0"], np.float32), np.array(jo["t0"], np.float32)
            c = c.copy()
            P.PPG = "tau0.5" if v == "tau05" else "soft"
            if v.startswith("pbias"):
                al = float(v[5:] or 100) / 100

                def pb(fr: Path) -> np.ndarray:
                    lgr = np.fromfile(fr, "<f4").reshape(-1, 224)[:, :200]
                    q = torch.from_numpy(lgr).softmax(1).numpy()
                    return np.log(q.mean(0) + 1e-6)
                b_op = pb(P.H / "e2e" / "opref_dev" / f"op_{pi:04d}.front")
                b_tg = pb(P.H / "e2e" / "eval_dev" / f"ref_{ti:03d}.front")
                lg = lg + al * (b_tg - b_op)[None, :].astype(np.float32)
            if v == "c0lvl":
                p = torch.from_numpy(lg).softmax(1).numpy()
                c[:, 0] += p @ (t0t - t0o)
            elif v in ("c0comp", "f0comp", "c0exp"):
                j_ = 0 if v != "f0comp" else 25
                fac = 1.3 if v == "c0exp" else 0.6
                vo = c[:, 26] > 0.5
                m, out = None, c[:, j_].copy()
                for t in range(len(c)):
                    if vo[t] or v in ("c0comp", "c0exp"):
                        m = c[t, j_] if m is None else 0.99 * m + 0.01 * c[t, j_]
                        out[t] = m + fac * (c[t, j_] - m)
                c[:, j_] = out
            elif v == "c0op":
                cal = json.loads((P.H / "e2e" / "opref_dev" / "c0_med.json").read_text())
                c[:, 0] += cal["target_ref_median"] - cal["op"][f"op_{pi:04d}"]
            elif v.startswith("c0p"):
                c[:, 0] += float(v[3:]) / 100
            elif v == "c0glob":
                wt, wo = np.array(jt["counts"], np.float64), np.array(jo["counts"], np.float64)
                c[:, 0] += float((t0t * wt).sum() / wt.sum() - (t0o * wo).sum() / wo.sum())
            sf.write(od / f"{f.stem}_PV.wav", P.render(g, c, lg, spk, dev), N.SR)
        groups.append(f"{v}={od}/p*_PV.wav")
        print("rendered", v, flush=True)
    P.PPG = "soft"
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "eval_v1perp.py"), "--groups", *groups, "--out", str(O / "v1perp.json")], capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-1500:] if r.returncode else "", flush=True)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
