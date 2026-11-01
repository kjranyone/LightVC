"""D341: 遅い音色の揺らぎの倍率 σ と乱数の種ごとに SR(保留 30)・FF(devff 30)・FF2(devff2 30)の v1⊥ を測る。揺らぎを学習していない ckpt にも --force_slow で同じ統計・同じ種の乱数を足す(推論だけの対照)。
種は SLOW["seed"](項目の長さを足して項目ごとに変わる)= 同じ種なら ckpt 間で同じ乱数。
    uv run python d341_sigma.py --ckpts PPG21=../results/diag_ppgvoc21/last_ema.pt PPG18=../results/diag_ppgvoc18/last_ema.pt --force_slow ../results/diag_ppgvoc21/slow_stats.npz --out ../results/conv_p0/d341
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P


def items(s_: str, held: list) -> list:
    if s_ == "sr":
        return P.eval_items(held, "table", "cuda")[0]
    return P.ff_items([s_])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", nargs="+", default=["sr", "devff", "devff2"])
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.0, 1.0, 1.5])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1000])
    ap.add_argument("--force_slow", default="")
    ap.add_argument("--causal", action="store_true", help="乱数の揺らぎを因果な最小位相 FIR で作る(製品の経路と同じ)")
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    held = P.e2e_held()
    its = {s_: items(s_, held) for s_ in a.sets}
    groups: dict = {s_: [] for s_ in a.sets}
    for spec in a.ckpts:
        name, ck = spec.split("=", 1)
        st = torch.load(ck, map_location="cpu", weights_only=False)
        P.PPG = st.get("ppg", "soft")
        P.CTX["on"], P.CTX["ch"], P.CTX["adapt"] = bool(st.get("ctx", False)), int(st.get("ctx_ch", 192)), bool(st.get("adapt", False))
        P.slow_from_ckpt(st)
        if not P.SLOW["on"] and a.force_slow:
            P.slow_load(a.force_slow, 1.0)
        P.SLOW["causal"] = a.causal
        if P.OPSLOW["on"]:
            raise SystemExit("d341_sigma は OPSLOW(操作者の遅い揺らぎ)に非対応: d444_ffgate で測る")
        g = P.init_gen(dev, st.get("spk", "table"))
        g.load_state_dict(st["ema"])
        g.eval()
        for sg in a.sigmas:
            for sd in (a.seeds if sg > 0 else a.seeds[:1]):
                P.SLOW["sigma"], P.SLOW["seed"] = sg, sd
                on = P.SLOW["on"]
                P.SLOW["on"] = on and sg > 0
                tag = f"{name}_s{sg}_r{sd}"
                for s_ in a.sets:
                    od = O / s_ / tag
                    od.mkdir(parents=True, exist_ok=True)
                    for nm, c, lg, spk in its[s_]:
                        sf.write(od / f"{nm}.wav", P.render(g, c, lg, spk, dev), N.SR)
                    groups[s_].append(f"{tag}={od}/*.wav")
                P.SLOW["on"] = on
                print("rendered", tag, flush=True)
    for s_ in a.sets:
        r = subprocess.run([sys.executable, str(Path(__file__).parent / "eval_v1perp.py"), "--groups", *groups[s_], "--out", str(O / f"{s_}_v1perp.json")], capture_output=True, text=True)
        print(s_, r.stdout[-6000:], r.stderr[-1500:] if r.returncode else "", flush=True)
    rows = {}
    for s_ in a.sets:
        j = json.loads((O / f"{s_}_v1perp.json").read_text())
        rows[s_] = {k: v["v1perp"]["logit"] for k, v in j.items()}
    (O / "summary.json").write_text(json.dumps(rows, indent=1))
    print(json.dumps(rows), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
