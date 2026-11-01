"""D464: D463 の事後の写像 f(男声の事後 → 女声の事後の分布・内容を保つ)を男声の操作者の入力にだけ当てて候補(P43OP)で描画し、男声の名残と自然さ・明瞭さが動くかを測る(推論の入口だけ・生成器はそのまま)。
組: dev(男声 TTS 60・f あり)・devf(同じ文の女声 TTS 60・f なし)→ v1⊥ と対の判別(D328 と同じ)・devvctkm(内容をそろえた VCTK: 男声 p01* に f・女声 p00* はなし)→ 対の判別・dev の CER。
    uv run python d464_postmap_render.py --ckpt ../results/diag_ppgvoc43/product_002500_op15.pt --map ../results/conv_p0/d463/postmap.pt --out ../results/conv_p0/d464
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
from d463_postmap import PostMap

T_ = Path(__file__).parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="PM")
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    P.slow_from_ckpt(st)
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    fm = PostMap().to(dev)
    fm.load_state_dict(torch.load(a.map, map_location="cpu", weights_only=False)["net"])
    fm.eval()

    @torch.no_grad()
    def mapped(lg: np.ndarray) -> np.ndarray:
        lp = torch.log_softmax(torch.from_numpy(lg.T.copy()).float().to(dev), 0)[None]
        return fm(lp)[0].cpu().numpy().T.copy()

    def rend(s: str, k: str, ti: int, use_map: bool) -> np.ndarray:
        c, lg = P.load_dump(P.H / "e2e" / f"eval_{s}", k)
        if use_map:
            lg = mapped(lg)
        return P.render(g, c, lg, P.load_enroll(P.H / s / "d85_work" / f"en_{ti:03d}"), dev, T_op=P.op_tab(P.H / s, k) if P.OPSLOW["on"] else None)

    od = O / "conv"
    od.mkdir(parents=True, exist_ok=True)
    for s_, off, um in (("dev", 0, True), ("devf", 100, False)):
        for r in json.loads((P.H / s_ / "devset.json").read_text())["pairs"]:
            k = f"p{r['pi']:04d}_t{r['ti']:03d}"
            sf.write(od / f"p{r['pi'] + off:04d}_t{r['ti']:03d}_CV.wav", rend(s_, k, r["ti"], um), N.SR)
    vd = O / "vctkm"
    vd.mkdir(parents=True, exist_ok=True)
    for f in sorted((P.H / "e2e" / "eval_devvctkm").glob("p0*.cond")):
        ti = int(f.stem.split("_t")[1][:3])
        pi = int(f.stem[1:5])
        sf.write(vd / f"{f.stem}_CV.wav", rend("devvctkm", f.stem, ti, pi >= 100), N.SR)
    print("rendered", flush=True)
    r1 = subprocess.run([sys.executable, str(T_ / "eval_v1perp.py"), "--groups", f"MT={od}/p00*_CV.wav", f"FT={od}/p01*_CV.wav", "--out", str(O / "v1perp.json")], capture_output=True, text=True)
    print(r1.stdout[-1500:], flush=True)
    spk = P.ROOT / "results/conv_p0/d438m_002500/spk.json"
    r2 = subprocess.run([sys.executable, str(T_ / "d326_pair.py"), "--perm", "100", "--pairs", f"TTSM_PM={od}/p01*_CV.wav|{od}/p00*_CV.wav|{spk}",
                         f"VCTKM_PM={vd}/p01*_CV.wav|{vd}/p00*_CV.wav|{P.H / 'devvctkm' / 'devset.json'}", "--out", str(O / "pair.json")], capture_output=True, text=True)
    print(r2.stdout[-1500:], r2.stderr[-800:] if r2.returncode else "", flush=True)
    cd = O / "cer_dev"
    cd.mkdir(parents=True, exist_ok=True)
    for f in od.glob("p00*_CV.wav"):
        dst = cd / f"{f.name[:10]}_{a.name}.wav"
        if not dst.exists():
            dst.symlink_to(f.resolve())
    for f in (P.H / "dev" / "d56").glob("p00*_SRC.wav"):
        dst = cd / f.name
        if not dst.exists():
            dst.symlink_to(f)
    r3 = subprocess.run([sys.executable, str(T_ / "d23_cer.py"), "--dirs", str(cd), "--names", "dev", "--out", str(O / "cer.json"), "--device", "cuda"], capture_output=True, text=True)
    print(r3.stdout[-600:], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
