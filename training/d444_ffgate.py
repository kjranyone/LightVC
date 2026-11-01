"""D444: 女声 → 女声(実女声の入力)の解析の関門の束。候補を devff(90 組)・devff2(60 組)の全部で描画し、
(1) v1 / v1⊥(群の大きさを 30 にそろえる: 目標ごとに 1 本の部分集合 j = 0, 1, 2 ごと・D290)・(2) d109 v2 の 4 軸(f0・ガビ・こもり・目標との差)・
(3) CER(入力そのものの文字起こしを基準)・(4) 本人らしさ(d443 と同じ識別の余白)を測る。入力そのもの(SRC)・写し合成(CP = rvoc3hi で入力そのものの条件を鳴らす = 耳で合格した型の基準点)・製品・kNN-VC(あるもの)を同じ物差しで並べる。
描画は scratch の <set>/d444/<NAME>/ に(d109 が読む)。
    uv run python d444_ffgate.py --ckpt D2=../results/diag_ppgvoc41/product_002500_m0.pt --out ../results/conv_p0/d444_D2
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

T = Path(__file__).parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True, help="NAME=path")
    ap.add_argument("--sets", nargs="+", default=["devff", "devff2"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip_render", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    O = Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    names = [c.split("=", 1)[0] for c in a.ckpt]
    for s in a.sets:
        pairs = json.loads((P.H / s / "devset.json").read_text())["pairs"]
        for c in a.ckpt:
            nm, path = c.split("=", 1)
            od = P.H / s / "d444" / nm
            od.mkdir(parents=True, exist_ok=True)
            if a.skip_render and len(list(od.glob("p*.wav"))) == len(pairs):
                continue
            st = torch.load(path, map_location="cpu", weights_only=False)
            P.PPG = st.get("ppg", "soft")
            P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
            P.slow_from_ckpt(st)
            g = P.init_gen(dev, st.get("spk", "table"))
            g.load_state_dict(st["ema"])
            g.eval()
            for r in pairs:
                k = f"p{r['pi']:04d}_t{r['ti']:03d}"
                c_, lg = P.load_dump(P.H / "e2e" / f"eval_{s}", k)
                sf.write(od / f"{k}_{nm}.wav", P.render(g, c_, lg, P.load_enroll(P.H / s / "d85_work" / f"en_{r['ti']:03d}"), dev, T_op=P.op_tab(P.H / s, k) if P.OPSLOW["on"] else None), N.SR)
            del g
            torch.cuda.empty_cache()
        cpd = P.H / s / "d444" / "CP"
        cpd.mkdir(parents=True, exist_ok=True)
        if len(list(cpd.glob("p*_CP.wav"))) < len(pairs):
            import rvoc as R
            import train_rvoc as TR
            s_ = torch.load(P.INIT / "snap/ema_330k.pt", map_location="cpu", weights_only=False)
            gcp = R.RVoc(ch=s_["cfg"]["ch"], kernels=tuple(s_["cfg"]["kernels"]), dils=tuple(s_["cfg"]["dils"]), d_cond=s_["cfg"]["d_cond"]).to(dev)
            gcp.load_state_dict(s_["ema"])
            gcp.eval()
            with torch.no_grad():
                for r in pairs:
                    k = f"p{r['pi']:04d}_t{r['ti']:03d}"
                    c_, _ = P.load_dump(P.H / "e2e" / f"eval_{s}", k)
                    cond = torch.from_numpy(c_.T.astype("float32"))[None].to(dev)
                    f0 = torch.where(cond[:, 26] > 0.5, 200.0 * torch.exp(cond[:, 25]), torch.zeros_like(cond[:, 25]))
                    yy = gcp(cond, TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
                    sf.write(cpd / f"{k}_CP.wav", yy / (abs(yy).max() + 1e-9) * 0.9, N.SR)
            del gcp
        print("rendered", s, len(pairs), flush=True)
    rep: dict = {}
    for s in a.sets:
        pairs = json.loads((P.H / s / "devset.json").read_text())["pairs"]
        by_t: dict = {}
        for r in pairs:
            by_t.setdefault(r["ti"], []).append(f"p{r['pi']:04d}_t{r['ti']:03d}")
        nsub = min(len(v) for v in by_t.values())
        sysdirs = {nm: (P.H / s / "d444" / nm, nm) for nm in names} | {"SRC": (P.H / s / "d56", "SRC"), "CP": (P.H / s / "d444" / "CP", "CP")}
        if (P.H / s / "d135_real").exists():
            sysdirs["KNN"] = (P.H / s / "d135_real", "KNN")
        if (P.H / s / "d114").exists() and list((P.H / s / "d114").glob("p*_G03S10ZH.wav")):
            sysdirs["PROD"] = (P.H / s / "d114", "G03S10ZH")
        groups = []
        for nm, (d, suf) in sysdirs.items():
            for j in range(nsub):
                sd = O / s / "sub" / f"{nm}_{j}"
                sd.mkdir(parents=True, exist_ok=True)
                for t, ks in by_t.items():
                    src = d / f"{sorted(ks)[j]}_{suf}.wav"
                    dst = sd / src.name
                    if src.exists() and not dst.exists():
                        dst.symlink_to(src)
                groups.append(f"{nm}_{j}={sd}/p*.wav")
        r1 = subprocess.run([sys.executable, str(T / "eval_v1perp.py"), "--groups", *groups, "--out", str(O / f"{s}_v1perp.json")], capture_output=True, text=True)
        if r1.returncode:
            print(r1.stderr[-1500:], flush=True)
        j1 = json.loads((O / f"{s}_v1perp.json").read_text())
        v1 = {}
        for nm in sysdirs:
            ks = [k for k in j1 if k.rsplit("_", 1)[0] == nm]
            v1[nm] = {m: {"auc_mean": round(sum(j1[k][m]["auc"] for k in ks) / len(ks), 4), "logit_mean": round(sum(j1[k][m]["logit"] for k in ks) / len(ks), 3),
                          "logit_each": [j1[k][m]["logit"] for k in ks]} for m in ("v1", "v1perp")}
        rep[s] = {"v1": v1}
        print(s, "v1", json.dumps({k: (v["v1"]["auc_mean"], v["v1"]["logit_mean"]) for k, v in v1.items()}), flush=True)
        ex = [f"d444/{nm}:{nm}" for nm in names] + [f"d56:SRC", "d444/CP:CP"] + ([f"d135_real:KNN"] if "KNN" in sysdirs else []) + ([f"d114:G03S10ZH"] if "PROD" in sysdirs else [])
        r2 = subprocess.run([sys.executable, str(T / "d109_axis_gate.py"), "--s", str(P.H / s), "--no_base", "--tag", f"d444_{s}_{'_'.join(names)}", "--extra", *ex], capture_output=True, text=True)
        if r2.returncode:
            print(r2.stderr[-1500:], flush=True)
        j2 = json.loads((P.ROOT / "results/conv_p0" / f"d444_{s}_{'_'.join(names)}.json").read_text())
        rep[s]["axes"] = {k: {"axes": v["axes"], "fail": {st: x["median"] for st, x in v["stats"].items() if not x["pass"]}} for k, v in j2.items() if isinstance(v, dict) and "axes" in v}
        print(s, "axes", json.dumps(rep[s]["axes"], ensure_ascii=False), flush=True)
        cd = O / s / "cer"
        cd.mkdir(parents=True, exist_ok=True)
        for nm, (d, suf) in sysdirs.items():
            if nm == "SRC":
                continue
            for f in d.glob(f"p*_{suf}.wav"):
                dst = cd / f"{f.name[:10]}_{nm}.wav"
                if not dst.exists():
                    dst.symlink_to(f.resolve())
        for f in (P.H / s / "d56").glob("p*_SRC.wav"):
            dst = cd / f.name
            if not dst.exists():
                dst.symlink_to(f)
        r3 = subprocess.run([sys.executable, str(T / "d23_cer.py"), "--dirs", str(cd), "--names", s, "--out", str(O / f"{s}_cer.json"), "--device", "cuda"], capture_output=True, text=True)
        if r3.returncode:
            print(r3.stderr[-1500:], flush=True)
        print(s, "cer", r3.stdout[-800:], flush=True)
    (O / "summary.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
