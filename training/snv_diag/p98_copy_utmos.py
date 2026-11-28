import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import snv_dev_eval as DE
import snv_p1_eval as EV
from snv_a0_measure import to16

ROOT = Path(__file__).resolve().parent.parent.parent
FORMS = {
    "product": ("", {}),
    "product_nosm": ("", {"SNV_ESM": "0"}),
    "f3_0": ("results/snv_p98_f3_0/snap/ema_10k.pt", {}),
    "f3_0_know": ("results/snv_p98_f3_0/snap/ema_10k.pt", {"SNV_EK": "0.25,0.5,0.25,0"}),
    "f3_240": ("results/snv_p98_f3_240/snap/ema_10k.pt", {}),
    "y0": ("results/snv_p97_y0/snap/ema_30k.pt", {}),
    "y0_know": ("results/snv_p97_y0/snap/ema_30k.pt", {"SNV_EK": "0.25,0.5,0.25,0"}),
    "y240": ("results/snv_p97_y240/snap/ema_30k.pt", {}),
    "y240_know": ("results/snv_p97_y240/snap/ema_30k.pt", {"SNV_EK": "0.25,0.5,0.25,0"}),
    "n0": ("results/snv_p97_n0/snap/ema_30k.pt", {}),
    "n240": ("results/snv_p97_n240/snap/ema_30k.pt", {}),
    "s240": ("results/snv_p97_s240/snap/ema_30k.pt", {}),
}
CKS = {"y240": "results/snv_p97_y240/snap/ema_30k.pt", "n240": "results/snv_p97_n240/snap/ema_30k.pt", "y0": "results/snv_p97_y0/snap/ema_30k.pt", "f3_0": "results/snv_p98_f3_0/snap/ema_10k.pt", "f3_240": "results/snv_p98_f3_240/snap/ema_10k.pt", "n0": "results/snv_p97_n0/snap/ema_30k.pt"}
for _a, _c in CKS.items():
    for _m in ("uv", "v", "uvb", "all", "c0", "sh", "big", "small", "lo", "hi"):
        FORMS[f"{_a}_m{_m}"] = (_c, {"MIX": _m})


def mix(env: torch.Tensor, v: np.ndarray, mode: str) -> torch.Tensor:
    q, n = torch.nn.functional.pad(env, (2, 0), mode="replicate"), env.shape[-1]
    sm = 0.25 * q[..., 2 : 2 + n] + 0.5 * q[..., 1 : 1 + n] + 0.25 * q[..., :n]
    uv = v < 0.5
    if mode == "uvb":
        uv = uv | np.concatenate([uv[1:], uv[-1:]]) | np.concatenate([uv[:1], uv[:-1]])
    if mode in ("d1", "d2"):
        k = int(mode[1])
        return torch.cat([env[..., :1].expand(-1, -1, k), env[..., :-k]], -1)
    if mode in ("f1", "f2"):
        return env
    if mode in ("c0", "sh", "lo", "hi"):
        r = torch.zeros(25, dtype=torch.bool, device=env.device)
        r[{"c0": slice(0, 1), "sh": slice(1, 25), "lo": slice(1, 9), "hi": slice(9, 25)}[mode]] = True
        return torch.where(r[None, :, None], sm, env)
    if mode in ("big", "small"):
        c = env[0, 0].cpu().numpy()
        d = np.abs(np.diff(c, prepend=c[0]))
        b = d > np.percentile(d, 80)
        b = b | np.concatenate([b[:1], b[:-1]])
        return torch.where(torch.from_numpy(b if mode == "big" else ~b).to(env.device)[None, None], sm, env)
    m = {"uv": uv, "uvb": uv, "v": ~uv, "all": np.ones_like(uv)}[mode]
    return torch.where(torch.from_numpy(m).to(env.device)[None, None], sm, env)
SW = ["SNV_ESM", "SNV_EK", "SNV_EK0", "SNV_RCK"]


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    names = sys.argv[2].split(",") if len(sys.argv) > 2 else list(FORMS)
    st = sys.argv[3] if len(sys.argv) > 3 else "devA"
    out = ROOT / ("results/snv_p98/copy_utmos.json" if st == "devA" else f"results/snv_p98/copy_utmos_{st}.json")
    res = json.load(open(out)) if out.exists() else {}
    if st == "devA":
        items = DE.cv_items("devA", n)
    elif st.startswith("p73_"):
        import snv_p0_front as P
        import train_snv_p1 as T

        def utt(wp: str) -> dict:
            x = librosa.load(ROOT / wp, sr=EV.SR, mono=True)[0].astype(np.float32)
            x = x[: P.n_frames(len(x)) * P.H]
            d = P.analyze(x)
            f0 = d["f0"].astype(np.float32)
            return {"y": T.to_y(d["env"].astype(np.float32), d["per"].astype(np.float32), f0), "f0": f0, "wav": x}

        sets = json.loads((ROOT / "data/snv_p73/sets.json").read_text())[st[4:]]
        items = [{"src": f"{e['voice']}_{Path(c['wav']).stem}", "u": utt(c["wav"])} for e in sets for c in e["src"][:2]]
    else:
        import train_snv_p1 as T
        vs = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())["val"]
        items = [{"src": f"{v}_{t}", "u": T.load_utt(ROOT / f"data/snv_p_feat/e1/{v}/{t}.npz") | {"wav_path": str(ROOT / f"data/snv_e1/{v}/{t}.wav")}} for v in vs for t in ("t03", "t06", "t09")]
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(EV.DEV).eval()
    mos = lambda w: float(utmos(torch.from_numpy(to16(w.astype(np.float64), EV.SR).astype(np.float32))[None].to(EV.DEV), 16000).item())
    wav48 = lambda u: u["wav"] if "wav" in u else librosa.load(u["wav_path"], sr=EV.SR, mono=True)[0].astype(np.float32)
    if "orig" not in res:
        res["orig"] = {it["src"]: round(mos(wav48(it["u"])), 4) for it in items}
        print("orig", round(float(np.mean(list(res["orig"].values()))), 4), flush=True)
    for name in names:
        if "@" in name:
            tag, k = name.split("+")[0].split("@")
            x = name.split("+")[1] if "+" in name else ""
            FORMS[name] = (f"results/{tag}/snap/ema_{k}k.pt", {"SNV_EK": x[3:].replace(";", ",")} if x.startswith("ek:") else {"MIX": x} if x else {})
        ck, sw = FORMS[name]
        for k in SW:
            os.environ.pop(k, None)
        mode = sw.get("MIX")
        os.environ.update({k: x for k, x in sw.items() if k != "MIX"})
        if ck:
            os.environ["SNV_RCK"] = ck
        gen = EV.load_gen()
        r = {}
        for it in items:
            u = it["u"]
            y, f0 = u["y"], u["f0"]
            if mode in ("f1", "f2"):
                k = int(mode[1])
                f0 = np.concatenate([f0[k:], np.repeat(f0[-1:], k)])
                y = np.concatenate([y[:25], np.concatenate([y[25:, k:], np.repeat(y[25:, -1:], k, 1)], 1)], 0)
            v = (f0 > 0).astype(np.float32)
            lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
            cond = torch.from_numpy(np.concatenate([y[:25], lf[None], v[None], np.clip(y[25:29], 0, 1.5) * v[None]], 0))[None].to(EV.DEV)
            cond = torch.cat([mix(cond[:, :25], v, mode) if mode else EV.smooth_env(gen, cond[:, :25]), cond[:, 25:]], 1)
            w = EV.draw(gen, cond, torch.from_numpy(f0.astype(np.float32))[None].to(EV.DEV), wav48(u) if getattr(gen, "fast", 0) else None)
            r[it["src"]] = round(mos(w), 4)
        res[name] = r
        d = np.array([r[k] - res["product"][k] for k in r]) if "product" in res else np.zeros(1)
        print(name, "mean", round(float(np.mean(list(r.values()))), 4), "vs product", round(float(d.mean()), 4), "±", round(float(1.96 * d.std() / np.sqrt(len(d))), 4), flush=True)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    with torch.no_grad():
        main()
