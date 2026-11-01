"""A2b の項目ごとの読み(diag_snv_a2_xs/a2b_items.json)を作る。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import soundfile as sf
import torch
from snv_a0_measure import DEV, ROOT, Embedders, load, to16
from snv_a0b_measure import fit_lda, proj, splits
from snv_a2_eval import map_pitch, vocoder
from train_snv_a2 import FEAT, G, playing

A1 = ROOT / "results/snv_a1"
A0 = ROOT / "results/snv_a0"
meta = json.loads((A1 / "convert_meta.json").read_text())
srcs, tgts = meta["sources"], meta["targets"]
drows = json.loads((A1 / "diag/meta.json").read_text())
sp = splits()
emb = Embedders()
X, yy = [], []
for i, s in enumerate(sp["lda"]):
    for p in sp["pool"][s]:
        X.append(emb(to16(*load(p)))["ecapa"]); yy.append(i)
T = fit_lda(np.stack(X), np.array(yy))
Et = {v: proj(emb(np.concatenate([to16(*load(A0 / f"wav/{v}_{j}.wav")) for j in (0, 1, 2)]))["ecapa"], T) for v in tgts}
voc = vocoder()
tstats = {v: np.load(FEAT / f"sparc/a1src/tref_{v}.npz")["pitch_stats"].reshape(-1)[:2] for v in tgts}
items = {}
for tag in ("diag_snv_a2_xs", "diag_snv_a2_x"):
    d = ROOT / "results" / tag
    ck = torch.load(d / "last.pt", map_location="cpu", weights_only=True)
    g = G(ck["mode"]).to(DEV).eval(); g.load_state_dict(ck["g"])
    nz = np.load(d / "norm.npz"); mu, sd = nz["mu"], nz["sd"]
    with torch.no_grad():
        S = {}
        for v in tgts:
            rmel = np.load(FEAT / f"mel/a0enroll/{v}.npy").astype(np.float32)
            rp = (playing(FEAT / f"sparc/a1src/tref_{v}.npz", rmel.shape[1]) - mu) / sd
            S[v] = g.structure(torch.from_numpy(rmel)[None].to(DEV), torch.from_numpy(rp)[None].to(DEV), torch.ones(1, rmel.shape[1], dtype=torch.bool, device=DEV))

        def top1(p_npz, n_mel, v, stats):
            p = playing(p_npz, n_mel)
            if stats is not None:
                p = map_pitch(p, stats, tstats[v])
            y = g(torch.from_numpy((p - mu) / sd)[None].to(DEV), S[v])
            w = voc(y).squeeze().cpu().numpy()
            e = proj(emb(to16(w.astype(np.float64), 44100))["ecapa"], T)
            return max(tgts, key=lambda u: float(e @ Et[u]))

        for r in drows:
            v, gi = r["target"], r["g"]
            n_mel = np.load(FEAT / f"mel/a1diag/D0_{v}_{gi}.npy").shape[1]
            items[f"{tag}|S|{v}|{gi}"] = top1(FEAT / f"sparc/a1diag/D0_{v}_{gi}.npz", n_mel, v, None) == v
            o = FEAT / f"sparc/a1diag/src_{r['d2_src']}_{gi}.npz"
            items[f"{tag}|O|{v}|{gi}"] = top1(o, n_mel, v, np.load(o)["pitch_stats"].reshape(-1)[:2]) == v
        for s in srcs:
            f = FEAT / f"sparc/a1src/{s}.npz"
            for v in tgts:
                items[f"{tag}|M|{v}|{s}"] = top1(f, int(10 * 44100 / 512) + 1, v, np.load(f)["pitch_stats"].reshape(-1)[:2]) == v
    del g
out = ROOT / "results/diag_snv_a2_xs/a2b_items.json"
out.write_text(json.dumps({k: bool(b) for k, b in items.items()}, indent=0))
r_g = json.loads((ROOT / "results/diag_snv_a2_g/a2_eval.json").read_text())
gvec = np.array(r_g["S_diag_snv_a2_g_vec"]).reshape(24, 3)
rng = np.random.default_rng(0)

def vec(tag, c):
    ks = sorted(k for k in items if k.startswith(f"{tag}|{c}|"))
    by = {}
    for k in ks:
        by.setdefault(k.split("|")[2], []).append(float(items[k]))
    return np.array([by[v] for v in sorted(by)])

res = {}
for c in ("S", "O", "M"):
    a, b = vec("diag_snv_a2_x", c), vec("diag_snv_a2_xs", c)
    d = (a - b)
    bs = [d[rng.integers(0, 24, 24)].mean() for _ in range(10000)]
    res[f"{c}_x_minus_xs"] = {"x": float(a.mean()), "xs": float(b.mean()), "diff": float(d.mean()), "ci95_voice": [float(x) for x in np.percentile(bs, [2.5, 97.5])], "x_only": int(((a == 1) & (b == 0)).sum()), "xs_only": int(((a == 0) & (b == 1)).sum())}
xs_S = vec("diag_snv_a2_xs", "S")
d = xs_S - gvec
bs = [d[rng.integers(0, 24, 24)].mean() for _ in range(10000)]
res["S_xs_minus_g"] = {"diff": float(d.mean()), "ci95_voice": [float(x) for x in np.percentile(bs, [2.5, 97.5])], "xs_only": int(((xs_S == 1) & (gvec == 0)).sum()), "g_only": int(((xs_S == 0) & (gvec == 1)).sum())}
(ROOT / "results/diag_snv_a2_xs/a2b_ci.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
