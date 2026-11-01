"""話者を跨いだ状態の対応: 注意が読んだ参照の調音の状態(SPARC の EMA・物差しとしてだけ)と、今の入力の調音の状態の相関。S = 本人の演奏・O = 別の女声の演奏・M = 本物の男声。
A2 の x(注意の鍵 = EMA)・A3 の n・A3c(鍵 = 自前の E)。"""
import json, sys
import numpy as np, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from snv_a0_measure import DEV, ROOT
from snv_a2_eval import map_pitch
from snv_a3_eval import A1, male_segments, wav44
from train_snv_a2 import FEAT, G, playing
from train_snv_a3 import HOP, Model, envelope

meta = json.loads((A1 / "convert_meta.json").read_text())
srcs, tgts = meta["sources"], meta["targets"]
drows = json.loads((A1 / "diag/meta.json").read_text())
males = male_segments(srcs)
tstats = {v: np.load(FEAT / f"sparc/a1src/tref_{v}.npz")["pitch_stats"].reshape(-1)[:2] for v in tgts}
rng = np.random.default_rng(0)


def items():
    for r in drows:
        v, gi, o = r["target"], r["g"], r["d2_src"]
        n = np.load(FEAT / f"mel/a1diag/D0_{v}_{gi}.npy").shape[1]
        yield "S", v, FEAT / f"wav44/a1diag__D0_{v}_{gi}.wav", FEAT / f"sparc/a1diag/D0_{v}_{gi}.npz", n, False
        yield "O", v, FEAT / f"wav44/a1diag__D0_{o}_{gi}.wav", FEAT / f"sparc/a1diag/src_{o}_{gi}.npz", n, True
    for s in srcs[::3]:
        for v in tgts[::3]:
            yield "M", v, s, FEAT / f"sparc/a1src/{s}.npz", len(males[s]) // HOP, True


def corr(pa, q):
    c = [np.corrcoef(pa[:, j], q[:, j])[0, 1] for j in range(12)]
    perm = rng.permutation(len(q))
    f = [np.corrcoef(pa[perm, j], q[:, j])[0, 1] for j in range(12)]
    return float(np.nanmean(c)), float(np.nanmean(f))


res = {}
for tag in ("diag_snv_a2_x", "diag_snv_a3_n", "diag_snv_a3c_w", "diag_snv_a3d_pre_g", "diag_snv_a3d_pre_s", "diag_snv_a3d_pre_sc"):
    d = ROOT / "results" / tag
    ck = torch.load(d / "last.pt", map_location="cpu", weights_only=True)
    a2 = "g" in ck and "mode" in ck
    if a2:
        m = G(ck["mode"]).to(DEV).eval()
        m.load_state_dict(ck["g"])
    else:
        m = Model().to(DEV).eval()
        m.load_state_dict(ck["model"])
    nz = np.load(d / "norm.npz")
    mu, sd = nz["mu"], nz["sd"]
    out = {}
    with torch.no_grad():
        S = {}
        for v in tgts:
            rmel = np.load(FEAT / f"mel/a0enroll/{v}.npy").astype(np.float32)
            n = rmel.shape[1]
            rp = playing(FEAT / f"sparc/a1src/tref_{v}.npz", n)
            mk = torch.ones(1, n, dtype=torch.bool, device=DEV)
            if a2:
                S[v] = (m.structure(torch.from_numpy(rmel)[None].to(DEV), torch.from_numpy((rp - mu) / sd)[None].to(DEV), mk), rp[:, :12])
            else:
                rw = wav44(FEAT / f"wav44/a0enroll__{v}.wav", n)
                S[v] = (m.structure(torch.from_numpy(rmel)[None].to(DEV), envelope(torch.from_numpy(rw)[None].to(DEV)), torch.from_numpy((rp[:, 12:15] - mu) / sd)[None].to(DEV), mk), rp[:, :12])
        for c, v, y, pz, n, z in items():
            p = playing(pz, n)
            q = map_pitch(p, np.load(pz)["pitch_stats"].reshape(-1)[:2], tstats[v]) if z else p
            if a2:
                _, att = m(torch.from_numpy((q - mu) / sd)[None].to(DEV), S[v][0], return_attn=True)
            else:
                w = males[y] if c == "M" else wav44(y, n)
                _, att = m(envelope(torch.from_numpy(w)[None].to(DEV)), torch.from_numpy((q[:, 12:15] - mu) / sd)[None].to(DEV), S[v][0], return_attn=True)
            a = att[0].double().cpu().numpy()
            pa = np.einsum("htr,rd->htd", a, S[v][1]).mean(0)
            out.setdefault(c, []).append(corr(pa, p[:, :12]))
    res[tag] = {c: {"corr": round(float(np.mean([x[0] for x in v_])), 3), "shuffled": round(float(np.mean([x[1] for x in v_])), 3), "n": len(v_)} for c, v_ in out.items()}
    print(tag, res[tag], flush=True)
    del m
    torch.cuda.empty_cache()
