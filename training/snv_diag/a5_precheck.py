"""A5 の起動前の検査 I の探り(学習なし・記録だけ): A4 の c2 の重みを固定し、A5 の当てはめの形(初期値 + 層ごとに正規化した勾配の K 歩・歩幅は固定)で
開発の組 E1V(学習に無い 30 声)の Φ を作り、普通の発話の本人らしさ(30 声中の top-1)と音素の無い発声の TEX を A4b の iso100 / iso400 / x2 と並べる。
問い = 8 歩の固定の歩みで声を運べるか(運べなければ A5 の内側の予算が足りない)。出力 = results/diag_snv_a5/precheck.json。
round2 = 二階の走行が発散した後(2026-11-03)、一階で使う固定の減衰する歩幅の列を学習の声の平均の Φ から始めて比べる。出力 = results/diag_snv_a5/precheck2.json。
    cd training && uv run --no-sync python snv_diag/a5_precheck.py [round2]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snv_a0_measure import DEV, ROOT, load, to16
from snv_a2_eval import vocoder
from snv_a4b_fit import ENROLL_T, HELD_T, N_E1
from snv_n0_measure import trim
from snv_n2c_tex import tex_stats
from train_snv_a2 import FEAT, WARM, playing
from train_snv_a4 import CH, N_LAYER, N_MEL, PHI, C

OUT = ROOT / "results/diag_snv_a5"
K = 8
VARIANTS = (("z0_a03", 0, [0.03] * K), ("z0_a05", 0, [0.05] * K), ("z0_a08", 0, [0.08] * K), ("mu_a05", 1, [0.05] * K))
VARIANTS2 = (("mu_a03", 1, [0.03] * K), ("mu_d10_70", 1, [0.10 * 0.7 ** k for k in range(K)]), ("mu_d08_75", 1, [0.08 * 0.75 ** k for k in range(K)]),
             ("mu_d15_65", 1, [0.15 * 0.65 ** k for k in range(K)]), ("mu_d20_60", 1, [0.20 * 0.6 ** k for k in range(K)]))


def fit_norm(net: C, rp: torch.Tensor, rmel: torch.Tensor, phi0: torch.Tensor, sched: list[float]) -> torch.Tensor:
    w = torch.ones(1, 1, rmel.shape[2], device=DEV)
    w[..., :WARM] = 0.0
    phi = phi0.clone()
    for a in sched:
        phi.requires_grad_(True)
        loss = ((net(rp, phi) - rmel).abs() * w).sum() / (w.sum() * N_MEL)
        g, = torch.autograd.grad(loss, phi)
        g = g.view(1, N_LAYER, 2 * CH)
        phi = (phi - a * (g / (g.pow(2).mean(-1, keepdim=True) + 1e-14).sqrt()).view(1, PHI)).detach()
    return phi


def main() -> int:
    r2 = len(sys.argv) > 1 and sys.argv[1] == "round2"
    OUT.mkdir(parents=True, exist_ok=True)
    scale = np.load(ROOT / "results/snv_n2c/tex_scale.npy")
    tag = "diag_snv_a4_c_gan"
    ck = torch.load(ROOT / "results" / tag / "last.pt", map_location="cpu", weights_only=True)
    net = C().to(DEV).eval()
    net.load_state_dict(ck["g"])
    for p in net.parameters():
        p.requires_grad_(False)
    nz = np.load(ROOT / "results" / tag / "norm.npz")
    mu, sd = nz["mu"], nz["sd"]
    codes = ck["codes"]["weight"].to(DEV)
    pm = codes.mean(0, keepdim=True)
    names_tr = json.loads((ROOT / "results" / tag / "train_log.json").read_text())["names_train"]
    man = sorted(v["voice"] for v in json.loads((ROOT / "data/snv_e1/manifest.json").read_text())["voices"])
    e1v = [v for v in man if v not in names_tr][-N_E1:]
    nvman = [it for it in json.loads((ROOT / "data/snv_e1nv/manifest.json").read_text())["items"] if it["k"] == 0 and it["voice"] in e1v]
    voc = vocoder()
    from speechbrain.inference.speaker import EncoderClassifier
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})

    def emb(y16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(y16).float()[None].to(DEV)).reshape(-1)
        return torch.nn.functional.normalize(e, dim=0).cpu().numpy()

    enroll, items = {}, []
    for v in e1v:
        ms = [np.load(FEAT / f"mel/e1/{v}/{t}.npy").astype(np.float32) for t in ENROLL_T]
        ps = [playing(FEAT / f"sparc/e1/{v}/{t}.npz", m.shape[1]) for t, m in zip(ENROLL_T, ms)]
        enroll[v] = (torch.from_numpy((np.concatenate(ps, 0) - mu) / sd)[None].to(DEV), torch.from_numpy(np.concatenate(ms, 1))[None].to(DEV),
                     emb(np.concatenate([to16(*load(ROOT / f"data/snv_e1/{v}/{t}.wav")) for t in ENROLL_T])))
        items += [("normal", v, FEAT / f"mel/e1/{v}/{t}.npy", FEAT / f"sparc/e1/{v}/{t}.npz", None) for t in HELD_T]
    for it in nvman:
        n = f"{it['cat']}_0"
        items.append(("nv", it["voice"], FEAT / f"mel/e1nv/{it['voice']}/{n}.npy", FEAT / f"sparc/e1nv/{it['voice']}/{n}.npz",
                      tex_stats(to16(*load(ROOT / f"data/snv_e1nv/{it['voice']}/{n}.wav")))))
    res: dict = {"note": "記録だけ・A5 の起動前の検査 I の探り(A4 の c2 の重みは固定・歩幅は学ばない)", "K": K,
                 "ref_a4b_dev": {"iso100": {"nv_tex": 0.694, "normal_top1": 0.978, "phi_rms": 0.310}, "iso400": {"nv_tex": 0.715, "normal_top1": 0.978, "phi_rms": 0.599},
                                 "x2": {"nv_tex": 0.606, "normal_top1": 0.978}}, "learned_phi_rms_median": round(float(codes.pow(2).mean(1).sqrt().median()), 4), "variants": {}}
    if r2:
        res["note"] = "記録だけ・二階の走行の発散の後に一階の固定の歩幅の列を選ぶ(A4 の c2 の重みは固定)"
        res["round1"] = json.loads((OUT / "precheck.json").read_text())["variants"]
    for name, init, a in (VARIANTS2 if r2 else VARIANTS):
        phis, rms, fl = {}, [], []
        for v, (rp, rmel, _) in enroll.items():
            phis[v] = fit_norm(net, rp, rmel, pm if init else torch.zeros(1, PHI, device=DEV), a)
            rms.append(float(phis[v].pow(2).mean().sqrt()))
            with torch.no_grad():
                fl.append(float((net(rp, phis[v]) - rmel)[..., WARM:].abs().mean()))
        top, txs = [], []
        for kind, v, mp, spp, tst in items:
            p = torch.from_numpy((playing(spp, np.load(mp).shape[1]) - mu) / sd)[None].to(DEV)
            with torch.no_grad():
                w = voc(net(p, phis[v])).squeeze().cpu().numpy()
            if kind == "nv":
                txs.append(float(np.mean(np.abs(tex_stats(to16(w, 44100)) - tst) / scale)))
            else:
                e = emb(trim(to16(w, 44100)))
                top.append(max(e1v, key=lambda u: float(e @ enroll[u][2])) == v)
        res["variants"][name] = {"sched": [round(x, 4) for x in a], "nv_tex": round(float(np.mean(txs)), 4), "normal_top1": round(float(np.mean(top)), 3),
                                 "phi_rms": round(float(np.median(rms)), 4), "enroll_l1": round(float(np.mean(fl)), 4)}
        print(name, res["variants"][name], flush=True)
    (OUT / ("precheck2.json" if r2 else "precheck.json")).write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
