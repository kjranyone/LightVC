"""A4b(diag・学習なし): A4 の c2 の重みを固定し、登録での Φ の当てはめ方だけを替える(0 から・学習の Φ の平均から・学習の Φ の主成分の部分空間)。
事前登録 = results/diag_snv_a4b/prereg.yaml。開発の組 E1V で 1 つ選び、評価の組 A0F で x2 と 1 度だけ比べる。出力 = results/diag_snv_a4b/a4b.json。
    cd training && uv run --no-sync python snv_a4b_fit.py [smoke]
smoke = 声・本数・変種を絞って経路だけを通す(結果は書かない)。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from snv_a0_measure import DEV, ROOT, cer, kana, load, to16
from snv_a0b_measure import proj
from snv_a2_eval import A1, map_pitch, vocoder
from snv_a4_eval import A0, N0R, cluster_ci
from snv_n0_measure import trim
from snv_n2c_tex import GROUPS, tex_stats
from train_snv_a2 import FEAT, WARM, G, playing
from train_snv_a4 import C, FIT_LR, N_MEL, PRIOR, fit_phi

OUT = ROOT / "results/diag_snv_a4b"
VARIANTS = ("iso100", "iso400", "mu400", "pca32_400", "pca128_400", "pca569_400")
ENROLL_T = ("t00", "t01", "t02")
HELD_T = ("t07", "t08", "t09")
N_E1 = 30
ID_TOL = 0.05
FIX_HI = 0.03


def fit_sub(net: C, rp: torch.Tensor, rmel: torch.Tensor, base: torch.Tensor, B: torch.Tensor | None, steps: int) -> torch.Tensor:
    w = torch.ones(1, 1, rmel.shape[2], device=rmel.device)
    w[..., :WARM] = 0.0
    a = torch.zeros(1, B.shape[0] if B is not None else base.shape[1], device=rp.device, requires_grad=True)
    opt = torch.optim.Adam([a], lr=FIT_LR)
    for _ in range(steps):
        phi = base + (a @ B if B is not None else a)
        loss = ((net(rp, phi) - rmel).abs() * w).sum() / (w.sum() * N_MEL) + PRIOR * a.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.zero_grad(set_to_none=True)
    with torch.no_grad():
        return base + (a @ B if B is not None else a)


def main() -> int:
    global VARIANTS
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    OUT.mkdir(parents=True, exist_ok=True)
    scale = np.load(ROOT / "results/snv_n2c/tex_scale.npy")
    meta = json.loads((A1 / "convert_meta.json").read_text())
    srcs, tgts = meta["sources"], meta["targets"]
    drows = json.loads((A1 / "diag/meta.json").read_text())
    tstats = {v: np.load(FEAT / f"sparc/a1src/tref_{v}.npz")["pitch_stats"].reshape(-1)[:2] for v in tgts}
    n0r = [it for it in json.loads((N0R / "items.json").read_text()) if it["sex"] == "F"]
    names_tr = json.loads((ROOT / "results/diag_snv_a4_c_gan/train_log.json").read_text())["names_train"]
    man = sorted(v["voice"] for v in json.loads((ROOT / "data/snv_e1/manifest.json").read_text())["voices"])
    e1v = [v for v in man if v not in names_tr][-N_E1:]
    nvman = [it for it in json.loads((ROOT / "data/snv_e1nv/manifest.json").read_text())["items"] if it["k"] == 0 and it["voice"] in e1v]

    if smoke:
        VARIANTS = ("iso400", "pca32_400")
        e1v, srcs, drows = e1v[:3], srcs[:1], drows[:4]
        nvman = [it for it in nvman if it["voice"] in e1v]
        n0r = [it for it in n0r if it["voice"] in tgts[:2]][:6]
    voc = vocoder()
    render = lambda mel: voc(mel).squeeze().detach().cpu().numpy()
    nets = {}
    for s, tag in (("c2", "diag_snv_a4_c_gan"), ("x2", "diag_snv_a4_x_gan")):
        ck = torch.load(ROOT / "results" / tag / "last.pt", map_location="cpu", weights_only=True)
        net = (C() if ck["mode"] == "c" else G("x")).to(DEV).eval()
        net.load_state_dict(ck["g"])
        for p in net.parameters():
            p.requires_grad_(False)
        nz = np.load(ROOT / "results" / tag / "norm.npz")
        nets[s] = (net, nz["mu"], nz["sd"], ck.get("codes"))
    assert np.allclose(nets["c2"][1], nets["x2"][1]) and np.allclose(nets["c2"][2], nets["x2"][2])
    mu, sd = nets["c2"][1], nets["c2"][2]
    codes = nets["c2"][3]["weight"].to(DEV)
    pm = codes.mean(0, keepdim=True)
    U, S, Vh = torch.linalg.svd(codes - pm, full_matrices=False)
    std = S / np.sqrt(codes.shape[0] - 1)
    learned_rms = codes.pow(2).mean(1).sqrt()

    def get_phi(var: str, rp: torch.Tensor, rmel: torch.Tensor) -> torch.Tensor | dict:
        net = nets["c2"][0]
        mask = torch.ones(1, rmel.shape[2], dtype=torch.bool, device=DEV)
        if var == "x2":
            with torch.no_grad():
                return nets["x2"][0].structure(rmel, rp, mask)
        if var.startswith("iso"):
            return fit_phi(net, rp, rmel, mask, steps=int(var[3:]))
        if var == "mu400":
            return fit_sub(net, rp, rmel, pm, None, 400)
        k = int(var[3:].split("_")[0])
        return fit_sub(net, rp, rmel, pm, std[:k, None] * Vh[:k], 400)

    def forward(var: str, p: np.ndarray, phi) -> np.ndarray:
        net = nets["x2" if var == "x2" else "c2"][0]
        with torch.no_grad():
            return render(net(torch.from_numpy((p - mu) / sd)[None].to(DEV), phi))

    from speechbrain.inference.speaker import EncoderClassifier
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})
    T = np.load(FEAT / "ecapa_lda_T.npz")
    Tt = (T["mu"], T["scalings"])

    def emb(y16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(y16).float()[None].to(DEV)).reshape(-1)
        return torch.nn.functional.normalize(e, dim=0).cpu().numpy()

    from faster_whisper import WhisperModel
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16")

    def tr(w: np.ndarray) -> str:
        segs, _ = asr.transcribe(w, language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(x.text for x in segs))

    def tex_row(w44: np.ndarray, tst: np.ndarray) -> dict:
        o = tex_stats(to16(w44, 44100))
        d = o - tst
        r = {"tex": float(np.mean(np.abs(d) / scale))} | {g: float(np.mean(np.abs(d[s]) / scale[s])) for g, s in GROUPS.items()}
        r["gain_db"] = float(d[GROUPS["mean"]].mean() * 10 / np.log(10))
        r["mod_signed"] = d[GROUPS["mod"]].reshape(32, 4).mean(0).tolist()
        return r

    e1_enroll, e1_items = {}, {}
    for v in e1v:
        ms = [np.load(FEAT / f"mel/e1/{v}/{t}.npy").astype(np.float32) for t in ENROLL_T]
        ps = [playing(FEAT / f"sparc/e1/{v}/{t}.npz", m.shape[1]) for t, m in zip(ENROLL_T, ms)]
        e1_enroll[v] = (np.concatenate(ms, 1), np.concatenate(ps, 0), emb(np.concatenate([to16(*load(ROOT / f"data/snv_e1/{v}/{t}.wav")) for t in ENROLL_T])))
        for t in HELD_T:
            e1_items[("normal", v, t)] = (FEAT / f"mel/e1/{v}/{t}.npy", FEAT / f"sparc/e1/{v}/{t}.npz", None)
    for it in nvman:
        n = f"{it['cat']}_0"
        e1_items[("nv", it["voice"], n)] = (FEAT / f"mel/e1nv/{it['voice']}/{n}.npy", FEAT / f"sparc/e1nv/{it['voice']}/{n}.npz",
                                           tex_stats(to16(*load(ROOT / f"data/snv_e1nv/{it['voice']}/{n}.wav"))))
    a0_tst = {f"{it['voice']}_{it['cat']}_{it['k']}": tex_stats(to16(*load(FEAT / f"wav44/n0r__{it['voice']}_{it['cat']}_{it['k']}.wav"))) for it in n0r}
    Et = {v: proj(emb(np.concatenate([to16(*load(A0 / f"wav/{v}_{j}.wav")) for j in (0, 1, 2)])), Tt) for v in tgts}
    ref_txt = {s: tr(sf.read(str(A1 / f"src/{s}.wav"))[0].astype(np.float32)) for s in srcs}
    print("prep done", len(e1_items), len(a0_tst), flush=True)

    res: dict = {"prereg": "results/diag_snv_a4b/prereg.yaml", "learned_phi_rms": {"median": round(float(learned_rms.median()), 4)},
                 "pca_var_frac": {k: round(float((S[:k] ** 2).sum() / (S ** 2).sum()), 4) for k in (32, 128, 569)}, "dev": {}, "test": {}}
    for var in VARIANTS + ("x2",):
        phis, rms, top, txs = {}, [], [], []
        for v, (rm, rp, _) in e1_enroll.items():
            phis[v] = get_phi(var, torch.from_numpy((rp - mu) / sd)[None].to(DEV), torch.from_numpy(rm)[None].to(DEV))
            if var != "x2":
                rms.append(float(phis[v].pow(2).mean().sqrt()))
        for (kind, v, n), (mp, spp, tst) in e1_items.items():
            w = forward(var, playing(spp, np.load(mp).shape[1]), phis[v])
            if kind == "nv":
                txs.append(tex_row(w, tst)["tex"])
            else:
                e = emb(trim(to16(w, 44100)))
                top.append(max(e1v, key=lambda u: float(e @ e1_enroll[u][2])) == v)
        res["dev"][var] = {"nv_tex": round(float(np.mean(txs)), 4), "normal_top1": round(float(np.mean(top)), 3), "phi_rms": round(float(np.median(rms)), 4) if rms else None}
        print("dev", var, res["dev"][var], flush=True)
    floor = res["dev"]["iso400"]["normal_top1"] - ID_TOL
    ok = [v for v in VARIANTS if res["dev"][v]["normal_top1"] >= floor]
    sel = min(ok, key=lambda v: res["dev"][v]["nv_tex"])
    res["selected"] = {"variant": sel, "eligible": ok, "id_floor": round(floor, 3)}
    print("selected", res["selected"], flush=True)

    per: dict = defaultdict(dict)
    for var in VARIANTS + ("x2",):
        phis, rms = {}, []
        for v in tgts:
            rmel = np.load(FEAT / f"mel/a0enroll/{v}.npy").astype(np.float32)
            rp = playing(FEAT / f"sparc/a1src/tref_{v}.npz", rmel.shape[1])
            phis[v] = get_phi(var, torch.from_numpy((rp - mu) / sd)[None].to(DEV), torch.from_numpy(rmel)[None].to(DEV))
            if var != "x2":
                rms.append(float(phis[v].pow(2).mean().sqrt()))
        rows = []
        for it in n0r:
            n = f"{it['voice']}_{it['cat']}_{it['k']}"
            w = forward(var, playing(FEAT / f"sparc/n0r/{n}.npz", np.load(FEAT / f"mel/n0r/{n}.npy").shape[1]), phis[it["voice"]])
            r = tex_row(w, a0_tst[n])
            per[var][n] = r["tex"]
            rows.append(r)
        sh = []
        for r in drows:
            n_mel = np.load(FEAT / f"mel/a1diag/D0_{r['target']}_{r['g']}.npy").shape[1]
            w = forward(var, playing(FEAT / f"sparc/a1diag/D0_{r['target']}_{r['g']}.npz", n_mel), phis[r["target"]])
            e = proj(emb(trim(to16(w, 44100))), Tt)
            sh.append(max(tgts, key=lambda u: float(e @ Et[u])) == r["target"])
        mh, cers = [], []
        for s in srcs:
            f = FEAT / f"sparc/a1src/{s}.npz"
            st = np.load(f)["pitch_stats"].reshape(-1)[:2]
            for v in tgts:
                w = forward(var, map_pitch(playing(f, int(10 * 44100 / 512) + 1), st, tstats[v]), phis[v])
                w16 = to16(w, 44100)
                e = proj(emb(trim(w16)), Tt)
                mh.append(max(tgts, key=lambda u: float(e @ Et[u])) == v)
                if v in tgts[:8]:
                    cers.append(cer(ref_txt[s], tr(w16)))
        res["test"][var] = {"nv_tex": round(float(np.mean([r["tex"] for r in rows])), 4),
                            "nv_groups": {g: round(float(np.mean([r[g] for r in rows])), 4) for g in GROUPS},
                            "nv_gain_db": round(float(np.mean([r["gain_db"] for r in rows])), 3),
                            "nv_mod_signed_4_8_16_32_50": [round(float(x), 4) for x in np.mean([r["mod_signed"] for r in rows], 0)],
                            "S": round(float(np.mean(sh)), 3), "M": round(float(np.mean(mh)), 3), "M_cer_median_8tgt": round(float(np.median(cers)), 3),
                            "phi_rms": round(float(np.median(rms)), 4) if rms else None}
        print("test", var, res["test"][var], flush=True)
    names = sorted(per["x2"])
    grp = np.array([n.split("_")[0] for n in names])
    for var in VARIANTS:
        d = np.array([per[var][n] - per["x2"][n] for n in names])
        res["test"][var]["d_tex_minus_x2"] = {"mean": round(float(d.mean()), 4), "ci95_voice_cluster": cluster_ci(d, grp, 4000)}
        di = np.array([per[var][n] - per["iso400"][n] for n in names])
        res["test"][var]["d_tex_minus_iso400"] = {"mean": round(float(di.mean()), 4), "ci95_voice_cluster": cluster_ci(di, grp, 4000)}
    t, x = res["test"][sel], res["test"]["x2"]
    id_ok = t["S"] >= x["S"] - ID_TOL and t["M"] >= x["M"] - ID_TOL
    if t["d_tex_minus_x2"]["ci95_voice_cluster"][1] < FIX_HI and id_ok:
        res["verdict"] = "H_fit_fixes"
    elif sel != "iso400" and t["d_tex_minus_iso400"]["ci95_voice_cluster"][1] < 0:
        res["verdict"] = "H_fit_partial"
    else:
        res["verdict"] = "H_fit_none"
    res["id_ok_selected"] = id_ok
    if not smoke:
        (OUT / "a4b.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("verdict", res["verdict"], "selected", sel, t["d_tex_minus_x2"], "id_ok", id_ok, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
