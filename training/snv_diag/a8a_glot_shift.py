"""A8a(学習なし・事前検査): 演奏 PG の声門の 9 次元を、操作者の登録の平均から目標の登録の平均へずらすだけで、本物の男声 → 目標の M が戻るか(results/snv_a8a/prereg.yaml)。
基準 = 有声のフレーム(SPARC の周期性 ≥ 0.5)の次元ごとの平均・標準偏差。操作者 = 元の mp3 の先頭 30 s(glot/a1src/{源}_enroll)・O の女声の源 = その声の登録(tref)・目標 = tref。
変種 raw / shift / affine を A7ax2(主)・A7ac2(記録)で、SPARC だけの A4x2 / A4c2 を参考値に。次元ごとの shift・M_alt・M7・O_read_as_source は記録。出力 = results/snv_a8a/a8a.json。
    cd training && uv run --no-sync python snv_diag/a8a_glot_shift.py [smoke]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snv_a0_measure import DEV, ROOT, cer, kana, load, to16
from snv_a0b_measure import proj
from snv_a2_eval import A1, map_pitch, vocoder
from snv_a4_eval import A0, cluster_ci
from snv_glot import ENR_T, G_DIM, NAMES, SRC_T
from snv_n0_measure import trim
from train_snv_a2 import FEAT, P_DIM, G
from train_snv_a4 import FIT_STEPS, C, fit_phi, play

OUT = ROOT / "results/snv_a8a"
SYS = {"A7ax2": ("diag_snv_a7_ax_gan", 100), "A7ac2": ("diag_snv_a7_ac_gan", 100), "A4x2": ("diag_snv_a4_x_gan", FIT_STEPS), "A4c2": ("diag_snv_a4_c_gan", FIT_STEPS)}
VOICED = 0.5
SEMI = 7
CER_GUARD = 0.05
O_GUARD = -0.05
NEAR_P = 0.05
GL = slice(P_DIM, P_DIM + G_DIM)


def baseline(glot: Path, sp: Path, n: int) -> tuple[np.ndarray, np.ndarray]:
    g = np.load(glot).astype(np.float64)
    v = play(sp, n)[:, 14] >= VOICED
    assert g.shape == (n, G_DIM) and v.sum() > 200, (glot, g.shape, int(v.sum()))
    return g[v].mean(0), g[v].std(0) + 1e-6


def adjust(p: np.ndarray, kind: str, bs: tuple, bt: tuple, dims: list | None = None) -> np.ndarray:
    if kind == "raw":
        return p
    q = p.copy()
    g = q[:, GL].astype(np.float64)
    (ms, ss), (mt, st) = bs, bt
    new = g + (mt - ms) if kind == "shift" else (g - ms) / ss * st + mt
    keep = np.ones(G_DIM, bool) if dims is None else np.isin(np.arange(G_DIM), dims)
    q[:, GL] = np.where(keep, new, g).astype(np.float32)
    return q


def main() -> int:
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((A1 / "convert_meta.json").read_text())
    srcs, tgts = meta["sources"], meta["targets"]
    drows = json.loads((A1 / "diag/meta.json").read_text())
    cer_t = tgts[:8]
    dim_list = list(range(G_DIM))
    if smoke:
        srcs, drows, cer_t, dim_list = [srcs[0], srcs[-1]], drows[:4], tgts[:1], [0, 7]
    conv_t = tgts[:3] if smoke else tgts
    nets = {}
    for s, (tag, fit) in SYS.items():
        ck = torch.load(ROOT / "results" / tag / "last.pt", map_location="cpu", weights_only=True)
        pdim = ck.get("pdim", P_DIM)
        assert pdim == (P_DIM + G_DIM if s.startswith("A7") else P_DIM), (s, pdim)
        net = (G("x", pdim=pdim) if ck["mode"] == "x" else C(pdim=pdim)).to(DEV).eval()
        net.load_state_dict(ck["g"])
        for prm in net.parameters():
            prm.requires_grad_(False)
        nz = np.load(ROOT / "results" / tag / "norm.npz")
        assert len(nz["mu"]) == pdim
        nets[s] = (net, ck["mode"], nz["mu"], nz["sd"], pdim > P_DIM, fit)
    voc = vocoder()

    def pg(rel: str, n: int, glot: bool) -> np.ndarray:
        return play(FEAT / f"sparc/{rel}.npz", n, glot)

    def get_phi(s: str, v: str):
        net, kind, mu, sd, gl, fit = nets[s]
        rmel = np.load(FEAT / f"mel/a0enroll/{v}.npy").astype(np.float32)
        rp_t = torch.from_numpy((pg(f"a1src/tref_{v}", rmel.shape[1], gl) - mu) / sd)[None].to(DEV)
        rm_t = torch.from_numpy(rmel)[None].to(DEV)
        mask = torch.ones(1, rmel.shape[1], dtype=torch.bool, device=DEV)
        if kind == "x":
            with torch.no_grad():
                return net.structure(rm_t, rp_t, mask)
        return fit_phi(net, rp_t, rm_t, mask, steps=fit)

    def forward(s: str, p: np.ndarray, phi) -> np.ndarray:
        net, _, mu, sd, _, _ = nets[s]
        with torch.no_grad():
            return voc(net(torch.from_numpy((p - mu) / sd)[None].to(DEV), phi)).squeeze().cpu().numpy()

    from speechbrain.inference.speaker import EncoderClassifier
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})
    T = np.load(FEAT / "ecapa_lda_T.npz")
    Tt = (T["mu"], T["scalings"])

    def emb(y16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(y16).float()[None].to(DEV)).reshape(-1)
        return proj(torch.nn.functional.normalize(e, dim=0).cpu().numpy(), Tt)

    Et = {v: emb(np.concatenate([to16(*load(A0 / f"wav/{v}_{j}.wav")) for j in (0, 1, 2)])) for v in tgts}
    Et2 = {v: emb(np.concatenate([to16(*load(A0 / f"wav/{v}_{j}.wav")) for j in (6, 7)])) for v in tgts}
    top = lambda e, ref: max(tgts, key=lambda u: float(e @ ref[u]))

    from faster_whisper import WhisperModel
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16")

    def tr(w: np.ndarray) -> str:
        segs, _ = asr.transcribe(w, language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(x.text for x in segs))

    ref_txt = {s: tr(sf.read(str(A1 / f"src/{s}.wav"))[0].astype(np.float32)) for s in srcs}
    tstats = {v: np.load(FEAT / f"sparc/a1src/tref_{v}.npz")["pitch_stats"].reshape(-1)[:2] for v in tgts}
    base_t = {v: baseline(FEAT / f"glot/a1src/tref_{v}.npy", FEAT / f"sparc/a1src/tref_{v}.npz", np.load(FEAT / f"mel/a0enroll/{v}.npy").shape[1]) for v in tgts}
    base_op = {nm: baseline(FEAT / f"glot/a1src/{nm}_enroll.npy", FEAT / f"sparc/a1src/{nm}_enroll.npz", ENR_T) for nm in ("uirou", "namikawa")}
    res: dict = {"prereg": "results/snv_a8a/prereg.yaml", "smoke": smoke, "tags": {s: t for s, (t, _) in SYS.items()},
                 "baseline_gap_mean_target_minus_operator": {nm: dict(zip(NAMES, np.round(np.mean([base_t[v][0] for v in tgts], 0) - b[0], 3).tolist())) for nm, b in base_op.items()},
                 "baseline_target_sd_over_voices": dict(zip(NAMES, np.round(np.std([base_t[v][0] for v in tgts], 0), 3).tolist())),
                 "sys": {}}
    print("baseline gap", json.dumps(res["baseline_gap_mean_target_minus_operator"], ensure_ascii=False), flush=True)
    hits: dict = {}

    def run_m(s: str, phis: dict, kind: str, dims: list | None = None, semi: bool = False, do_cer: bool = False, alt: bool = False) -> dict:
        gl = nets[s][4]
        h, h2, cs, grp = [], [], [], []
        for src in srcs:
            stt = np.load(FEAT / f"sparc/a1src/{src}.npz")["pitch_stats"].reshape(-1)[:2]
            p0 = pg(f"a1src/{src}", SRC_T, gl)
            for v in conv_t:
                if semi:
                    p = p0.copy()
                    p[:, 12] += SEMI * np.log(2) / 12
                else:
                    p = map_pitch(p0, stt, tstats[v])
                if gl:
                    p = adjust(p, kind, base_op[src.rsplit("_", 1)[0]], base_t[v], dims)
                w16 = to16(forward(s, p, phis[v]), 44100)
                e = emb(trim(w16))
                h.append(top(e, Et) == v)
                grp.append(v)
                if alt:
                    h2.append(top(e, Et2) == v)
                if do_cer and v in cer_t:
                    cs.append(cer(ref_txt[src], tr(w16)))
        out = {"M": round(float(np.mean(h)), 3), "n_M": len(h)}
        if alt:
            out["M_alt_j67"] = round(float(np.mean(h2)), 3)
        if do_cer:
            out["cer_median_8tgt"] = round(float(np.median(cs)), 3)
        return out, np.array(h, float), np.array(grp)

    def run_o(s: str, phis: dict, kind: str) -> dict:
        gl = nets[s][4]
        h, hs = [], []
        for r in drows:
            v, o = r["target"], f"D0_{r['d2_src']}_{r['g']}"
            n = np.load(FEAT / f"mel/a1diag/{o}.npy").shape[1]
            p = map_pitch(pg(f"a1diag/{o}", n, gl), np.load(FEAT / f"sparc/a1diag/{o}.npz")["pitch_stats"].reshape(-1)[:2], tstats[v])
            if gl:
                p = adjust(p, kind, base_t[r["d2_src"]], base_t[v])
            e = emb(trim(to16(forward(s, p, phis[v]), 44100)))
            t1 = top(e, Et)
            h.append(t1 == v)
            hs.append(t1 == r["d2_src"])
        return {"O": round(float(np.mean(h)), 3), "O_read_as_source": round(float(np.mean(hs)), 3), "n_O": len(h)}

    for s in SYS:
        phis = {v: get_phi(s, v) for v in sorted(set(conv_t) | {r["target"] for r in drows})}
        gl = nets[s][4]
        main_x = s == "A7ax2"
        kinds = ("raw", "shift", "affine") if gl else ("raw",)
        row = {}
        for kd in kinds:
            r, h, grp = run_m(s, phis, kd, do_cer=s in ("A7ax2", "A4x2"), alt=s in ("A7ax2", "A4x2") and kd != "affine")
            hits[(s, kd)] = (h, grp)
            if s in ("A7ax2", "A4x2"):
                r |= run_o(s, phis, kd)
            if s in ("A7ax2", "A4x2") and kd != "affine":
                r["M7"] = run_m(s, phis, kd, semi=True)[0]["M"]
            row[kd] = r
            print(s, kd, r, flush=True)
        if main_x:
            row["per_dim_shift_M"] = {}
            for k in dim_list:
                m = run_m(s, phis, "shift", dims=[k])[0]["M"]
                row["per_dim_shift_M"][NAMES[k]] = m
                print(s, "shift only", NAMES[k], m, flush=True)
        res["sys"][s] = row
        if nets[s][1] == "c":
            del phis
            torch.cuda.empty_cache()

    hx, grp = hits[("A7ax2", "raw")]
    dm = {kd: {"mean": round(float((hits[("A7ax2", kd)][0] - hx).mean()), 4), "ci95_target_cluster": cluster_ci(hits[("A7ax2", kd)][0] - hx, grp)} for kd in ("shift", "affine")}
    dm_c = {kd: {"mean": round(float((hits[("A7ac2", kd)][0] - hits[("A7ac2", "raw")][0]).mean()), 4), "ci95_target_cluster": cluster_ci(hits[("A7ac2", kd)][0] - hits[("A7ac2", "raw")][0], grp)} for kd in ("shift", "affine")}
    x = res["sys"]["A7ax2"]
    m_p = res["sys"]["A4x2"]["raw"]["M"]
    guards = {"cer": round(x["shift"]["cer_median_8tgt"] - x["raw"]["cer_median_8tgt"], 3), "o": round(x["shift"]["O"] - x["raw"]["O"], 3)}
    g_ok = guards["cer"] <= CER_GUARD and guards["o"] >= O_GUARD
    lo = dm["shift"]["ci95_target_cluster"][0]
    if not g_ok:
        verdict = "INVALID"
    elif lo > 0 and x["shift"]["M"] >= m_p - NEAR_P:
        verdict = "H_baseline"
    elif lo > 0:
        verdict = "H_partial"
    else:
        verdict = "H_dynamic"
    res["summary"] = {"dM_x": dm, "dM_c_record": dm_c, "M_P_A4x2": m_p, "M_x_shift": x["shift"]["M"], "guards": guards | {"ok": g_ok}, "verdict": verdict}
    print("summary", json.dumps(res["summary"], ensure_ascii=False), flush=True)
    if not smoke:
        (OUT / "a8a.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
