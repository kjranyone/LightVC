"""A5(diag)の評価。事前登録 = results/diag_snv_a5/prereg.yaml。
dev1 = 段 1 の途中信号(開発の組 E1V = 学習に無い 30 声): a5_1・c1(A4 の 400 回の当てはめ)・x1 の音素の無い発声の TEX と普通の発話の top-1 → 段 2 へ進むか(dev1.json)。
test = 最終の判定(評価の組 A0F = 女 24 声): a5_1・a5_2・x1・x2・c2 の S・M・CER・自己の reamp の音素の無い発声 336 本の TEX(a5_eval.json)。
    cd training && uv run --no-sync python snv_a5_eval.py dev1|test [smoke]
smoke = 声・本数を絞り、段 2 の名前に段 1 の重みを当てて経路だけを通す(結果は書かない)。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch

from snv_a0_measure import DEV, ROOT, cer, kana, load, to16
from snv_a0b_measure import proj
from snv_a2_eval import A1, map_pitch, vocoder
from snv_a4_eval import A0, N0R, cluster_ci
from snv_a4b_fit import ENROLL_T, HELD_T, N_E1
from snv_n0_measure import trim
from snv_n2c_tex import GROUPS, tex_stats
from train_snv_a2 import FEAT, G, playing
from train_snv_a4 import C, fit_phi
from train_snv_a5 import Fit

OUT = ROOT / "results/diag_snv_a5"
TAGS = {"a5_1": "diag_snv_a5_s1", "a5_2": "diag_snv_a5_s2", "c1": "diag_snv_a4_c", "c2": "diag_snv_a4_c_gan", "x1": "diag_snv_a4_x", "x2": "diag_snv_a4_x_gan"}
ID_TOL = 0.05
GO_FRAC = 0.5
CER_GUARD = 0.05


def main() -> int:
    mode = sys.argv[1]
    smoke = len(sys.argv) > 2 and sys.argv[2] == "smoke"
    systems = ("a5_1", "c1", "x1") if mode == "dev1" else ("a5_1", "a5_2", "x1", "x2", "c2")
    tags = dict(TAGS)
    if smoke:
        tags.update({"a5_1": sys.argv[3], "a5_2": sys.argv[3]})
    OUT.mkdir(parents=True, exist_ok=True)
    scale = np.load(ROOT / "results/snv_n2c/tex_scale.npy")
    nets = {}
    for s in systems:
        ck = torch.load(ROOT / "results" / tags[s] / "last.pt", map_location="cpu", weights_only=True)
        net = (G("x") if ck["mode"] == "x" else C()).to(DEV).eval()
        net.load_state_dict(ck["g"])
        for p in net.parameters():
            p.requires_grad_(False)
        fit = None
        if ck["mode"] == "a5":
            fit = Fit().to(DEV).eval()
            fit.load_state_dict(ck["fit"])
            for p in fit.parameters():
                p.requires_grad_(False)
        nz = np.load(ROOT / "results" / tags[s] / "norm.npz")
        nets[s] = (net, fit, ck["mode"], nz["mu"], nz["sd"])
    mu, sd = nets["x1"][3], nets["x1"][4]
    assert all(np.allclose(v[3], mu) and np.allclose(v[4], sd) for v in nets.values()), "系どうしで演奏の正規化が違う"
    voc = vocoder()
    render = lambda mel: voc(mel).squeeze().detach().cpu().numpy()

    def get_phi(s: str, rp: np.ndarray, rmel: np.ndarray):
        net, fit, kind, _, _ = nets[s]
        rp_t = torch.from_numpy((rp - mu) / sd)[None].to(DEV)
        rm_t = torch.from_numpy(rmel)[None].to(DEV)
        mask = torch.ones(1, rmel.shape[1], dtype=torch.bool, device=DEV)
        if kind == "x":
            with torch.no_grad():
                return net.structure(rm_t, rp_t, mask), None
        phi = fit(net, rp_t, rm_t, mask) if kind == "a5" else fit_phi(net, rp_t, rm_t, mask)
        return phi, float(phi.pow(2).mean().sqrt())

    def forward(s: str, p: np.ndarray, phi) -> np.ndarray:
        with torch.no_grad():
            return render(nets[s][0](torch.from_numpy((p - mu) / sd)[None].to(DEV), phi))

    def tex_row(w44: np.ndarray, tst: np.ndarray) -> dict:
        d = tex_stats(to16(w44, 44100)) - tst
        r = {"tex": float(np.mean(np.abs(d) / scale))} | {g: float(np.mean(np.abs(d[sl]) / scale[sl])) for g, sl in GROUPS.items()}
        r["gain_db"] = float(d[GROUPS["mean"]].mean() * 10 / np.log(10))
        r["mod_signed"] = d[GROUPS["mod"]].reshape(32, 4).mean(0).tolist()
        return r

    from speechbrain.inference.speaker import EncoderClassifier
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})

    def emb(y16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(y16).float()[None].to(DEV)).reshape(-1)
        return torch.nn.functional.normalize(e, dim=0).cpu().numpy()

    res: dict = {"prereg": "results/diag_snv_a5/prereg.yaml", "mode": mode, "tags": {s: tags[s] for s in systems}, "sys": {}}
    if mode == "dev1":
        names_tr = json.loads((ROOT / "results/diag_snv_a4_c/train_log.json").read_text())["names_train"]
        man = sorted(v["voice"] for v in json.loads((ROOT / "data/snv_e1/manifest.json").read_text())["voices"])
        e1v = [v for v in man if v not in names_tr][-N_E1:]
        a5_val = json.loads((ROOT / "results" / tags["a5_1"] / "train_log.json").read_text()).get("names_val") if not smoke else None
        assert smoke or sorted(a5_val) == sorted(e1v), "開発の組が A5 の検証の声と違う"
        nvman = [it for it in json.loads((ROOT / "data/snv_e1nv/manifest.json").read_text())["items"] if it["k"] == 0 and it["voice"] in e1v]
        if smoke:
            e1v = e1v[:3]
            nvman = [it for it in nvman if it["voice"] in e1v]
        enroll, items = {}, []
        for v in e1v:
            ms = [np.load(FEAT / f"mel/e1/{v}/{t}.npy").astype(np.float32) for t in ENROLL_T]
            ps = [playing(FEAT / f"sparc/e1/{v}/{t}.npz", m.shape[1]) for t, m in zip(ENROLL_T, ms)]
            enroll[v] = (np.concatenate(ms, 1), np.concatenate(ps, 0), emb(np.concatenate([to16(*load(ROOT / f"data/snv_e1/{v}/{t}.wav")) for t in ENROLL_T])))
            items += [("normal", v, f"{t}", FEAT / f"mel/e1/{v}/{t}.npy", FEAT / f"sparc/e1/{v}/{t}.npz", None) for t in HELD_T]
        for it in nvman:
            n = f"{it['cat']}_0"
            items.append(("nv", it["voice"], n, FEAT / f"mel/e1nv/{it['voice']}/{n}.npy", FEAT / f"sparc/e1nv/{it['voice']}/{n}.npz",
                          tex_stats(to16(*load(ROOT / f"data/snv_e1nv/{it['voice']}/{n}.wav")))))
        per: dict = defaultdict(dict)
        for s in systems:
            phis, rms, top = {}, [], []
            for v, (rm, rp, _) in enroll.items():
                phis[v], r = get_phi(s, rp, rm)
                if r is not None:
                    rms.append(r)
            for kind, v, n, mp, spp, tst in items:
                w = forward(s, playing(spp, np.load(mp).shape[1]), phis[v])
                if kind == "nv":
                    per[s][(v, n)] = tex_row(w, tst)["tex"]
                else:
                    e = emb(trim(to16(w, 44100)))
                    top.append(max(e1v, key=lambda u: float(e @ enroll[u][2])) == v)
            res["sys"][s] = {"nv_tex": round(float(np.mean(list(per[s].values()))), 4), "normal_top1": round(float(np.mean(top)), 3),
                             "phi_rms": round(float(np.median(rms)), 4) if rms else None}
            print("dev1", s, res["sys"][s], flush=True)
        keys = sorted(per["x1"])
        grp = np.array([k[0] for k in keys])
        for s in ("a5_1", "c1"):
            d = np.array([per[s][k] - per["x1"][k] for k in keys])
            res["sys"][s]["d_tex_minus_x1"] = {"mean": round(float(d.mean()), 4), "ci95_voice_cluster": cluster_ci(d, grp, 4000)}
        da, dc = res["sys"]["a5_1"]["d_tex_minus_x1"]["mean"], res["sys"]["c1"]["d_tex_minus_x1"]["mean"]
        go_tex = da <= GO_FRAC * max(dc, 0.0)
        go_id = res["sys"]["a5_1"]["normal_top1"] >= res["sys"]["x1"]["normal_top1"] - ID_TOL
        res["rule"] = {"gap_closed_frac": round(1 - da / dc, 3) if dc > 0 else None, "go_tex": go_tex, "go_id": go_id, "go_stage2": go_tex and go_id}
        print("rule", res["rule"], flush=True)
        if not smoke:
            (OUT / "dev1.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        return 0

    meta = json.loads((A1 / "convert_meta.json").read_text())
    srcs, tgts = meta["sources"], meta["targets"]
    drows = json.loads((A1 / "diag/meta.json").read_text())
    tstats = {v: np.load(FEAT / f"sparc/a1src/tref_{v}.npz")["pitch_stats"].reshape(-1)[:2] for v in tgts}
    n0r = [it for it in json.loads((N0R / "items.json").read_text()) if it["sex"] == "F"]
    if smoke:
        srcs, drows = srcs[:1], drows[:4]
        n0r = [it for it in n0r if it["voice"] in tgts[:2]][:6]
    T = np.load(FEAT / "ecapa_lda_T.npz")
    Tt = (T["mu"], T["scalings"])
    from faster_whisper import WhisperModel
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16")

    def tr(w: np.ndarray) -> str:
        segs, _ = asr.transcribe(w, language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(x.text for x in segs))

    a0_tst = {f"{it['voice']}_{it['cat']}_{it['k']}": tex_stats(to16(*load(FEAT / f"wav44/n0r__{it['voice']}_{it['cat']}_{it['k']}.wav"))) for it in n0r}
    Et = {v: proj(emb(np.concatenate([to16(*load(A0 / f"wav/{v}_{j}.wav")) for j in (0, 1, 2)])), Tt) for v in tgts}
    ref_txt = {s: tr(sf.read(str(A1 / f"src/{s}.wav"))[0].astype(np.float32)) for s in srcs}
    per = defaultdict(dict)
    for s in systems:
        phis, rms = {}, []
        for v in tgts:
            rmel = np.load(FEAT / f"mel/a0enroll/{v}.npy").astype(np.float32)
            phis[v], r = get_phi(s, playing(FEAT / f"sparc/a1src/tref_{v}.npz", rmel.shape[1]), rmel)
            if r is not None:
                rms.append(r)
        rows = []
        for it in n0r:
            n = f"{it['voice']}_{it['cat']}_{it['k']}"
            w = forward(s, playing(FEAT / f"sparc/n0r/{n}.npz", np.load(FEAT / f"mel/n0r/{n}.npy").shape[1]), phis[it["voice"]])
            r = tex_row(w, a0_tst[n])
            per[s][n] = r["tex"]
            rows.append(r)
        sh = []
        for r in drows:
            n_mel = np.load(FEAT / f"mel/a1diag/D0_{r['target']}_{r['g']}.npy").shape[1]
            w = forward(s, playing(FEAT / f"sparc/a1diag/D0_{r['target']}_{r['g']}.npz", n_mel), phis[r["target"]])
            e = proj(emb(trim(to16(w, 44100))), Tt)
            sh.append(max(tgts, key=lambda u: float(e @ Et[u])) == r["target"])
        mh, cers = [], []
        for src in srcs:
            f = FEAT / f"sparc/a1src/{src}.npz"
            st = np.load(f)["pitch_stats"].reshape(-1)[:2]
            for v in tgts:
                w16 = to16(forward(s, map_pitch(playing(f, int(10 * 44100 / 512) + 1), st, tstats[v]), phis[v]), 44100)
                e = proj(emb(trim(w16)), Tt)
                mh.append(max(tgts, key=lambda u: float(e @ Et[u])) == v)
                if v in tgts[:8]:
                    cers.append(cer(ref_txt[src], tr(w16)))
        res["sys"][s] = {"nv_tex": round(float(np.mean([r["tex"] for r in rows])), 4),
                         "nv_groups": {g: round(float(np.mean([r[g] for r in rows])), 4) for g in GROUPS},
                         "nv_gain_db": round(float(np.mean([r["gain_db"] for r in rows])), 3),
                         "nv_mod_signed_4_8_16_32_50": [round(float(x), 4) for x in np.mean([r["mod_signed"] for r in rows], 0)],
                         "S": round(float(np.mean(sh)), 3), "M": round(float(np.mean(mh)), 3), "M_cer_median_8tgt": round(float(np.median(cers)), 3),
                         "phi_rms": round(float(np.median(rms)), 4) if rms else None}
        print("test", s, res["sys"][s], flush=True)
    names = sorted(per["x2"])
    grp = np.array([n.split("_")[0] for n in names])

    def diff(a_: str, b_: str) -> dict:
        d = np.array([per[a_][n] - per[b_][n] for n in names])
        return {"mean": round(float(d.mean()), 4), "ci95_voice_cluster": cluster_ci(d, grp, 4000)}

    res["diff"] = {f"{a_}_minus_{b_}": diff(a_, b_) for a_, b_ in (("a5_2", "x2"), ("a5_2", "c2"), ("a5_1", "x1"), ("a5_2", "a5_1"), ("x2", "x1"), ("c2", "x2"))}
    t, x = res["sys"]["a5_2"], res["sys"]["x2"]
    dx, dc = res["diff"]["a5_2_minus_x2"], res["diff"]["a5_2_minus_c2"]
    v = {"H_id_ok": t["S"] >= x["S"] - ID_TOL and t["M"] >= x["M"] - ID_TOL,
         "H_ext": "H_circuit_ext" if dx["mean"] < 0 and dx["ci95_voice_cluster"][1] < 0 else "H_table_ext" if dx["mean"] > 0 and dx["ci95_voice_cluster"][0] > 0 else "H_equal_ext",
         "H_a5_gain": dc["ci95_voice_cluster"][1] < 0,
         "guard_cer_ok": t["M_cer_median_8tgt"] - res["sys"]["a5_1"]["M_cer_median_8tgt"] <= CER_GUARD,
         "H_gan_ok_record": res["diff"]["a5_2_minus_a5_1"]["ci95_voice_cluster"][1] < 0}
    res["verdict"] = v
    print("verdict", v, flush=True)
    if not smoke:
        (OUT / "a5_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
