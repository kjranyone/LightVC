"""A7(diag)の評価。事前登録 = results/diag_snv_a7/prereg.yaml。演奏 PG(SPARC 15 + 声門 9)で、叫び・喘ぎ・混ざりを学習から抜いた c / x(A7h)と、全種類で学んだ c / x(A7a)を並べ、
叫びでの c − x が見た種類での c − x よりどれだけ動くか(交互作用)を A7a の同じ量と比べる(DiD_scream・主)。抜いた 3 種まとめての DiD は記録。c の登録の当てはめは 100 回。
dev1 = 段 1 の途中信号(開発の組 E1V 30 声 × 7 種 × 1 本)→ dev1.json。test = 段 2 の判定(A0F の女 24 声の N0r 336 本・A7h の系の S・M・CER)→ a7_eval.json。
    cd training && uv run --no-sync python snv_a7_eval.py dev1|test [smoke]
smoke = 声・本数を絞り、学習の名前の検査を飛ばして経路だけを通す(結果は書かない)。
"""
from __future__ import annotations

import json
import sys

import numpy as np
import soundfile as sf
import torch

from snv_a0_measure import DEV, ROOT, cer, kana, load, to16
from snv_a0b_measure import proj
from snv_a2_eval import A1, map_pitch, vocoder
from snv_a4_eval import A0, N0R, cluster_ci
from snv_a4b_fit import ENROLL_T, HELD_T, N_E1
from snv_n0_measure import trim
from snv_n2c_tex import tex_stats
from train_snv_a2 import FEAT, G
from train_snv_a4 import C, fit_phi, play

OUT = ROOT / "results/diag_snv_a7"
HELD = ("scream", "moan", "mixed")
FOCUS = ("scream",)
FIT = 100
PDIM = 24
TAGS = {"A7hc1": "diag_snv_a7_hc", "A7hx1": "diag_snv_a7_hx", "A7hc2": "diag_snv_a7_hc_gan", "A7hx2": "diag_snv_a7_hx_gan",
        "A7ac1": "diag_snv_a7_ac", "A7ax1": "diag_snv_a7_ax", "A7ac2": "diag_snv_a7_ac_gan", "A7ax2": "diag_snv_a7_ax_gan"}
ID_TOL = 0.05
CER_GUARD = 0.05
INERT = 0.02


def interaction(tex: dict, items: list, c: str, x: str, focus: tuple) -> dict:
    out = {}
    for v in sorted({it[0] for it in items}):
        h = [tex[c][n] - tex[x][n] for vv, cat, n in items if vv == v and cat in focus]
        s = [tex[c][n] - tex[x][n] for vv, cat, n in items if vv == v and cat not in HELD]
        out[v] = float(np.mean(h) - np.mean(s))
    return out


def did_block(tex: dict, items: list, st: str, focus: tuple) -> dict:
    vs = sorted({it[0] for it in items})
    ih, ia = interaction(tex, items, f"A7hc{st}", f"A7hx{st}", focus), interaction(tex, items, f"A7ac{st}", f"A7ax{st}", focus)
    did = np.array([ih[v] - ia[v] for v in vs])
    grp = np.array(vs)
    sel = [it for it in items if it[1] in focus]
    sg = np.array([it[0] for it in sel])
    r = {"I_h": {"mean": round(float(np.mean([ih[v] for v in vs])), 4), "ci95_voice_cluster": cluster_ci(np.array([ih[v] for v in vs]), grp)},
         "I_a": {"mean": round(float(np.mean([ia[v] for v in vs])), 4), "ci95_voice_cluster": cluster_ci(np.array([ia[v] for v in vs]), grp)},
         "DiD": {"mean": round(float(did.mean()), 4), "ci95_voice_cluster": cluster_ci(did, grp)}}
    for sy in ("c", "x"):
        d = np.array([tex[f"A7h{sy}{st}"][n] - tex[f"A7a{sy}{st}"][n] for _, _, n in sel])
        r[f"cost_{sy}"] = {"mean": round(float(d.mean()), 4), "ci95_voice_cluster": cluster_ci(d, sg)}
    d = np.array([tex[f"A7hc{st}"][n] - tex[f"A7hx{st}"][n] for _, _, n in sel])
    r["h_c_minus_x"] = {"mean": round(float(d.mean()), 4), "ci95_voice_cluster": cluster_ci(d, sg)}
    return r


def summarize(tex: dict, items: list, st: str) -> dict:
    r = {"scream": did_block(tex, items, st, FOCUS), "held3": did_block(tex, items, st, HELD)}
    cats = sorted({it[1] for it in items})
    r["tex_by_cat"] = {s: {c: round(float(np.mean([tex[s][n] for _, cc, n in items if cc == c])), 4) for c in cats} for s in tex}
    allv = np.array([it[0] for it in items])
    d = np.array([tex[f"A7ac{st}"][n] - tex[f"A7ax{st}"][n] for _, _, n in items])
    r["a_c_minus_x_all_cats"] = {"mean": round(float(d.mean()), 4), "ci95_voice_cluster": cluster_ci(d, allv)}
    p = r["scream"]
    inert = all(p[f"cost_{sy}"]["ci95_voice_cluster"][0] <= 0 <= p[f"cost_{sy}"]["ci95_voice_cluster"][1] and abs(p[f"cost_{sy}"]["mean"]) < INERT for sy in ("c", "x"))
    ci = p["DiD"]["ci95_voice_cluster"]
    r["reading"] = {"H_holdout_inert": inert,
                    "H_ext": "H_circuit_extrap" if p["DiD"]["mean"] < 0 and ci[1] < 0 else "H_table_extrap" if p["DiD"]["mean"] > 0 and ci[0] > 0 else "H_flat",
                    "c_wins_outright_scream": p["h_c_minus_x"]["ci95_voice_cluster"][1] < 0}
    return r


def main() -> int:
    mode = sys.argv[1]
    smoke = len(sys.argv) > 2 and sys.argv[2] == "smoke"
    st = "1" if mode == "dev1" else "2"
    systems = [f"A7{g}{sy}{st}" for g in ("h", "a") for sy in ("c", "x")] + (["A7hc1", "A7hx1"] if mode == "test" else [])
    tags = dict(TAGS)
    if smoke:
        tags = {k: sys.argv[3] if k[3] == "c" else sys.argv[4] for k in TAGS}
    OUT.mkdir(parents=True, exist_ok=True)
    scale = np.load(ROOT / "results/snv_n2c/tex_scale.npy")
    nets = {}
    for s in systems:
        ck = torch.load(ROOT / "results" / tags[s] / "last.pt", map_location="cpu", weights_only=True)
        assert ck.get("pdim") == PDIM, f"{s} の演奏が PG でない"
        net = (G("x", pdim=PDIM) if ck["mode"] == "x" else C(pdim=PDIM)).to(DEV).eval()
        net.load_state_dict(ck["g"])
        for p in net.parameters():
            p.requires_grad_(False)
        nz = np.load(ROOT / "results" / tags[s] / "norm.npz")
        nets[s] = (net, ck["mode"], nz["mu"], nz["sd"])
    for s in systems:
        if not smoke:
            log = (ROOT / "results" / tags[s] / "train.log").read_text()
            assert "pdim 24" in log, f"{s} の演奏の次元のログが無い"
            want = "hold ['mixed', 'moan', 'scream']" if s.startswith("A7h") else "hold []"
            assert want in log, f"{s} の抜いた種類が事前登録と違う"
    voc = vocoder()

    def get_phi(s: str, rp: np.ndarray, rmel: np.ndarray):
        net, kind, mu, sd = nets[s]
        rp_t = torch.from_numpy((rp - mu) / sd)[None].to(DEV)
        rm_t = torch.from_numpy(rmel)[None].to(DEV)
        mask = torch.ones(1, rmel.shape[1], dtype=torch.bool, device=DEV)
        if kind == "x":
            with torch.no_grad():
                return net.structure(rm_t, rp_t, mask)
        return fit_phi(net, rp_t, rm_t, mask, steps=FIT)

    def forward(s: str, p: np.ndarray, phi) -> np.ndarray:
        net, _, mu, sd = nets[s]
        with torch.no_grad():
            return voc(net(torch.from_numpy((p - mu) / sd)[None].to(DEV), phi)).squeeze().cpu().numpy()

    def tex1(w44: np.ndarray, tst: np.ndarray) -> float:
        return float(np.mean(np.abs(tex_stats(to16(w44, 44100)) - tst) / scale))

    def pg(rel: str, n: int) -> np.ndarray:
        return play(FEAT / f"sparc/{rel}.npz", n, True)

    from speechbrain.inference.speaker import EncoderClassifier
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})

    def emb(y16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(y16).float()[None].to(DEV)).reshape(-1)
        return torch.nn.functional.normalize(e, dim=0).cpu().numpy()

    res: dict = {"prereg": "results/diag_snv_a7/prereg.yaml", "mode": mode, "held": HELD, "focus": FOCUS, "fit_steps_c": FIT, "tags": {s: tags[s] for s in systems}, "sys": {}}
    tex: dict = {s: {} for s in systems}
    if mode == "dev1":
        names_tr = json.loads((ROOT / "results/diag_snv_a4_c/train_log.json").read_text())["names_train"]
        man = sorted(v["voice"] for v in json.loads((ROOT / "data/snv_e1/manifest.json").read_text())["voices"])
        e1v = [v for v in man if v not in names_tr][-N_E1:]
        if not smoke:
            for s in ("A7hc1", "A7hx1", "A7ac1", "A7ax1"):
                assert sorted(json.loads((ROOT / "results" / tags[s] / "train_log.json").read_text())["names_val"]) == sorted(e1v), "開発の組が検証の声と違う"
        nvman = [it for it in json.loads((ROOT / "data/snv_e1nv/manifest.json").read_text())["items"] if it["k"] == 0 and it["voice"] in e1v]
        if smoke:
            e1v = e1v[:3]
            nvman = [it for it in nvman if it["voice"] in e1v]
        enroll = {}
        for v in e1v:
            ms = [np.load(FEAT / f"mel/e1/{v}/{t}.npy").astype(np.float32) for t in ENROLL_T]
            ps = [pg(f"e1/{v}/{t}", m.shape[1]) for t, m in zip(ENROLL_T, ms)]
            enroll[v] = (np.concatenate(ms, 1), np.concatenate(ps, 0), emb(np.concatenate([to16(*load(ROOT / f"data/snv_e1/{v}/{t}.wav")) for t in ENROLL_T])))
        items, tst = [], {}
        for it in nvman:
            n = f"{it['voice']}/{it['cat']}_0"
            items.append((it["voice"], it["cat"], n))
            tst[n] = tex_stats(to16(*load(ROOT / f"data/snv_e1nv/{n}.wav")))
        for s in systems:
            phis = {v: get_phi(s, rp, rm) for v, (rm, rp, _) in enroll.items()}
            for v, _, n in items:
                tex[s][n] = tex1(forward(s, pg(f"e1nv/{n}", np.load(FEAT / f"mel/e1nv/{n}.npy").shape[1]), phis[v]), tst[n])
            top = []
            for v in e1v:
                for t in HELD_T:
                    e = emb(trim(to16(forward(s, pg(f"e1/{v}/{t}", np.load(FEAT / f"mel/e1/{v}/{t}.npy").shape[1]), phis[v]), 44100)))
                    top.append(max(e1v, key=lambda u: float(e @ enroll[u][2])) == v)
            res["sys"][s] = {"nv_tex": round(float(np.mean(list(tex[s].values()))), 4), "normal_top1": round(float(np.mean(top)), 3)}
            print("dev1", s, res["sys"][s], flush=True)
        res["summary"] = summarize(tex, items, "1")
        r = res["summary"]
        res["rule"] = {"stop_table": r["reading"]["H_ext"] == "H_table_extrap", "stop_inert": r["reading"]["H_holdout_inert"]}
        res["rule"]["go_stage2"] = not (res["rule"]["stop_table"] or res["rule"]["stop_inert"])
        print("summary", json.dumps({k: v for k, v in r.items() if k != "tex_by_cat"}, ensure_ascii=False), flush=True)
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
        n0r = [it for it in n0r if it["voice"] in tgts[:2]]
    T = np.load(FEAT / "ecapa_lda_T.npz")
    Tt = (T["mu"], T["scalings"])
    from faster_whisper import WhisperModel
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16")

    def tr(w: np.ndarray) -> str:
        segs, _ = asr.transcribe(w, language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(x.text for x in segs))

    items = [(it["voice"], it["cat"], f"{it['voice']}_{it['cat']}_{it['k']}") for it in n0r]
    tst = {n: tex_stats(to16(*load(FEAT / f"wav44/n0r__{n}.wav"))) for _, _, n in items}
    Et = {v: proj(emb(np.concatenate([to16(*load(A0 / f"wav/{v}_{j}.wav")) for j in (0, 1, 2)])), Tt) for v in tgts}
    ref_txt = {s: tr(sf.read(str(A1 / f"src/{s}.wav"))[0].astype(np.float32)) for s in srcs}
    n_src = int(10 * 44100 / 512) + 1
    for s in systems:
        phis = {}
        for v in tgts:
            rmel = np.load(FEAT / f"mel/a0enroll/{v}.npy").astype(np.float32)
            phis[v] = get_phi(s, pg(f"a1src/tref_{v}", rmel.shape[1]), rmel)
        for v, _, n in items:
            tex[s][n] = tex1(forward(s, pg(f"n0r/{n}", np.load(FEAT / f"mel/n0r/{n}.npy").shape[1]), phis[v]), tst[n])
        res["sys"][s] = {"nv_tex": round(float(np.mean(list(tex[s].values()))), 4)}
        if s.startswith("A7h") or s in ("A7ac2", "A7ax2"):
            sh = []
            for r in drows:
                n_mel = np.load(FEAT / f"mel/a1diag/D0_{r['target']}_{r['g']}.npy").shape[1]
                e = proj(emb(trim(to16(forward(s, pg(f"a1diag/D0_{r['target']}_{r['g']}", n_mel), phis[r["target"]]), 44100))), Tt)
                sh.append(max(tgts, key=lambda u: float(e @ Et[u])) == r["target"])
            mh, cers = [], []
            for src in srcs:
                stt = np.load(FEAT / f"sparc/a1src/{src}.npz")["pitch_stats"].reshape(-1)[:2]
                for v in tgts:
                    w16 = to16(forward(s, map_pitch(pg(f"a1src/{src}", n_src), stt, tstats[v]), phis[v]), 44100)
                    e = proj(emb(trim(w16)), Tt)
                    mh.append(max(tgts, key=lambda u: float(e @ Et[u])) == v)
                    if v in tgts[:8]:
                        cers.append(cer(ref_txt[src], tr(w16)))
            res["sys"][s] |= {"S": round(float(np.mean(sh)), 3), "M": round(float(np.mean(mh)), 3), "M_cer_median_8tgt": round(float(np.median(cers)), 3)}
        print("test", s, res["sys"][s], flush=True)
    res["summary"] = summarize(tex, items, "2")
    c2, x2 = res["sys"]["A7hc2"], res["sys"]["A7hx2"]
    res["guards"] = {"H_id_ok_h": c2["S"] >= x2["S"] - ID_TOL and c2["M"] >= x2["M"] - ID_TOL,
                     "cer_ok": {sy: res["sys"][f"A7h{sy}2"]["M_cer_median_8tgt"] - res["sys"][f"A7h{sy}1"]["M_cer_median_8tgt"] <= CER_GUARD for sy in ("c", "x")}}
    print("summary", json.dumps({k: v for k, v in res["summary"].items() if k != "tex_by_cat"}, ensure_ascii=False), flush=True)
    print("guards", res["guards"], flush=True)
    if not smoke:
        (OUT / "a7_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
