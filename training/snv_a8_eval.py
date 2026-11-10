"""A8(diag)の評価。事前登録 = results/diag_snv_a8/prereg.yaml。声門の 9 次元をその声の登録の平均からの外れで学んだ x(A8ax)と、
生の声門の次元で学んだ x(A7ax)に A8a の登録のずらしを推論で当てたものを並べる。基準 = 有声のフレームの平均(train_snv_a4.voiced_base・学習と同じ関数)。
操作者 = 本物の男声の登録(先頭 30 s)・女声の源 = その声の登録・自己(S・nv TEX)= 目標の登録。
dev1 = 段 1 の途中信号(開発の組 E1V 30 声・本物の男声 12 区間 → 30 声の top-1 と nv TEX)→ dev1.json。test = 段 2 の判定(A0F の女 24 声)→ a8_eval.json。
dev1raw = 記録だけ(判定の後に足した): A7ax1 にずらしを当てない dev M → dev1_raw7_record.json。
a9 = A9(diag・results/diag_snv_a9/prereg.yaml)= 声門の次元の部分集合で学んだ x(A9x)と A7ax を、両方に登録のずらしを当てて同じ量で並べる → results/diag_snv_a9。
    cd training && uv run --no-sync python snv_a8_eval.py [a9] dev1|dev1raw|test [smoke <tag>]
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
from snv_a4b_fit import ENROLL_T, N_E1
from snv_glot import ENR_T, NAMES, SRC_T
from snv_n0_measure import trim
from snv_n2c_tex import tex_stats
from train_snv_a2 import FEAT, P_DIM, G
from train_snv_a4 import GL, play, voiced_base

ARMS = {"a8": {"out": "results/diag_snv_a8", "new": "A8ax", "label": "rel", "tags": {"A8ax1": "diag_snv_a8_x", "A8ax2": "diag_snv_a8_x_gan"}},
        "a9": {"out": "results/diag_snv_a9", "new": "A9x", "label": "sub", "tags": {"A9x1": "diag_snv_a9_x", "A9x2": "diag_snv_a9_x_gan"}}}
BASE_TAGS = {"A7ax1": "diag_snv_a7_ax", "A7ax2": "diag_snv_a7_ax_gan"}
PDIM = 24
SEMI = 7
TEX_GUARD = 0.03
CER_GUARD = 0.05
S_GUARD = 0.05
NEAR_P = 0.05


def main() -> int:
    args = sys.argv[1:]
    arm = args.pop(0) if args[0] in ARMS else "a8"
    cfg = ARMS[arm]
    OUT, NEW = ROOT / cfg["out"], cfg["new"]
    mode = args[0]
    smoke = len(args) > 1 and args[1] == "smoke"
    raw7 = mode == "dev1raw"
    mode = "dev1" if raw7 else mode
    st = "1" if mode == "dev1" else "2"
    systems = ["A7ax1"] if raw7 else [f"{NEW}{st}", f"A7ax{st}"] + ([f"{NEW}1"] if mode == "test" else [])
    tags = cfg["tags"] | BASE_TAGS
    if smoke:
        tags = {k: args[2] if k.startswith(NEW) else v for k, v in tags.items()}
    OUT.mkdir(parents=True, exist_ok=True)
    scale = np.load(ROOT / "results/snv_n2c/tex_scale.npy")
    nets = {}
    for s in systems:
        ck = torch.load(ROOT / "results" / tags[s] / "last.pt", map_location="cpu", weights_only=True)
        dims = ck.get("glot_dims")
        assert (dims is not None) == s.startswith("A9"), f"{s} の声門の次元の部分集合が事前登録と違う"
        cols = None if dims is None else list(range(P_DIM)) + [P_DIM + NAMES.index(d) for d in dims]
        assert ck.get("pdim") == (PDIM if cols is None else len(cols)) and ck["mode"] == "x", s
        rel = bool(ck.get("glot_rel", False))
        assert rel == s.startswith("A8"), f"{s} の声門の次元の持ち方が事前登録と違う"
        net = G("x", pdim=ck["pdim"]).to(DEV).eval()
        net.load_state_dict(ck["g"])
        for prm in net.parameters():
            prm.requires_grad_(False)
        nz = np.load(ROOT / "results" / tags[s] / "norm.npz")
        nets[s] = (net, rel, nz["mu"], nz["sd"], cols)
        if not smoke:
            log = (ROOT / "results" / tags[s] / "train.log").read_text()
            assert f"pdim {ck['pdim']}" in log and "hold []" in log, s
            assert ("glot relative" in log) == rel and (f"glot dims {dims}" in log) == (dims is not None), s
    voc = vocoder()

    def pg(rel: str, n: int) -> np.ndarray:
        return play(FEAT / f"sparc/{rel}.npz", n, True)

    def base_of(p: np.ndarray) -> np.ndarray:
        b, ok = voiced_base(p)
        assert ok, "登録の有声のフレームが足りない"
        return b

    def inp(s: str, p: np.ndarray, b_src: np.ndarray, b_tgt: np.ndarray) -> np.ndarray:
        q = p.copy()
        if not raw7:
            q[:, GL] += (-b_src) if nets[s][1] else (b_tgt - b_src)
        return q

    def get_phi(s: str, rp: np.ndarray, rmel: np.ndarray, b_tgt: np.ndarray):
        net, rel, mu, sd, cols = nets[s]
        q = rp.copy()
        if rel:
            q[:, GL] -= b_tgt
        if cols is not None:
            q = q[:, cols]
        with torch.no_grad():
            return net.structure(torch.from_numpy(rmel)[None].to(DEV), torch.from_numpy((q - mu) / sd)[None].to(DEV), torch.ones(1, rmel.shape[1], dtype=torch.bool, device=DEV))

    def forward(s: str, p: np.ndarray, phi) -> np.ndarray:
        net, _, mu, sd, cols = nets[s]
        if cols is not None:
            p = p[:, cols]
        with torch.no_grad():
            return voc(net(torch.from_numpy((p - mu) / sd)[None].to(DEV), phi)).squeeze().cpu().numpy()

    def tex1(w44: np.ndarray, tst: np.ndarray) -> float:
        return float(np.mean(np.abs(tex_stats(to16(w44, 44100)) - tst) / scale))

    from speechbrain.inference.speaker import EncoderClassifier
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})

    def emb(y16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(y16).float()[None].to(DEV)).reshape(-1)
        return torch.nn.functional.normalize(e, dim=0).cpu().numpy()

    meta = json.loads((A1 / "convert_meta.json").read_text())
    srcs, tgts = meta["sources"], meta["targets"]
    if smoke:
        srcs = [srcs[0], srcs[-1]]
    b_op = {nm: base_of(pg(f"a1src/{nm}_enroll", ENR_T)) for nm in ("uirou", "namikawa")}
    res: dict = {"prereg": f"{cfg['out']}/prereg.yaml", "mode": mode, "tags": {s: tags[s] for s in systems}, "sys": {}}

    if mode == "dev1":
        names_tr = json.loads((ROOT / "results/diag_snv_a4_c/train_log.json").read_text())["names_train"]
        man = sorted(v["voice"] for v in json.loads((ROOT / "data/snv_e1/manifest.json").read_text())["voices"])
        e1v = [v for v in man if v not in names_tr][-N_E1:]
        if not smoke:
            for s in systems:
                assert sorted(json.loads((ROOT / "results" / tags[s] / "train_log.json").read_text())["names_val"]) == sorted(e1v), "開発の組が検証の声と違う"
        nvman = [it for it in json.loads((ROOT / "data/snv_e1nv/manifest.json").read_text())["items"] if it["k"] == 0 and it["voice"] in e1v]
        if smoke:
            e1v = e1v[:4]
            nvman = [it for it in nvman if it["voice"] in e1v]
        enroll = {}
        for v in e1v:
            ms = [np.load(FEAT / f"mel/e1/{v}/{t}.npy").astype(np.float32) for t in ENROLL_T]
            ps = np.concatenate([pg(f"e1/{v}/{t}", m.shape[1]) for t, m in zip(ENROLL_T, ms)], 0)
            stt = np.mean([np.load(FEAT / f"sparc/e1/{v}/{t}.npz")["pitch_stats"].reshape(-1)[:2] for t in ENROLL_T], 0)
            enroll[v] = (np.concatenate(ms, 1), ps, base_of(ps), stt, emb(np.concatenate([to16(*load(ROOT / f"data/snv_e1/{v}/{t}.wav")) for t in ENROLL_T])))
        tst = {f"{it['voice']}/{it['cat']}_0": tex_stats(to16(*load(ROOT / f"data/snv_e1nv/{it['voice']}/{it['cat']}_0.wav"))) for it in nvman}
        hits = {}
        for s in systems:
            phis = {v: get_phi(s, rp, rm, b) for v, (rm, rp, b, _, _) in enroll.items()}
            tx = [tex1(forward(s, inp(s, pg(f"e1nv/{n}", np.load(FEAT / f"mel/e1nv/{n}.npy").shape[1]), enroll[n.split("/")[0]][2], enroll[n.split("/")[0]][2]), phis[n.split("/")[0]]), t) for n, t in tst.items()]
            h, grp = [], []
            for src in srcs:
                stt = np.load(FEAT / f"sparc/a1src/{src}.npz")["pitch_stats"].reshape(-1)[:2]
                p0 = pg(f"a1src/{src}", SRC_T)
                for v in e1v:
                    p = inp(s, map_pitch(p0, stt, enroll[v][3]), b_op[src.rsplit("_", 1)[0]], enroll[v][2])
                    e = emb(trim(to16(forward(s, p, phis[v]), 44100)))
                    h.append(max(e1v, key=lambda u: float(e @ enroll[u][4])) == v)
                    grp.append(v)
            hits[s] = (np.array(h, float), np.array(grp))
            res["sys"][s] = {"dev_M": round(float(np.mean(h)), 3), "n_M": len(h), "nv_tex": round(float(np.mean(tx)), 4), "n_tex": len(tx)}
            print("dev1", s, res["sys"][s], flush=True)
        if raw7:
            res["note"] = "記録だけ(dev1 の判定の後に足した・判定に使わない): A7ax1 の声門の次元を生のまま(ずらしなし)"
            if not smoke:
                (OUT / "dev1_raw7_record.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
            return 0
        d = hits[f"{NEW}1"][0] - hits["A7ax1"][0]
        res["summary"] = {"dM_dev": {"mean": round(float(d.mean()), 4), "ci95_target_cluster": cluster_ci(d, hits[f"{NEW}1"][1])},
                          "dTEX_dev": round(res["sys"][f"{NEW}1"]["nv_tex"] - res["sys"]["A7ax1"]["nv_tex"], 4)}
        r = res["summary"]
        res["rule"] = {"stop_no_gain": r["dM_dev"]["mean"] <= 0.0, "stop_tex": r["dTEX_dev"] > TEX_GUARD}
        res["rule"]["go_stage2"] = not (res["rule"]["stop_no_gain"] or res["rule"]["stop_tex"])
        print("summary", json.dumps(r, ensure_ascii=False), "rule", res["rule"], flush=True)
        if not smoke:
            (OUT / "dev1.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        return 0

    drows = json.loads((A1 / "diag/meta.json").read_text())
    n0r = [it for it in json.loads((N0R / "items.json").read_text()) if it["sex"] == "F"]
    cer_t = tgts[:8]
    if smoke:
        drows, cer_t = drows[:4], tgts[:1]
        n0r = [it for it in n0r if it["voice"] in tgts[:2]][:6]
    conv_t = tgts[:3] if smoke else tgts
    T = np.load(FEAT / "ecapa_lda_T.npz")
    Tt = (T["mu"], T["scalings"])
    Et = {v: proj(emb(np.concatenate([to16(*load(A0 / f"wav/{v}_{j}.wav")) for j in (0, 1, 2)])), Tt) for v in tgts}
    Et2 = {v: proj(emb(np.concatenate([to16(*load(A0 / f"wav/{v}_{j}.wav")) for j in (6, 7)])), Tt) for v in tgts}
    top = lambda e, ref: max(tgts, key=lambda u: float(e @ ref[u]))
    from faster_whisper import WhisperModel
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16")

    def tr(w: np.ndarray) -> str:
        segs, _ = asr.transcribe(w, language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(x.text for x in segs))

    ref_txt = {s: tr(sf.read(str(A1 / f"src/{s}.wav"))[0].astype(np.float32)) for s in srcs}
    tstats = {v: np.load(FEAT / f"sparc/a1src/tref_{v}.npz")["pitch_stats"].reshape(-1)[:2] for v in tgts}
    tref = {}
    for v in tgts:
        rmel = np.load(FEAT / f"mel/a0enroll/{v}.npy").astype(np.float32)
        rp = pg(f"a1src/tref_{v}", rmel.shape[1])
        tref[v] = (rmel, rp, base_of(rp))
    items = [(it["voice"], f"{it['voice']}_{it['cat']}_{it['k']}") for it in n0r]
    tst = {n: tex_stats(to16(*load(FEAT / f"wav44/n0r__{n}.wav"))) for _, n in items}
    hits = {}
    for s in systems:
        phis = {v: get_phi(s, tref[v][1], tref[v][0], tref[v][2]) for v in tgts}
        tx = [tex1(forward(s, inp(s, pg(f"n0r/{n}", np.load(FEAT / f"mel/n0r/{n}.npy").shape[1]), tref[v][2], tref[v][2]), phis[v]), tst[n]) for v, n in items]
        sh = []
        for r in drows:
            o = f"D0_{r['target']}_{r['g']}"
            p = inp(s, pg(f"a1diag/{o}", np.load(FEAT / f"mel/a1diag/{o}.npy").shape[1]), tref[r["target"]][2], tref[r["target"]][2])
            sh.append(top(proj(emb(trim(to16(forward(s, p, phis[r["target"]]), 44100))), Tt), Et) == r["target"])
        oh, osrc = [], []
        for r in drows:
            v, o = r["target"], f"D0_{r['d2_src']}_{r['g']}"
            p = map_pitch(pg(f"a1diag/{o}", np.load(FEAT / f"mel/a1diag/{o}.npy").shape[1]), np.load(FEAT / f"sparc/a1diag/{o}.npz")["pitch_stats"].reshape(-1)[:2], tstats[v])
            t1 = top(proj(emb(trim(to16(forward(s, inp(s, p, tref[r["d2_src"]][2], tref[v][2]), phis[v]), 44100))), Tt), Et)
            oh.append(t1 == v)
            osrc.append(t1 == r["d2_src"])
        mh, mh2, m7, cs, grp = [], [], [], [], []
        for src in srcs:
            stt = np.load(FEAT / f"sparc/a1src/{src}.npz")["pitch_stats"].reshape(-1)[:2]
            p0 = pg(f"a1src/{src}", SRC_T)
            bs = b_op[src.rsplit("_", 1)[0]]
            for v in conv_t:
                w16 = to16(forward(s, inp(s, map_pitch(p0, stt, tstats[v]), bs, tref[v][2]), phis[v]), 44100)
                e = proj(emb(trim(w16)), Tt)
                mh.append(top(e, Et) == v)
                mh2.append(top(e, Et2) == v)
                grp.append(v)
                if v in cer_t:
                    cs.append(cer(ref_txt[src], tr(w16)))
                p7 = p0.copy()
                p7[:, 12] += SEMI * np.log(2) / 12
                m7.append(top(proj(emb(trim(to16(forward(s, inp(s, p7, bs, tref[v][2]), phis[v]), 44100))), Tt), Et) == v)
        hits[s] = (np.array(mh, float), np.array(grp))
        res["sys"][s] = {"nv_tex": round(float(np.mean(tx)), 4), "n_tex": len(tx), "S": round(float(np.mean(sh)), 3), "O": round(float(np.mean(oh)), 3),
                         "O_read_as_source": round(float(np.mean(osrc)), 3), "M": round(float(np.mean(mh)), 3), "n_M": len(mh), "M_alt_j67": round(float(np.mean(mh2)), 3),
                         "M7": round(float(np.mean(m7)), 3), "M_cer_median_8tgt": round(float(np.median(cs)), 3)}
        print("test", s, res["sys"][s], flush=True)
    a8, a7 = res["sys"][f"{NEW}2"], res["sys"]["A7ax2"]
    d = hits[f"{NEW}2"][0] - hits["A7ax2"][0]
    m_p = json.loads((ROOT / "results/snv_a8a/a8a.json").read_text())["sys"]["A4x2"]["raw"]["M"]
    guards = {"tex": round(a8["nv_tex"] - a7["nv_tex"], 4), "cer": round(a8["M_cer_median_8tgt"] - a7["M_cer_median_8tgt"], 3), "S": round(a8["S"] - a7["S"], 3)}
    g_ok = guards["tex"] <= TEX_GUARD and guards["cer"] <= CER_GUARD and guards["S"] >= -S_GUARD
    lo = cluster_ci(d, hits[f"{NEW}2"][1])
    lab = cfg["label"]
    if lo[0] > 0 and a8["M"] >= m_p - NEAR_P:
        reading = f"H_{lab}_full"
    elif lo[0] > 0:
        reading = f"H_{lab}_gain"
    else:
        reading = f"H_{lab}_none"
    res["summary"] = {"dM_vs_A7ax2_shift": {"mean": round(float(d.mean()), 4), "ci95_target_cluster": lo}, "M_P_A4x2": m_p, "guards": guards | {"ok": g_ok}, "reading": reading}
    print("summary", json.dumps(res["summary"], ensure_ascii=False), flush=True)
    if not smoke:
        (OUT / "a8_eval.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
