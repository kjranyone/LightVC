"""A4a の走行後の確かめ(事前登録の外・記録だけ): 当てはめた系の本人らしさが「当てはめに使った登録の録音そのもの」に寄っただけかを分ける。
物差しの目標の登録を、当てはめに使っていない発話(A0 の j6・j7)に替えて、S・O・M の top-1 を読み直す。当てはめと描画は snv_a4a_fit.py と同じ。
    cd training && uv run --no-sync python snv_diag/a4a_alt_enroll.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from snv_a0_measure import DEV, ROOT, load, to16
from snv_a0b_measure import proj
from snv_a2_eval import A1, map_pitch, vocoder
from snv_a4a_fit import A0, OUT, film_forward, fit, load_g
from train_snv_a2 import FEAT, playing


def main() -> int:
    meta = json.loads((A1 / "convert_meta.json").read_text())
    srcs, tgts = meta["sources"], meta["targets"]
    drows = json.loads((A1 / "diag/meta.json").read_text())
    tstats = {v: np.load(FEAT / f"sparc/a1src/tref_{v}.npz")["pitch_stats"].reshape(-1)[:2] for v in tgts}
    voc = vocoder()
    gnet, mu, sd = load_g("g")
    xnet, _, _ = load_g("x")
    enr = {}
    for v in tgts:
        rmel = torch.from_numpy(np.load(FEAT / f"mel/a0enroll/{v}.npy").astype(np.float32))[None].to(DEV)
        rp = torch.from_numpy((playing(FEAT / f"sparc/a1src/tref_{v}.npz", rmel.shape[2]) - mu) / sd)[None].to(DEV)
        mask = torch.ones(1, rmel.shape[2], dtype=torch.bool, device=DEV)
        with torch.no_grad():
            sg, sx = gnet.structure(rmel, rp, mask), xnet.structure(rmel, rp, mask)
        z, _, _ = fit(gnet, "z", sg["z"], rp, rmel)
        f, _, _ = fit(gnet, "f", sg["z"], rp, rmel)
        enr[v] = {"g_enc": sg, "g_fitz": {"z": z}, "g_fitf": f, "x_enc": sx}

    @torch.no_grad()
    def run(cond: str, p_npz: Path, n_mel: int, v: str, stats: np.ndarray | None) -> np.ndarray:
        p = playing(p_npz, n_mel)
        if stats is not None:
            p = map_pitch(p, stats, tstats[v])
        q = torch.from_numpy((p - mu) / sd)[None].to(DEV)
        y = xnet(q, enr[v]["x_enc"]) if cond == "x_enc" else film_forward(gnet, q, enr[v]["g_fitf"]) if cond == "g_fitf" else gnet(q, enr[v][cond])
        return to16(voc(y).squeeze().cpu().numpy(), 44100)

    conds = ("g_enc", "g_fitz", "g_fitf", "x_enc")
    wavs = {}
    for r in drows:
        n_mel = np.load(FEAT / f"mel/a1diag/D0_{r['target']}_{r['g']}.npy").shape[1]
        o = FEAT / f"sparc/a1diag/src_{r['d2_src']}_{r['g']}.npz"
        for c in conds:
            wavs[(f"S_{c}", r["target"], r["g"])] = run(c, FEAT / f"sparc/a1diag/D0_{r['target']}_{r['g']}.npz", n_mel, r["target"], None)
            wavs[(f"O_{c}", r["target"], r["g"])] = run(c, o, n_mel, r["target"], np.load(o)["pitch_stats"].reshape(-1)[:2])
    for s in srcs:
        f = FEAT / f"sparc/a1src/{s}.npz"
        for v in tgts:
            for c in conds:
                wavs[(f"M_{c}", s, v)] = run(c, f, int(10 * 44100 / 512) + 1, v, np.load(f)["pitch_stats"].reshape(-1)[:2])
    del voc, gnet, xnet
    torch.cuda.empty_cache()
    from speechbrain.inference.speaker import EncoderClassifier
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})
    T = np.load(FEAT / "ecapa_lda_T.npz")
    Tt = (T["mu"], T["scalings"])

    def E(y16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(y16).float()[None].to(DEV)).reshape(-1)
        return proj(torch.nn.functional.normalize(e, dim=0).cpu().numpy(), Tt)

    Ew = {k: E(w) for k, w in wavs.items()}
    res = {"note": "事前登録の外・記録だけ", "enroll_for_ruler": {}}
    for name, js in (("fit_j012", (0, 1, 2)), ("alt_j67", (6, 7))):
        Et = {v: E(np.concatenate([to16(*load(A0 / f"wav/{v}_{j}.wav")) for j in js])) for v in tgts}
        row = {}
        for c in conds:
            for part in ("S", "O", "M"):
                ks = [k for k in wavs if k[0] == f"{part}_{c}"]
                tr = (lambda k: k[2]) if part == "M" else (lambda k: k[1])
                row[f"{part}_{c}"] = round(float(np.mean([max(tgts, key=lambda u: float(Ew[k] @ Et[u])) == tr(k) for k in ks])), 3)
        res["enroll_for_ruler"][name] = row
        print(name, row, flush=True)
    (OUT / "a4a_alt_enroll.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
