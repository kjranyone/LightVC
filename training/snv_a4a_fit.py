"""A4a(diag・G は学習し直さない): 連続の回路 g の本人らしさの不足は、参照を 1 本のベクトルへ縮める符号器(償却の推論)の差か、回路の族の表現の限界かを分ける。
A2 の g の重みは固定し、目標の声ごとに登録(A0 の j0〜j2)だけで構造の値を合わせる(自己の reamp の L1・登録の演奏 → 登録の mel)。
条件: g_enc = 符号器の z(A2 のまま)・g_fitz = z(256)を登録で合わせる・g_fitf = 12 層の FiLM の出力(12 × 512)を登録で合わせる・x_enc = A2 の x(表・参考値)。
S = 自己の reamp(D0 の 72 本・登録と別の発話)・O = 別の女声の演奏 → 目標・M = 本物の男声 12 区間 → 目標 24 声(高さは A2 と同じ平均・標準偏差の写像)。
事前登録 = results/diag_snv_a4a/prereg.yaml。
    cd training && uv run --no-sync python snv_a4a_fit.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from snv_a0_measure import DEV, ROOT, cer, kana, load, to16
from snv_a0b_measure import proj
from snv_a2_eval import A1, map_pitch, vocoder
from train_snv_a2 import FEAT, G, WARM, playing

OUT = ROOT / "results/diag_snv_a4a"
A0 = ROOT / "results/snv_a0"
STEPS = 400
LR = {"z": 3e-2, "f": 1e-2}


def cluster_ci(d: np.ndarray, groups: np.ndarray, n: int = 4000) -> list:
    rng = np.random.default_rng(0)
    gs = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in gs}
    bs = [np.mean(np.concatenate([d[idx[g]] for g in rng.choice(gs, len(gs))])) for _ in range(n)]
    return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]


def load_g(m: str) -> tuple:
    d = ROOT / f"results/diag_snv_a2_{m}"
    ck = torch.load(d / "last.pt", map_location="cpu", weights_only=True)
    net = G(ck["mode"]).to(DEV).eval()
    net.load_state_dict(ck["g"])
    for q in net.parameters():
        q.requires_grad_(False)
    nz = np.load(d / "norm.npz")
    return net, nz["mu"], nz["sd"]


def film_forward(net: G, p: torch.Tensor, gs: torch.Tensor) -> torch.Tensor:
    h = net.inp(p.transpose(1, 2))
    for blk, gsl in zip(net.blocks, gs):
        a, b = blk.conv(torch.nn.functional.pad(h, (2 * blk.dil, 0))).chunk(2, 1)
        g, s = gsl[None].chunk(2, -1)
        h = h + blk.out(torch.tanh(a * (1 + g[..., None]) + s[..., None]) * torch.sigmoid(b))
    return net.head(h)


def fit(net: G, kind: str, z0: torch.Tensor, rp: torch.Tensor, rmel: torch.Tensor) -> tuple:
    w = torch.zeros_like(rmel[:, :1])
    w[..., WARM:] = 1.0
    if kind == "z":
        prm = z0.detach().clone().requires_grad_(True)
        f = lambda: net(rp, {"z": prm})
    else:
        prm = torch.stack([blk.film(z0[0]) for blk in net.blocks]).detach().clone().requires_grad_(True)
        f = lambda: film_forward(net, rp, prm)
    opt = torch.optim.Adam([prm], lr=LR[kind])
    l0 = None
    for _ in range(STEPS):
        loss = ((f() - rmel).abs() * w).sum() / (w.sum() * rmel.shape[1])
        l0 = float(loss) if l0 is None else l0
        opt.zero_grad()
        loss.backward()
        opt.step()
    return prm.detach(), l0, float(loss)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((A1 / "convert_meta.json").read_text())
    srcs, tgts = meta["sources"], meta["targets"]
    drows = json.loads((A1 / "diag/meta.json").read_text())
    tstats = {v: np.load(FEAT / f"sparc/a1src/tref_{v}.npz")["pitch_stats"].reshape(-1)[:2] for v in tgts}
    voc = vocoder()
    render = lambda mel: voc(mel).squeeze().detach().cpu().numpy()
    gnet, gmu, gsd = load_g("g")
    xnet, xmu, xsd = load_g("x")
    assert np.allclose(gmu, xmu) and np.allclose(gsd, xsd)
    mu, sd = gmu, gsd
    enr = {}
    fitlog = {}
    for v in tgts:
        rmel = torch.from_numpy(np.load(FEAT / f"mel/a0enroll/{v}.npy").astype(np.float32))[None].to(DEV)
        rp = torch.from_numpy((playing(FEAT / f"sparc/a1src/tref_{v}.npz", rmel.shape[2]) - mu) / sd)[None].to(DEV)
        mask = torch.ones(1, rmel.shape[2], dtype=torch.bool, device=DEV)
        with torch.no_grad():
            sg = gnet.structure(rmel, rp, mask)
            sx = xnet.structure(rmel, rp, mask)
        with torch.no_grad():
            init = torch.stack([blk.film(sg["z"][0]) for blk in gnet.blocks])
            assert torch.allclose(film_forward(gnet, rp, init), gnet(rp, sg), atol=1e-4), "FiLM の直接の値の経路が G と一致しない"
        z, lz0, lz = fit(gnet, "z", sg["z"], rp, rmel)
        f, lf0, lf = fit(gnet, "f", sg["z"], rp, rmel)
        with torch.no_grad():
            lx = float((xnet(rp, sx)[..., WARM:] - rmel[..., WARM:]).abs().mean())
        enr[v] = {"g_enc": sg, "g_fitz": {"z": z}, "g_fitf": f, "x_enc": sx}
        fitlog[v] = {"enroll_l1_g_enc": round(lz0, 4), "enroll_l1_g_fitz": round(lz, 4), "enroll_l1_g_fitf": round(lf, 4), "enroll_l1_x_enc": round(lx, 4)}
        print(v, fitlog[v], flush=True)

    @torch.no_grad()
    def run(cond: str, p_npz: Path, n_mel: int, v: str, stats: np.ndarray | None) -> np.ndarray:
        p = playing(p_npz, n_mel)
        if stats is not None:
            p = map_pitch(p, stats, tstats[v])
        q = torch.from_numpy((p - mu) / sd)[None].to(DEV)
        if cond == "x_enc":
            y = xnet(q, enr[v]["x_enc"])
        elif cond == "g_fitf":
            y = film_forward(gnet, q, enr[v]["g_fitf"])
        else:
            y = gnet(q, enr[v][cond])
        return render(y)

    conds = ("g_enc", "g_fitz", "g_fitf", "x_enc")
    wavs: dict = {}
    keep: dict = {}
    for r in drows:
        n_mel = np.load(FEAT / f"mel/a1diag/D0_{r['target']}_{r['g']}.npy").shape[1]
        o = FEAT / f"sparc/a1diag/src_{r['d2_src']}_{r['g']}.npz"
        for c in conds:
            wavs[(f"S_{c}", r["target"], r["g"])] = to16(run(c, FEAT / f"sparc/a1diag/D0_{r['target']}_{r['g']}.npz", n_mel, r["target"], None), 44100)
            wavs[(f"O_{c}", r["target"], r["g"])] = to16(run(c, o, n_mel, r["target"], np.load(o)["pitch_stats"].reshape(-1)[:2]), 44100)
    for s in srcs:
        f = FEAT / f"sparc/a1src/{s}.npz"
        n_mel = int(10 * 44100 / 512) + 1
        for v in tgts:
            for c in conds:
                w = run(c, f, n_mel, v, np.load(f)["pitch_stats"].reshape(-1)[:2])
                if s in srcs[:2] and v in tgts[:2]:
                    keep[(f"M_{c}", s, v)] = w
                wavs[(f"M_{c}", s, v)] = to16(w, 44100)
    del voc, gnet, xnet
    torch.cuda.empty_cache()
    print("rendered", len(wavs), flush=True)

    from speechbrain.inference.speaker import EncoderClassifier
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})
    T = np.load(FEAT / "ecapa_lda_T.npz")
    Tt = (T["mu"], T["scalings"])

    def E(y16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(y16).float()[None].to(DEV)).reshape(-1)
        return proj(torch.nn.functional.normalize(e, dim=0).cpu().numpy(), Tt)

    Et = {v: E(np.concatenate([to16(*load(A0 / f"wav/{v}_{j}.wav")) for j in (0, 1, 2)])) for v in tgts}
    import soundfile as sf
    Es = {s: E(sf.read(str(A1 / f"src/{s.rsplit('_', 1)[0]}_enroll.wav"))[0].astype(np.float32)) for s in srcs}
    Ew = {k: E(w) for k, w in wavs.items()}
    del ec
    torch.cuda.empty_cache()
    src_of = {(r["target"], r["g"]): r["d2_src"] for r in drows}
    hit = {k: max(tgts, key=lambda u: float(Ew[k] @ Et[u])) for k in wavs}
    res: dict = {"prereg": "results/diag_snv_a4a/prereg.yaml", "fit": {"steps": STEPS, "lr": LR, "per_voice": fitlog},
                 "ref_a2": {"S": {"g": 0.542, "x": 0.750}, "O": {"g": 0.361, "x": 0.681}, "M": {"g": 0.281, "x": 0.556}, "V": 0.806}}
    vec: dict = {}
    for c in conds:
        for part in ("S", "O", "M"):
            ks = sorted(k for k in wavs if k[0] == f"{part}_{c}")
            truth = (lambda k: k[2]) if part == "M" else (lambda k: k[1])
            vec[(part, c)] = (np.array([hit[k] == truth(k) for k in ks], float), np.array([k[2] if part == "M" else k[1] for k in ks]), ks)
            row = {"top1": round(float(vec[(part, c)][0].mean()), 3), "n": len(ks)}
            if part == "O":
                row["read_as_source"] = round(float(np.mean([hit[k] == src_of[(k[1], k[2])] for k in ks])), 3)
            if part == "M":
                row["move_mean"] = round(float(np.mean([Ew[k] @ Et[k[2]] - Ew[k] @ Es[k[1]] for k in ks])), 3)
            res[f"{part}_{c}"] = row
            print(part, c, row, flush=True)
    res["diff"] = {}
    for part in ("S", "O", "M"):
        for a, b in (("g_fitz", "g_enc"), ("g_fitf", "g_enc"), ("x_enc", "g_fitz"), ("x_enc", "g_fitf")):
            d = vec[(part, a)][0] - vec[(part, b)][0]
            res["diff"][f"{part}:{a}-{b}"] = {"mean": round(float(d.mean()), 3), "ci95_voice_cluster": cluster_ci(d, vec[(part, a)][1])}
    print("diff", res["diff"], flush=True)

    from faster_whisper import WhisperModel
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16")

    def tr(w: np.ndarray) -> str:
        segs, _ = asr.transcribe(w, language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(s.text for s in segs))

    ref = {s: tr(sf.read(str(A1 / f"src/{s}.wav"))[0].astype(np.float32)) for s in srcs}
    for c in conds:
        ks = [k for k in wavs if k[0] == f"M_{c}" and k[2] in tgts[:8]]
        vals = [cer(ref[k[1]], tr(wavs[k])) for k in ks]
        res[f"M_{c}"]["cer_median_8tgt"] = round(float(np.median(vals)), 3)
        print("CER", c, res[f"M_{c}"]["cer_median_8tgt"], flush=True)
    (OUT / "wav").mkdir(exist_ok=True)
    for k, w in keep.items():
        sf.write(OUT / f"wav/{k[0]}__{k[1]}__{k[2]}.wav", w, 44100)
    (OUT / "a4a.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
