"""P8a の物差し(synth_nam_vc §8・results/snv_p8a/prereg.yaml): 演奏から描ける声の量を C を学ばずに測る。
参照なしの復号器 D(因果の 0〜100 ms の積み重ねの演奏 + 高さ・有声 → Y の MLP)を E1 の学習の声の 1/4 で学び、検証の女声 30 声の t08・t09 を D で描いて
その声の高さで rvoc3hi で鳴らし、ECAPA(素・登録 = t03〜t05 の元の音)の 30 声中の 1 位が本人の率 = self_id。演奏は登録(t00〜t02 の有声)の平均を引く。
表現の名 = [変換_]元[@long]。変換(全て登録の定数): w = 登録の共分散で白くする(収縮 0.1)・w01 / w3 = 収縮 0.01 / 0.3・w64 = 池の PCA 64 次元で白くする・
wg = 白くした後に次元ごとに登録の経験分布 → 標準正規の分位・wst = 池の PCA 32 次元を 0・40・80・120 ms に積んで白くする。@long = D の文脈 0〜240 ms(results/snv_p8b/prereg.yaml)。
wst64 = PCA 64 次元・wstL = 0〜240 ms の 7 枚・wst64L = 両方。w32 = PCA 32 + ZCA・f32 = PCA 32 で、話者に依らない混合正規 G(池の登録・話者適応の学習 4 巡)の下の
尤度 + log|det A| を最大にする話者ごとのアフィン A・b(初期値 = ZCA・results/snv_p9b/prereg.yaml)。
r = 平均も引かない素の表現(P14)・元 aE = 演奏の取り出しの回路の逆 EncS(es=<重み>・操作者の登録 = その声の t00〜t02 で構造を読む)の演奏の平均(paper_snv 段 I)・a32 / s32 = PCA 32 + ZCA の後、時間をずらした共分散(a = 8 フレーム・s = 4 + 8 + 16 の和)の対称の部分の固有ベクトルで話者ごとの正準の向きへ(固有値の大きい順・3 次のモーメントで符号・results/snv_p9c/prereg.yaml)。cer=1 = 本物の男声 12 区間を D で描いて large-v3 の CER(描ける内容・results/snv_p9a/prereg.yaml)。
    cd training && uv run --no-sync python snv_diag/p8a_draw_voice.py [R=pitch,EMA,Y,c_P2b,c_P2e,w_c_P2b,w_c_P2e] [out=results/snv_p8a/draw_voice.json] [cer=1]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nvoc as N
import snv_p1_eval as EV
import snv_p2_nnc as NN
import train_rvoc as TR
import train_snv_p1 as T
import train_snv_p2_cpc as CP

ROOT = EV.ROOT
DEV = EV.DEV
SP = ROOT / "data/snv_e1_feat/sparc"
LAGS = (0, 4, 8, 12, 16, 20)
LAGS_LONG = (0, 8, 16, 24, 32, 40, 48)
ST_LAGS = (0, 8, 16, 24)
ST_LAGS_L = (0, 8, 16, 24, 32, 40, 48)
SHR = {"w": 0.1, "w01": 0.01, "w3": 0.3, "w64": 0.1, "wg": 0.1, "wst": 0.1, "wst64": 0.1, "wstL": 0.1, "wst64L": 0.1, "w32": 0.1, "f32": 0.1, "a32": 0.1, "s32": 0.1, "r": 0.0}
K_G = 64
SAT_IT = 4
EM_IT = 30
AFF_STEPS = 300
SUB = 6
STEPS = 4000
SHRINK = 0.1
CPCS = {"c_P2b": "results/snv_p2b_cpc/last.pt", "c_P2e": "results/snv_p2e_cpc/last.pt", "c_P10": "results/snv_p10_wcpc/last.pt", "c_P14": "results/snv_p14_pcpc/last.pt"}
REPS = ["pitch", "EMA", "Y", "c_P2b", "c_P2e", "w_c_P2b", "w_c_P2e"]


def lagged(x: np.ndarray, lags: tuple = LAGS) -> np.ndarray:
    return np.concatenate([np.concatenate([np.repeat(x[:1], l, 0), x[:len(x) - l]]) if l else x for l in lags], 1)


def whiten(x: np.ndarray, shrink: float = SHRINK) -> np.ndarray:
    s = x.T @ x / len(x)
    s = (1 - shrink) * s + shrink * np.trace(s) / len(s) * np.eye(len(s))
    w, q = np.linalg.eigh(s)
    return (q / np.sqrt(np.maximum(w, 1e-8))) @ q.T


def gmm_logp(g: tuple, Y: torch.Tensor) -> torch.Tensor:
    m, v, w = g
    q = (Y * Y) @ (1 / v).T - 2 * Y @ (m / v).T + ((m * m) / v).sum(-1)
    return (-0.5 * (q + v.log().sum(-1) + m.shape[1] * math.log(2 * math.pi)) + w.log()).logsumexp(-1)


def gmm_fit(Y: torch.Tensor, g: tuple | None) -> tuple:
    if g is None:
        gen = torch.Generator(device=Y.device).manual_seed(0)
        m = Y[torch.randperm(len(Y), generator=gen, device=Y.device)[:K_G]].clone()
        g = (m, Y.var(0, keepdim=True).expand(K_G, -1).clone(), torch.full((K_G,), 1.0 / K_G, device=Y.device))
    for _ in range(EM_IT):
        m, v, w = g
        q = (Y * Y) @ (1 / v).T - 2 * Y @ (m / v).T + ((m * m) / v).sum(-1)
        R = (-0.5 * (q + v.log().sum(-1)) + w.log()).softmax(-1)
        nk = R.sum(0) + 1e-6
        m = (R.T @ Y) / nk[:, None]
        v = ((R.T @ (Y * Y)) / nk[:, None] - m * m).clamp(min=1e-3)
        g = (m, v, nk / nk.sum())
    return g


def fit_affine(X: torch.Tensor, M: torch.Tensor, A: torch.Tensor, b: torch.Tensor, g: tuple) -> tuple[torch.Tensor, torch.Tensor]:
    A, b = A.clone().requires_grad_(True), b.clone().requires_grad_(True)
    op = torch.optim.Adam([A, b], 3e-3)
    n = M.sum(1).float()
    for _ in range(AFF_STEPS):
        lp = gmm_logp(g, X @ A + b[:, None])
        loss = -((lp * M).sum() + (n * torch.linalg.slogdet(A)[1]).sum()) / n.sum()
        op.zero_grad()
        loss.backward()
        op.step()
    return A.detach(), b.detach()


def sat(xs: list[np.ndarray]) -> tuple:
    S, N, d = len(xs), max(len(x) for x in xs), xs[0].shape[1]
    X, M = torch.zeros(S, N, d, device=DEV), torch.zeros(S, N, dtype=torch.bool, device=DEV)
    for i, x in enumerate(xs):
        X[i, :len(x)], M[i, :len(x)] = torch.from_numpy(x).float(), True
    A = torch.stack([torch.from_numpy(whiten(x)).float() for x in xs]).to(DEV)
    b = torch.zeros(S, d, device=DEV)
    g = None
    for it in range(SAT_IT):
        sc = (X @ A + b[:, None])[M].std()
        A, b = A / sc, b / sc
        g = gmm_fit((X @ A + b[:, None])[M], g)
        A, b = fit_affine(X, M, A, b, g)
        with torch.no_grad():
            ll = float((gmm_logp(g, X @ A + b[:, None]) * M).sum() / M.sum() + (M.sum(1) * torch.linalg.slogdet(A)[1]).sum() / M.sum())
        print("sat", it, "loglik/frame", round(ll, 3), flush=True)
    return g


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    reps = opt.get("R", ",".join(REPS)).split(",")
    from snv_a0_measure import load, to16
    from snv_n0_measure import trim
    from speechbrain.inference.speaker import EncoderClassifier
    torch.manual_seed(0)
    split = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())
    pool = split["train"][::4]
    tg = [v for v in split["val"] if v.startswith("ef")]
    cpcs = {}
    for k, ck in CPCS.items():
        if not any(k in r for r in reps):
            continue
        st = torch.load(ROOT / opt.get(k, ck), map_location="cpu", weights_only=False)
        cm = CP.CPC().to(DEV).eval()
        cm.load_state_dict(st["cpc"])
        pc = None
        if "white" in st:
            wh = CP.WHead(st["dw"]).to(DEV).eval()
            wh.load_state_dict(st["white"])
            pc = wh.pc
        cpcs[k] = (cm, st["sig"], pc)

    def raw(v: str, j: int, need: set) -> dict:
        return raw_rel(f"e1/{v}/t0{j}", SP / f"e1/{v}/t0{j}.npz", need)

    def raw_rel(rel: str, ema: Path, need: set) -> dict:
        u = T.load_utt(T.FEAT / f"{rel}.npz")
        f = {"u": u, "vm": u["v"] > 0, "Y": u["y"][1:29].T}
        if "EMA" in need:
            f["EMA"] = NN.ema_at(ema, u["y"].shape[1])
        for k, (cm, sig, pc) in cpcs.items():
            if k in need:
                with torch.no_grad():
                    c = cm.encode(torch.from_numpy(CP.x_of(u, sig))[None].to(DEV))[1]
                    f[k] = (pc(c.transpose(1, 2))[0] if pc is not None else c[0].T).cpu().numpy()
        return f

    def parse(r: str) -> tuple[str, str, tuple]:
        name, _, lg = r.partition("@")
        tf, _, b = name.partition("_") if name.split("_", 1)[0] in SHR else ("", "", name)
        return tf, b, LAGS_LONG if lg == "long" else LAGS

    need = {parse(r)[1] for r in reps}
    es_net = None
    if "aE" in need:
        import train_snv_p11 as P11
        st_es = torch.load(ROOT / opt["es"], map_location="cpu", weights_only=False)
        es_net = P11.EncS().to(DEV).eval()
        es_net.load_state_dict(st_es["Es"])

    def add_ae(enr: list[dict], fs: list[dict]) -> None:
        tt = lambda f: (torch.from_numpy(f["u"]["y"])[None].to(DEV), torch.from_numpy(f["u"]["v"])[None].to(DEV))
        with torch.no_grad():
            ry = torch.cat([tt(f)[0] for f in enr], -1)
            rv = torch.cat([tt(f)[1] for f in enr], -1)
            S = es_net.structure(P11.xin(ry, rv), torch.ones(rv.shape, dtype=torch.bool, device=DEV))
            for f in enr + fs:
                if "aE" not in f:
                    y, v = tt(f)
                    f["aE"] = es_net(P11.xin(y, v), S)[0][0].cpu().numpy().T
    pca: dict = {}
    for b in {parse(r)[1] for r in reps if parse(r)[0] in ("w64", "w32", "f32", "a32", "s32") or parse(r)[0].startswith("wst")}:
        xs = []
        for v in pool:
            enr = [raw(v, j, {b}) for j in (0, 1, 2)]
            x = np.concatenate([f[b][f["vm"]] for f in enr])
            xs.append((x - x.mean(0))[::SUB])
        x = np.concatenate(xs)
        w, q = np.linalg.eigh(x.T @ x / len(x))
        pca[b] = q[:, ::-1]
        print("pca", b, "var64", round(float(w[::-1][:64].sum() / w.sum()), 3), "var32", round(float(w[::-1][:32].sum() / w.sum()), 3), flush=True)

    gmm = None
    fb = [parse(r)[1] for r in reps if parse(r)[0] == "f32"]
    if fb:
        P32 = pca[fb[0]][:, :32]
        xs = []
        for v in pool:
            enr = [raw(v, j, {fb[0]}) for j in (0, 1, 2)]
            x = np.concatenate([f[fb[0]][f["vm"]] for f in enr])
            xs.append(np.ascontiguousarray(((x - x.mean(0)) @ P32)[::2]))
        gmm = sat(xs)

    def make_tf(tf: str, b: str, enr: list[dict]):
        vx = lambda g: np.concatenate([g(f[b])[f["vm"]] for f in enr])
        mu = vx(lambda x: x).mean(0)
        if tf == "":
            return lambda x: x - mu
        if tf == "r":
            return lambda x: x
        if tf in ("w", "w01", "w3"):
            W = whiten(vx(lambda x: x - mu), SHR[tf])
            return lambda x: (x - mu) @ W
        if tf == "w64":
            P = pca[b][:, :64]
            W = whiten(vx(lambda x: (x - mu) @ P), SHR[tf])
            return lambda x: (x - mu) @ P @ W
        if tf in ("a32", "s32"):
            P = pca[b][:, :32]
            W = whiten(vx(lambda x: (x - mu) @ P), SHR[tf])
            K = np.zeros((32, 32))
            for tau in ((8,) if tf == "a32" else (4, 8, 16)):
                for f in enr:
                    z, m = (f[b] - mu) @ P @ W, f["vm"]
                    ok = m[:-tau] & m[tau:]
                    c = z[:-tau][ok].T @ z[tau:][ok] / max(1, ok.sum())
                    K += (c + c.T) / 2 / len(enr)
            e, U = np.linalg.eigh(K)
            U = U[:, ::-1]
            zz = vx(lambda x: (x - mu) @ P @ W) @ U
            U = U * np.where((zz ** 3).mean(0) >= 0, 1.0, -1.0)
            return lambda x: (x - mu) @ P @ W @ U
        if tf in ("w32", "f32"):
            P = pca[b][:, :32]
            xe = vx(lambda x: (x - mu) @ P)
            W = whiten(xe, SHR[tf])
            if tf == "w32":
                return lambda x: (x - mu) @ P @ W
            Xt = torch.from_numpy(np.ascontiguousarray(xe[::2])).float().to(DEV)[None]
            A, bb = fit_affine(Xt, torch.ones(Xt.shape[:2], dtype=torch.bool, device=DEV), torch.from_numpy(W).float().to(DEV)[None], torch.zeros(1, 32, device=DEV), gmm)
            A, bb = A[0].cpu().numpy().astype(np.float64), bb[0].cpu().numpy().astype(np.float64)
            return lambda x: (x - mu) @ P @ A + bb
        if tf == "wg":
            W = whiten(vx(lambda x: x - mu), SHR[tf])
            E = np.sort(vx(lambda x: (x - mu) @ W), 0)
            from scipy.special import ndtri
            return lambda x: np.stack([ndtri((np.searchsorted(E[:, d], u) + 0.5) / (len(E) + 1)) for d, u in enumerate(((x - mu) @ W).T)], 1)
        P = pca[b][:, :64 if "64" in tf else 32]
        stk = lambda x: lagged((x - mu) @ P, ST_LAGS_L if tf.endswith("L") else ST_LAGS)
        W = whiten(vx(stk), SHR[tf])
        return lambda x: stk(x) @ W

    def voice_feats(v: str, js: tuple, need: set) -> list[np.ndarray]:
        enr = [raw(v, j, need) for j in (0, 1, 2)]
        return rows_for(enr, [raw(v, j, need) if j > 2 else enr[j] for j in js])

    def rows_for(enr: list[dict], fs: list[dict]) -> list[dict]:
        if es_net is not None:
            add_ae(enr, fs)
        tfs = {r: make_tf(parse(r)[0], parse(r)[1], enr) for r in reps if r != "pitch"}
        out = []
        for f in fs:
            pv = np.stack([f["u"]["lf"], f["u"]["v"]], 1)
            row = {"u": f["u"]}
            for r in reps:
                x = pv if r == "pitch" else np.concatenate([tfs[r](f[parse(r)[1]]), pv], 1)
                row[r] = lagged(x, parse(r)[2]).astype(np.float32)
            out.append(row)
        return out

    print("features (pool)", len(pool), flush=True)
    trn = {r: [] for r in reps}
    tgt = []
    for v in pool:
        for row in voice_feats(v, (0, 1, 2, 3, 4), need):
            for r in reps:
                trn[r].append(np.ascontiguousarray(row[r][::SUB]))
            tgt.append(np.ascontiguousarray(row["u"]["y"].T[::SUB]))
    Yt = torch.from_numpy(np.concatenate(tgt)).to(DEV)
    print("frames", len(Yt), flush=True)
    tests = {v: voice_feats(v, (8, 9), need) for v in tg}
    gen = EV.load_gen()
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})

    def emb(w16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(w16).float()[None].to(DEV)).reshape(-1)
        return torch.nn.functional.normalize(e, dim=0).cpu().numpy()

    w16 = lambda w: to16(w.astype(np.float64), EV.SR).astype(np.float32)
    Et = {v: emb(np.concatenate([to16(*load(ROOT / f"data/snv_e1/{v}/t0{j}.wav")).astype(np.float32) for j in (3, 4, 5)])) for v in tg}

    @torch.no_grad()
    def render(yo: np.ndarray, u: dict) -> np.ndarray:
        y = torch.from_numpy(np.ascontiguousarray(yo.T))[None].to(DEV)
        lf = torch.from_numpy(u["lf"])[None].to(DEV)
        vv = torch.from_numpy(u["v"])[None].to(DEV)
        cond = torch.cat([y[:, :25], lf[:, None], vv[:, None], y[:, 25:29].clamp(0, 1.5) * vv[:, None]], 1)
        f0 = torch.from_numpy(u["f0"])[None].to(DEV)
        return gen(cond, TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]

    def self_id(dec) -> tuple[float, float]:
        hit, l1 = [], []
        for v in tg:
            for row in tests[v]:
                yo = dec(row)
                l1.append(float(np.abs(yo - row["u"]["y"].T).mean()))
                e = emb(trim(w16(render(yo, row["u"]))))
                hit.append(max(tg, key=lambda x: float(e @ Et[x])) == v)
        return round(float(np.mean(hit)), 4), round(float(np.mean(l1)), 4)

    res: dict = {"prereg": "results/snv_p8a/prereg.yaml", "pool_voices": len(pool), "test_voices": len(tg), "trials": 2 * len(tg), "chance": round(1 / len(tg), 4), "self_id": {}, "val_l1": {}}
    res["self_id"]["COPY"], res["val_l1"]["COPY"] = self_id(lambda row: row["u"]["y"].T)
    print("COPY", res["self_id"]["COPY"], flush=True)
    do_cer = opt.get("cer", "0") == "1"
    if do_cer:
        from faster_whisper import WhisperModel
        from snv_a0_measure import cer, kana
        asr = WhisperModel("large-v3", device=DEV, compute_type="float16")

        def tr(w: np.ndarray) -> str:
            segs, _ = asr.transcribe(w, language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
            return kana("".join(x.text for x in segs))

        srcs = json.loads((EV.A1 / "convert_meta.json").read_text())["sources"]
        ref_txt = {s_: tr(to16(*load(T.FEAT / f"real/{s_}.wav")).astype(np.float32)) for s_ in srcs}
        rrows = []
        for sp in sorted({s_.rsplit("_", 1)[0] for s_ in srcs}):
            ss = [s_ for s_ in srcs if s_.startswith(sp + "_")]
            rrows += list(zip(ss, rows_for([raw_rel(f"real/{sp}_enroll", SP / f"a1src/{sp}_enroll.npz", need)], [raw_rel(f"real/{s_}", SP / f"a1src/{s_}.npz", need) for s_ in ss])))
        res["cer"] = {}

        def d_cer(dec) -> float:
            return round(float(np.mean([cer(ref_txt[s_], tr(w16(render(dec(row), row["u"])))) for s_, row in rrows])), 4)

        res["cer"]["COPY"] = d_cer(lambda row: row["u"]["y"].T)
        print("COPY cer", res["cer"]["COPY"], flush=True)
    for r in reps:
        X = torch.from_numpy(np.concatenate(trn[r])).to(DEV)
        mx, sx = X.mean(0), X.std(0) + 1e-5
        net = nn.Sequential(nn.Linear(X.shape[1], 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, T.D_Y)).to(DEV)
        op = torch.optim.Adam(net.parameters(), 1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(op, STEPS)
        g = torch.Generator(device=DEV).manual_seed(0)
        for _ in range(STEPS):
            i = torch.randint(0, len(X), (4096,), device=DEV, generator=g)
            loss = (net((X[i] - mx) / sx) - Yt[i]).abs().mean()
            op.zero_grad()
            loss.backward()
            op.step()
            sch.step()
        net.eval()
        dec = lambda row: net((torch.from_numpy(row[r]).to(DEV) - mx) / sx).detach().cpu().numpy()
        res["self_id"][r], res["val_l1"][r] = self_id(dec)
        if do_cer:
            res["cer"][r] = d_cer(dec)
        print(r, "train_l1", round(float(loss), 4), "self_id", res["self_id"][r], "val_l1", res["val_l1"][r], "cer", res.get("cer", {}).get(r), flush=True)
        del X, net
        torch.cuda.empty_cache()
    out = ROOT / opt.get("out", "results/snv_p8a/draw_voice.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
