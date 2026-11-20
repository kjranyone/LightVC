"""段 P20 step a(results/snv_p20/prereg.yaml・学習は探りの復号器 D だけ): 声の抜けを学習の分布で取る乱しの族を、C を学ぶ前に物差し D で比べる。
演奏 = 段 P19 で選んだ式(凍結した P2b の CPC の文脈 → PCA 32 → 登録の共分散で白くする・遅れなし・縮み 0.6)。D = 演奏(0〜100 ms の積み重ね)+ 高さ・有声 → Y の MLP(段 P8a と同じ)。
乱しは D の学習の入力だけに掛け(当てる先は加工しない Y)、測るときは掛けない = 「その乱しで学んだ復号器が、乱しの無い演奏から描ける声」を測る。
族:
  none = 乱しなし(段 P19 step a の w32s6 の再現)
  feat = 特徴の上の乱し(包絡の一律の伸縮 0.8〜1.25・傾き ±1・滑らかな音色 ±0.25 = train_snv_p2_cpc の族)を源と登録に同じ値で掛けてから符号にする
  shr = 白くする縮みを発話ごとに一様 [0.2, 0.95] で振る(操作者自身の 2 次の構造の残り方を振る)
  col4・col8 = 白くした符号に、発話ごとの対称の正定値の写像 exp(ε S)(S は対称の乱数・スペクトルのノルム 1・ε = 0.4 / 0.8)を掛ける(2 次の構造の置き違いを振る)
  cmn32・cmn128・cmn512(補足 2・乱しではなく演奏の式)= 内容ごとの平均の正規化: 族の学習の声の白くした符号で K 個の級の符号帳(k 平均)を作り、操作者の登録のフレームの級ごとの平均(柔らかい割り当て・事前の重み 20 フレーム)と
    族の級の中心の差を、フレームごとに級の事後確率で重みづけて引く(閉じた形・登録の定数・因果)。学習でも測るときでも掛ける
量: self_id = 検証の女声 30 声 × t08・t09 を描いた音の本人の率(段 P8a と同じ)/ self_id_m = 生成した対照の男声 24 声 × t01・t02 の本人の率(24 声の中)/
    D の CER = devA の先頭 n 人の有効な源を描いた large-v3 の CER(中央値と平均)/ val_l1。
段 P33(results/snv_p33/prereg.yaml): xc0=1 = 明示の強さ(登録の有声の c0 の平均と標準偏差で標準化した c0)を演奏に 1 次元足す(§5 M34)。PCA の基底の列の数が次元。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/p20_draw.py [F=none,feat,shr,col4,col8] [n=64] [shrink=0.6] [xc0=1] [cpc=results/snv_p2b_cpc/last.pt] [pca=results/snv_p8b/pca_c_P2b.npy] [out=results/snv_p20/draw.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nvoc as N
import p8a_draw_voice as DV
import snv_p1_eval as EV
import train_rvoc as TR
import train_snv_p1 as T
import train_snv_p2_cpc as CP

ROOT = EV.ROOT
DEV = EV.DEV
FAMS = ("none", "feat", "shr", "col4", "col8", "cmn32", "cmn128", "cmn512")
COPIES = {"none": 1, "feat": 2, "shr": 3, "col4": 3, "col8": 3, "cmn32": 1, "cmn128": 1, "cmn512": 1}
TAU = 20.0
EQ = 0.25
TILT = 1.0


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    fams = opt.get("F", ",".join(FAMS)).split(",")
    assert all(f in FAMS for f in fams), fams
    n, shrink, xc0 = int(opt.get("n", 64)), float(opt.get("shrink", 0.6)), opt.get("xc0") == "1"
    tj = tuple(int(x) for x in opt.get("tests", "8,9").split(","))
    out = ROOT / opt.get("out", "results/snv_p20/draw.json")
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, kana, load, to16
    from snv_n0_measure import trim
    from speechbrain.inference.speaker import EncoderClassifier
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    split = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())
    pool = split["train"][::4]
    tg = [v for v in split["val"] if v.startswith("ef")]
    st = torch.load(ROOT / opt.get("cpc", "results/snv_p2b_cpc/last.pt"), map_location="cpu", weights_only=False)
    cm = CP.CPC().to(DEV).eval()
    cm.load_state_dict(st["cpc"])
    sig = st["sig"]
    P = np.load(ROOT / opt.get("pca", "results/snv_p8b/pca_c_P2b.npy"))
    P = P[:, : P.shape[1] if P.shape[1] < P.shape[0] else 32].astype(np.float64)
    Ms, tv, eqb = CP.warp_mats(), CP.tilt_vec(), CP.eq_basis()

    def perturb(u: dict, par: tuple) -> dict:
        ia, tau, amp = par
        y = u["y"].copy()
        env = np.concatenate([y[:1] * 10 + T.PA.C0_SIL, y[1:25] * 10], 0)
        ew = Ms[ia] @ env + tau * tv[:, None] + (amp @ eqb)[:, None]
        y[0], y[1:25] = (ew[0] - T.PA.C0_SIL) / 10, ew[1:25] / 10
        return u | {"y": y.astype(np.float32)}

    def code(u: dict) -> np.ndarray:
        with torch.no_grad():
            return cm.encode(torch.from_numpy(CP.x_of(u, sig))[None].to(DEV))[1][0].T.cpu().numpy().astype(np.float64)

    def color(eps: float) -> np.ndarray:
        g = rng.standard_normal((32, 32))
        w, q = np.linalg.eigh((g + g.T) / 2)
        return (q * np.exp(eps * w / np.abs(w).max())) @ q.T

    CB: dict = {}

    def post(x: np.ndarray, cb: tuple) -> np.ndarray:
        c, s2 = cb
        d = ((x ** 2).sum(1)[:, None] - 2 * x @ c.T + (c ** 2).sum(1)[None]) / (2 * s2)
        g = np.exp(-(d - d.min(1, keepdims=True)))
        return g / g.sum(1, keepdims=True)

    def cmn(x: np.ndarray, xe: np.ndarray, cb: tuple) -> np.ndarray:
        ge = post(xe, cb)
        delta = (ge.T @ xe + TAU * cb[0]) / (ge.sum(0)[:, None] + TAU) - cb[0]
        return x - post(x, cb) @ delta

    def fit_cb(K: int) -> tuple:
        xs = []
        for v in pool:
            us = base[v]
            cs = [code(u) for u in us]
            mu = np.concatenate([c[u["v"] > 0] for c, u in zip(cs[:3], us[:3])]).mean(0)
            W = DV.whiten(np.concatenate([(c - mu)[u["v"] > 0] for c, u in zip(cs[:3], us[:3])]) @ P, shrink)
            xs += [((c - mu) @ P @ W)[u["v"] > 0][::8] for c, u in zip(cs, us)]
        X = torch.from_numpy(np.concatenate(xs)).to(DEV)
        g = torch.Generator(device="cpu").manual_seed(0)
        c = X[torch.randperm(len(X), generator=g)[:K].to(DEV)].clone()
        for _ in range(40):
            a = torch.cdist(X, c).argmin(1)
            for k in range(K):
                m = a == k
                if m.any():
                    c[k] = X[m].mean(0)
        s2 = float(((X - c[torch.cdist(X, c).argmin(1)]) ** 2).sum(1).mean() / X.shape[1])
        print("符号帳", K, "フレーム", len(X), "級の中の分散(次元あたり)", round(s2, 3), flush=True)
        return c.double().cpu().numpy(), s2

    def rows(enr: list[dict], fs: list[dict], fam: str = "none") -> list[dict]:
        out_ = []
        for f in fs:
            par = (int(rng.integers(0, len(Ms))), float(rng.uniform(-TILT, TILT)), rng.uniform(-EQ, EQ, 3)) if fam == "feat" else None
            ee, ff = ([perturb(u, par) for u in enr], perturb(f, par)) if par is not None else (enr, f)
            ce, cf = [code(u) for u in ee], code(ff)
            mu = np.concatenate([c[u["v"] > 0] for c, u in zip(ce, ee)]).mean(0)
            W = DV.whiten(np.concatenate([(c - mu)[u["v"] > 0] for c, u in zip(ce, ee)]) @ P, float(rng.uniform(0.2, 0.95)) if fam == "shr" else shrink)
            x = (cf - mu) @ P @ W
            if fam.startswith("col"):
                x = x @ color(int(fam[3:]) / 10)
            if fam.startswith("cmn"):
                x = cmn(x, np.concatenate([((c - mu) @ P @ W)[u["v"] > 0] for c, u in zip(ce, ee)]), CB[int(fam[3:])])
            if xc0:
                c0e = np.concatenate([u["y"][0][u["v"] > 0] for u in ee])
                x = np.concatenate([x, ((ff["y"][0] - c0e.mean()) / max(float(c0e.std()), 1e-3))[:, None]], 1)
            out_.append({"u": f, "x": DV.lagged(np.concatenate([x, np.stack([f["lf"], f["v"]], 1)], 1)).astype(np.float32)})
        return out_

    lu = T.load_utt
    e1 = lambda v, j: lu(T.FEAT / f"e1/{v}/t0{j}.npz")
    print("features (pool)", len(pool), flush=True)
    base = {v: [e1(v, j) for j in range(5)] for v in pool}
    tests = {v: rows([e1(v, j) for j in (0, 1, 2)], [e1(v, j) for j in tj]) for v in tg}
    dm = json.loads((ROOT / "data/snv_d5/manifest.json").read_text())
    males = [v["voice"] for v in dm["voices"] if (v["type"], v["cap"]) == ("g0", "c1")]
    import probe_rig as G
    org = G.org_items("g0")
    mtests = [(it["spk"], rows(it["enr"], [it["u"]])[0]) for it in org]
    dv = json.loads((ROOT / "results/snv_d5/dev_rows_devA.json").read_text())["ref"]
    man = json.loads((ROOT / "data/cv22_ja/dev_manifest.json").read_text())
    real_src = []
    for sp in [x for x in man["speakers"] if x["split"] == "devA"][:n]:
        ss = [f"{sp['speaker']}_{c['name']}" for c in sp["src"][:2] if dv[f"{sp['speaker']}_{c['name']}"]["text_vs_orig"] <= 0.1]
        en = [lu(T.FEAT / f"cv/{sp['speaker']}_{e['name']}.npz") for e in sp["enroll"]]
        real_src += [(s_, en, lu(T.FEAT / f"cv/{s_}.npz")) for s_ in ss]
    real = [(s_, rows(en, [f_])[0]) for s_, en, f_ in real_src]
    gen = EV.load_gen()
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16")
    w16 = lambda w: to16(w.astype(np.float64), EV.SR).astype(np.float32)

    def emb(x16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(x16).float()[None].to(DEV)).reshape(-1)
        return torch.nn.functional.normalize(e, dim=0).cpu().numpy()

    def tr(w: np.ndarray) -> str:
        segs, _ = asr.transcribe(w, language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(x.text for x in segs))

    Et = {v: emb(np.concatenate([to16(*load(ROOT / f"data/snv_e1/{v}/t0{j}.wav")).astype(np.float32) for j in (3, 4, 5)])) for v in tg}
    Em = {v: emb(to16(*load(ROOT / f"data/snv_d5/{v}/t00.wav")).astype(np.float32)) for v in males}

    @torch.no_grad()
    def render(yo: np.ndarray, u: dict) -> np.ndarray:
        y = torch.from_numpy(np.ascontiguousarray(yo.T))[None].to(DEV)
        lf, vv = torch.from_numpy(u["lf"])[None].to(DEV), torch.from_numpy(u["v"])[None].to(DEV)
        cond = torch.cat([y[:, :25], lf[:, None], vv[:, None], y[:, 25:29].clamp(0, 1.5) * vv[:, None]], 1)
        return gen(cond, TR.excitation(torch.from_numpy(u["f0"])[None].to(DEV), cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]

    res = json.loads(out.read_text()) if out.exists() else {"prereg": "results/snv_p20/prereg.yaml", "shrink": shrink, "n_real": len(real), "fam": {}}
    for fam in fams:
        if fam in res["fam"] and opt.get("redo") != "1":
            continue
        tests_f, mtests_f, real_f = tests, mtests, real
        if fam.startswith("cmn"):
            CB[int(fam[3:])] = fit_cb(int(fam[3:]))
            tests_f = {v: rows([e1(v, j) for j in (0, 1, 2)], [e1(v, j) for j in tj], fam) for v in tg}
            mtests_f = [(it["spk"], rows(it["enr"], [it["u"]], fam)[0]) for it in org]
            real_f = [(s_, rows(en, [f_], fam)[0]) for s_, en, f_ in real_src]
        X, Y = [], []
        for v in pool:
            for _ in range(COPIES[fam]):
                for row in rows(base[v][:3], base[v], fam):
                    X.append(row["x"][:: DV.SUB])
                    Y.append(row["u"]["y"].T[:: DV.SUB])
        X, Yt = torch.from_numpy(np.concatenate(X)).to(DEV), torch.from_numpy(np.concatenate(Y)).to(DEV)
        mx, sx = X.mean(0), X.std(0) + 1e-5
        net = nn.Sequential(nn.Linear(X.shape[1], 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, T.D_Y)).to(DEV)
        op = torch.optim.Adam(net.parameters(), 1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(op, DV.STEPS)
        g = torch.Generator(device=DEV).manual_seed(0)
        for _ in range(DV.STEPS):
            i = torch.randint(0, len(X), (4096,), device=DEV, generator=g)
            loss = (net((X[i] - mx) / sx) - Yt[i]).abs().mean()
            op.zero_grad()
            loss.backward()
            op.step()
            sch.step()
        net.eval()
        dec = lambda row: net((torch.from_numpy(row["x"]).to(DEV) - mx) / sx).detach().cpu().numpy()
        hit, l1, hv = [], [], []
        for v in tg:
            for row in tests_f[v]:
                yo = dec(row)
                l1.append(float(np.abs(yo - row["u"]["y"].T).mean()))
                e = emb(trim(w16(render(yo, row["u"]))))
                hit.append(max(tg, key=lambda x_: float(e @ Et[x_])) == v)
                hv.append(v)
        hm = []
        for v, row in mtests_f:
            e = emb(trim(w16(render(dec(row), row["u"]))))
            hm.append(max(males, key=lambda x_: float(e @ Em[x_])) == v)
        cs = [cer(dv[s_]["orig"], tr(w16(render(dec(row), row["u"])))) for s_, row in real_f] or [float("nan")]
        res["fam"][fam] = {"frames": int(len(X)), "train_l1": round(float(loss), 4), "self_id": round(float(np.mean(hit)), 4), "self_id_m": round(float(np.mean(hm)), 4), "val_l1": round(float(np.mean(l1)), 4),
                           "cer_median": round(float(np.median(cs)), 4), "cer_mean": round(float(np.mean(cs)), 4), "cer_clip1": round(float(np.minimum(cs, 1).mean()), 4), "cer_gt0.5": round(float(np.mean(np.asarray(cs) > 0.5)), 4),
                           "cers": {s_: round(float(c), 4) for (s_, _), c in zip(real_f, cs)}, "hits": [[v_, bool(h_)] for v_, h_ in zip(hv, hit)]}
        print(fam, json.dumps({k: v for k, v in res["fam"][fam].items() if k not in ("cers", "hits")}), flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        del X, Yt, net
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
