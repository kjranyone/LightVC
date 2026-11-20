"""段 P18 補足 2(results/snv_p18/prereg.yaml の supplement_2・学習なし・ASR なし): 凍結した符号の、声を跨いだ対応(NNC)が、どの段・どの閉じた形の正規化で動くか。
NNC = 源の有声のフレームごとに、目標の声の参照(j 0〜5 の有声のフレーム)から表現で最も近いフレームを選び、両者の EMA(SPARC・物差しとしてだけ・話者ごとに標準化)の相関を次元ごとに取って平均(snv_p2_nnc.py と同じ)。
条件(名前 = 表現 + 入力の正規化):
  表現: Yn = 包絡 c1..c24 と周期性を登録の平均と標準偏差で正規化 / c = CPC の文脈 − 登録の平均(256)/ pw = その PCA 32(白くしない・遅れなし)/ w0 = PCA 32 を白くする(遅れなし)/
        W = PCA 32 × 遅れ 4 を白くする(P8 の符号・縮み 0.1)/ W_s3・W_s6 = 縮み 0.3・0.6 / W_k16 = PCA 16 × 遅れ 4 / W_all = 統計を有声でなく全てのフレームで
        w0_s3・w0_s6 = 遅れなしで縮み 0.3・0.6 / W_l2 = 遅れ 0・40 ms の 2 枚(段 P19 の候補)
        w0_s6_cmn128 = w0_s6 に内容ごとの平均の正規化(snv_cmn.py・符号帳 results/snv_p21/codebook_k128.npz・段 P20 補足 2)
        段 P23 step a(results/snv_p23/prereg.yaml): cmn_it2・cmn_it3 = 級 128 を 2 回・3 回繰り返す / cmn_c2f = 級 8 → 32 → 128 を順に / cmn_c2f_it2 = その後に級 128 をもう 1 回 /
        cmn_t4 = 事後確率の温度 4 倍 / cmn_tau5・cmn_tau80 = 事前の重み 5・80 フレーム
  入力(符号の前・源と登録に同じ): cmn_m・cmvn_m = 包絡 c1..c24 の登録の平均(と標準偏差)を、E1 の学習の男声の平均に合わせる / cmn_t・cmvn_t = 目標の声の参照の値に合わせる /
        warp110・warp120・warp130 = 包絡の周波数の一律の伸縮 α(snv_p1_eval.warp_utt)
        dc03・dc06・dc15・dc20(段 P25 補足 2・c と w0_s6_cmn128 の上)= 音量 c0 の、発話の中央値からのずれを 0.3 / 0.6 / 1.5 / 2.0 倍に(音量の動きの圧縮と伸長・符号の入力では c0 の差分だけが変わる)
組: valf・valm・org_g0・org_g0_ch・org_w・org_w_ch・jv・cv。出力 = results/snv_p18/nnc_variants.json。sets=rec2,valf,valm などで組を選ぶ(rec2 = 検証の声の、本物の非音声の音源を足した版 data/snv_p_feat_rec2・段 P25)。
cpc=<results の下の名前>[@<保存の名前>](段 P24)= 符号だけを別の CPC の重みに替える(行の名前は <条件>@<名前>)。pca= を付けないと条件は c と Yn だけ。
段 P71 step a(results/snv_p71/prereg.yaml): h8・h16・h32・h64 = 文脈 c を、直前の 8 / 16 / 32 / 64 フレーム(40〜320 ms)の入力だけから計算する(それより前は発話の頭と同じ 0 の詰め・全ての受容野 133 フレームで c と一致)/
        z = 符号器の出力(受容野 7 フレーム)・a1〜a5 = 文脈の網の 1〜5 段目の後(受容野 9・13・21・37・69 フレーム・c = 6 段目の後)。どれも c と同じ扱い(登録の平均を引く・PCA なし)。
段 P71 step b: c+<入力>・z+<入力>(入力 = cmn_m・cmvn_m・cmn_t・cmvn_t・warp90・110・120・130・140)= 文脈と符号器の出力の上で、同じ入力の写しを測る。
段 P72: c+w2_<lo>_<hi>(10〜14 = 1.0〜1.4)= 包絡の周波数の 2 区間の伸縮(源の 900 Hz より下を lo 倍・上を hi 倍・lo = hi は warp と一致)。
段 P73(results/snv_p73/prereg.yaml): tpool=mi / mv / r / x:<組>= 目標の参照を、女声の 4 声(既定)から、E1 の学習の男声 4 声(49〜52 番目)/ E1 の検証の男声 / devA の 8 人(先頭 64 人の 4・12・…・60 番目・登録 e0〜e5)/
        data/snv_p73/sets.json の組の声に替える(源と同じ話者の目標は 1 つずらす)。tpool=self = 源の話者自身の登録の発話。tn=<本数>(既定 6)= 目標の参照の発話の数(self は登録の先頭から)。pf=<数>(self だけ)= 参照の有声のフレームを先頭からその数に切る(足りない源は除く・行の名前に :pf<数>)。行の名前は <条件>><tpool>。条件 E = EMA(遅れ 4 枚)そのもので最も近いフレームを選ぶ(物差しの天井)。sets=x:<組> = その組を源にする。
pca=<その符号の PCA の npy>(snv_diag/p8b_pca.py)で白くする条件も測る。cb128=<その符号の級 128 の符号帳>(snv_cmn.py fit)で級の正規化の条件の符号帳を替える。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/p18_nnc.py [sys=snv_p8_wst] [n=64] [conds=...] [cpc=snv_p2d_cpc]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p18_align as AL
import probe_rig as G
import snv_p1_eval as EV
import train_snv_p1 as T

ROOT = G.ROOT
DEV = EV.DEV
SETS = ("valf", "valm", "org_g0", "org_g0_ch", "org_w", "org_w_ch", "jv", "cv")
REPS = {"Yn": {}, "c": {}, "pw": {"lags": (0,), "white": False}, "w0": {"lags": (0,)}, "W": {}, "W_s3": {"shrink": 0.3}, "W_s6": {"shrink": 0.6}, "W_k16": {"k": 16}, "W_all": {"voiced": False},
        "w0_s3": {"lags": (0,), "shrink": 0.3}, "w0_s6": {"lags": (0,), "shrink": 0.6}, "W_l2": {"lags": (0, 8)},
        "w0_s6_cmn128": {"lags": (0,), "shrink": 0.6, "cb": "results/snv_p21/codebook_k128.npz"}}
K128, K32, K8 = "results/snv_p21/codebook_k128.npz", "results/snv_p23/codebook_k32.npz", "results/snv_p23/codebook_k8.npz"
REPS |= {f"cmn_{n}": {"lags": (0,), "shrink": 0.6} | o for n, o in {"it2": {"cb": (K128, K128)}, "it3": {"cb": (K128, K128, K128)}, "c2f": {"cb": (K8, K32, K128)}, "c2f_it2": {"cb": (K8, K32, K128, K128)},
                                                              "t4": {"cb": K128, "temp": 4.0}, "tau5": {"cb": K128, "tau": 5.0}, "tau80": {"cb": K128, "tau": 80.0}}.items()}
HIST = {f"h{n}": n for n in (8, 16, 32, 64)}
TAPS = {"z": 0} | {f"a{i}": i for i in range(1, 6)}
CL = ("c",) + tuple(HIST) + tuple(TAPS)
REPS |= {r: {} for r in CL[1:]}
INPUTS = ("cmn_m", "cmvn_m", "cmn_t", "cmvn_t", "warp110", "warp120", "warp130")
DYN = ("dc03", "dc06", "dc15", "dc20")
CONDS = tuple(REPS) + tuple(f"W+{i}" for i in INPUTS) + tuple(f"{r}+{i}" for r in ("c", "w0_s6_cmn128") for i in DYN)
CONDS += tuple(f"{r}+{i}" for r in ("c", "z") for i in INPUTS + ("warp90", "warp140"))
W2 = (10, 11, 12, 13, 14)
W2_HZ = 900.0
CONDS += tuple(f"c+w2_{lo}_{hi}" for lo in W2 for hi in W2) + ("E",)


def warp2_utt(u: dict, lo: float, hi: float) -> dict:
    import snv_p0c_anchor as A
    _, dct = A.PA._np_consts()
    y = u["y"].copy()
    env = np.concatenate([y[:1] * 10 + T.PA.C0_SIL, y[1:25] * 10], 0).astype(np.float64)
    L = dct[:25].T @ env
    fm = A.mel_hz()
    g = np.where(fm < lo * W2_HZ, fm / lo, W2_HZ + (fm - lo * W2_HZ) / hi)
    idx = np.interp(g, fm, np.arange(len(fm)))
    i0 = np.clip(np.floor(idx).astype(int), 0, len(fm) - 2)
    w = (idx - i0)[:, None]
    ew = (dct[:25] @ (L[i0] * (1 - w) + L[i0 + 1] * w)).astype(np.float32)
    y[0], y[1:25] = (ew[0] - T.PA.C0_SIL) / 10, ew[1:25] / 10
    return u | {"y": y}


def vstat(us: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    y = np.concatenate([u["y"][1:25][:, u["v"] > 0.5] for u in us], 1)
    return y.mean(1, keepdims=True), y.std(1, keepdims=True) + 1e-6


def norm_in(us: list[dict], enr: list[int], ref: tuple, var: bool) -> list[dict]:
    m, s = vstat([us[i] for i in enr])
    out = []
    for u in us:
        y = u["y"].copy()
        y[1:25] = ((y[1:25] - m) / s * ref[1] + ref[0]) if var else (y[1:25] - m + ref[0])
        out.append(u | {"y": y.astype(np.float32)})
    return out


@torch.no_grad()
def ctx_of(Pl, y: torch.Tensor, v: torch.Tensor, rep: str) -> torch.Tensor:
    if rep not in HIST and rep not in TAPS:
        return Pl.ctx(y, v)[0].T
    import torch.nn.functional as F
    import train_snv_p2_cpc as CP
    x = CP.x_torch(torch.cat([y, v[:, None]], 1), Pl.sig)
    if rep in HIST:
        n = HIST[rep]
        w = F.pad(x, (n - 1, 0)).unfold(2, n, 1)[0].permute(1, 0, 2)
        h = torch.cat([Pl.cpc.encode(w[i:i + 512])[1][:, :, -1] for i in range(0, len(w), 512)])
        h[:n - 1] = Pl.cpc.encode(x)[1][0].T[:n - 1]
        return h
    m = Pl.cpc
    h = F.relu(m.e1(F.pad(x, (2, 0))))
    h = F.relu(m.e2(F.pad(h, (4, 0))))
    c = F.layer_norm(m.e3(h).transpose(1, 2), (CP.CH,)).transpose(1, 2)
    if TAPS[rep]:
        c = m.ar_in(c)
        for blk in m.ar[:TAPS[rep]]:
            c = blk(c)
    return c[0].T


@torch.no_grad()
def code(Pl, us: list[dict], enr: list[int], rep: str) -> list[torch.Tensor]:
    o = {"k": None, "lags": (0, 8, 16, 24), "shrink": 0.1, "white": True, "voiced": True, "cb": None, "temp": 1.0, "tau": None} | REPS[rep]
    ys, vs = [AL.tt(u, "y") for u in us], [AL.tt(u, "v") for u in us]
    wt = lambda v: v[0] if o["voiced"] else torch.ones_like(v[0])
    if rep == "Yn":
        X = [y[0, 1:29].T for y in ys]
    else:
        cs = [ctx_of(Pl, y, v, rep) for y, v in zip(ys, vs)]
        w = torch.cat([wt(vs[i]) for i in enr])
        mu = (torch.cat([cs[i] for i in enr]) * w[:, None]).sum(0) / w.sum()
        X = [c - mu for c in cs]
        if rep not in CL:
            X = [x @ Pl.P[:, : o["k"] or Pl.P.shape[1]] for x in X]
            X = [torch.cat([torch.cat([x[:1].expand(l, -1), x[: len(x) - l]]) if l else x for l in o["lags"]], 1) for x in X]
    if rep == "Yn" or o["white"] and rep not in CL:
        w = torch.cat([wt(vs[i]) for i in enr]).double()
        E = torch.cat([X[i] for i in enr]).double()
        mu = (E * w[:, None]).sum(0) / w.sum()
        if rep == "Yn":
            sd = (((E - mu) ** 2 * w[:, None]).sum(0) / w.sum()).sqrt().clamp(min=1e-6)
            return [((x.double() - mu) / sd).float() for x in X]
        cov = ((E - mu) * w[:, None]).T @ (E - mu) / w.sum()
        D = cov.shape[0]
        cov = (1 - o["shrink"]) * cov + o["shrink"] * cov.diagonal().sum() / D * torch.eye(D, dtype=cov.dtype, device=cov.device)
        e, q = torch.linalg.eigh(cov)
        Wm = (q / e.clamp(min=1e-8).sqrt()) @ q.T
        X = [((x.double() - mu) @ Wm).float() for x in X]
        if o["cb"]:
            import snv_cmn as CM
            for cb in (o["cb"],) if isinstance(o["cb"], str) else o["cb"]:
                c, s2 = CM.load(ROOT / cb)
                c, s2 = c.to(DEV), s2 * o["temp"]
                delta = CM.offsets(torch.cat([X[i] for i in enr]), torch.cat([wt(vs[i]) for i in enr]), c, s2, CM.TAU if o["tau"] is None else o["tau"])
                X = [CM.apply(x, c, s2, delta) for x in X]
        return X
    return X


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    n, tag = int(opt.get("n", 64)), opt.get("sys", "snv_p8_wst")
    conds = opt.get("conds", ",".join(CONDS)).split(",")
    assert all(c in CONDS for c in conds), conds
    out = ROOT / opt.get("out", "results/snv_p18/nnc_variants.json")
    split = json.loads((ROOT / "results" / tag.partition("@")[0] / "split.json").read_text())
    sets = AL.build_sets(opt["sets"].split(",") if "sets" in opt else list(SETS), n, split)
    s = EV.System(tag, None)
    Pl = s.Pl
    cpc = opt.get("cpc")
    if cpc:
        import train_snv_p3 as P3
        ck = ROOT / "results" / cpc.partition("@")[0] / f"{cpc.partition('@')[2] or 'last'}.pt"
        assert "pca" in opt or all(c.partition("+")[0] in CL + ("Yn", "E") for c in conds), conds
        Pl = (P3.PlayW(ck, ROOT / opt["pca"], lags=(0,), shrink=0.6) if "pca" in opt else P3.Play(ck)).to(DEV).eval()
    if "cb128" in opt:
        sw = lambda x: opt["cb128"] if x == K128 else x
        for o in REPS.values():
            if "cb" in o:
                o["cb"] = sw(o["cb"]) if isinstance(o["cb"], str) else tuple(sw(x) for x in o["cb"])
    tp, tn, pf = opt.get("tpool", "f"), int(opt.get("tn", 6)), int(opt.get("pf", 0))
    if tp == "self":
        tg4, tref, tema = [], {}, {}
    elif tp == "f":
        _, tg4 = G.targets()
        tref = {v: [EV.utt(f"a0/{v}_{j}") for j in range(6)] for v in tg4}
        tema = {v: [str(G.SP / f"a0/{v}_{j}.npz") for j in range(6)] for v in tg4}
    elif tp in ("mi", "mv"):
        tg4 = [x for x in split["train"] if x.startswith("em")][48:52] if tp == "mi" else [x for x in split["val"] if x.startswith("em")]
        tref = {v: [T.load_utt(T.FEAT / f"e1/{v}/t0{j}.npz") for j in range(6)] for v in tg4}
        tema = {v: [str(G.SP / f"e1/{v}/t0{j}.npz") for j in range(6)] for v in tg4}
    elif tp == "r":
        man = [x for x in json.loads((ROOT / "data/cv22_ja/dev_manifest.json").read_text())["speakers"] if x["split"] == "devA"][3:64:8]
        tg4 = [x["speaker"] for x in man]
        tref = {x["speaker"]: [T.load_utt(T.FEAT / f"cv/{x['speaker']}_{c['name']}.npz") for c in x["enroll"][:6]] for x in man}
        tema = {x["speaker"]: [str(G.SP / f"cv/{x['speaker']}_{c['name']}.npz") for c in x["enroll"][:6]] for x in man}
    else:
        its = G.ext_items(tp[2:])
        tg4 = sorted({it["spk"] for it in its})
        one = {v: [it for it in its if it["spk"] == v] for v in tg4}
        tref = {v: ([it["u"] for it in one[v]] + one[v][0]["enr"])[:6] for v in tg4}
        tema = {v: ([it["ema"][0] for it in one[v]] + one[v][0]["ema"][1:])[:6] for v in tg4}
    tref, tema = {v: u[:tn] for v, u in tref.items()}, {v: u[:tn] for v, u in tema.items()}
    K = len(tg4)

    def pair(it: dict) -> tuple:
        out = []
        if tp == "self":
            return ("self",)
        for j in (it["idx"] % K, (it["idx"] + K // 2) % K):
            while tg4[j] == it["spk"] or tg4[j] in out:
                j = (j + 1) % K
            out.append(tg4[j])
        return tuple(out)

    fam = [T.load_utt(T.FEAT / f"e1/{v}/{t}.npz") for v in [x for x in split["train"] if x.startswith("em")][::4] for t in ("t00", "t01")]
    ref_m = vstat(fam)
    ref_t = {v: vstat(tref[v][:3]) for v in tg4}

    def ema(paths: list[str], us: list[dict], lag: bool = False) -> list[torch.Tensor]:
        M = [torch.from_numpy(u["v"] > 0.5).to(DEV) for u in us]
        E = [AL.ema_at(p, u["y"].shape[1]) for p, u in zip(paths, us)]
        A = torch.cat([e[m] for e, m in zip(E, M)])
        E = [(e - A.mean(0)) / A.std(0).clamp(min=1e-6) for e in E]
        return [(AL.stack(e) if lag else e)[m] for e, m in zip(E, M)]

    def prep(us: list[dict], enr: list[int], inp: str | None, v: str) -> list[dict]:
        if inp is None:
            return us
        if inp.startswith("w2_"):
            return [warp2_utt(u, int(inp.split("_")[1]) / 10, int(inp.split("_")[2]) / 10) for u in us]
        if inp.startswith("warp"):
            return [EV.warp_utt(u, int(inp[4:]) / 100) for u in us]
        if inp.startswith("dc"):
            out = []
            for u in us:
                y = u["y"].copy()
                y[0] = np.median(y[0]) + int(inp[2:]) / 10 * (y[0] - np.median(y[0]))
                out.append(u | {"y": y.astype(np.float32)})
            return out
        return norm_in(us, enr, ref_m if inp.endswith("_m") else ref_t[v], inp.startswith("cmvn"))

    temaz = {v: torch.cat(ema(tema[v], tref[v])) for v in tg4}
    temac = {v: torch.cat(ema(tema[v], tref[v], True)) for v in tg4}
    tmask = {v: [torch.from_numpy(u["v"] > 0.5).to(DEV) for u in tref[v]] for v in tg4}
    res = json.loads(out.read_text()) if out.exists() else {"prereg": "results/snv_p18/prereg.yaml(supplement_2)", "sys": tag, "pop(中央値)": {}, "cv_speakers": {}, "rows": {}}
    cache_e: dict = {}
    for cond in conds:
        name = (f"{cond}@{cpc}" if cpc else cond) + ("" if tp == "f" else f">{tp}") + (f":pf{pf}" if pf else "")
        if name in res["rows"] and opt.get("redo") != "1":
            continue
        rep, _, inp = cond.partition("+")
        inp = inp or None
        pool = temac if rep == "E" else {v: torch.cat([x[m] for x, m in zip(code(Pl, tref[v], [0, 1, 2][:len(tref[v])], rep), tmask[v])]) for v in tg4}
        rows = []
        for nm, its in sets.items():
            for it in its:
                us = [it["u"]] + it["enr"]
                enr = list(range(1, len(us)))
                key = (nm, it["src"])
                if key not in cache_e:
                    cache_e[key] = ema(it["ema"], us)[0]
                e = cache_e[key]
                m = torch.from_numpy(it["u"]["v"] > 0.5).to(DEV)
                for v in pair(it):
                    xs = ema(it["ema"], us, True) if rep == "E" else [c[torch.from_numpy(u["v"] > 0.5).to(DEV)] for c, u in zip(code(Pl, prep(us, enr, inp, v), enr, rep), us)]
                    pl, te = (torch.cat(xs[1:1 + tn]), torch.cat(ema(it["ema"], us)[1:1 + tn])) if tp == "self" else (pool[v], temaz[v])
                    if pf and tp == "self":
                        if len(pl) < pf:
                            continue
                        pl, te = pl[:pf], te[:pf]
                    rows.append({"set": nm, "spk": it["spk"], "src": it["src"], "valid": bool(it["valid"]), "tgt": v, "nnc": AL.corr(e, te[AL.nn(xs[0], pl)])})
        cond = name
        res["rows"][cond] = rows
        res["pop(中央値)"][cond] = {nm: round(float(np.median([r["nnc"] for r in rows if r["set"] == nm and r["valid"]])), 3) for nm in sets}
        cv = [r for r in rows if r["set"] == "cv" and r["valid"]]
        sk = sorted({r["spk"] for r in cv})
        x = np.array([np.mean([r["nnc"] for r in cv if r["spk"] == sp]) for sp in sk])
        if len(x):
            res["cv_speakers"][cond] = {"p5_25_50_75_95": [round(float(q), 3) for q in np.percentile(x, [5, 25, 50, 75, 95])], "frac_ge_0.30": round(float((x >= 0.30).mean()), 3)}
        print(cond, json.dumps(res["pop(中央値)"][cond], ensure_ascii=False), json.dumps(res["cv_speakers"].get(cond)), flush=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
