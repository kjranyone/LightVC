"""P1 の評価(synth_nam_vc §8・運用の条件): 操作者の音の前段 v3 → E(操作者の登録の平均を引く)→ C(目標の参照の構造・高さ = 操作者の高さ + 固定の k 半音)→ rvoc3hi → 48 kHz。
test: M = 本物の男声 12 区間(外郎売・namikawa・登録 = 先頭 30 s)→ A0F の目標 24 声(構造の参照 = j 0〜2)・k ∈ {0, 5, 7, 12}・ECAPA + LDA の 24 声中の 1 位(登録 = j 0〜2 の元の音・M_alt = j 6〜7)。
      S = 目標の j 3〜5 の自己の reamp(参照 j 0〜2)の 1 位・CER = k 0 / 7 × 目標 4 声の large-v3 の仮名の CER(源の書き起こし 対 出力)・
      TEX = N0r の女声の音素の無い発声 84 本の自己の reamp(参照 = その声の j 0〜2)・UTMOS = M の出力の平均。差 = 系の対ごとに目標の声で束ねた 95% 区間。
dev: VCTK の男声 6 人 × 2 本(登録 = 別の 2 本)→ P1 の検証の女声 30 人(構造 = t00〜t02・登録 = t03〜t05 の元の音)・k = 7・ECAPA の 30 声中の 1 位。
    cd training && uv run --no-sync python snv_p1_eval.py dev|test <tag> [<tag> ...] [judge=results/snv_p1c_judge/judge.pt](dev で判定器の 30 声中の 1 位も)
test の warp=α = 操作者の源と登録の包絡を先に一律の伸縮 α で(演奏の入力だけが変わる探り・高さと有声は元のまま)。
系の名前に +sep[:<tag>[@ck]](例 snv_i10_c_pr+sep:snv_i14_sep・= は物差しの引数と衝突するので :)= paper_snv §5 M20: 操作者の側(源・操作者の登録・自己の構造 = 鍵が self| の参照)を前段 D(train_snv_sep.features・発話の辞書の wav か wav_path から)の出力に置き換える。目標の参照は元のまま。
+mask:<tag>[@ck](paper_snv §5 M23)= 同じ操作者の側を、前段 v3 に時間周波数の重みを掛けた特徴(train_snv_mask.features)に置き換える。+maskd:<tag>[@ck] = 同じで、有声と高さを重みで濾した YIN(snv_p0_front.f0_track_myin)で決める(paper_snv §9 O13)。
系の名前に +vtl(例 snv_i10_c_pr+vtl)= paper_snv §5 M17: 操作者の登録だけで声道の長さの伸縮 α を当てはめる(格子 0.7〜1.45・推論の E の内容の出力の有声の平均を、登録の高さの中央値で決めた性の学習の声の分布で z にしたノルムを最小に・母集団は results/<名>/vtl_pop.npz)。源と E の構造に同じ α・目標の参照はそのまま。
保存に e_lift があれば(paper_snv §5 M19)、推論の E が読む全ての入力(源・操作者の登録・目標の参照の鍵)の包絡の c_K〜c24 を保存の lift_mean に置き換える(C が読む包絡は元のまま)。
装置 = 環境変数 SNV_DEV か、無ければ GPU の空きが 4 GiB 以上なら cuda・未満なら cpu(学習と並ぶ途中信号)。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import nvoc as N
import rvoc as R
import train_rvoc as TR
import train_snv_p1 as T

ROOT = Path(__file__).resolve().parent.parent
FEAT = T.FEAT
A0 = ROOT / "results/snv_a0"
A1 = ROOT / "results/snv_a1"
N0R = ROOT / "results/snv_n0r/wav"
CK_R = ROOT / "results/rvoc3hi/snap/ema_330k.pt"


def pick_dev() -> str:
    if os.environ.get("SNV_DEV"):
        return os.environ["SNV_DEV"]
    if not torch.cuda.is_available():
        return "cpu"
    free = torch.cuda.mem_get_info()[0]
    if free < 4 * 2**30:
        print(f"[snv_p1_eval] GPU の空き {free / 2**30:.1f} GiB < 4 GiB → cpu で評価", file=sys.stderr, flush=True)
        return "cpu"
    return "cuda"


DEV = pick_dev()
KS = (0, 5, 7, 12)
SR = 48000
VTL_DCT = 59
VTL_F0_SPLIT = 165.0
VTL_GRID = tuple(sorted(set(np.round(np.exp(np.linspace(np.log(0.7), np.log(1.45), 33)), 4).tolist()) | {1.0}))


def wav_path(rel: str) -> Path | None:
    kind, _, rest = rel.partition("/")
    base = {"e1": ROOT / "data/snv_e1", "d": ROOT / "data/snv_d", "e1nv": ROOT / "data/snv_e1nv", "a0": A0 / "wav", "n0r": N0R}.get(kind)
    for q in (FEAT / f"{rel}.wav",) + ((base / f"{rest}.wav",) if base is not None else ()):
        if q.exists():
            return q
    return None


def utt(rel: str) -> dict:
    u = T.load_utt(FEAT / f"{rel}.npz")
    w = wav_path(rel)
    return u | {"wav_path": str(w)} if w is not None else u


def warp_utt(u: dict, alpha: float) -> dict:
    import snv_p0c_anchor as A
    y = u["y"].copy()
    env = np.concatenate([y[:1] * 10 + T.PA.C0_SIL, y[1:25] * 10], 0)
    ew = A.warp_env(env, alpha)
    y[0], y[1:25] = (ew[0] - T.PA.C0_SIL) / 10, ew[1:25] / 10
    return u | {"y": y}


def struct_inputs(refs: list[dict]) -> tuple:
    ry = torch.from_numpy(np.concatenate([r["y"] for r in refs], 1))[None].to(DEV)
    rv = torch.from_numpy(np.concatenate([r["v"] for r in refs]))[None].to(DEV)
    mu = torch.from_numpy(sum(r["vs"] for r in refs) / max(1, sum(r["vn"] for r in refs)))[None].to(DEV)
    return ry, rv, torch.ones(rv.shape, dtype=torch.bool, device=DEV), mu


def mu_of(refs: list[dict]) -> torch.Tensor:
    return torch.from_numpy(sum(r["vs"] for r in refs) / max(1, sum(r["vn"] for r in refs)))[None].to(DEV)


class System:
    def __init__(self, tag: str, gen: torch.nn.Module) -> None:
        base, _, extra = tag.partition("+")
        name, _, ck = base.partition("@")
        self.name, self.vtl, self.vtl_alpha = name, extra == "vtl", {}
        self.sep, self.pcache = None, {}
        if extra.startswith("sep"):
            import train_snv_sep as SEP
            sname, _, sck = (extra.partition(":")[2] or "snv_i14_sep").partition("@")
            self.SEP, self.sep = SEP, SEP.load(ROOT / "results" / sname / f"{sck or 'last'}.pt", DEV)
        elif extra.startswith("mask"):
            import train_snv_mask as MK
            mname, _, mck = (extra.partition(":")[2] or "snv_i17_mask").partition("@")
            self.SEP, self.sep = MK, MK.load(ROOT / "results" / mname / f"{mck or 'last'}.pt", DEV)
            if extra.startswith("maskd"):
                self.sep.dec = "myin"
        st = torch.load(ROOT / "results" / name / f"{ck or 'last'}.pt", map_location="cpu", weights_only=True)
        self.p12 = "TH" in st
        if self.p12:
            import train_snv_p12 as P12
            self.P12 = P12
            self.M12 = P12.Circuit().to(DEV).eval()
            self.M12.load_state_dict(st["M"])
            self.C = lambda a, lf, v, th: self.M12.dec(a, lf, v, th)
        else:
            self.C = (T.CircQ() if st.get("circ") == "q" else T.Circ()).to(DEV).eval()
            self.C.load_state_dict(st["C"])
        self.p11 = "Es" in st
        self.p3 = "proj" in st
        if self.p12:
            pass
        elif self.p11:
            import train_snv_p11 as P11
            self.P11 = P11
            self.Es = P11.EncS().to(DEV).eval()
            self.Es.load_state_dict(st["Es"])
            self.lift = int(st.get("e_lift") or 0)
            self.lmean = torch.tensor(st["lift_mean"], dtype=torch.float32, device=DEV) if self.lift else None
        elif self.p3:
            import train_snv_p3 as P3
            self.Pl = P3.make_play(ROOT / st["cpc"], st.get("play", "c"), ROOT / st["pca"] if st.get("pca") else None).to(DEV).eval()
            self.Pl.proj.load_state_dict(st["proj"])
        else:
            self.E = T.Enc().to(DEV).eval()
            self.E.load_state_dict(st["E"])
        self.gen, self.tag, self.step = gen, tag, st["step"]
        self.cache: dict = {}
        self.scache: dict = {}

    def ex(self, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return self.P11.xin(self.P11.lift(y, self.lift, self.lmean), v)

    def prep(self, u: dict) -> dict:
        if self.sep is None:
            return u
        if id(u) not in self.pcache:
            x = u.get("wav")
            if x is None:
                assert "wav_path" in u, "前段 D は操作者の波形が要る(発話の辞書に wav か wav_path)"
                import librosa
                x = librosa.load(u["wav_path"], sr=SR, mono=True)[0]
            self.pcache[id(u)] = (u, self.SEP.features(self.sep, np.asarray(x, np.float32), DEV))
        return self.pcache[id(u)][1]

    def fit(self, utts: list[dict]) -> torch.Tensor:
        tt = lambda u, k: torch.from_numpy(u[k]).to(DEV)
        return self.P12.fit_theta(self.M12, [tt(u, "y") for u in utts], [tt(u, "lf") for u in utts], [tt(u, "v") for u in utts])

    @torch.no_grad()
    def e_vmean(self, utts: list[dict]) -> np.ndarray:
        ry, rv, rm, _ = struct_inputs(utts)
        x = self.ex(ry, rv)
        a = self.Es(x, self.Es.structure(x, rm))[0][0, :VTL_DCT]
        return a[:, rv[0] > 0.5].mean(1).cpu().numpy()

    def vtl_pop(self, sex: str) -> tuple[np.ndarray, np.ndarray]:
        p = ROOT / "results" / self.name / "vtl_pop.npz"
        if not p.exists():
            split = json.loads((ROOT / "results" / self.name / "split.json").read_text())
            out = {}
            for sx in ("em", "ef"):
                ms = np.stack([self.e_vmean([T.load_utt(q) for q in sorted((FEAT / "e1" / v).glob("t*.npz"))[:4]]) for v in split["train"] if v.startswith(sx)])
                out[f"mu_{sx}"], out[f"sd_{sx}"] = ms.mean(0), ms.std(0) + 1e-4
            tmp = p.with_name(f"vtl_pop.{os.getpid()}.tmp.npz")
            np.savez(tmp, **out)
            os.replace(tmp, p)
        z = np.load(p)
        return z[f"mu_{sex}"], z[f"sd_{sex}"]

    def fit_vtl(self, utts: list[dict]) -> float:
        key = tuple(id(u) for u in utts)
        if key not in self.vtl_alpha:
            f0 = np.concatenate([u["f0"][u["v"] > 0.5] for u in utts])
            mu, sd = self.vtl_pop("em" if np.median(f0) < VTL_F0_SPLIT else "ef")
            self.vtl_alpha[key] = min(VTL_GRID, key=lambda a: float(np.linalg.norm((self.e_vmean([warp_utt(u, a) for u in utts]) - mu) / sd)))
        return self.vtl_alpha[key]

    @torch.no_grad()
    def op_stat(self, utts: list[dict]) -> torch.Tensor:
        key = tuple(id(u) for u in utts)
        if key not in self.scache:
            utts = [self.prep(u) for u in utts]
            if self.p12:
                self.scache[key] = self.fit(utts)
            elif self.p11 and self.vtl:
                al = self.fit_vtl(utts)
                ry, rv, rm, _ = struct_inputs([warp_utt(u, al) for u in utts])
                self.scache[key] = self.Es.structure(self.ex(ry, rv), rm)
            elif self.p11:
                ry, rv, rm, _ = struct_inputs(utts)
                self.scache[key] = self.Es.structure(self.ex(ry, rv), rm)
            elif self.p3:
                ry, rv, rm, _ = struct_inputs(utts)
                self.scache[key] = self.Pl.mean_of(ry, rv, rm)
            else:
                self.scache[key] = mu_of(utts)
        return self.scache[key]

    @torch.no_grad()
    def structure(self, key: str, refs: list[dict]) -> dict:
        if key not in self.cache:
            if key.startswith("self|"):
                refs = [self.prep(u) for u in refs]
            ry, rv, rm, mu = struct_inputs(refs)
            if self.p12:
                self.cache[key] = self.fit(refs)
            elif self.p11:
                rx = self.ex(ry, rv)
                self.cache[key] = self.C.structure(ry, self.Es(rx, self.Es.structure(rx, rm))[0], rm)
            elif self.p3:
                self.cache[key] = self.C.structure(ry, self.Pl(ry, rv, self.op_stat(refs)), rm)
            else:
                self.cache[key] = T.structure_of(self.E, self.C, ry, rv, rm, mu)
        return self.cache[key]

    @torch.no_grad()
    def play(self, y: torch.Tensor, v: torch.Tensor, stat: torch.Tensor) -> torch.Tensor:
        if self.p12:
            return self.M12.enc(y, v, stat)[0]
        if self.p11:
            return self.Es(self.ex(y, v), stat)[0]
        return self.Pl(y, v, stat) if self.p3 else self.E(T.e_in(y, v, stat))

    @torch.no_grad()
    def convert(self, src: dict, stat: list[dict], S: dict, k: int, with_y: bool = False, dlf_c: float | None = None):
        if self.vtl and self.p11:
            src = warp_utt(src, self.fit_vtl(stat))
        src = self.prep(src)
        y = torch.from_numpy(src["y"])[None].to(DEV)
        v = torch.from_numpy(src["v"])[None].to(DEV)
        lf = torch.from_numpy(src["lf"])[None].to(DEV) + k * T.LN2_12 * v
        lfc = lf if dlf_c is None else (torch.from_numpy(src["lf"])[None].to(DEV) + dlf_c) * v
        yo = self.C(self.play(y, v, self.op_stat(stat)), lfc, v, S)
        cond = torch.cat([yo[:, :25], lf[:, None], v[:, None], yo[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
        f0 = torch.from_numpy(src["f0"] * 2 ** (k / 12))[None].to(DEV)
        w = self.gen(cond, TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return (w.astype(np.float32), yo) if with_y else w.astype(np.float32)


def load_gen() -> torch.nn.Module:
    st = torch.load(CK_R, map_location="cpu", weights_only=False)
    cfg = st["cfg"]
    g = R.RVoc(ch=cfg["ch"], kernels=tuple(cfg["kernels"]), dils=tuple(cfg["dils"]), d_cond=cfg["d_cond"]).to(DEV)
    g.load_state_dict(st["ema"])
    return g.eval().requires_grad_(False)


def main() -> int:
    mode, tags = sys.argv[1], [t for t in sys.argv[2:] if t != "smoke" and "=" not in t]
    smoke = "smoke" in sys.argv[2:]
    opt = dict(t.split("=", 1) for t in sys.argv[2:] if "=" in t)
    from snv_a0_measure import cer, kana, load, to16
    from snv_a0b_measure import proj
    from snv_a4_eval import cluster_ci
    from snv_n0_measure import trim
    from snv_n2c_tex import tex_dist, tex_stats
    from speechbrain.inference.speaker import EncoderClassifier
    gen = load_gen()
    systems = [System(t, gen) for t in tags]
    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})
    Tz = np.load(ROOT / "data/snv_e1_feat/ecapa_lda_T.npz")
    Tt = (Tz["mu"], Tz["scalings"])

    def emb(w16: np.ndarray, lda: bool = True) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(w16).float()[None].to(DEV)).reshape(-1)
        e = torch.nn.functional.normalize(e, dim=0).cpu().numpy()
        return proj(e, Tt) if lda else e

    w16 = lambda w: to16(w.astype(np.float64), SR).astype(np.float32)
    lw = lambda p: to16(*load(p)).astype(np.float32)
    res: dict = {"mode": mode, "systems": {s.tag: {"step": s.step} for s in systems}, "prereg": "results/snv_p1/prereg.yaml"}
    out = ROOT / "results/snv_p1" / (("smoke_" if smoke else "") + f"{mode}_" + "_".join(t.replace("snv_p1_", "").replace("@", "-") for t in tags) + (f"_warp{opt['warp']}" if "warp" in opt else "") + ".json")
    if mode == "dev":
        split = json.loads((ROOT / "results" / tags[0].partition("@")[0] / "split.json").read_text())
        tg = [v for v in split["val"] if v.startswith("ef")]
        spk = sorted({p.stem.split("_")[0] for p in (FEAT / "vctk").glob("*.npz")})
        srcs, enr = [], {}
        for s_ in spk:
            fs = sorted((FEAT / "vctk").glob(f"{s_}_*.npz"))
            enr[s_] = [T.load_utt(f) for f in fs[:2]]
            srcs += [(s_, T.load_utt(f)) for f in fs[2:4]]
        if smoke:
            tg, srcs = tg[:5], srcs[:2]
        Et = {v: emb(np.concatenate([lw(ROOT / f"data/snv_e1/{v}/t0{j}.wav") for j in (3, 4, 5)]), lda=False) for v in tg}
        J = None
        if "judge" in opt:
            from train_snv_p1j import Judge
            sj = torch.load(ROOT / opt["judge"], map_location="cpu", weights_only=False)
            J = Judge(len(sj["classes"]), norm=sj.get("norm", "none")).to(DEV).eval().requires_grad_(False)
            J.load_state_dict(sj["J"])
            Jt = {v: J.embed(torch.from_numpy(np.concatenate([utt(f"e1/{v}/t0{j}")["y"] for j in (3, 4, 5)], 1))[None].to(DEV))[0] for v in tg}
            Jm = torch.stack([Jt[v] for v in tg])
            jhits = {s.tag: [] for s in systems}
            res["judge"] = opt["judge"]
        hits = {s.tag: [] for s in systems}
        grp = []
        for si, (s_, src) in enumerate(srcs):
            for v in tg:
                grp.append(v)
                for s in systems:
                    S = s.structure(f"dev|{v}", [utt(f"e1/{v}/t0{j}") for j in (0, 1, 2)])
                    w, yo = s.convert(src, enr[s_], S, 7, with_y=True)
                    e = emb(trim(w16(w)), lda=False)
                    hits[s.tag].append(max(tg, key=lambda u: float(e @ Et[u])) == v)
                    if J is not None:
                        with torch.no_grad():
                            jhits[s.tag].append(tg[int((Jm @ J.embed(yo)[0]).argmax())] == v)
        res["dev_M_k7"] = {t: round(float(np.mean(h)), 4) for t, h in hits.items()}
        if J is not None:
            res["dev_judge_top1_k7"] = {t: round(float(np.mean(h)), 4) for t, h in jhits.items()}
        res["n_trials"] = len(grp)
        if len(systems) > 1:
            a_, b_ = np.array(hits[systems[-1].tag], float), np.array(hits[systems[0].tag], float)
            res["dev_dM_last_minus_first"] = {"mean": round(float((a_ - b_).mean()), 4), "ci95_target_cluster": cluster_ci(a_ - b_, np.array(grp))}
        print(json.dumps(res, ensure_ascii=False), flush=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
        return 0
    meta = json.loads((A1 / "convert_meta.json").read_text())
    tgts, srcs = meta["targets"], meta["sources"]
    ks = (0, 7) if smoke else KS
    if smoke:
        tgts, srcs = tgts[:3], srcs[:2]
    Et = {v: emb(np.concatenate([lw(A0 / f"wav/{v}_{j}.wav") for j in (0, 1, 2)])) for v in tgts}
    Et2 = {v: emb(np.concatenate([lw(A0 / f"wav/{v}_{j}.wav") for j in (6, 7)])) for v in tgts}
    top = lambda e, ref: max(tgts, key=lambda u: float(e @ ref[u]))
    Es = {sp: emb(lw(FEAT / f"real/{sp}_enroll.wav")) for sp in ("uirou", "namikawa")}
    trefs = {v: [utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tgts}
    refs_of = lambda v: trefs[v] if v in trefs else [utt(f"a0/{v}_{j}") for j in (0, 1, 2)]
    src_u = {s_: utt(f"real/{s_}") for s_ in srcs}
    mu_op = {sp: [utt(f"real/{sp}_enroll")] for sp in ("uirou", "namikawa")}
    if "warp" in opt:
        src_u = {k: warp_utt(u, float(opt["warp"])) for k, u in src_u.items()}
        mu_op = {sp: [warp_utt(u, float(opt["warp"])) for u in us] for sp, us in mu_op.items()}
        res["warp_operator_input"] = float(opt["warp"])
    from faster_whisper import WhisperModel
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16" if DEV == "cuda" else "int8")

    def tr(w: np.ndarray) -> str:
        segs, _ = asr.transcribe(w, language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(x.text for x in segs))

    ref_txt = {s_: tr(lw(FEAT / f"real/{s_}.wav")) for s_ in srcs}
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).to(DEV).eval()
    mos = lambda w: float(utmos(torch.from_numpy(w).float()[None].to(DEV), 16000).item())
    scale = np.load(ROOT / "results/snv_n2c/tex_scale.npy")
    n0r = [it for it in json.loads((N0R / "items.json").read_text()) if it["sex"] == "F"][::4]
    n0r = n0r[:4] if smoke else n0r
    vec: dict = {}
    wd = ROOT / ("results/snv_p1/wav_smoke" if smoke else "results/snv_p1/wav")
    wd.mkdir(parents=True, exist_ok=True)
    for s in systems:
        row: dict = {}
        for k in ks:
            hit, hit2, grp, ms_, sc = [], [], [], [], []
            for s_ in srcs:
                for v in tgts:
                    w = s.convert(src_u[s_], mu_op[s_.rsplit("_", 1)[0]], s.structure(f"t|{v}", trefs[v]), k)
                    y16 = w16(w)
                    e = emb(trim(y16))
                    hit.append(top(e, Et) == v), hit2.append(top(e, Et2) == v), grp.append(v)
                    sc.append(float(e @ Es[s_.rsplit("_", 1)[0]]) > float(e @ Et[v]))
                    ms_.append(mos(y16))
                    if s_ in srcs[:1] and v in tgts[:2]:
                        sf.write(wd / f"{s.tag}__{s_}__{v}__k{k}.wav", w, SR)
            vec[(s.tag, k)] = (np.array(hit, float), np.array(grp))
            res.setdefault("trials", {}).setdefault(s.tag, {})[f"k{k}"] = {"hit": [int(h) for h in hit], "target": grp}
            row[f"M_k{k}"] = round(float(np.mean(hit)), 4)
            row[f"M_alt_k{k}"] = round(float(np.mean(hit2)), 4)
            row[f"utmos_k{k}"] = round(float(np.mean(ms_)), 3)
            row[f"src_closer_k{k}"] = round(float(np.mean(sc)), 4)
        row["M_mean_4k"] = round(float(np.mean([row[f"M_k{k}"] for k in ks])), 4)
        sh = []
        for v in tgts:
            for j in (3, 4, 5):
                u = utt(f"a0/{v}_{j}")
                w = s.convert(u, trefs[v], s.structure(f"t|{v}", trefs[v]), 0)
                sh.append(top(emb(trim(w16(w))), Et) == v)
        row["S"] = round(float(np.mean(sh)), 4)
        cs = []
        for k in (0, 7):
            for s_ in srcs:
                for v in tgts[:4]:
                    w = s.convert(src_u[s_], mu_op[s_.rsplit("_", 1)[0]], s.structure(f"t|{v}", trefs[v]), k)
                    cs.append(cer(ref_txt[s_], tr(w16(w))))
        row["cer_k0_k7"] = round(float(np.mean(cs)), 4)
        tx = []
        for it in n0r:
            nme = f"{it['voice']}_{it['cat']}_{it['k']}"
            u = utt(f"n0r/{nme}")
            rf = refs_of(it["voice"])
            w = s.convert(u, rf, s.structure(f"t|{it['voice']}", rf), 0)
            tx.append(tex_dist(tex_stats(w16(w)), tex_stats(lw(N0R / f"{nme}.wav")), scale))
        row["nv_tex"] = round(float(np.mean(tx)), 4)
        res["systems"][s.tag].update(row)
        print(s.tag, json.dumps(row), flush=True)
    if len(systems) > 1:
        base = systems[0].tag
        res["diff_vs_first"] = {}
        for s in systems[1:]:
            d = {}
            for k in ks:
                a_, g_ = vec[(s.tag, k)]
                b_, _ = vec[(base, k)]
                d[f"dM_k{k}"] = {"mean": round(float((a_ - b_).mean()), 4), "ci95_target_cluster": cluster_ci(a_ - b_, g_)}
            a4 = np.mean([vec[(s.tag, k)][0] for k in ks], 0)
            b4 = np.mean([vec[(base, k)][0] for k in ks], 0)
            d["dM_mean_4k"] = {"mean": round(float((a4 - b4).mean()), 4), "ci95_target_cluster": cluster_ci(a4 - b4, vec[(base, ks[0])][1])}
            res["diff_vs_first"][s.tag] = d
        print("diff", json.dumps(res["diff_vs_first"]), flush=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
