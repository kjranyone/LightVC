"""参照を読む実現生成器 R の学習(current/ref_reader.md)。特徴 = prep_rr.py の data/rr_feat。
交差ペア: 目標 A(女声 TTS)の文 tK を正解(CheapTrick 包絡 c1..c24 と c0・伸縮しない)、操作者 B(別の女声 TTS か学習用の男声)の同じ文 tK の入力特徴を DTW で A の時間へ(入力側だけ)、
参照 = A の別の文(5〜9 本 ≈ 20〜36s)。出力 = 表の引き Σp_B T_A[u] + Δ包絡・操作者の c0 + Δc0(最後の層は 0 初期化 = 学習前は表 + 操作者の c0)。
損失 = 有声の包絡 L1 + 0.3·無声の包絡 L1 + λ_c0·有声の c0 の形(発話平均を引く)L1(生の単位)。検証 = 評価外の女声 A・評価外の男声で 表 / 床 と比べ、注意の有効フレーム数と写しの監視を印字。
    uv run python train_rr.py --tag diag_rr1 --steps 15000
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data/rr_feat"
K = 200
C0_SIL = math.log(1e-5) * math.sqrt(128)
EVAL_MALES = {f"m{i:03d}" for i in range(30)}
VAL_MALES = {"m046", "m047", "m048"}
STY = ["neutral", "breathy", "low_tension", "soft", "warm"]


def held_females() -> set:
    fem = ROOT / "data/female_tts_corpus"
    fems = sorted(d.name for d in fem.iterdir() if d.is_dir() and all((d / f"t{i:02d}_{s}.wav").exists() for i in range(10) for s in STY))
    random.Random(80).shuffle(fems)
    held = set(fems[:48])
    rest = sorted(set(fems) - held)
    random.Random(7).shuffle(rest)
    return held | set(rest[:52])


CFG = {"align": "rr"}


def dense(pk: torch.Tensor, pv: torch.Tensor) -> np.ndarray:
    p = torch.zeros(pk.shape[0], K)
    p.scatter_(1, pk.long(), pv.float())
    return p.T.numpy()


class Store:
    def __init__(self, extra: tuple = (), keep=None):
        self.f: dict = {}
        self.wav: dict = {}
        ex_keys: set = set()
        for corpus in ("fem_neutral", "male_par", "male_ref", "real_fem", "male_realx", "realx_clone") + tuple(extra):
            man = FEAT / corpus / "manifest.json"
            if not man.exists():
                continue
            kc = "real_fem" if corpus == "real_fem2" else corpus
            for it in json.loads(man.read_text())["items"]:
                if keep is not None and not keep(kc, it["spk"], it["stem"]):
                    continue
                d = torch.load(FEAT / it["pt"], weights_only=False)
                if corpus == "real_fem2":
                    ex_keys.add((kc, it["spk"], it["stem"]))
                self.wav[(kc, it["spk"], it["stem"])] = it["wav"]
                self.f[(kc, it["spk"], it["stem"])] = {"pk": d["pk"], "pv": d["pv"], "u": d["pk"][:, 0].long().numpy(), "e3": d["e3"].float().numpy(),
                                                           "c0f": d["c0f"].numpy(), "f1": d["f1"].numpy(), "key": d["key"].float().numpy() if kc not in ("real_fem", "male_realx", "realx_clone") else None, "env": d["env"].float().numpy(), "ft": d["ft"].numpy(),
                                                           "p4": d["p4"].float().numpy()}
        self.real: dict = {}
        for (c, s, stem) in self.f:
            if c == "real_fem" and (c, s, stem) not in ex_keys:
                self.real.setdefault(s, []).append(stem)
        self.stats: dict = {}
        for (c, s, stem), d in self.f.items():
            if c in ("male_par", "male_realx", "realx_clone") or (c, s, stem) in ex_keys:
                continue
            self.stats.setdefault(s, []).append(d)
        for s, ds in list(self.stats.items()):
            e3 = np.concatenate([d["e3"][:, d["f1"] > 0] for d in ds], 1); c0 = np.concatenate([d["c0f"][d["f1"] > 0] for d in ds])
            lf = np.concatenate([np.log(d["f1"][d["f1"] > 0]) for d in ds])
            self.stats[s] = {"e3": e3.mean(1) if e3.shape[1] else np.zeros(24, np.float32), "c0": float(c0.mean()) if len(c0) else 0.0, "lf": float(np.median(lf)) if len(lf) else 0.0}


class Pairs(torch.utils.data.Dataset):
    def __init__(self, st: Store, females: list, males: list, n: int, p_male: float, seed: int, fixed: bool = False, crop: int = 0, max_ref: int = 5000,
                 real: list | None = None, p_real: float = 0.0, ecapa: str = "", realx: list | None = None, p_realx: float = 0.0,
                 clonex: list | None = None, p_clonex: float = 0.0, tks: list | None = None, mem_tks: list | None = None, swap_mem: bool = False):
        self.st, self.fem, self.mal, self.n, self.pm, self.seed, self.fixed, self.crop, self.max_ref = st, females, males, n, p_male, seed, fixed, crop, max_ref
        self.real, self.p_real, self.ecapa_path, self.ze = real or [], p_real, ecapa, None
        self.realx, self.p_realx = realx or [], p_realx
        self.clonex, self.p_clonex = clonex or [], p_clonex
        self.tks, self.mem_tks, self.swap_mem = tks, mem_tks, swap_mem
        import conv_c0 as C0
        self.table = C0.table
        self.Cb = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()
        self.pbase = None

    def __len__(self) -> int:
        return self.n

    def eref(self, keys: list) -> np.ndarray:
        if not self.ecapa_path:
            return np.zeros(192, np.float32)
        if self.ze is None:
            self.ze = dict(np.load(self.ecapa_path))
        v = np.mean([self.ze[k] for k in keys if k in self.ze], 0)
        return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)

    def real_item(self, rng: random.Random) -> dict:
        from d88_perturb_identity import wmat
        A = rng.choice(self.real)
        stems = list(self.st.real[A])
        rng.shuffle(stems)
        tgt = stems[0]
        a = self.st.f[("real_fem", A, tgt)]
        refs, tot = [], 0
        for st_ in stems[1:]:
            if tot >= self.max_ref:
                break
            refs.append(st_); tot += self.st.f[("real_fem", A, st_)]["env"].shape[1]
        rr = [self.st.f[("real_fem", A, x)] for x in refs]
        sA = self.st.stats[A]
        pA = dense(a["pk"], a["pv"])
        vA = a["f1"] > 0
        e = (a["e3"] - sA["e3"][:, None])
        W = wmat(rng.uniform(0.8, 1.25))
        Mx = np.eye(24) + 0.3 * np.array([[rng.gauss(0, 1) for _ in range(24)] for _ in range(24)]) / np.sqrt(24)
        Ds = np.diag([rng.uniform(0.5, 1.5) for _ in range(24)])
        e = (Ds @ Mx @ W @ e).astype(np.float32)
        q = np.concatenate([pA, e / 10, (a["c0f"] - sA["c0"])[None], np.where(vA, np.log(np.maximum(a["f1"], 1.0)) - sA["lf"], 0.0)[None], vA[None].astype(np.float32)], 0).astype(np.float32)
        rp = np.concatenate([dense(r["pk"], r["pv"]) for r in rr], 1)[:, : self.max_ref]
        renv = np.concatenate([r["env"] for r in rr], 1)[:, : self.max_ref]
        ru = np.concatenate([r["u"] for r in rr])[: self.max_ref]
        T = self.table(renv[1:25], ru, K, self.Cb).astype(np.float32)
        rc0 = (renv[0] - C0_SIL) / 10
        rv = np.concatenate([renv[1:25] / 10, (rc0 - rc0.mean())[None]], 0).astype(np.float32)
        n = q.shape[1]
        s0 = rng.randrange(0, max(1, n - self.crop)) if self.crop and n > self.crop else 0
        s1 = s0 + self.crop if self.crop else n
        return {"q": q[:, s0:s1], "tab": (T @ pA).astype(np.float32)[:, s0:s1], "c0b": a["c0f"].astype(np.float32)[s0:s1], "rp": rp, "rv": rv,
                "y": a["env"][1:25].astype(np.float32)[:, s0:s1], "yc0": ((a["env"][0] - C0_SIL) / 10).astype(np.float32)[s0:s1], "yv": (a["ft"] > 0)[s0:s1], "A": A, "B": A, "male": False,
                "f0o": np.where(vA, a["f1"], 0.0).astype(np.float32)[s0:s1], "per": np.clip(a["p4"], 0, 1.5).astype(np.float32)[:, s0:s1], "tk": -1, "kind": "rf",
                "eref": self.eref([f"real/{A}/{x}" for x in refs])}

    def realx_item(self, rng: random.Random) -> dict:
        it = rng.choice(self.realx)
        A = it["target"]
        stems = [x for x in self.st.real[A] if x != it["stem"]]
        rng.shuffle(stems)
        refs, tot = [], 0
        for st_ in stems:
            if tot >= self.max_ref:
                break
            refs.append(self.st.f[("real_fem", A, st_)]); tot += refs[-1]["env"].shape[1]
        return self.cross_item(rng, self.st.f[("real_fem", A, it["stem"])], self.st.f[("male_realx", "wav", f"p{it['item']:04d}_t{it['ti']:03d}_SRC")], A, it["spk"], refs, "xr")

    def clonex_item(self, rng: random.Random) -> dict:
        it = rng.choice(self.clonex)
        A = it["target"]
        refs = [self.st.f[("real_fem", A, x)] for x in it["ref_stems"] if ("real_fem", A, x) in self.st.f]
        if self.swap_mem:
            o = rng.choice([m for m in self.clonex if m["target"] != A])
            refs = [self.st.f[("real_fem", o["target"], x)] for x in o["ref_stems"] if ("real_fem", o["target"], x) in self.st.f]
            return self.cross_item(rng, self.st.f[("realx_clone", "clone", it["clone"])], self.st.f[("male_realx", "wav", f"p{it['item']:04d}_t{it['ti']:03d}_SRC")], A, it["spk"], refs, "xc",
                                   tab_refs=[self.st.f[("real_fem", A, x)] for x in it["ref_stems"] if ("real_fem", A, x) in self.st.f], pb=(f"p{it['item']:04d}_t{it['ti']:03d}_SRC", it["spk"], it["item"], A))
        return self.cross_item(rng, self.st.f[("realx_clone", "clone", it["clone"])], self.st.f[("male_realx", "wav", f"p{it['item']:04d}_t{it['ti']:03d}_SRC")], A, it["spk"], refs, "xc",
                               pb=(f"p{it['item']:04d}_t{it['ti']:03d}_SRC", it["spk"], it["item"], A))

    def cross_item(self, rng: random.Random, a: dict, b: dict, A: str, B: str, refs: list, kind: str, tab_refs: list | None = None, pb: tuple | None = None) -> dict:
        import librosa
        _, wp = librosa.sequence.dtw(X=np.sqrt(dense(a["pk"], a["pv"])), Y=np.sqrt(dense(b["pk"], b["pv"])), metric="euclidean")
        j = np.zeros(a["pk"].shape[0], int)
        for i_, j_ in wp[::-1]:
            j[i_] = j_
        n = min(len(j), a["env"].shape[1], len(a["ft"]))
        j = j[:n]
        sB = self.st.stats[B]
        pB = dense(b["pk"], b["pv"])[:, j]
        vB = (b["f1"] > 0)[j]
        q = np.concatenate([pB, (b["e3"] - sB["e3"][:, None])[:, j] / 10, ((b["c0f"] - sB["c0"])[j])[None],
                            np.where(vB, np.log(np.maximum(b["f1"][j], 1.0)) - sB["lf"], 0.0)[None], vB[None].astype(np.float32)], 0).astype(np.float32)
        rp = np.concatenate([dense(r["pk"], r["pv"]) for r in refs], 1)[:, : self.max_ref]
        renv = np.concatenate([r["env"] for r in refs], 1)[:, : self.max_ref]
        tr_ = tab_refs or refs
        tenv = np.concatenate([r["env"] for r in tr_], 1)[:, : self.max_ref]
        tu = np.concatenate([r["u"] for r in tr_])[: self.max_ref]
        T = self.table(tenv[1:25], tu, K, self.Cb).astype(np.float32)
        rc0 = (renv[0] - C0_SIL) / 10
        rv = np.concatenate([renv[1:25] / 10, (rc0 - rc0.mean())[None]], 0).astype(np.float32)
        fts = np.concatenate([r["ft"] for r in tr_])
        muA = float(np.median(np.log(fts[fts > 0])))
        f0o = np.where(vB, np.exp(np.log(np.maximum(b["f1"][j], 1.0)) - sB["lf"] + muA), 0.0).astype(np.float32)
        per = np.clip(b["p4"][:, j], 0, 1.5).astype(np.float32)
        if self.pbase is not None and pb is not None:
            be = self.pbase.env(*pb).T * 10
            nb = b["pk"].shape[0]
            assert abs(be.shape[1] - nb) <= 2, (pb, be.shape, nb)
            tab = np.pad(be, ((0, 0), (0, max(0, nb - be.shape[1]))), mode="edge")[:, :nb][:, j].astype(np.float32)
        else:
            tab = (T @ pB).astype(np.float32)
        s0 = rng.randrange(0, max(1, n - self.crop)) if self.crop and n > self.crop else 0
        s1 = s0 + self.crop if self.crop else n
        return {"q": q[:, s0:s1], "tab": tab[:, s0:s1], "c0b": b["c0f"][j].astype(np.float32)[s0:s1], "rp": rp, "rv": rv,
                "y": a["env"][1:25, :n].astype(np.float32)[:, s0:s1], "yc0": ((a["env"][0, :n] - C0_SIL) / 10).astype(np.float32)[s0:s1], "yv": (a["ft"][:n] > 0)[s0:s1], "A": A, "B": B, "male": True,
                "f0o": f0o[s0:s1], "per": per[:, s0:s1], "tk": -1, "kind": kind, "eref": np.zeros(192, np.float32)}

    def __getitem__(self, i: int) -> dict:
        import librosa
        rng = random.Random(self.seed * 100003 + i if self.fixed else None)
        if self.clonex and rng.random() < self.p_clonex:
            return self.clonex_item(rng)
        if self.realx and rng.random() < self.p_realx:
            return self.realx_item(rng)
        if self.real and rng.random() < self.p_real:
            return self.real_item(rng)
        A = rng.choice(self.fem)
        tks = self.tks if self.tks is not None else list(range(10))
        tk = rng.choice(tks)
        if self.mal and rng.random() < self.pm:
            B = rng.choice(self.mal)
            b = self.st.f[("male_par", B, f"f{tk:02d}")]
        else:
            B = rng.choice([f for f in self.fem if f != A])
            b = self.st.f[("fem_neutral", B, f"t{tk:02d}_neutral")]
        a = self.st.f[("fem_neutral", A, f"t{tk:02d}_neutral")]
        others = [t for t in (self.mem_tks if self.mem_tks is not None else range(10)) if t != tk]
        rng.shuffle(others)
        refs = [self.st.f[("fem_neutral", A, f"t{t:02d}_neutral")] for t in others[: rng.randint(min(5, len(others)), min(9, len(others)))]]
        refs_m = refs
        if self.swap_mem:
            A2 = rng.choice([f for f in self.fem if f != A])
            refs_m = [self.st.f[("fem_neutral", A2, f"t{t:02d}_neutral")] for t in others[: len(refs)]]
        if CFG["align"] == "c1":
            _, wp = librosa.sequence.dtw(X=np.sqrt(dense(a["pk"], a["pv"])), Y=np.sqrt(dense(b["pk"], b["pv"])), metric="euclidean")
        else:
            an = a["key"] / (np.linalg.norm(a["key"], axis=0, keepdims=True) + 1e-9)
            bn = b["key"] / (np.linalg.norm(b["key"], axis=0, keepdims=True) + 1e-9)
            _, wp = librosa.sequence.dtw(C=1 - an.T @ bn)
        j = np.zeros(a["key"].shape[1], int)
        for i_, j_ in wp[::-1]:
            j[i_] = j_
        sB = self.st.stats[B]
        pB = dense(b["pk"], b["pv"])[:, j]
        vB = (b["f1"] > 0)[j]
        q = np.concatenate([pB, (b["e3"] - sB["e3"][:, None])[:, j] / 10, ((b["c0f"] - sB["c0"])[j])[None],
                            np.where(vB, np.log(np.maximum(b["f1"][j], 1.0)) - sB["lf"], 0.0)[None], vB[None].astype(np.float32)], 0).astype(np.float32)
        rp = np.concatenate([dense(r["pk"], r["pv"]) for r in refs_m], 1)[:, : self.max_ref]
        renv = np.concatenate([r["env"] for r in refs], 1)[:, : self.max_ref]
        ru = np.concatenate([r["u"] for r in refs])[: self.max_ref]
        T = self.table(renv[1:25], ru, K, self.Cb).astype(np.float32)
        menv = np.concatenate([r["env"] for r in refs_m], 1)[:, : self.max_ref]
        rc0 = (menv[0] - C0_SIL) / 10
        rv = np.concatenate([np.concatenate([menv[1:25] / 10, (rc0 - rc0.mean())[None]], 0)], 0).astype(np.float32)
        fts = np.concatenate([r["ft"] for r in refs])
        muA = float(np.median(np.log(fts[fts > 0])))
        f0o = np.where(vB, np.exp(np.log(np.maximum(b["f1"][j], 1.0)) - sB["lf"] + muA), 0.0).astype(np.float32)
        per = np.clip(b["p4"][:, j], 0, 1.5).astype(np.float32)
        n = q.shape[1]
        s0 = rng.randrange(0, max(1, n - self.crop)) if self.crop and n > self.crop else 0
        s1 = s0 + self.crop if self.crop else n
        return {"q": q[:, s0:s1], "tab": (T @ pB).astype(np.float32)[:, s0:s1], "c0b": b["c0f"][j].astype(np.float32)[s0:s1], "rp": rp, "rv": rv,
                "y": a["env"][1:25].astype(np.float32)[:, s0:s1], "yc0": ((a["env"][0] - C0_SIL) / 10).astype(np.float32)[s0:s1], "yv": (a["ft"] > 0)[s0:s1], "A": A, "B": B, "male": B.startswith("m"),
                "f0o": f0o[s0:s1], "per": per[:, s0:s1], "tk": tk, "kind": "mf" if B.startswith("m") else "ff",
                "eref": self.eref([f"{A}/t{t:02d}_neutral" for t in range(10) if t != tk])}


def collate(bs: list) -> dict:
    T = max(b["q"].shape[1] for b in bs); M = max(b["rp"].shape[1] for b in bs)
    out = {"q": torch.zeros(len(bs), bs[0]["q"].shape[0], T), "tab": torch.zeros(len(bs), 24, T), "c0b": torch.zeros(len(bs), T), "y": torch.zeros(len(bs), 24, T),
           "yc0": torch.zeros(len(bs), T), "yv": torch.zeros(len(bs), T, dtype=torch.bool), "tm": torch.zeros(len(bs), T, dtype=torch.bool),
           "rp": torch.zeros(len(bs), K, M), "rv": torch.zeros(len(bs), 25, M), "rm": torch.zeros(len(bs), M, dtype=torch.bool), "male": torch.tensor([b["male"] for b in bs]),
           "f0o": torch.zeros(len(bs), T), "per": torch.zeros(len(bs), 4, T), "A": [b["A"] for b in bs], "tk": [b["tk"] for b in bs],
           "kind": [b["kind"] for b in bs], "eref": torch.from_numpy(np.stack([b["eref"] for b in bs]))}
    for i, b in enumerate(bs):
        t, m = b["q"].shape[1], b["rp"].shape[1]
        out["q"][i, :, :t] = torch.from_numpy(b["q"]); out["tab"][i, :, :t] = torch.from_numpy(b["tab"]); out["c0b"][i, :t] = torch.from_numpy(b["c0b"])
        out["y"][i, :, :t] = torch.from_numpy(b["y"]); out["yc0"][i, :t] = torch.from_numpy(b["yc0"]); out["yv"][i, :t] = torch.from_numpy(b["yv"]); out["tm"][i, :t] = True
        out["rp"][i, :, :m] = torch.from_numpy(b["rp"]); out["rv"][i, :, :m] = torch.from_numpy(b["rv"]); out["rm"][i, :m] = True
        out["f0o"][i, :t] = torch.from_numpy(b["f0o"]); out["per"][i, :, :t] = torch.from_numpy(b["per"])
    return out


class CConv(nn.Module):
    def __init__(self, ci: int, co: int, k: int = 1, d: int = 1, causal: bool = True):
        super().__init__()
        self.pad = ((k - 1) * d, 0) if causal else ((k - 1) * d // 2, (k - 1) * d // 2)
        self.c = nn.Conv1d(ci, co, k, dilation=d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c(F.pad(x, self.pad))


class Blk(nn.Module):
    def __init__(self, ch: int, d: int, causal: bool):
        super().__init__()
        self.a = CConv(ch, ch, 3, d, causal); self.b = CConv(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.b(F.leaky_relu(self.a(F.leaky_relu(x, 0.1)), 0.1))


class RR(nn.Module):
    def __init__(self, nq: int, ch: int = 192, heads: int = 4, dk: int = 32, qd: tuple = (1, 2, 4, 8, 16, 32), kd: tuple = (1, 2, 4)):
        super().__init__()
        self.cfg = {"nq": nq, "ch": ch, "heads": heads, "dk": dk, "qd": list(qd), "kd": list(kd)}
        self.h, self.dk = heads, dk
        self.kin = CConv(K, ch, 1, causal=False); self.kb = nn.Sequential(*[Blk(ch, d, False) for d in kd]); self.kout = CConv(ch, heads * dk, 1)
        self.vin = CConv(K + 25, ch, 1, causal=False); self.vb = nn.Sequential(*[Blk(ch, d, False) for d in kd]); self.vout = CConv(ch, heads * dk, 1)
        self.qin = CConv(nq, ch, 1); self.qb = nn.Sequential(*[Blk(ch, d, True) for d in qd]); self.qout = CConv(ch, heads * dk, 1)
        self.din = CConv(ch + heads * dk + 24, ch, 1); self.db = nn.Sequential(*[Blk(ch, d, True) for d in (1, 2, 4)]); self.dout = CConv(ch, 25, 1)
        nn.init.zeros_(self.dout.c.weight); nn.init.zeros_(self.dout.c.bias)

    def memory(self, rp: torch.Tensor, rv: torch.Tensor) -> tuple:
        if CFG.get("ckpt") and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint_sequential
            k = self.kout(checkpoint_sequential(self.kb, len(self.kb), self.kin(rp), use_reentrant=False))
            v = self.vout(checkpoint_sequential(self.vb, len(self.vb), self.vin(torch.cat([rp, rv], 1)), use_reentrant=False))
        else:
            k = self.kout(self.kb(self.kin(rp)))
            v = self.vout(self.vb(self.vin(torch.cat([rp, rv], 1))))
        B, _, M = k.shape
        return k.view(B, self.h, self.dk, M), v.view(B, self.h, self.dk, M)

    def forward(self, q: torch.Tensor, tab: torch.Tensor, mem: tuple, rm: torch.Tensor, null: torch.Tensor | None = None) -> tuple:
        hq = self.qb(self.qin(q))
        qq = self.qout(hq)
        B, _, T = qq.shape
        qq = qq.view(B, self.h, self.dk, T)
        k, v = mem
        s = torch.einsum("bhdt,bhdm->bhtm", qq, k) / math.sqrt(self.dk)
        s = s.masked_fill(~rm[:, None, None, :], -1e4)
        w = s.softmax(-1)
        r = torch.einsum("bhtm,bhdm->bhdt", w, v).reshape(B, self.h * self.dk, T)
        if null is not None:
            r = r.masked_fill(null[:, None, None], 0.0)
        o = self.dout(self.db(self.din(torch.cat([hq, r, tab / 10], 1))))
        eff = 1.0 / (w ** 2).sum(-1)
        return tab + 10 * o[:, :24], o[:, 24] , eff


class RR2(nn.Module):
    """diag_rr11c: 読みを多層に。記憶の符号 hm(事後 + 包絡 + c0 の非因果 TCN = 内容と実現の両方で鍵を作る)に対し、層ごとの交差注意を因果の問いの流れに足す。"""

    def __init__(self, nq: int, ch: int = 192, heads: int = 4, dk: int = 32, qd: tuple = (1, 2, 4, 8, 16, 32), kd: tuple = (1, 2, 4), layers: int = 3):
        super().__init__()
        self.cfg = {"nq": nq, "ch": ch, "heads": heads, "dk": dk, "qd": list(qd), "kd": list(kd), "layers": layers}
        self.h, self.dk, self.L = heads, dk, layers
        self.min = CConv(K + 25, ch, 1, causal=False); self.mb = nn.Sequential(*[Blk(ch, d, False) for d in kd])
        self.kp = nn.ModuleList([CConv(ch, heads * dk, 1) for _ in range(layers)]); self.vp = nn.ModuleList([CConv(ch, heads * dk, 1) for _ in range(layers)])
        self.qin = CConv(nq, ch, 1); self.qb = nn.Sequential(*[Blk(ch, d, True) for d in qd])
        self.qp = nn.ModuleList([CConv(ch, heads * dk, 1) for _ in range(layers)]); self.op = nn.ModuleList([CConv(heads * dk, ch, 1) for _ in range(layers)])
        self.lb = nn.ModuleList([Blk(ch, d, True) for d in (1, 2, 4, 1, 2, 4)[:layers]])
        self.din = CConv(ch + 24, ch, 1); self.db = nn.Sequential(*[Blk(ch, d, True) for d in (1, 2, 4)]); self.dout = CConv(ch, 25, 1)
        nn.init.zeros_(self.dout.c.weight); nn.init.zeros_(self.dout.c.bias)

    def memory(self, rp: torch.Tensor, rv: torch.Tensor) -> list:
        hm = self.mb(self.min(torch.cat([rp, rv], 1)))
        B, _, M = hm.shape
        return [(self.kp[l](hm).view(B, self.h, self.dk, M), self.vp[l](hm).view(B, self.h, self.dk, M)) for l in range(self.L)]

    def forward(self, q: torch.Tensor, tab: torch.Tensor, mem: list, rm: torch.Tensor, null: torch.Tensor | None = None) -> tuple:
        x = self.qb(self.qin(q))
        B, _, T = x.shape
        eff = None
        for l, (k, v) in enumerate(mem):
            qq = self.qp[l](x).view(B, self.h, self.dk, T)
            sc = torch.einsum("bhdt,bhdm->bhtm", qq, k) / math.sqrt(self.dk)
            w = sc.masked_fill(~rm[:, None, None, :], -1e4).softmax(-1)
            rd = torch.einsum("bhtm,bhdm->bhdt", w, v).reshape(B, self.h * self.dk, T)
            if null is not None:
                rd = rd.masked_fill(null[:, None, None], 0.0)
            x = x + self.op[l](rd)
            x = self.lb[l](x)
            eff = 1.0 / (w ** 2).sum(-1)
        o = self.dout(self.db(self.din(torch.cat([x, tab / 10], 1))))
        return tab + 10 * o[:, :24], o[:, 24], eff


def build(cfg: dict) -> nn.Module:
    kw = {k: cfg[k] for k in ("nq", "ch", "heads", "dk")} | {k_: tuple(cfg[k_]) for k_ in ("qd", "kd") if k_ in cfg}
    return RR2(**kw, layers=cfg["layers"]) if cfg.get("layers", 1) > 1 else RR(**kw)


def lossf(env: torch.Tensor, dc0: torch.Tensor, b: dict, lam: float) -> tuple:
    v, u = (b["yv"] & b["tm"]), (~b["yv"] & b["tm"])
    le = (env - b["y"]).abs().mean(1)
    lv = (le * v).sum() / v.sum().clamp(min=1); lu = (le * u).sum() / u.sum().clamp(min=1)
    c0 = b["c0b"] + dc0
    vf = v.float()
    mo = (c0 * vf).sum(1, keepdim=True) / vf.sum(1, keepdim=True).clamp(min=1)
    mt = (b["yc0"] * vf).sum(1, keepdim=True) / vf.sum(1, keepdim=True).clamp(min=1)
    lc = (((c0 - mo) - (b["yc0"] - mt)).abs() * vf).sum() / vf.sum().clamp(min=1)
    return lv + 0.3 * lu + lam * lc, lv, lc


@torch.no_grad()
def evaluate(net: RR, dl, dev: str, no_op_env: bool = False) -> dict:
    net.eval()
    acc: dict = {"ff": [], "mf": []}
    for b in dl:
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        if no_op_env:
            b["q"][:, K:K + 24] = 0
        env, dc0, eff = net(b["q"], b["tab"], net.memory(b["rp"], b["rv"]), b["rm"])
        for i in range(env.shape[0]):
            v = b["yv"][i] & b["tm"][i]
            l_r = float((env[i][:, v] - b["y"][i][:, v]).abs().mean()); l_t = float((b["tab"][i][:, v] - b["y"][i][:, v]).abs().mean())
            yc, c_r, c_b = b["yc0"][i][v], (b["c0b"][i] + dc0[i])[v], b["c0b"][i][v]
            cr = float(torch.corrcoef(torch.stack([c_r, yc]))[0, 1]); cb = float(torch.corrcoef(torch.stack([c_b, yc]))[0, 1])
            rv = b["rv"][i][:24, b["rm"][i]] * 10
            dn = torch.cdist(env[i][:, v].T, rv.T).min(1).values.median()
            dy = torch.cdist(b["y"][i][:, v].T, rv.T).min(1).values.median()
            acc.setdefault(b["kind"][i], []).append([l_r, l_t, cr, cb, float(eff[i][:, b["tm"][i]].median()), float(dn / dy.clamp(min=1e-6))])
    net.train()
    out = {}
    for k_, rows in acc.items():
        if rows:
            r = np.array(rows)
            out[k_] = {"l1_rr": round(float(np.median(r[:, 0])), 4), "l1_tab": round(float(np.median(r[:, 1])), 4), "c0corr_rr": round(float(np.median(r[:, 2])), 3),
                       "c0corr_op": round(float(np.median(r[:, 3])), 3), "eff": round(float(np.median(r[:, 4])), 1), "copy_ratio": round(float(np.median(r[:, 5])), 3), "n": len(rows)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lam_c0", type=float, default=1.0)
    ap.add_argument("--p_male", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--id_w", type=float, default=0.0, help="出力部(--rvoc・凍結)で鳴らした出力の ECAPA と目標の参照(その文を除く 9 文)の 1 − cos の重み(diag_rr3)")
    ap.add_argument("--rvoc", default="results/rvtab_x/last.pt")
    ap.add_argument("--init", default="", help="R の初期重み(results/<tag>/last.pt)")
    ap.add_argument("--pbase", action="store_true", help="クローンの交差ペアの土台(tab)を製品の経路の包絡の行にする(pbase.py・data/realx/pbase・d217_pbase_prep.py・較正 B なし・diag_rr18p)")
    ap.add_argument("--p_real", type=float, default=0.0, help="実女声の自己再構成の標本の割合(入力側の話者の摂動つき・rr2)")
    ap.add_argument("--eval_bs", type=int, default=4)
    ap.add_argument("--p_realx", type=float, default=0.0, help="男声 TTS の読み → 実女声の交差ペア(data/realx・prep_realx.py・D177)の割合(diag_rr8c)")
    ap.add_argument("--p_clonex", type=float, default=0.0, help="男声 TTS の読み → 目標 A のクローン(Irodori・A の参照を手本に同じ文・gen_realx_clone.py・オーナー判断 10-14)の交差ペアの割合・記憶 = 手本と同じクリップ(diag_rr9c)")
    ap.add_argument("--layers", type=int, default=1, help="読みの交差注意の層の数(1 = RR・2 以上 = RR2: 記憶の鍵に包絡・c0 も入れ層ごとに読む・diag_rr11c)")
    ap.add_argument("--hold_tk", default="", help="TTS の文 tK を学習の目標・記憶から外す(例 8,9・既知の話者の保留文の検査 diag_rr10c)。検証(未知の話者)も目標 = 保留文・記憶 = 残りの文")
    ap.add_argument("--clone_scale", action="store_true", help="data/realx/items_clone_scale.json の既知の話者(seen2)の学習用の文(held でない)を --p_clonex の組に足す(diag_rr13c)")
    ap.add_argument("--clone_full", action="store_true", help="data/realx/items_clone_full.json(make_clone_full.py・学習側の残り 574 人 × 10 文)の学習用の文を --p_clonex の組に足す(全話者の本走行)")
    ap.add_argument("--clone_full2", action="store_true", help="data/realx/items_clone_full2.json(make_clone_full2.py・real_fem に入らなかった残りの話者 × 8 文・記憶 = rr_feat/real_fem2)の学習用の文を --p_clonex の組に足す(diag_rr20d)")
    ap.add_argument("--clone_multi", action="store_true", help="data/realx/items_clone_multi.json の既知の話者の学習用の文(held でない)を --p_clonex の組に足す(diag_rr12c)")
    ap.add_argument("--ckpt", action="store_true", help="参照の符号器の勾配チェックポイント(計算は同じ・VRAM だけ節約・diag_rr6c)")
    ap.add_argument("--kd", default="1,2,4", help="参照の鍵・値の符号器(非因果・登録時)の dilation(diag_rr6c は句の水準まで)")
    ap.add_argument("--qd", default="1,2,4,8,16,32", help="問いの符号器の因果 TCN の dilation(受容野 = 1 + 2Σd フレーム・diag_rr5c は 64,128 を足して 2.56 秒)")
    ap.add_argument("--align", default="rr", choices=["rr", "c1"], help="交差ペアの入力側の整列: rr = log-mel の DCT 1..20 のコサイン(diag_rr1〜4)・c1 = C1 の事後の平方根(D155 の三角閉包 5.9 vs 34.7)")
    ap.add_argument("--p_null", type=float, default=0.0, help="記憶の読みを 0 にする標本の割合(記憶の脱落・diag_rr15c: 推論の出力 = 表 + 記憶あり − 記憶なし = 一般の分を出さない・D198)")
    ap.add_argument("--null_both", action="store_true", help="各標本で記憶あり・記憶なし(読みを 0)の両方の損失を足す(記憶ありの学習量を減らさない記憶の脱落・diag_rr15c)")
    ap.add_argument("--w_null", type=float, default=1.0)
    ap.add_argument("--no_op_env", action="store_true", help="問いの入力から操作者の包絡の動き(e3 の 24 行)を 0 にする(漏れの切り分け・diag_rr2)")
    a = ap.parse_args()
    CFG["align"] = a.align
    CFG["ckpt"] = a.ckpt
    if a.selftest:
        net = RR(nq=K + 24 + 3, qd=tuple(int(z) for z in a.qd.split(",")), kd=tuple(int(z) for z in a.kd.split(","))).eval()
        for prm in net.dout.parameters():
            nn.init.normal_(prm, std=0.1)
        q = torch.randn(1, K + 27, 300); tab = torch.randn(1, 24, 300); rp = torch.rand(1, K, 500); rv = torch.randn(1, 25, 500); rm = torch.ones(1, 500, dtype=torch.bool)
        mem = net.memory(rp, rv)
        e1, c1_, _ = net(q, tab, mem, rm)
        q2 = q.clone(); q2[:, :, 200:] = torch.randn(1, K + 27, 100) * 5
        tab2 = tab.clone(); tab2[:, :, 200:] += 3
        e2, c2_, _ = net(q2, tab2, mem, rm)
        d_past = float((e1[..., :200] - e2[..., :200]).abs().max()) + float((c1_[..., :200] - c2_[..., :200]).abs().max())
        d_fut = float((e1[..., 200:] - e2[..., 200:]).abs().max())
        print("selftest 未来不変性: 過去の最大差", d_past, "未来の最大差", d_fut)
        assert d_past == 0.0 and d_fut > 1e-3, "因果性 FAIL または INCONCLUSIVE"
        print("selftest PASS")
        return 0
    od = ROOT / "results" / a.tag
    od.mkdir(parents=True, exist_ok=True)
    pre = od / "prereg.yaml"
    assert pre.exists(), f"事前登録 {pre} が無い(design_laws §1)"
    log = open(od / "train.log", "a")
    print(f"prereg: {pre}", file=log, flush=True)
    torch.manual_seed(0)
    dev = "cuda"
    st = Store(("real_fem2",) if a.clone_full2 else ())
    held = held_females()
    fems = sorted(s for s in {s for (c, s, _) in st.f if c == "fem_neutral"} if all(("fem_neutral", s, f"t{t:02d}_neutral") in st.f for t in range(10)))
    tr_f, va_f = [s for s in fems if s not in held], [s for s in fems if s in held]
    males = sorted(s for s in {s for (c, s, _) in st.f if c == "male_par"} if all(("male_par", s, f"f{t:02d}") in st.f for t in range(10)))
    tr_m = [m for m in males if m not in EVAL_MALES and m not in VAL_MALES and m in st.stats]
    va_m = [m for m in males if m in VAL_MALES and m in st.stats]
    print(json.dumps({"train_f": len(tr_f), "val_f": len(va_f), "train_m": tr_m, "val_m": va_m}), file=log, flush=True)
    print("train_f", len(tr_f), "val_f", len(va_f), "train_m", tr_m, "val_m", va_m, flush=True)
    assert not (set(tr_m) & EVAL_MALES) and not (set(tr_f) & held)
    reals = sorted(s for s, v in st.real.items() if len(v) >= 8)
    rv_ = set(reals[::20])
    tr_r, va_r = [s for s in reals if s not in rv_], sorted(rv_)
    print("real train", len(tr_r), "real val", len(va_r), flush=True)
    eca = ""
    if a.id_w > 0:
        eca = str(FEAT / "ecapa_all.npz")
    rx = [it for it in json.loads((ROOT / "data/realx/items.json").read_text()) if ("male_realx", "wav", f"p{it['item']:04d}_t{it['ti']:03d}_SRC") in st.f and it["target"] in st.real] if a.p_realx > 0 else []
    rx_tr, rx_va = [it for it in rx if not it["val"]], [it for it in rx if it["val"]]
    if rx:
        print("realx train", len(rx_tr), "val", len(rx_va), flush=True)
    cx = [it for it in json.loads((ROOT / "data/realx/items_clone.json").read_text()) if ("male_realx", "wav", f"p{it['item']:04d}_t{it['ti']:03d}_SRC") in st.f
          and ("realx_clone", "clone", it["clone"]) in st.f and it["target"] in st.real] if a.p_clonex > 0 else []
    cx_tr, cx_va = [it for it in cx if not it["val"]], [it for it in cx if it["val"]]
    if a.clone_multi:
        cm = [it for it in json.loads((ROOT / "data/realx/items_clone_multi.json").read_text()) if it["group"] == "seen" and not it["held"]
              and ("realx_clone", "clone", it["clone"]) in st.f and ("male_realx", "wav", f"p{it['item']:04d}_t{it['ti']:03d}_SRC") in st.f]
        cx_tr = cx_tr + cm
        print("clone_multi train", len(cm), flush=True)
    if a.clone_scale:
        cs = [it for it in json.loads((ROOT / "data/realx/items_clone_scale.json").read_text()) if it["group"] == "seen2" and not it["held"]
              and ("realx_clone", "clone", it["clone"]) in st.f and ("male_realx", "wav", f"p{it['item']:04d}_t{it['ti']:03d}_SRC") in st.f]
        cx_tr = cx_tr + cs
        print("clone_scale train", len(cs), flush=True)
    if a.clone_full:
        cf = [it for it in json.loads((ROOT / "data/realx/items_clone_full.json").read_text()) if it["group"] == "seen3" and not it["held"]
              and ("realx_clone", "clone", it["clone"]) in st.f and ("male_realx", "wav", f"p{it['item']:04d}_t{it['ti']:03d}_SRC") in st.f]
        cx_tr = cx_tr + cf
        print("clone_full train", len(cf), flush=True)
    if a.clone_full2:
        cf2 = [it for it in json.loads((ROOT / "data/realx/items_clone_full2.json").read_text()) if it["group"] == "seen4" and not it["held"]
               and ("realx_clone", "clone", it["clone"]) in st.f and ("male_realx", "wav", f"p{it['item']:04d}_t{it['ti']:03d}_SRC") in st.f
               and all(("real_fem", it["target"], x) in st.f for x in it["ref_stems"])]
        cx_tr = cx_tr + cf2
        print("clone_full2 train", len(cf2), "speakers", len({it["target"] for it in cf2}), flush=True)
    if cx:
        print("clonex train", len(cx_tr), "val", len(cx_va), flush=True)
    hold = [int(z) for z in a.hold_tk.split(",")] if a.hold_tk else []
    keep = [t for t in range(10) if t not in hold] if hold else None
    ds = Pairs(st, tr_f, tr_m, 10 ** 9, a.p_male, 0, crop=400, real=tr_r, p_real=a.p_real, ecapa=eca, realx=rx_tr, p_realx=a.p_realx, clonex=cx_tr, p_clonex=a.p_clonex,
               tks=keep, mem_tks=keep)
    vds = Pairs(st, va_f, va_m, 144 if va_r else 96, 0.5 if va_m else 0.0, 1, fixed=True, real=va_r if a.p_real > 0 or not (rx or cx) else [], p_real=(1 / 3 if va_r else 0.0) if a.p_real > 0 or not (rx or cx) else 0.0,
                realx=rx_va, p_realx=1 / 3 if rx_va else 0.0, clonex=cx_va, p_clonex=1 / 3 if cx_va else 0.0, tks=hold or None, mem_tks=keep)
    if a.pbase:
        import pbase as PB
        ds.pbase = vds.pbase = PB.PBase(ROOT / "data/realx/pbase")
    dl = torch.utils.data.DataLoader(ds, batch_size=a.bs, num_workers=a.workers, collate_fn=collate, persistent_workers=True, prefetch_factor=4)
    vdl = torch.utils.data.DataLoader(vds, batch_size=a.eval_bs, num_workers=2, collate_fn=collate)
    net = build({"nq": K + 24 + 3, "ch": 192, "heads": 4, "dk": 32, "qd": [int(z) for z in a.qd.split(",")], "kd": [int(z) for z in a.kd.split(",")], "layers": a.layers}).to(dev)
    if a.init:
        net.load_state_dict(torch.load(ROOT / a.init, map_location="cpu", weights_only=False)["net"])
    gen = ecap = None
    if a.id_w > 0:
        import idloss as ID
        import nvoc as N
        import rvoc as Rv
        import train_rvoc as TR
        sr_ = torch.load(ROOT / a.rvoc, map_location="cpu", weights_only=False)
        gen = Rv.RVoc(ch=sr_["cfg"]["ch"], kernels=tuple(sr_["cfg"]["kernels"]), dils=tuple(sr_["cfg"]["dils"]), d_cond=sr_["cfg"]["d_cond"]).to(dev)
        gen.load_state_dict(sr_["ema"]); gen.eval().requires_grad_(False)
        ecap = ID.Ecapa(dev)
        cpath = FEAT / "ecapa_files.npz"
        if cpath.exists():
            ze = dict(np.load(cpath))
        else:
            from d6_unit_cov import load48
            ze = {}
            with torch.no_grad():
                for (c, sp, stem), w in st.wav.items():
                    if c == "fem_neutral":
                        ze[f"{sp}/{stem}"] = ecap(torch.from_numpy(load48(ROOT / w).astype(np.float32))[None].to(dev))[0].cpu().numpy()
            np.savez(cpath, **ze)
        rpath = FEAT / "ecapa_real.npz"
        if st.real and not rpath.exists():
            from d6_unit_cov import load48
            zr = {}
            with torch.no_grad():
                for (c, sp, stem), w in st.wav.items():
                    if c == "real_fem":
                        x = load48(ROOT / w if not Path(w).is_absolute() else Path(w)).astype(np.float32)[: 20 * 48000]
                        zr[f"real/{sp}/{stem}"] = ecap(torch.from_numpy(x)[None].to(dev))[0].cpu().numpy()
            np.savez(rpath, **zr)
        if rpath.exists():
            ze.update(dict(np.load(rpath)))
        np.savez(FEAT / "ecapa_all.npz", **ze)
        print("ecapa files", len(ze), flush=True)

        def e_ref(A: str, tk: int) -> torch.Tensor:
            v = np.mean([ze[f"{A}/t{t:02d}_neutral"] for t in range(10) if t != tk], 0)
            return F.normalize(torch.from_numpy(v).float(), dim=0)

        def render_id(env: torch.Tensor, c0: torch.Tensor, b: dict) -> torch.Tensor:
            f0 = b["f0o"]
            v = (f0 > 0).float()
            lf = torch.where(f0 > 0, torch.log(f0.clamp(min=1.0) / 200.0), torch.zeros_like(f0))
            cond = torch.cat([c0[:, None], env / 10, lf[:, None], v[:, None], b["per"] * v[:, None]], 1)
            exc = TR.excitation(f0, f0.shape[1] * N.HOP, torch.Generator(device="cpu").manual_seed(int(time.time() * 1000) % 100000))
            y = gen(cond, exc.to(dev))[:, N.DELAY:]
            e = ecap(y)
            er = b["eref"]
            return (1 - (e * er).sum(-1)).mean()
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    ev0 = evaluate(net, vdl, dev, a.no_op_env)
    print("init", json.dumps(ev0), file=log, flush=True); print("init", ev0, flush=True)
    t0 = time.time()
    it = iter(dl)
    for step in range(1, a.steps + 1):
        b = next(it)
        b = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
        if a.no_op_env:
            b["q"][:, K:K + 24] = 0
        nl = (torch.rand(b["q"].shape[0], device=dev) < a.p_null) if a.p_null > 0 else None
        mem = net.memory(b["rp"], b["rv"])
        env, dc0, _ = net(b["q"], b["tab"], mem, b["rm"], nl)
        loss, lv, lc = lossf(env, dc0, b, a.lam_c0)
        if a.null_both:
            env_n, dc0_n, _ = net(b["q"], b["tab"], mem, b["rm"], torch.ones(b["q"].shape[0], dtype=torch.bool, device=dev))
            loss = loss + a.w_null * lossf(env_n, dc0_n, b, a.lam_c0)[0]
        lid = torch.zeros(())
        if a.id_w > 0:
            lid = render_id(env, b["c0b"] + dc0, b)
            loss = loss + a.id_w * lid
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        if not torch.isfinite(gn):
            continue
        opt.step(); sched.step()
        if step % 100 == 0:
            msg = {"step": step, "loss": round(float(loss.detach()), 4), "env_v": round(float(lv.detach()), 4), "c0": round(float(lc.detach()), 4), "id": round(float(lid.detach()), 4), "s_per_step": round((time.time() - t0) / step, 3)}
            print(json.dumps(msg), file=log, flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(net, vdl, dev, a.no_op_env)
            print("eval", step, json.dumps(ev), file=log, flush=True); print("eval", step, ev, flush=True)
            torch.save({"net": net.state_dict(), "cfg": net.cfg | {"no_op_env": a.no_op_env}, "step": step, "args": vars(a)}, od / "last.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
