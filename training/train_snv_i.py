"""段 I(paper_snv §5 M2・M3・§6 段 I): 原点の生成のモデル y = C(u, lf, 有声; 目標の参照)に、内容で条件づけた共有の演奏の事前分布(識別の項)を入れて Irodori の全量で学ぶ。
E = 操作者の回路の逆(train_snv_p11.EncS: 操作者自身の参照を読む)→ q(u | y, S)・C = 表の型 x(T.Circ: 目標の参照を読む)。
事前分布: 2 声以上が発する内容 w は共有の型 μ_w(内容の中の位置 × 64・長さ = その内容の発話の長さの中央値 / 8 フレーム)と共通の対角の幅 Σ で N(μ_w[τ_t], Σ)。
τ = 発話ごとの単調な対応(4 フレームの平均の演奏と型の最尤・切り出しなので始まりと終わりは型の途中でよい)。1 声だけの内容は N(0, I)(段 T: 1 声の型は N(0, I) より悪い)。
損失 = 自己の reamp の ELBO(ラプラス・尺度を学ぶ・段 T の T4)= NLL + β KL(β は 5k で 0 → 1)。
参照(rev2・T5): 同じ声の普通の発話 3 本を最大 10 秒ずつ(約 27 秒 = 推論の目標の参照 約 20 秒・操作者の登録 30 秒にそろえる)。内容 = E1 の文・段 D の文・E1-NV の(種類・説明文の番号)。推論では内容も型も使わない。
事前登録 = results/snv_i/prereg.yaml。
rev3(paper_snv §5 M3・results/snv_i/verdict.yaml): 型は内容が束に現れるたびに、対応づけた事後の平均(4 フレームの平均・勾配を止める)で移動平均(--tpl ema・γ = --tpl_gamma)。
最初の出現は発話の中の時刻の比で位置へ置いて初期化(観測の無い位置は最も近い観測で埋める)・初めて観測した位置はその値を置く。--tpl sgd = 旧来(型を勾配の引数として学ぶ)。
対応は有効なフレーム(検証から外す冒頭 WARM_SELF の後)の区間で数える。型・位置ごとの観測の数・Σ を重みと一緒に保存する。tpl_r2 = 共有の内容のフレームで型が事後の平均の分散を説明する割合。
rev9(paper_snv §5 M7・--share hard): 固い共有。復号器 C の入力 = 内容の共有の軌跡 μ_w[τ](引数・再現の勾配で学ぶ・lr = --tpl_lr)+ 演技の残差 r(EncR = EncS に軌跡を足した入力・KL N(0, I)・--critic で r に判別器)。
1 声だけの内容は軌跡 0(r だけ)。E は y と操作者の構造から sg(μ_w[τ] + r の平均)を当てる(推論の演奏)・C の鍵 = sg(E(参照))。τ は最初の ALIGN_WARM step は発話の中の時刻の比・その後は E の出力と型の単調な対応。
rev11(paper_snv §5 M8・--share hard --residual none): 学習の残差なし。C の入力 = [共有の軌跡(D_CT = 59 次元)・明示の演技 5(音量 c0 と周期性 4 帯をその声の参照の平均と標準偏差で標準化・無声の周期性は 0)]。
E はこの 64 次元を当てる。内容の型の無い発話は C と E の学習から外す。
rev12〜14(paper_snv 系 2g・2h・§5 M9・M10): --align band(対応を発話の中の時刻の比 ±--band × L の帯に拘束)・--tpl_est ste(軌跡 = E の内容の出力の対応づけた移動平均 + straight-through)・
--tpl_est gste --group K(M10: 束 = 同じ内容の K 声を同じ時刻の比の窓で組み、軌跡は束の中で先に平均してから移動平均・E の勾配は組の同じ位置に対応づけた声どうしで平均)・
--tpl_est gswap(M11: M10 + C の内容の入力の値に同じ組の別の声の同じ位置のずれを足す・勾配は通さない)。
--tpl_est gloo(M12・rev17: M10 の勾配から自分を除く = 声 s の出力は同じ組の同じ位置に対応づけた他の声の再現の勾配だけを受ける・値は M10 と同じ)。ログの rho = 勾配の平均のうち自分のフレームの割合(M10 で声ごとの押しとして残る分)。
--student(M13・rev18): 推論の E は sg([軌跡, 明示の演技]) への回帰だけで学ぶ(再現の勾配を受けない)・軌跡と再現の勾配は学習だけの G(= 上の E)・C の参照の鍵 = sg(推論の E(参照))。保存の Es = 推論の E・Gs = G。ログの fit_s = 推論の E の当ての誤差 / 軌跡の分散。
ログの vr_ct = 軌跡の切り出しの中の分散 / 束の分散・ct_sq = 軌跡の大きさ・fit = E の内容の当ての誤差 / 軌跡の分散。
--e_only <ckpt>(M14・rev20): 保存した M13 の G・C・軌跡(型)を凍結し、推論の E だけを新しく sg([軌跡, 明示の演技]) への回帰で学ぶ(M13 の推論の E は G・C・軌跡へ勾配を返さないので同じ目的)。
--e_aug(M14): 推論の E の入力(発話とその声の参照)に、声の乱し = 包絡の周波数の一律の伸縮 α ∈ [0.8, 1.25](対数一様)+ 傾き ±0.5 を同じ値で掛ける(train_snv_p2_cpc.augment・P2b と同じ族)。当てる先は乱していない音から。
--rec K(rev21): 録音の条件で乱した音の特徴(snv_rec_feat.py・data/snv_p_feat_rec・版 0..K−1)を読み、標本ごとに版を 1 つ選ぶ(確率 --p_clean で元の音・源と参照は同じ版 = 同じ録音の場・束の選び方の乱数とは別の乱数)。
--e_rec(M16・rev21): --e_only の推論の E の入力(発話・有声・その声の参照)を選んだ版から(当てる先 = 軌跡と明示の演技は元の音から)。
--c_only <ckpt>(M15・rev21): 保存した M13 の G・推論の E・軌跡を凍結し、C だけを学ぶ(入力の値 = [軌跡, 明示の演技]・鍵 = 推論の E(参照)・保存の C から始める)。
--c_pitch S(M15): C の高さの入力を発話ごとに一律 U[−S, S] 半音ずらす(確率 --p_pitch・有声のフレームだけ)。--c_rec p(M15): 確率 p で C の高さと有声を選んだ版から。当てる先は元の y。
--rec_dir(M18・rev24): 録音の版の置き場(既定 data/snv_p_feat_rec = 合成の族・data/snv_p_feat_rec2 = 本物の非音声の音源の族)。--aug_lo / --aug_hi(M18): --e_aug の伸縮の域(既定 0.8 / 1.25)。
--e_play wst(M28・rev42): --e_only の推論の E を snv_encw.EncW に替える(入力 = P8 の白くした演奏 = 凍結した --cpc の文脈を --pca で 32 次元 × 4 つの遅れに積み、その声の参照の 2 次の統計で白くしたもの・参照を読む注意なし)。
--e_lift K(M19・rev25): --e_only の推論の E が読む全ての入力(発話と構造の参照)の包絡の c_K〜c24 を学習の入力の平均の定数(rec の版 400 本 + 元の音 130 本・seed 0・保存の lift_mean)に置き換える。当てる先と C の入力は元の音の全ての次数。
rev5(paper_snv §5 M5・--critic λ): 共有の内容の発話で、型からのずれ r = μ_q − μ_w[τ](型は止めた値)から学習の声を当てる因果の判別器 Q を交互に学び(Adam lr 1e-3)、演奏の側の損失に −λ × Q のフレームの交差エントロピー(I(u; s | w) を直接下げる)。
    cd training && uv run --no-sync python train_snv_i.py --tag snv_i_m3 --prereg results/snv_i/prereg.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_

import train_snv_p1 as T
import train_snv_p11 as P11

ROOT = Path(__file__).resolve().parent.parent
POOL = 4
TPL_STEP = 8
BETA_RAMP = 5000
MIN_VOICES = 2
REF_N = 3
REF_LEN = 2000
ALIGN_WARM = 2000
D_ACT = 5
D_CT = T.D_A - D_ACT


REC = ROOT / "data/snv_p_feat_rec"


def rec_path(p: Path, k: int) -> Path:
    return REC / p.relative_to(T.FEAT).with_name(f"{p.stem}_k{k}.npz")


def set_rec_dir(d: str) -> None:
    global REC
    REC = ROOT / d


def lift_mean(voices: list[str]) -> np.ndarray:
    rng = random.Random(0)
    tr = [p for v in voices for p in sorted((T.FEAT / "e1" / v).glob("t*.npz"))]
    pick = rng.sample(tr, 400)
    ys = [T.load_utt(rec_path(p, i % 2))["y"][:25] for i, p in enumerate(pick)] + [T.load_utt(p)["y"][:25] for p in pick[:130]]
    return np.concatenate(ys, 1).mean(1)


def load_data() -> tuple[dict, dict, list]:
    e1 = json.loads((ROOT / "data/snv_e1/manifest.json").read_text())
    dm = json.loads((ROOT / "data/snv_d/manifest.json").read_text())
    nv = json.loads((ROOT / "data/snv_e1nv/manifest.json").read_text())
    keys: dict = {}
    data: dict = {}
    for v in e1["voices"]:
        data.setdefault(v["voice"], {"norm": [], "nv": []})
        for f in v["files"]:
            data[v["voice"]]["norm"].append((T.FEAT / f"e1/{v['voice']}/{f['t']}.npz", ("txt", f["text"])))
    for v in dm["voices"]:
        for f in v["files"]:
            p = T.FEAT / f"d/{v['voice']}/{f['s']}.npz"
            if p.exists():
                data[v["voice"]]["norm"].append((p, ("txt", f["text"])))
    for it in nv["items"]:
        p = T.FEAT / f"e1nv/{it['voice']}/{it['cat']}_{it['k']}.npz"
        if p.exists():
            data[it["voice"]]["nv"].append((p, ("nv", it["cat"], it["text_id"])))
    return data, keys, []


class Corpus:
    def __init__(self, names: list[str], rec: int = 0) -> None:
        raw, _, _ = load_data()
        self.names = names
        self.rec, self.rec_missing = rec, 0
        self.u: dict = {}
        voices_of: dict = {}
        lens: dict = {}
        for n in names:
            self.u[n] = {"norm": [], "nv": []}
            for kind in ("norm", "nv"):
                for p, key in raw[n][kind]:
                    u = T.load_utt(p)
                    u["key"] = key
                    u["rec"] = []
                    for k in range(rec):
                        q = rec_path(p, k)
                        if not q.exists():
                            self.rec_missing += 1
                            continue
                        uk = T.load_utt(q)
                        assert uk["y"].shape == u["y"].shape, (q, uk["y"].shape, u["y"].shape)
                        u["rec"].append({"y": uk["y"], "v": uk["v"], "lf": uk["lf"]})
                    self.u[n][kind].append(u)
                    voices_of.setdefault(key, set()).add(n)
                    lens.setdefault(key, []).append(u["y"].shape[1])
        shared = sorted((k for k, vs in voices_of.items() if len(vs) >= MIN_VOICES), key=str)
        self.cid = {k: i for i, k in enumerate(shared)}
        self.tlen = np.array([max(2, int(math.ceil(np.median(lens[k]) / TPL_STEP))) for k in shared])
        self.n_voices = {k: len(voices_of[k]) for k in shared}
        for n in names:
            for kind in ("norm", "nv"):
                for u in self.u[n][kind]:
                    u["cid"] = self.cid.get(u["key"], -1)

    def stats(self) -> dict:
        utts = [u for n in self.names for k in ("norm", "nv") for u in self.u[n][k]]
        return {"voices": len(self.names), "utts": len(utts), "utts_shared": sum(u["cid"] >= 0 for u in utts), "contents_shared": len(self.cid), "rec": self.rec, "rec_missing": self.rec_missing,
                "voices_per_shared_content_median": float(np.median(list(self.n_voices.values()))) if self.n_voices else 0.0, "tpl_len_max": int(self.tlen.max()) if len(self.tlen) else 0}


class Batches:
    def __init__(self, c: Corpus, seed: int, p_nv: float = T.P_NV, p_clean: float = 1.0) -> None:
        self.c, self.rng, self.p_nv = c, random.Random(seed), p_nv
        self.rec_on, self.p_clean, self.rng_v = c.rec > 0, p_clean, random.Random(seed + 99)

    def _ver(self) -> int:
        return -1 if not self.rec_on or self.rng_v.random() < self.p_clean else self.rng_v.randrange(self.c.rec)

    def _crop(self, u: dict, L: int, at: int | None = None, ver: int = -1) -> tuple:
        n_all = u["y"].shape[1]
        s = self.rng.randint(0, max(0, n_all - L)) if at is None else min(max(0, at), max(0, n_all - L))
        n = min(L, n_all - s)
        y, lf, v, ok = np.zeros((T.D_Y, L), np.float32), np.zeros(L, np.float32), np.zeros(L, np.float32), np.zeros(L, np.float32)
        y[:, :n], lf[:n], v[:n], ok[:n] = u["y"][:, s:s + n], u["lf"][s:s + n], u["v"][s:s + n], 1.0
        self.last = (s, n_all)
        if self.rec_on:
            src = u["rec"][ver] if 0 <= ver < len(u["rec"]) else u
            yi, lfi, vi = np.zeros((T.D_Y, L), np.float32), np.zeros(L, np.float32), np.zeros(L, np.float32)
            yi[:, :n], lfi[:n], vi[:n] = src["y"][:, s:s + n], src["lf"][s:s + n], src["v"][s:s + n]
            self.last_in = (yi, lfi, vi)
        return y, lf, v, ok

    def _pack_in(self, YI: list, LI: list, VI: list, RYI: list, RVI: list, B: int, Lr: int) -> dict:
        ry = np.zeros((B, T.D_Y, Lr), np.float32)
        rv = np.zeros((B, Lr), np.float32)
        for i, (a, b) in enumerate(zip(RYI, RVI)):
            ry[i, :, :a.shape[1]], rv[i, :len(b)] = a, b
        t = torch.from_numpy
        return {"y_in": t(np.stack(YI)), "lf_in": t(np.stack(LI)), "v_in": t(np.stack(VI)), "ry_in": t(ry), "rv_in": t(rv)}

    def sample(self, bs: int) -> dict:
        Y, LF, V, OK, CI, RY, RV, RM, OFF, NA, SP = [], [], [], [], [], [], [], [], [], [], []
        YI, LI, VI, RYI, RVI = [], [], [], [], []
        for _ in range(bs):
            sp = self.rng.randrange(len(self.c.names))
            SP.append(sp)
            d = self.c.u[self.c.names[sp]]
            idx = list(range(len(d["norm"])))
            self.rng.shuffle(idx)
            use_nv = bool(d["nv"]) and self.rng.random() < self.p_nv
            tgt = self.rng.choice(d["nv"]) if use_nv else d["norm"][idx[0]]
            ver = self._ver()
            y, lf, v, ok = self._crop(tgt, T.CROP, ver=ver)
            Y.append(y), LF.append(lf), V.append(v), OK.append(ok), CI.append(tgt["cid"]), OFF.append(self.last[0]), NA.append(self.last[1])
            if self.rec_on:
                YI.append(self.last_in[0]), LI.append(self.last_in[1]), VI.append(self.last_in[2])
            refs = [d["norm"][i] for i in idx[1:1 + REF_N]] if not use_nv else [d["norm"][i] for i in idx[:REF_N]]
            ry, rv, ryi, rvi = [], [], [], []
            for r in refs:
                yy, _, vv, okk = self._crop(r, REF_LEN, ver=ver)
                n = int(okk.sum())
                ry.append(yy[:, :n]), rv.append(vv[:n])
                if self.rec_on:
                    ryi.append(self.last_in[0][:, :n]), rvi.append(self.last_in[2][:n])
            RY.append(np.concatenate(ry, 1)), RV.append(np.concatenate(rv))
            if self.rec_on:
                RYI.append(np.concatenate(ryi, 1)), RVI.append(np.concatenate(rvi))
        Lr = max(r.shape[1] for r in RY)
        ry = np.zeros((bs, T.D_Y, Lr), np.float32)
        rv = np.zeros((bs, Lr), np.float32)
        rm = np.zeros((bs, Lr), bool)
        for i, (a, b) in enumerate(zip(RY, RV)):
            ry[i, :, :a.shape[1]], rv[i, :len(b)], rm[i, :len(b)] = a, b, True
        t = torch.from_numpy
        out = {"y": t(np.stack(Y)), "lf": t(np.stack(LF)), "v": t(np.stack(V)), "ok": t(np.stack(OK)), "cid": torch.tensor(CI), "off": torch.tensor(OFF), "nall": torch.tensor(NA), "spk": torch.tensor(SP), "ry": t(ry), "rv": t(rv), "rmask": t(rm)}
        return out | (self._pack_in(YI, LI, VI, RYI, RVI, bs, Lr) if self.rec_on else {})

    def sample_grouped(self, bs: int, K: int) -> dict:
        if not hasattr(self, "by_cid"):
            self.by_cid: dict = {}
            kind_of: dict = {}
            for sp, n in enumerate(self.c.names):
                for kind in ("norm", "nv"):
                    for j, u in enumerate(self.c.u[n][kind]):
                        if u["cid"] >= 0:
                            self.by_cid.setdefault(u["cid"], {}).setdefault(sp, []).append((kind, j))
                            kind_of[u["cid"]] = u["key"][0]
            ks = sorted(c for c, vs in self.by_cid.items() if len(vs) >= 2)
            self.cid_nv = [c for c in ks if kind_of[c] == "nv"]
            self.cid_txt = [c for c in ks if kind_of[c] != "nv"]
        Y, LF, V, OK, CI, RY, RV, RM, OFF, NA, SP, GI = [], [], [], [], [], [], [], [], [], [], [], []
        YI, LI, VI, RYI, RVI = [], [], [], [], []
        g = 0
        while bs - len(Y) >= 2:
            pool = self.cid_nv if (self.cid_nv and self.rng.random() < self.p_nv) else self.cid_txt
            c = self.rng.choice(pool)
            vs = list(self.by_cid[c])
            self.rng.shuffle(vs)
            vs = vs[:min(K, bs - len(Y))]
            tg = [(sp, self.rng.choice(self.by_cid[c][sp])) for sp in vs]
            us = [self.c.u[self.c.names[sp]][kind][j] for sp, (kind, j) in tg]
            nmin = min(u["y"].shape[1] for u in us)
            r = self.rng.random() * max(0.0, 1.0 - T.CROP / nmin)
            for (sp, (kind, j)), tgt in zip(tg, us):
                d = self.c.u[self.c.names[sp]]
                ver = self._ver()
                y, lf, v, ok = self._crop(tgt, T.CROP, at=int(round(r * tgt["y"].shape[1])), ver=ver)
                Y.append(y), LF.append(lf), V.append(v), OK.append(ok), CI.append(tgt["cid"]), OFF.append(self.last[0]), NA.append(self.last[1]), SP.append(sp), GI.append(g)
                if self.rec_on:
                    YI.append(self.last_in[0]), LI.append(self.last_in[1]), VI.append(self.last_in[2])
                idx = [i for i in range(len(d["norm"])) if not (kind == "norm" and i == j)]
                self.rng.shuffle(idx)
                ry, rv, ryi, rvi = [], [], [], []
                for i in idx[:REF_N]:
                    yy, _, vv, okk = self._crop(d["norm"][i], REF_LEN, ver=ver)
                    n = int(okk.sum())
                    ry.append(yy[:, :n]), rv.append(vv[:n])
                    if self.rec_on:
                        ryi.append(self.last_in[0][:, :n]), rvi.append(self.last_in[2][:n])
                RY.append(np.concatenate(ry, 1)), RV.append(np.concatenate(rv))
                if self.rec_on:
                    RYI.append(np.concatenate(ryi, 1)), RVI.append(np.concatenate(rvi))
            g += 1
        B = len(Y)
        Lr = max(r_.shape[1] for r_ in RY)
        ry = np.zeros((B, T.D_Y, Lr), np.float32)
        rv = np.zeros((B, Lr), np.float32)
        rm = np.zeros((B, Lr), bool)
        for i, (a, b) in enumerate(zip(RY, RV)):
            ry[i, :, :a.shape[1]], rv[i, :len(b)], rm[i, :len(b)] = a, b, True
        t = torch.from_numpy
        out = {"y": t(np.stack(Y)), "lf": t(np.stack(LF)), "v": t(np.stack(V)), "ok": t(np.stack(OK)), "cid": torch.tensor(CI), "off": torch.tensor(OFF), "nall": torch.tensor(NA), "spk": torch.tensor(SP), "ry": t(ry), "rv": t(rv), "rmask": t(rm), "gid": torch.tensor(GI)}
        return out | (self._pack_in(YI, LI, VI, RYI, RVI, B, Lr) if self.rec_on else {})


def mas_free(logp: torch.Tensor, tl: torch.Tensor, ll: torch.Tensor) -> torch.Tensor:
    B, Tn, L = logp.shape
    neg = -1e9
    jj = torch.arange(L, device=logp.device)[None]
    lmask = jj < ll[:, None]
    Q = torch.where(lmask, logp[:, 0], torch.full_like(logp[:, 0], neg))
    back = torch.zeros(B, Tn, L, dtype=torch.bool, device=logp.device)
    for t in range(1, Tn):
        move = torch.cat([torch.full_like(Q[:, :1], neg), Q[:, :-1]], 1)
        cm = move > Q
        Qn = torch.where(cm, move, Q) + logp[:, t]
        Qn = torch.where(lmask, Qn, torch.full_like(Qn, neg))
        act = (t < tl)[:, None]
        Q = torch.where(act, Qn, Q)
        back[:, t] = cm & act
    j = Q.argmax(1)
    path = torch.zeros(B, Tn, dtype=torch.long, device=logp.device)
    ar = torch.arange(B, device=logp.device)
    for t in range(Tn - 1, -1, -1):
        act = t < tl
        path[:, t] = torch.where(act, j, path[:, t])
        mv = back[ar, t, j] & act & (t > 0)
        j = (j - mv.long()).clamp(min=0)
    return path


@torch.no_grad()
def ema_update(tpl: torch.Tensor, npos: torch.Tensor, cids: torch.Tensor, path: torch.Tensor, mp: torch.Tensor, tl: torch.Tensor, ll: torch.Tensor, init_pos: torch.Tensor, gamma: float) -> None:
    for b in range(len(cids)):
        c, n, L = int(cids[b]), int(tl[b]), int(ll[b])
        if n == 0:
            continue
        first = not bool((npos[c, :L] > 0).any())
        j = (init_pos[b, :n] if first else path[b, :n]).clamp(0, L - 1)
        cnt = torch.bincount(j, minlength=L)[:L]
        mean = torch.zeros(L, mp.shape[1], device=mp.device).index_add_(0, j, mp[b, :, :n].T) / cnt.clamp(min=1)[:, None].float()
        obs = cnt > 0
        if first:
            idx = torch.arange(L, device=mp.device)
            oi = idx[obs]
            tpl[c, :L] = mean[oi[(idx[:, None] - oi[None, :]).abs().argmin(1)]]
        else:
            new = obs & (npos[c, :L] == 0)
            upd = obs & (npos[c, :L] > 0)
            row = tpl[c, :L]
            row[new] = mean[new]
            row[upd] = gamma * row[upd] + (1 - gamma) * mean[upd]
        npos[c, :L] += cnt.to(npos.dtype)


@torch.no_grad()
def ema_update_batch(tpl: torch.Tensor, npos: torch.Tensor, cids: torch.Tensor, path: torch.Tensor, mp: torch.Tensor, tl: torch.Tensor, ll: torch.Tensor, init_pos: torch.Tensor, gamma: float) -> None:
    D, Tn, Lm = mp.shape[1], mp.shape[2], tpl.shape[1]
    valid = torch.arange(Tn, device=mp.device)[None] < tl[:, None]
    fresh = ~(npos[cids] > 0).any(1)
    j = torch.minimum(torch.where(fresh[:, None], init_pos, path).clamp(min=0), ll[:, None] - 1)
    uk, inv = torch.unique((cids[:, None] * Lm + j)[valid], return_inverse=True)
    cnt = torch.bincount(inv, minlength=len(uk))
    mean = torch.zeros(len(uk), D, device=mp.device).index_add_(0, inv, mp.transpose(1, 2)[valid]) / cnt[:, None].float()
    flat, nflat = tpl.view(-1, tpl.shape[-1]), npos.view(-1)
    old = nflat[uk] > 0
    flat[uk[old]] = gamma * flat[uk[old]] + (1 - gamma) * mean[old]
    flat[uk[~old]] = mean[~old]
    nflat[uk] += cnt.to(nflat.dtype)
    for c in torch.unique(cids[fresh]).tolist():
        L = int(ll[(cids == c).nonzero()[0, 0]])
        idx = torch.arange(L, device=mp.device)
        oi = idx[npos[c, :L] > 0]
        tpl[c, :L] = tpl[c, :L][oi[(idx[:, None] - oi[None, :]).abs().argmin(1)]]


def group_ste(mu: torch.Tensor, pv: torch.Tensor, tau: torch.Tensor, m: torch.Tensor, gid: torch.Tensor, Lm: int, swap: bool = False, loo: bool = False) -> torch.Tensor:
    B, D, _ = mu.shape
    valid = (m > 0) & (tau >= 0)
    t0 = tau.clamp(min=0)
    key = gid[:, None] * Lm + t0
    kv = key[valid]
    G = int(gid.max()) + 1
    mt = mu.transpose(1, 2)[valid]
    one = torch.ones(len(kv), device=mu.device, dtype=mu.dtype)
    if loo:
        rk = torch.arange(B, device=mu.device)[:, None] * Lm + t0
        sg = torch.zeros(G * Lm, D, device=mu.device, dtype=mu.dtype).index_add(0, kv, mt)
        ng = torch.zeros(G * Lm, device=mu.device, dtype=mu.dtype).index_add(0, kv, one)
        sr = torch.zeros(B * Lm, D, device=mu.device, dtype=mu.dtype).index_add(0, rk[valid], mt)
        nr = torch.zeros(B * Lm, device=mu.device, dtype=mu.dtype).index_add(0, rk[valid], one)
        no = ng[key] - nr[rk]
        h = ((sg[key] - sr[rk]) / no.clamp(min=1.0)[..., None] * ((no > 0) & valid)[..., None].to(mu.dtype)).transpose(1, 2)
        return pv + h - h.detach()
    ub = torch.zeros(G * Lm, D, device=mu.device, dtype=mu.dtype).index_add(0, kv, mt) / torch.zeros(G * Lm, device=mu.device, dtype=mu.dtype).index_add(0, kv, one).clamp(min=1.0)[:, None]
    g = ub[key].transpose(1, 2) * valid[:, None]
    out = pv + g - g.detach()
    if not swap:
        return out
    with torch.no_grad():
        ar = torch.arange(B, device=mu.device)
        kb = (ar[:, None] * Lm + t0)[valid]
        cb = torch.zeros(B * Lm, device=mu.device, dtype=mu.dtype).index_add_(0, kb, one)
        db = torch.zeros(B * Lm, D, device=mu.device, dtype=mu.dtype).index_add_(0, kb, mt.detach()) / cb.clamp(min=1.0)[:, None]
        db = (db - ub.detach()[(gid[:, None] * Lm + torch.arange(Lm, device=mu.device)[None]).reshape(-1)]) * (cb > 0)[:, None].to(mu.dtype)
        first = torch.zeros(G, dtype=torch.long, device=mu.device).scatter_reduce(0, gid, ar, reduce="amin", include_self=False)
        size = torch.bincount(gid, minlength=G)
        partner = first[gid] + (ar - first[gid] + 1) % size[gid]
        sw = db[partner[:, None] * Lm + t0] * (valid & (size[gid] > 1)[:, None])[..., None].to(mu.dtype)
    return out + sw.transpose(1, 2)


def own_share(tau: torch.Tensor, m: torch.Tensor, gid: torch.Tensor, Lm: int) -> float:
    valid = (m > 0) & (tau >= 0)
    t0 = tau.clamp(min=0)
    key = (gid[:, None] * Lm + t0)[valid]
    rk = (torch.arange(len(gid), device=tau.device)[:, None] * Lm + t0)[valid]
    one = torch.ones(len(key), device=tau.device)
    ng = torch.zeros((int(gid.max()) + 1) * Lm, device=tau.device).index_add_(0, key, one)
    nr = torch.zeros(len(gid) * Lm, device=tau.device).index_add_(0, rk, one)
    return float((nr[rk] / ng[key]).mean()) if len(key) else float("nan")

def kl_terms(mu: torch.Tensor, lv: torch.Tensor, m: torch.Tensor, cid: torch.Tensor, tpl: torch.Tensor, tlen: torch.Tensor, logsig: nn.Parameter,
             ema: float | None = None, npos: torch.Tensor | None = None, off: torch.Tensor | None = None, nall: torch.Tensor | None = None, aux: dict | None = None) -> tuple[torch.Tensor, float, float]:
    kl0 = 0.5 * (mu * mu + lv.exp() - 1.0 - lv).sum(1)
    sh = cid >= 0
    if not bool(sh.any()):
        return kl0, 0.0, 0.0
    i = sh.nonzero()[:, 0]
    mus, lvs, ms = mu[i], lv[i], m[i]
    ll = tlen[cid[i]]
    sig2 = (2 * logsig).exp()
    B_, D_, Tf = mus.shape
    Tn = Tf // POOL
    with torch.no_grad():
        wm = ms[:, :Tn * POOL].reshape(B_, Tn, POOL).sum(-1)
        mp_all = (mus.detach()[..., :Tn * POOL] * ms[:, None, :Tn * POOL]).reshape(B_, D_, Tn, POOL).sum(-1) / wm[:, None].clamp(min=1.0)
        valid = wm > 0
        tl = valid.sum(1)
        first = valid.float().argmax(1)
        ar = torch.arange(Tn, device=mu.device)
        mp = torch.gather(mp_all, 2, (first[:, None] + ar[None]).clamp(max=Tn - 1)[:, None].expand(-1, D_, -1))
        tp = tpl[cid[i]].detach()
        lp = -0.5 * (((mp.transpose(1, 2)[:, :, None, :] - tp[:, None, :, :]) ** 2) / sig2).sum(-1)
        path = mas_free(lp, tl, ll)
        if ema is not None:
            fr = (off[i][:, None] + (first[:, None] + ar[None]) * POOL + POOL // 2).float()
            init_pos = (fr / nall[i][:, None].float() * ll[:, None].float()).long()
            ema_update(tpl, npos, cid[i], path, mp, tl, ll, init_pos, ema)
        tau_p = torch.gather(path, 1, (ar[None] - first[:, None]).clamp(0, Tn - 1))
        tau = tau_p.repeat_interleave(POOL, 1)
        tau = F_.pad(tau, (0, Tf - tau.shape[1]), value=0) if tau.shape[1] < Tf else tau[:, :Tf]
    tp = tpl[cid[i]] if ema is None else tpl[cid[i]].detach()
    pm = torch.gather(tp, 1, tau[..., None].expand(-1, -1, tp.shape[-1])).transpose(1, 2)
    klt = (0.5 * (2 * logsig[None, :, None] - lvs + (lvs.exp() + (mus - pm) ** 2) / sig2[None, :, None] - 1)).sum(1)
    if aux is not None:
        aux["i"], aux["res"] = i, mus - pm.detach()
    with torch.no_grad():
        w = ms[:, None]
        mbar = (mus * w).sum((0, 2)) / w.sum().clamp(min=1.0)
        r2 = 1.0 - float((((mus - pm) ** 2) * w).sum() / (((mus - mbar[None, :, None]) ** 2) * w).sum().clamp(min=1e-12))
    kl = kl0.clone()
    kl[i] = klt
    return kl, float(sh.float().mean()), r2


class EncR(P11.EncS):
    def __init__(self) -> None:
        super().__init__()
        self.inp = nn.Conv1d(T.D_X + T.D_A, self.inp.out_channels, 1)


def share_path(mu: torch.Tensor, m: torch.Tensor, cid: torch.Tensor, tpl: torch.Tensor, tlen: torch.Tensor, off: torch.Tensor, nall: torch.Tensor, warm: bool,
               band: float = 0.0, ema: float | None = None, npos: torch.Tensor | None = None, batch_ema: bool = False, ret_tau: bool = False) -> tuple:
    B_, D_, Tf = mu.shape
    pm = torch.zeros_like(mu)
    sh = cid >= 0
    tau_full = torch.full((B_, Tf), -1, dtype=torch.long, device=mu.device)
    if not bool(sh.any()):
        return (pm, 0.0, tau_full) if ret_tau else (pm, 0.0)
    i = sh.nonzero()[:, 0]
    ms, ll = m[i], tlen[cid[i]]
    Tn = Tf // POOL
    with torch.no_grad():
        wm = ms[:, :Tn * POOL].reshape(len(i), Tn, POOL).sum(-1)
        mp_all = (mu[i][..., :Tn * POOL] * ms[:, None, :Tn * POOL]).reshape(len(i), D_, Tn, POOL).sum(-1) / wm[:, None].clamp(min=1.0)
        valid = wm > 0
        tl = valid.sum(1)
        first = valid.float().argmax(1)
        ar = torch.arange(Tn, device=mu.device)
        fr = (off[i][:, None] + (first[:, None] + ar[None]) * POOL + POOL // 2).float()
        ctr = fr / nall[i][:, None].float() * ll[:, None].float()
        mp = torch.gather(mp_all, 2, (first[:, None] + ar[None]).clamp(max=Tn - 1)[:, None].expand(-1, D_, -1))
        if warm:
            path = torch.minimum(ctr.long(), ll[:, None] - 1)
        else:
            tp = tpl[cid[i]].detach()
            lp = -0.5 * ((mp.transpose(1, 2)[:, :, None, :] - tp[:, None, :, :]) ** 2).sum(-1)
            if band:
                jj = torch.arange(lp.shape[2], device=mu.device)[None, None].float()
                lp = lp.masked_fill((jj - ctr[..., None]).abs() > torch.clamp(band * ll.float(), min=1.0)[:, None, None], -1e9)
            path = mas_free(lp, tl, ll)
        if ema is not None:
            fresh = ~(npos[cid[i]] > 0).any(1)
            path = torch.where(fresh[:, None], torch.minimum(ctr.long(), ll[:, None] - 1), path)
            (ema_update_batch if batch_ema else ema_update)(tpl, npos, cid[i], path, mp, tl, ll, ctr.long(), ema)
        tau = torch.gather(path, 1, (ar[None] - first[:, None]).clamp(0, Tn - 1)).repeat_interleave(POOL, 1)
        tau = F_.pad(tau, (0, Tf - tau.shape[1]), value=0) if tau.shape[1] < Tf else tau[:, :Tf]
    tp = tpl[cid[i]]
    pm[i] = torch.gather(tp, 1, tau[..., None].expand(-1, -1, tp.shape[-1])).transpose(1, 2)
    tau_full[i] = tau
    return (pm, float(sh.float().mean()), tau_full) if ret_tau else (pm, float(sh.float().mean()))


def var_ratio(x: torch.Tensor, m: torch.Tensor) -> float:
    w = m[:, None]
    n = w.sum(-1).clamp(min=1.0)
    mu = (x * w).sum(-1) / n
    within = ((((x - mu[..., None]) ** 2) * w).sum(-1) / n).sum(1)
    keep = m.sum(-1) > 0
    g = (x * w).sum((0, 2)) / w.sum().clamp(min=1.0)
    total = (((x - g[None, :, None]) ** 2) * w).sum() / w.sum().clamp(min=1.0)
    return float(within[keep].mean() / total.clamp(min=1e-12)) if bool(keep.any()) else float("nan")


def acting_feats(y: torch.Tensor, v: torch.Tensor, ry: torch.Tensor, rv: torch.Tensor, rmask: torch.Tensor) -> torch.Tensor:
    w = rmask.float()
    n = w.sum(-1).clamp(min=1.0)
    c0 = ry[:, 0]
    m0 = (c0 * w).sum(-1) / n
    s0 = ((((c0 - m0[:, None]) ** 2) * w).sum(-1) / n).sqrt() + 1e-3
    wv = rv * w
    nv = wv.sum(-1).clamp(min=1.0)[:, None]
    pr = ry[:, 25:29]
    mp = (pr * wv[:, None]).sum(-1) / nv
    sp = ((((pr - mp[..., None]) ** 2) * wv[:, None]).sum(-1) / nv).sqrt() + 1e-3
    return torch.cat([((y[:, 0] - m0[:, None]) / s0[:, None])[:, None], (y[:, 25:29] - mp[..., None]) / sp[..., None] * v[:, None]], 1)


class Critic(nn.Module):
    def __init__(self, n_spk: int, d: int = T.D_A, ch: int = 256) -> None:
        super().__init__()
        self.c1 = nn.Conv1d(d, ch, 5)
        self.c2 = nn.Conv1d(ch, ch, 5, dilation=2)
        self.out = nn.Conv1d(ch, n_spk, 1)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        h = F_.relu(self.c1(F_.pad(r, (4, 0))))
        h = F_.relu(self.c2(F_.pad(h, (8, 0))))
        return self.out(h)


def unit_tests() -> None:
    lp = torch.full((1, 6, 5), -10.0)
    for t, j in enumerate([1, 1, 2, 3, 3, 3]):
        lp[0, t, j] = 0.0
    got = mas_free(lp, torch.tensor([6]), torch.tensor([5])).tolist()
    assert got == [[1, 1, 2, 3, 3, 3]], f"始まりと終わりの自由な単調な対応がずれる {got}"
    lp2 = torch.full((1, 4, 3), -10.0)
    for t, j in enumerate([0, 1, 1, 2]):
        lp2[0, t, j] = 0.0
    assert mas_free(lp2, torch.tensor([4]), torch.tensor([3])).tolist() == [[0, 1, 1, 2]], "単調な対応がずれる"
    mu, lv = torch.randn(2, 4, 16), torch.zeros(2, 4, 16)
    tpl = nn.Parameter(torch.zeros(3, 5, 4))
    kl, frac, _ = kl_terms(mu, lv, torch.ones(2, 16), torch.tensor([-1, -1]), tpl, torch.tensor([5, 5, 5]), nn.Parameter(torch.zeros(4)))
    assert torch.allclose(kl, 0.5 * (mu * mu).sum(1)) and frac == 0.0, "共有の無い内容が N(0, I) にならない"
    kl2, frac2, _ = kl_terms(mu, lv, torch.ones(2, 16), torch.tensor([0, 2]), tpl, torch.tensor([5, 5, 5]), nn.Parameter(torch.zeros(4)))
    assert torch.allclose(kl2, kl, atol=1e-5) and frac2 == 1.0, "型 0・幅 1 の事前分布が N(0, I) と一致しない"
    tp = torch.zeros(2, 4, 2)
    npos = torch.zeros(2, 4, dtype=torch.long)
    mp = torch.arange(4.0).repeat(2, 1)[None]
    ema_update(tp, npos, torch.tensor([1]), torch.tensor([[0, 0, 0, 0]]), mp, torch.tensor([3]), torch.tensor([4]), torch.tensor([[0, 2, 2, 0]]), 0.5)
    assert tp[1, :, 0].tolist() == [0.0, 0.0, 1.5, 1.5] and npos[1].tolist() == [1, 0, 2, 0] and float(tp[0].abs().sum()) == 0.0, f"型の初期化がずれる {tp[1, :, 0].tolist()}"
    ema_update(tp, npos, torch.tensor([1]), torch.tensor([[1, 2, 2, 3]]), mp, torch.tensor([4]), torch.tensor([4]), torch.tensor([[0, 0, 0, 0]]), 0.5)
    assert tp[1, :, 0].tolist() == [0.0, 0.0, 1.5, 3.0] and npos[1].tolist() == [1, 1, 4, 1], f"型の移動平均がずれる {tp[1, :, 0].tolist()} {npos[1].tolist()}"
    tq, nq = torch.zeros(2, 4, 2), torch.zeros(2, 4, dtype=torch.long)
    ema_update_batch(tq, nq, torch.tensor([1]), torch.tensor([[0, 0, 0, 0]]), mp, torch.tensor([3]), torch.tensor([4]), torch.tensor([[0, 2, 2, 0]]), 0.5)
    ema_update_batch(tq, nq, torch.tensor([1]), torch.tensor([[1, 2, 2, 3]]), mp, torch.tensor([4]), torch.tensor([4]), torch.tensor([[0, 0, 0, 0]]), 0.5)
    assert torch.equal(tq, tp) and torch.equal(nq, npos), f"束の型の更新が 1 本ずつの更新とずれる {tq[1, :, 0].tolist()} {nq[1].tolist()}"
    tq2, nq2 = torch.zeros(1, 3, 1), torch.zeros(1, 3, dtype=torch.long)
    mp2 = torch.tensor([[[2.0, 4.0]], [[6.0, 8.0]]])
    ema_update_batch(tq2, nq2, torch.tensor([0, 0]), torch.tensor([[0, 1], [0, 1]]), mp2, torch.tensor([2, 2]), torch.tensor([3, 3]), torch.tensor([[0, 1], [0, 1]]), 0.5)
    assert tq2[0, :, 0].tolist() == [4.0, 6.0, 6.0] and nq2[0].tolist() == [2, 2, 0], f"同じ内容の声を先に平均する更新がずれる {tq2[0, :, 0].tolist()}"
    mg = torch.randn(4, 3, 5, requires_grad=True)
    pg = torch.randn(4, 3, 5)
    tg = torch.tensor([[0, 0, 1, 1, 2]] * 2 + [[0, 1, 1, 2, 2]] * 2)
    mk = torch.ones(4, 5)
    mk[3, 4] = 0
    xg = group_ste(mg, pg, tg, mk, torch.tensor([0, 0, 1, 1]), 3)
    wg = torch.randn(4, 3, 5)
    (xg * wg).sum().backward()
    want = torch.zeros_like(mg)
    for b_ in range(4):
        for t_ in range(5):
            if mk[b_, t_] == 0:
                continue
            sel = [(b2, t2) for b2 in range(4) if b2 // 2 == b_ // 2 for t2 in range(5) if tg[b2, t2] == tg[b_, t_] and mk[b2, t2] > 0]
            want[b_, :, t_] = sum(wg[b2, :, t2] for b2, t2 in sel) / len(sel)
    assert torch.allclose(xg, pg) and torch.allclose(mg.grad, want, atol=1e-6), "同じ内容の声どうしで平均した勾配がずれる"
    mg2 = mg.detach().clone().requires_grad_(True)
    xs = group_ste(mg2, pg, tg, mk, torch.tensor([0, 0, 1, 1]), 3, swap=True)
    (xs * wg).sum().backward()
    assert torch.allclose(mg2.grad, want, atol=1e-6), "ずれを見せても勾配が変わってはいけない"
    mu_bj = lambda b_, j_: torch.stack([mg[b_, :, t_] for t_ in range(5) if tg[b_, t_] == j_ and mk[b_, t_] > 0]).mean(0)
    ubar = lambda g_, j_: torch.stack([mg[b_, :, t_] for b_ in range(4) if b_ // 2 == g_ for t_ in range(5) if tg[b_, t_] == j_ and mk[b_, t_] > 0]).mean(0)
    for b_, t_ in ((0, 0), (1, 4), (2, 3), (3, 0)):
        pb_ = b_ ^ 1
        j_ = int(tg[b_, t_])
        has = any(tg[pb_, t2] == j_ and mk[pb_, t2] > 0 for t2 in range(5))
        d_ = (mu_bj(pb_, j_) - ubar(b_ // 2, j_)) if has else torch.zeros(3)
        assert torch.allclose(xs[b_, :, t_] - pg[b_, :, t_], d_.detach(), atol=1e-5), f"別の声のずれがずれる {b_} {t_}"
    x1 = group_ste(mg.detach(), pg, tg, mk, torch.tensor([0, 1, 2, 2]), 3, swap=True)
    assert torch.allclose(x1[:2], pg[:2]), "1 声だけの組にずれを足してはいけない"
    mg3 = mg.detach().clone().requires_grad_(True)
    xl = group_ste(mg3, pg, tg, mk, torch.tensor([0, 0, 1, 1]), 3, loo=True)
    (xl * wg).sum().backward()
    no_ = lambda b2, t2: sum(1 for b3 in range(4) if b3 // 2 == b2 // 2 and b3 != b2 for t3 in range(5) if tg[b3, t3] == tg[b2, t2] and mk[b3, t3] > 0)
    wl = torch.zeros_like(mg)
    for b_ in range(4):
        for t_ in range(5):
            if mk[b_, t_] == 0:
                continue
            for b2 in range(4):
                for t2 in range(5):
                    if b2 // 2 == b_ // 2 and b2 != b_ and tg[b2, t2] == tg[b_, t_] and mk[b2, t2] > 0:
                        wl[b_, :, t_] += wg[b2, :, t2] / no_(b2, t2)
    assert torch.allclose(xl, pg) and torch.allclose(mg3.grad, wl, atol=1e-6), "自分を除いた声どうしの勾配がずれる"
    assert abs(own_share(tg, mk, torch.tensor([0, 0, 1, 1]), 3) - (5 + 1.5 + 4 / 3 + 1.5 + 1 / 3) / 19) < 1e-6, "自己の分の数え方がずれる"
    m2 = torch.ones(1, 16)
    m2[:, :4] = 0
    tpl2 = torch.zeros(1, 4, 4)
    tpl2[0, :, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0])
    mu2 = torch.zeros(1, 4, 16)
    mu2[0, 0, 4:] = torch.tensor([0.0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    kl3, _, r2 = kl_terms(mu2, torch.zeros(1, 4, 16), m2, torch.tensor([0]), tpl2, torch.tensor([4]), nn.Parameter(torch.zeros(4)))
    assert abs(r2 - 1.0) < 1e-6, f"冒頭を外した区間の対応がずれる(型の説明率 {r2})"
    tplh = torch.zeros(1, 4, 2)
    tplh[0, :, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0])
    muh = torch.zeros(1, 2, 16)
    muh[0, 0, 4:] = torch.tensor([0.0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    mh = torch.ones(1, 16)
    mh[:, :4] = 0
    pmh, _ = share_path(muh, mh, torch.tensor([0]), tplh, torch.tensor([4]), torch.tensor([0]), torch.tensor([16]), False)
    assert torch.allclose(pmh[0, :, 4:], muh[0, :, 4:]), f"固い共有の対応がずれる {pmh[0, 0].tolist()}"
    pmw, _ = share_path(muh, mh, torch.tensor([0]), tplh, torch.tensor([4]), torch.tensor([0]), torch.tensor([16]), True)
    assert pmw[0, 0, 4:].tolist() == [1.0] * 4 + [2.0] * 4 + [3.0] * 4, f"時刻の比の対応がずれる {pmw[0, 0].tolist()}"
    pm0, f0 = share_path(muh, mh, torch.tensor([-1]), tplh, torch.tensor([4]), torch.tensor([0]), torch.tensor([16]), False)
    assert float(pm0.abs().sum()) == 0.0 and f0 == 0.0, "1 声だけの内容の軌跡が 0 にならない"
    muz = torch.zeros(1, 2, 16)
    pf, _ = share_path(muz, mh, torch.tensor([0]), tplh, torch.tensor([4]), torch.tensor([0]), torch.tensor([16]), False)
    assert float(pf[0, :, 4:].abs().sum()) == 0.0, f"端の自由な対応が定数に張り付かない(段 I5 の再現){pf[0, 0].tolist()}"
    pb, _ = share_path(muz, mh, torch.tensor([0]), tplh, torch.tensor([4]), torch.tensor([0]), torch.tensor([16]), False, band=0.1)
    vb = pb[0, 0, 4:]
    assert float(vb[0]) >= 1.0 and float(vb[-1]) == 3.0 and bool((vb[1:] >= vb[:-1]).all()), f"帯の対応が進まない {vb.tolist()}"
    tple, npe = torch.zeros(1, 4, 2), torch.zeros(1, 4, dtype=torch.int32)
    pe, _ = share_path(muh, mh, torch.tensor([0]), tple, torch.tensor([4]), torch.tensor([0]), torch.tensor([16]), False, band=0.1, ema=0.9, npos=npe)
    assert torch.allclose(pe[0, :, 4:], muh[0, :, 4:]) and int(npe.sum()) == 3 and tple[0, :, 0].tolist() == [0.0, 0.0, 1.0, 2.0], f"E の平均の軌跡(M9)がずれる {pe[0, 0].tolist()} {tple[0, :, 0].tolist()}"
    P11.unit_tests()
    print("unit tests ok(自由な単調な対応・事前分布の切り替え・固い共有の対応・帯の対応・E の平均の軌跡・P11 の E)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--prereg", default="results/snv_i/prereg.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tpl", default="ema", choices=["ema", "sgd"])
    ap.add_argument("--tpl_gamma", type=float, default=0.9)
    ap.add_argument("--critic", type=float, default=0.0)
    ap.add_argument("--share", default="soft", choices=["soft", "hard"])
    ap.add_argument("--residual", default="kl", choices=["kl", "none"])
    ap.add_argument("--tpl_lr", type=float, default=5e-3)
    ap.add_argument("--tpl_est", default="grad", choices=["grad", "ste", "gste", "gswap", "gloo"])
    ap.add_argument("--group", type=int, default=0)
    ap.add_argument("--align", default="free", choices=["free", "band"])
    ap.add_argument("--band", type=float, default=0.1)
    ap.add_argument("--align_warm", type=int, default=ALIGN_WARM)
    ap.add_argument("--student", action="store_true")
    ap.add_argument("--e_only", default="")
    ap.add_argument("--e_aug", action="store_true")
    ap.add_argument("--rec", type=int, default=0)
    ap.add_argument("--p_clean", type=float, default=0.25)
    ap.add_argument("--e_rec", action="store_true")
    ap.add_argument("--c_only", default="")
    ap.add_argument("--c_pitch", type=float, default=0.0)
    ap.add_argument("--p_pitch", type=float, default=0.75)
    ap.add_argument("--c_rec", type=float, default=0.0)
    ap.add_argument("--rec_dir", default="data/snv_p_feat_rec")
    ap.add_argument("--aug_lo", type=float, default=0.8)
    ap.add_argument("--aug_hi", type=float, default=1.25)
    ap.add_argument("--e_lift", type=int, default=0)
    ap.add_argument("--e_play", choices=["", "wst"], default="")
    ap.add_argument("--cpc", default="results/snv_p2b_cpc/last.pt")
    ap.add_argument("--pca", default="results/snv_p8b/pca_c_P2b.npy")
    a = ap.parse_args()
    set_rec_dir(a.rec_dir)
    assert a.tpl_est == "grad" or (a.share == "hard" and a.residual == "none"), "E の平均の軌跡(M9)は固い共有・残差なしだけ"
    assert not a.student or (a.share == "hard" and a.residual == "none" and a.tpl_est != "grad"), "推論の E の回帰(M13)は E の平均の軌跡の固い共有だけ"
    assert a.tpl_est not in ("gste", "gswap", "gloo") or (a.group >= 2 and a.align == "band"), "声どうしで平均した勾配(M10・M11)は束の組と帯の対応で"
    assert not (a.e_only or a.e_aug) or (a.student and a.e_only), "推論の E だけの学習(M14)は M13 の保存から"
    assert not (a.e_only and a.c_only), "推論の E だけと C だけは別の腕"
    assert not a.e_rec or (a.e_only and a.rec > 0), "M16 は --e_only と --rec で"
    assert not a.e_lift or (a.e_only and 0 < a.e_lift < 25), "M19 は --e_only で(0 < K < 25)"
    assert not a.e_play or (a.e_only and not a.e_lift), "M28 は --e_only で(--e_lift なし)"
    assert not (a.c_only or a.c_pitch or a.c_rec) or (a.student and a.c_only and (a.c_rec == 0 or a.rec > 0)), "M15 は M13 の保存から(--c_rec は --rec で)"
    out = ROOT / "results" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    if not a.smoke and not (ROOT / a.prereg).exists():
        print(f"{a.prereg} が無い: 起動しない", flush=True)
        return 1
    print("prereg", ROOT / a.prereg, "paper current/paper_snv.md", flush=True)
    torch.manual_seed(a.seed)
    unit_tests()
    t0 = time.time()
    names = sorted(json.loads((ROOT / "data/snv_e1/manifest.json").read_text())["voices"], key=lambda v: v["voice"])
    tr, va = T.split([v["voice"] for v in names])
    if a.smoke:
        tr = tr[:40]
    ctr, cva = Corpus(tr, rec=a.rec), Corpus(va)
    print("train", json.dumps(ctr.stats()), "val", json.dumps(cva.stats()), f"load {time.time() - t0:.0f}s", flush=True)
    (out / "split.json").write_text(json.dumps({"train": tr, "val": va}))
    dev = a.device
    if dev == "cpu":
        torch.set_num_threads(8)
    E, C = P11.EncS().to(dev), T.Circ().to(dev)
    E2 = P11.EncS().to(dev) if a.student else None
    if a.e_play == "wst":
        import snv_encw as EW
        EW.unit_tests(ROOT / a.cpc, ROOT / a.pca)
        E2 = EW.EncW(ROOT / a.cpc, ROOT / a.pca).to(dev)
        print("推論の E = 白くした演奏 → 軌跡(M28)", a.cpc, a.pca, sum(q.numel() for q in E2.parameters() if q.requires_grad), "params", flush=True)
    Einf = E2 if E2 is not None else E
    logb = nn.Parameter(torch.full((T.D_Y,), math.log(P11.B_INIT), device=dev))
    shape = (max(1, len(ctr.cid)), int(ctr.tlen.max()) if len(ctr.tlen) else 2, T.D_A)
    ema = a.tpl_gamma if a.tpl == "ema" else None
    tpl = torch.zeros(shape, device=dev) if ema is not None else nn.Parameter(0.1 * torch.randn(shape, device=dev))
    npos = torch.zeros(shape[:2], dtype=torch.int32, device=dev)
    logsig = nn.Parameter(torch.zeros(T.D_A, device=dev))
    tlen = torch.from_numpy(ctr.tlen if len(ctr.tlen) else np.array([2])).to(dev)
    hard = a.share == "hard"
    nores = hard and a.residual == "none"
    if hard:
        ema = None
        tpl = nn.Parameter(0.1 * torch.randn(shape[:2] + ((D_CT,) if nores else (T.D_A,)), device=dev)) if a.tpl_est == "grad" else torch.zeros(shape[:2] + (D_CT,), device=dev)
    Er = EncR().to(dev) if hard and not nores else None
    params = list(E.parameters()) + (list(E2.parameters()) if E2 is not None else []) + list(C.parameters()) + [logb, logsig] + ([tpl] if ema is None and not hard else []) + (list(Er.parameters()) if Er is not None else [])
    if a.e_only:
        st0 = torch.load(ROOT / a.e_only, map_location="cpu", weights_only=False)
        assert json.loads((ROOT / a.e_only).parent.joinpath("split.json").read_text()) == {"train": tr, "val": va}, "学習と検証の声が保存と違う"
        assert tuple(st0["tpl"].shape) == tuple(tpl.shape) and st0.get("student") and st0.get("tpl_est") == a.tpl_est, "保存の型の形・M13 の設定が違う"
        E.load_state_dict(st0["Gs"]), C.load_state_dict(st0["C"])
        with torch.no_grad():
            tpl.copy_(st0["tpl"].float().to(dev)), npos.copy_(st0["npos"].to(dev)), logb.copy_(st0["logb"].to(dev))
        E.requires_grad_(False), C.requires_grad_(False)
        params = [q for q in E2.parameters() if q.requires_grad]
        print("推論の E だけを学ぶ(G・C・型は凍結)", a.e_only, "step", st0["step"], "声の乱し", a.e_aug, (a.aug_lo, a.aug_hi), "録音の乱し", a.e_rec, a.rec, a.p_clean, a.rec_dir, flush=True)
    lm = torch.from_numpy(lift_mean(tr)).float().to(dev) if a.e_lift else None
    ex = lambda y, v: P11.xin(P11.lift(y, a.e_lift, lm), v)
    if a.e_lift:
        print("推論の E の入力の包絡を c0..c%d に(M19)" % (a.e_lift - 1), "定数 c%d..c24" % a.e_lift, [round(float(x), 4) for x in lm[a.e_lift:25]], flush=True)
    if a.c_only:
        st0 = torch.load(ROOT / a.c_only, map_location="cpu", weights_only=False)
        assert json.loads((ROOT / a.c_only).parent.joinpath("split.json").read_text()) == {"train": tr, "val": va}, "学習と検証の声が保存と違う"
        assert tuple(st0["tpl"].shape) == tuple(tpl.shape) and st0.get("student") and st0.get("tpl_est") == a.tpl_est, "保存の型の形・M13 の設定が違う"
        E.load_state_dict(st0["Gs"]), E2.load_state_dict(st0["Es"]), C.load_state_dict(st0["C"])
        with torch.no_grad():
            tpl.copy_(st0["tpl"].float().to(dev)), npos.copy_(st0["npos"].to(dev)), logb.copy_(st0["logb"].to(dev))
        E.requires_grad_(False), E2.requires_grad_(False)
        params = list(C.parameters())
        gc = torch.Generator(device=dev)
        gc.manual_seed(a.seed + 11)
        print("C だけを学ぶ(G・推論の E・型は凍結)", a.c_only, "step", st0["step"], "高さのずらし", a.c_pitch, a.p_pitch, "録音の高さと有声", a.c_rec, a.rec, a.p_clean, flush=True)
    if a.e_aug:
        import train_snv_p2_cpc as P2
        Ms, tvec = torch.from_numpy(P2.warp_mats(a.aug_lo, a.aug_hi)).to(dev), torch.from_numpy(P2.tilt_vec()).to(dev)
        gaug = torch.Generator(device=dev)
        gaug.manual_seed(a.seed + 7)
    print("型の推定", a.tpl, "γ", ema, "形", shape, "判別器 λ", a.critic, flush=True)
    Q = Critic(len(tr)).to(dev) if a.critic else None
    opt_q = torch.optim.Adam(Q.parameters(), lr=1e-3) if Q is not None else None
    qce = float("nan")
    opt = torch.optim.AdamW([{"params": params}] + ([{"params": [tpl], "lr": a.tpl_lr}] if hard and a.tpl_est == "grad" else []), lr=a.lr, weight_decay=0.0)
    if hard:
        print("固い共有 tpl_lr", a.tpl_lr, "型の推定", a.tpl_est, "対応", a.align, a.band, "暖機", a.align_warm, "残差", a.residual, sum(p_.numel() for p_ in Er.parameters()) if Er is not None else 0, "型", tuple(tpl.shape), flush=True)
    steps = 200 if a.smoke else a.steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 1000) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    trb, vab = Batches(ctr, a.seed, p_clean=a.p_clean), Batches(cva, 12345, p_nv=0.0)
    vbatches = [vab.sample(a.bs) for _ in range(8)]
    to_dev = lambda b: {k: x.to(dev, non_blocking=True) for k, x in b.items()}

    @torch.no_grad()
    def evaluate() -> dict:
        E.eval(), Einf.eval(), C.eval()
        r: dict = {k: [] for k in ("val_l1", "trivial", "kl", "fwd", "amax")}
        for vb in vbatches:
            b = to_dev(vb)
            m = b["ok"].clone()
            m[:, :T.WARM_SELF] = 0
            rx = ex(b["ry"], b["rv"])
            sE = Einf.structure(rx, b["rmask"])
            sC = C.structure(b["ry"], Einf(rx, sE)[0], b["rmask"])
            mu, lv = Einf(ex(b["y"], b["v"]), sE)
            r["val_l1"].append(float(T.masked_l1(C(mu, b["lf"], b["v"], sC), b["y"], m)))
            rmean = (b["ry"] * b["rmask"][:, None]).sum(-1, keepdim=True) / b["rmask"].sum(-1)[:, None, None].clamp(min=1)
            r["trivial"].append(float(T.masked_l1(rmean.expand_as(b["y"]), b["y"], m)))
            r["kl"].append(float((P11.kl_of(mu, lv) * m).sum() / m.sum()))
            perm = torch.roll(torch.arange(mu.shape[0], device=dev), 1)
            r["fwd"].append(float(T.masked_l1(C(mu, b["lf"], b["v"], {k: x[perm] for k, x in sC.items()}), b["y"], m)))
            h = C.inp(torch.cat([mu, b["lf"][:, None], b["v"][:, None]], 1))
            for i, blk in enumerate(C.blocks):
                h = blk(h, sC["z"])
                if i == 3:
                    B_, Ch, L = h.shape
                    q = C.q(h.transpose(1, 2)).view(B_, L, C.heads, C.dk).transpose(1, 2)
                    p = ((q @ sC["k"].transpose(-1, -2)) / math.sqrt(C.dk)).masked_fill(~sC["rmask"][:, None, None, :], -1e9).softmax(-1)
                    r["amax"].append(float(p.max(-1).values.mean()))
                    break
        E.train(), Einf.train(), C.train()
        return {k: float(np.mean(x)) for k, x in r.items()}

    ev0 = evaluate()
    print(f"floors: untrained val L1 {ev0['val_l1']:.4f} / trivial {ev0['trivial']:.4f}", flush=True)
    log = {"floors": ev0, "corpus": ctr.stats(), "evals": []}
    t0 = time.time()
    for step in range(1, steps + 1):
        b = to_dev(trb.sample_grouped(a.bs, a.group) if a.group else trb.sample(a.bs))
        m = b["ok"].clone()
        m[:, :T.WARM_SELF] = 0
        aux: dict = {}
        hv = (float("nan"), float("nan"), float("nan"))
        rho = float("nan")
        fs = float("nan")
        beta = min(1.0, step / BETA_RAMP)
        yh = None
        if a.e_only:
            with torch.no_grad():
                rx = P11.xin(b["ry"], b["rv"])
                mu = E(P11.xin(b["y"], b["v"]), E.structure(rx, b["rmask"]))[0]
                pm, frac = share_path(mu[:, :D_CT], m, b["cid"], tpl, tlen, b["off"], b["nall"], False, band=a.band if a.align == "band" else 0.0)
                md = m * (b["cid"] >= 0).float()[:, None]
                ain = torch.cat([pm, acting_feats(b["y"], b["v"], b["ry"], b["rv"], b["rmask"])], 1)
                gm = (pm * md[:, None]).sum((0, 2)) / md.sum().clamp(min=1.0)
                tv = (((pm - gm[None, :, None]) ** 2).sum(1) * md).sum() / md.sum().clamp(min=1.0)
                yin, vin, ryin, rvin = (b["y_in"], b["v_in"], b["ry_in"], b["rv_in"]) if a.e_rec else (b["y"], b["v"], b["ry"], b["rv"])
                if a.e_aug:
                    fx = (torch.randint(0, len(Ms), (yin.shape[0],), generator=gaug, device=dev), (torch.rand(yin.shape[0], generator=gaug, device=dev) * 2 - 1) * P2.TILT)
                    yin, ryin = P2.augment(yin, Ms, tvec, gaug, fixed=fx), P2.augment(ryin, Ms, tvec, gaug, fixed=fx)
            mu2 = E2(ex(yin, vin), E2.structure(ex(ryin, rvin), b["rmask"]))[0]
            le = (((mu2 - ain) ** 2).sum(1) * md).sum() / md.sum().clamp(min=1.0)
            loss, nll, kl, r2 = le, le.detach(), torch.zeros((), device=dev), float(le.detach())
            with torch.no_grad():
                fs = float((((mu2[:, :D_CT] - pm) ** 2).sum(1) * md).sum() / md.sum().clamp(min=1.0) / tv.clamp(min=1e-12))
                hv = (var_ratio(pm, md), float(((pm ** 2).sum(1) * md).sum() / md.sum().clamp(min=1.0)), float("nan"))
        elif a.c_only:
            with torch.no_grad():
                rx = P11.xin(b["ry"], b["rv"])
                mu = E(P11.xin(b["y"], b["v"]), E.structure(rx, b["rmask"]))[0]
                pm, frac = share_path(mu[:, :D_CT], m, b["cid"], tpl, tlen, b["off"], b["nall"], False, band=a.band if a.align == "band" else 0.0)
                md = m * (b["cid"] >= 0).float()[:, None]
                ain = torch.cat([pm, acting_feats(b["y"], b["v"], b["ry"], b["rv"], b["rmask"])], 1)
                keys = E2(rx, E2.structure(rx, b["rmask"]))[0]
                Bc = b["y"].shape[0]
                lfc, vc = b["lf"], b["v"]
                if a.c_rec > 0:
                    use = (torch.rand(Bc, generator=gc, device=dev) < a.c_rec).float()[:, None]
                    lfc, vc = use * b["lf_in"] + (1 - use) * lfc, use * b["v_in"] + (1 - use) * vc
                if a.c_pitch > 0:
                    sh = (torch.rand(Bc, generator=gc, device=dev) < a.p_pitch).float() * (torch.rand(Bc, generator=gc, device=dev) * 2 - 1) * a.c_pitch
                    lfc = lfc + sh[:, None] * T.LN2_12 * vc
            sC = C.structure(b["ry"], keys, b["rmask"])
            yh = C(ain, lfc, vc, sC)
            nll = (P11.nll_of(yh, b["y"], logb) * md).sum() / md.sum().clamp(min=1.0)
            loss, kl, r2 = nll, torch.zeros((), device=dev), float(nll.detach())
            with torch.no_grad():
                hv = (var_ratio(pm, md), float(((pm ** 2).sum(1) * md).sum() / md.sum().clamp(min=1.0)), float("nan"))
        elif hard:
            rx = P11.xin(b["ry"], b["rv"])
            sE = E.structure(rx, b["rmask"])
            mu, lv = E(P11.xin(b["y"], b["v"]), sE)
            sE2 = E2.structure(rx, b["rmask"]) if E2 is not None else None
            with torch.no_grad():
                keys = E2(rx, sE2)[0] if E2 is not None else E(rx, sE)[0]
            sC = C.structure(b["ry"], keys, b["rmask"])
            if nores:
                sp_ = share_path(mu.detach()[:, :D_CT], m, b["cid"], tpl, tlen, b["off"], b["nall"], step <= a.align_warm,
                                 band=a.band if a.align == "band" else 0.0, ema=a.tpl_gamma if a.tpl_est in ("ste", "gste", "gswap", "gloo") else None, npos=npos,
                                 batch_ema=a.tpl_est in ("gste", "gswap", "gloo"), ret_tau=a.tpl_est in ("gste", "gswap", "gloo"))
                pm, frac = sp_[0], sp_[1]
                md = m * (b["cid"] >= 0).float()[:, None]
                act = acting_feats(b["y"], b["v"], b["ry"], b["rv"], b["rmask"])
                ain = torch.cat([pm, act], 1)
                if a.tpl_est in ("gste", "gswap", "gloo"):
                    ct = group_ste(mu[:, :D_CT], pm, sp_[2], md, b["gid"], tpl.shape[1], swap=a.tpl_est == "gswap", loo=a.tpl_est == "gloo")
                    rho = own_share(sp_[2], md, b["gid"], tpl.shape[1])
                elif a.tpl_est == "ste":
                    ct = mu[:, :D_CT] + (pm - mu[:, :D_CT]).detach()
                else:
                    ct = pm
                yh = C(torch.cat([ct, act], 1), b["lf"], b["v"], sC)
                with torch.no_grad():
                    gm = (pm * md[:, None]).sum((0, 2)) / md.sum().clamp(min=1.0)
                    tv = (((pm - gm[None, :, None]) ** 2).sum(1) * md).sum() / md.sum().clamp(min=1.0)
                    fit = float((((mu[:, :D_CT] - pm) ** 2).sum(1) * md).sum() / md.sum().clamp(min=1.0) / tv.clamp(min=1e-12))
                    hv = (var_ratio(pm, md), float(((pm ** 2).sum(1) * md).sum() / md.sum().clamp(min=1.0)), fit)
                nll = (P11.nll_of(yh, b["y"], logb) * md).sum() / md.sum().clamp(min=1.0)
                kl = torch.zeros((), device=dev)
                le = (((mu - ain.detach()) ** 2).sum(1) * md).sum() / md.sum().clamp(min=1.0)
                r2 = float(le)
                loss = nll + le + 1e-3 * ((pm ** 2).sum(1) * md).sum() / md.sum().clamp(min=1.0)
                if E2 is not None:
                    mu2 = E2(P11.xin(b["y"], b["v"]), sE2)[0]
                    loss = loss + (((mu2 - ain.detach()) ** 2).sum(1) * md).sum() / md.sum().clamp(min=1.0)
                    with torch.no_grad():
                        fs = float((((mu2[:, :D_CT] - pm) ** 2).sum(1) * md).sum() / md.sum().clamp(min=1.0) / tv.clamp(min=1e-12))
            else:
                pm, frac = share_path(mu.detach(), m, b["cid"], tpl, tlen, b["off"], b["nall"], step <= a.align_warm)
                rmu, rlv = Er(torch.cat([P11.xin(b["y"], b["v"]), pm.detach()], 1), sE)
                yh = C(pm + rmu + (0.5 * rlv).exp() * torch.randn_like(rmu), b["lf"], b["v"], sC)
                nll = (P11.nll_of(yh, b["y"], logb) * m).sum() / m.sum()
                kl = (P11.kl_of(rmu, rlv) * m).sum() / m.sum()
                le = (((mu - (pm + rmu).detach()) ** 2).sum(1) * m).sum() / m.sum()
                r2 = float(le)
                loss = nll + beta * kl + le + 1e-3 * ((pm ** 2).sum(1) * m).sum() / m.sum()
                aux["i"], aux["res"] = torch.arange(mu.shape[0], device=dev), rmu
        else:
            sE, sC = P11.structures(E, C, b["ry"], b["rv"], b["rmask"])
            mu, lv = E(P11.xin(b["y"], b["v"]), sE)
            uu = mu + (0.5 * lv).exp() * torch.randn_like(mu)
            yh = C(uu, b["lf"], b["v"], sC)
            nll = (P11.nll_of(yh, b["y"], logb) * m).sum() / m.sum()
            klf, frac, r2 = kl_terms(mu, lv, m, b["cid"], tpl, tlen, logsig, ema, npos, b["off"], b["nall"], aux)
            kl = (klf * m).sum() / m.sum()
            loss = nll + beta * kl
        if Q is not None and "res" in aux:
            mq = m[aux["i"]]
            lab = b["spk"][aux["i"]][:, None].expand(-1, mq.shape[-1])
            ce = lambda lg: (F_.cross_entropy(lg, lab, reduction="none") * mq).sum() / mq.sum().clamp(min=1.0)
            loss = loss - a.critic * ce(Q(aux["res"]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if Q is not None and "res" in aux:
            opt_q.zero_grad(set_to_none=True)
            ql = ce(Q(aux["res"].detach()))
            ql.backward()
            opt_q.step()
            qce = float(ql)
        if step % 200 == 0:
            print(f"step {step} nll {float(nll):.2f} kl {float(kl):.2f} shared {frac:.2f} {'le' if hard else 'tpl_r2'} {r2:.3f} q_ce {qce:.3f} seen {float((npos > 0).any(1).float().mean()):.3f} beta {beta:.2f} l1 {float(T.masked_l1(yh, b['y'], m)) if yh is not None else float('nan'):.4f} sig {float(logsig.exp().mean()):.3f} gn {float(gn):.1f} vr_ct {hv[0]:.3f} ct_sq {hv[1]:.3f} fit {hv[2]:.3f} rho {rho:.3f} fit_s {fs:.3f} {(time.time() - t0) / step:.3f}s/step", flush=True)
        if step % 2000 == 0 or step == steps:
            ev = evaluate()
            rec = {"step": step, **ev, "train_nll": float(nll), "train_kl": float(kl), "tpl_r2": r2, "q_ce": qce, "q_chance": math.log(len(tr)), "contents_seen": float((npos > 0).any(1).float().mean()), "beta": beta, "sigma": float(logsig.exp().mean()), "vr_ct": hv[0], "ct_sq": hv[1], "fit": hv[2], "rho": rho, "fit_s": fs, "sec": time.time() - t0}
            log["evals"].append(rec)
            print("eval", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
            ck = {"Es": Einf.state_dict(), "Gs": E.state_dict() if E2 is not None else None, "student": a.student, "C": C.state_dict(), "logb": logb.detach().cpu(), "step": step, "tpl": tpl.detach().half().cpu(), "npos": npos.cpu(), "logsig": logsig.detach().cpu(), "tpl_mode": a.tpl, "critic": a.critic, "Q": Q.state_dict() if Q is not None else None,
                  "share": a.share, "residual": a.residual, "Er": Er.state_dict() if Er is not None else None, "tpl_est": a.tpl_est, "align": a.align, "band": a.band, "group": a.group, "e_only": a.e_only, "e_aug": a.e_aug,
                  "rec": a.rec, "p_clean": a.p_clean, "e_rec": a.e_rec, "c_only": a.c_only, "c_pitch": a.c_pitch, "p_pitch": a.p_pitch, "c_rec": a.c_rec, "rec_dir": a.rec_dir, "aug_lo": a.aug_lo, "aug_hi": a.aug_hi, "e_lift": a.e_lift, "lift_mean": lm.cpu().tolist() if lm is not None else None,
                  "e_play": a.e_play, "cpc": a.cpc if a.e_play else None, "pca": a.pca if a.e_play else None}
            torch.save(ck, out / "last.pt")
            if step == steps // 2:
                torch.save(ck, out / f"step{step // 1000}k.pt")
            (out / "train_log.json").write_text(json.dumps(log, indent=1))
            if ev["amax"] > 0.5:
                print(f"STOP: 注意が検索に寄る(最大 {ev['amax']:.3f})", flush=True)
                return 3
            if step == 10000 and ev["val_l1"] > 0.8 * ev["trivial"]:
                print(f"TRIPWIRE: val L1 {ev['val_l1']:.4f} > 0.8 × trivial at 10k → stop", flush=True)
                return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
