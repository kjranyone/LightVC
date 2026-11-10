"""P2 の起動前の関門(synth_nam_vc §8・学習なし): 演奏の表現の候補が話者を跨いだ調音の状態の対応を持つか = 最近傍の対応(NNC)。
源の有声のフレームごとに、目標の参照(t00〜t02 の有声のフレーム)から表現で最も近いフレームを選び、両者の調音の状態(SPARC の EMA 12 次元・物差しとしてだけ)の相関を次元ごとに取って平均する。
組 = RM(本物の男声 12 区間 → 検証の女声 10 声)・IM(Irodori の検証の男声 6 × t09 → 同じ 10 声)・FF(検証の女声 10 × t09 → 自分以外の 9 声 = 性別の中の上限)。
候補 = Y(包絡 c1..c24 + 周期性・コーパスの σ_d で標準化)・Ym(話者の登録の平均を引く)・Ys(話者ごとに平均と標準偏差で正規化)・E_<tag>(P 線の自前の E の出力)・
参考値 = WavLM-large の 6 層(既存のモデル・製品に入れない)・EMA 自身(物差しの上限)・shuffle(床)。
<表現>-c = 話者の登録の平均(有声のフレーム)を引いたもの(源 = 操作者の登録・目標 = 参照)= 登録の定数なので推論で使える。
    cd training && uv run --no-sync python snv_p2_nnc.py [E=snv_p1_n,snv_p1c_wc@step10k] [P=snv_p2_cpc]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

import snv_p1_eval as EV
import train_snv_p1 as T

ROOT = Path(__file__).resolve().parent.parent
SP = ROOT / "data/snv_e1_feat/sparc"
DEV = "cuda"
SUB = 4


def ema_at(npz: Path, n: int) -> np.ndarray:
    e = np.load(npz)["ema"][:, :12]
    t = np.arange(n) * T_HOP
    return np.stack([np.interp(t, np.arange(len(e)) / 50.0, e[:, k]) for k in range(12)], 1).astype(np.float32)


T_HOP = 240 / 48000


def corr(a: np.ndarray, b: np.ndarray) -> float:
    c = [np.corrcoef(a[:, j], b[:, j])[0, 1] for j in range(a.shape[1])]
    return float(np.nanmean(c))


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    etags = [t for t in opt.get("E", "snv_p1_n").split(",") if t]
    ptags = [t for t in opt.get("P", "").split(",") if t]
    import train_snv_p2_cpc as CP
    cpcs = {}
    for t in ptags:
        st = torch.load(ROOT / "results" / t / "last.pt", map_location="cpu", weights_only=False)
        cm = CP.CPC().to(DEV).eval()
        cm.load_state_dict(st["cpc"])
        cpcs[t] = (cm, st["sig"])
    split = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())
    tr_us = [T.load_utt(p) for v in split["train"][::12] for p in sorted((T.FEAT / "e1" / v).glob("t0[0-4].npz"))]
    sig = np.mean([u["y"].std(1) for u in tr_us], 0).astype(np.float32)
    fem = [v for v in split["val"] if v.startswith("ef")][::3]
    mal = [v for v in split["val"] if v.startswith("em")]
    meta = json.loads((EV.A1 / "convert_meta.json").read_text())
    from transformers import WavLMModel
    wl = WavLMModel.from_pretrained("microsoft/wavlm-large").to(DEV).eval()
    systems = {t: EV.System(t, None) for t in etags}

    def wavlm6(path: Path, n: int) -> np.ndarray:
        x, _ = librosa.load(str(path), sr=16000)
        with torch.no_grad():
            h = wl(torch.from_numpy(x)[None].to(DEV), output_hidden_states=True).hidden_states[6][0].cpu().numpy()
        t = np.arange(n) * T_HOP
        idx = np.clip(np.round(t * 50.0).astype(int), 0, len(h) - 1)
        return h[idx]

    def unit(name: str, rel: str, wav: Path, ema: Path, mu: np.ndarray, sd: np.ndarray) -> dict:
        u = EV.utt(rel)
        n = u["y"].shape[1]
        vm = u["v"] > 0
        y = u["y"][1:29].T
        env = T.y_env(torch.from_numpy(u["y"])[None])[0].numpy()
        out = {"v": vm, "ema": ema_at(ema, n) if ema is not None else np.zeros((n, 12), np.float32), "Y": y / sig[1:29], "Ym": np.concatenate([((env - mu[:, None]) / 10)[1:25].T, u["y"][25:29].T], 1) / sig[1:29],
               "Ys": np.concatenate([((env - mu[:, None]) / sd[:, None])[1:25].T, u["y"][25:29].T / sig[25:29]], 1), "WavLM6": wavlm6(wav, n)}
        for t, s in systems.items():
            with torch.no_grad():
                yt = torch.from_numpy(u["y"])[None].to(DEV)
                a = s.E(T.e_in(yt, torch.from_numpy(u["v"])[None].to(DEV), torch.from_numpy(mu)[None].to(DEV)))[0].cpu().numpy().T
            out[f"E_{t}"] = a
        for t, (cm, csig) in cpcs.items():
            with torch.no_grad():
                z, c = cm.encode(torch.from_numpy(CP.x_of(u, csig))[None].to(DEV))
            out[f"CPCz_{t}"], out[f"CPCc_{t}"] = z[0].cpu().numpy().T, c[0].cpu().numpy().T
        out["EMA"] = out["ema"]
        return out

    def stats(rels: list[str]) -> tuple[np.ndarray, np.ndarray]:
        us = [EV.utt(r) for r in rels]
        env = np.concatenate([T.y_env(torch.from_numpy(u["y"])[None])[0].numpy()[:, u["v"] > 0] for u in us], 1)
        return env.mean(1).astype(np.float32), env.std(1).astype(np.float32) + 1e-3

    pools = {}
    for v in fem:
        rels = [f"e1/{v}/t0{j}" for j in (0, 1, 2)]
        mu, sd = stats(rels)
        parts = [unit(v, r, ROOT / f"data/snv_e1/{v}/t0{j}.wav", SP / f"e1/{v}/t0{j}.npz", mu, sd) for j, r in zip((0, 1, 2), rels)]
        pools[v] = {k: np.concatenate([p[k][p["v"]] for p in parts]) for k in parts[0] if k != "v"}
    base = ["Y", "Ym", "Ys"] + [f"E_{t}" for t in etags] + [f"CPC{w}_{t}" for t in ptags for w in "zc"] + ["WavLM6"]
    pmean = {v: {r: pools[v][r].mean(0) for r in base} for v in fem}
    queries = []
    for s_ in meta["sources"]:
        sp = s_.rsplit("_", 1)[0]
        mu, sd = stats([f"real/{sp}_enroll"])
        en = unit(sp, f"real/{sp}_enroll", T.FEAT / f"real/{sp}_enroll.wav", None, mu, sd)
        queries.append(("RM", s_, unit(s_, f"real/{s_}", T.FEAT / f"real/{s_}.wav", SP / f"a1src/{s_}.npz", mu, sd), {r: en[r][en["v"]].mean(0) for r in base}))
    for v in mal:
        mu, sd = stats([f"e1/{v}/t0{j}" for j in (0, 1, 2)])
        ens = [unit(v, f"e1/{v}/t0{j}", ROOT / f"data/snv_e1/{v}/t0{j}.wav", None, mu, sd) for j in (0, 1, 2)]
        qm = {r: np.concatenate([e[r][e["v"]] for e in ens]).mean(0) for r in base}
        queries.append(("IM", v, unit(v, f"e1/{v}/t09", ROOT / f"data/snv_e1/{v}/t09.wav", SP / f"e1/{v}/t09.npz", mu, sd), qm))
    for v in fem:
        mu, sd = stats([f"e1/{v}/t0{j}" for j in (0, 1, 2)])
        queries.append(("FF", v, unit(v, f"e1/{v}/t09", ROOT / f"data/snv_e1/{v}/t09.wav", SP / f"e1/{v}/t09.npz", mu, sd), pmean[v]))
    for _, _, q, qm in queries:
        for r in base:
            q[r + "-c"] = q[r] - qm[r]
    for v in fem:
        for r in base:
            pools[v][r + "-c"] = pools[v][r] - pmean[v][r]
    reps = base + [r + "-c" for r in base] + ["EMA"]
    rng = np.random.default_rng(0)
    res: dict = {"prereg": "results/snv_p2/prereg.yaml(関門 NNC)", "ruler": "SPARC multi の EMA 12 次元(物差しとしてだけ)", "by": {}}
    for grp in ("RM", "IM", "FF"):
        row = {r: [] for r in reps + ["shuffle"]}
        for g, name, q, _ in queries:
            if g != grp:
                continue
            qm = q["v"].copy()
            qm[np.arange(len(qm)) % SUB != 0] = False
            for v in fem:
                if grp == "FF" and v == name:
                    continue
                P = pools[v]
                qe = q["ema"][qm]
                for r in reps:
                    a, b = q[r][qm], P[r]
                    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
                    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
                    nn = (torch.from_numpy(an).to(DEV).float() @ torch.from_numpy(bn).to(DEV).float().T).argmax(1).cpu().numpy()
                    row[r].append(corr(qe, P["ema"][nn]))
                row["shuffle"].append(corr(qe, P["ema"][rng.integers(0, len(P["ema"]), len(qe))]))
        res["by"][grp] = {r: round(float(np.mean(x)), 4) for r, x in row.items()}
        res["by"][grp]["n_pairs"] = len(row["Y"])
        print(grp, json.dumps(res["by"][grp]), flush=True)
    out = ROOT / ("results/snv_p2/nnc.json" if not ptags else f"results/snv_p2/nnc_{'_'.join(ptags)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
