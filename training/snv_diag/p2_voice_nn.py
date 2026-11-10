"""演奏の表現に残る声の量(synth_nam_vc §8.4・学習なし): 表現の最近傍が同じ話者のフレームに戻る率。
池 = 検証の女声 10 声と男声 6 声の t00〜t02 の有声のフレーム(話者ごとに同じ数に間引く)・問い = 各声の t09 の有声のフレーム(4 フレームおき)。
同じ声に戻る率(偶然 = 1/16)・同じ性別に戻る率(偶然 = 女声 10/16・男声 6/16)。表現 = Y・CPC の c(生・登録の平均を引いた c-c)・WavLM 6 層・SPARC の EMA 12 次元(参考・A 線で変換が成功した演奏)。
声の無い演奏なら同じ声の率は偶然に近い(状態だけで最近傍が決まる)。
    cd training && uv run --no-sync python snv_diag/p2_voice_nn.py P=snv_p2b_cpc[,snv_p2d_cpc]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import snv_p1_eval as EV
import snv_p2_nnc as NN
import train_snv_p2_cpc as CP

ROOT = EV.ROOT
DEV = EV.DEV
SUB = 4
PER = 1500


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    tags = [t for t in opt.get("P", "snv_p2b_cpc").split(",") if t]
    split = json.loads((ROOT / "results/snv_p1_n/split.json").read_text())
    fem = [v for v in split["val"] if v.startswith("ef")][::3]
    mal = [v for v in split["val"] if v.startswith("em")]
    voices = fem + mal
    rng = np.random.default_rng(0)
    from transformers import WavLMModel
    wl = WavLMModel.from_pretrained("microsoft/wavlm-large").to(DEV).eval()

    def wavlm6(path: Path, n: int) -> np.ndarray:
        x, _ = librosa.load(str(path), sr=16000)
        with torch.no_grad():
            h = wl(torch.from_numpy(x)[None].to(DEV), output_hidden_states=True).hidden_states[6][0].cpu().numpy()
        idx = np.clip(np.round(np.arange(n) * 240 / 48000 * 50.0).astype(int), 0, len(h) - 1)
        return h[idx]

    cpcs = {}
    for t in tags:
        name, _, ck = t.partition("@")
        st = torch.load(ROOT / "results" / name / f"{ck or 'last'}.pt", map_location="cpu", weights_only=False)
        cm = CP.CPC().to(DEV).eval()
        cm.load_state_dict(st["cpc"])
        cpcs[t] = (cm, st["sig"])

    def reps(v: str, j: str) -> dict:
        u = EV.utt(f"e1/{v}/{j}")
        vm = u["v"] > 0
        out = {"Y": u["y"][1:29].T[vm], "WavLM6": wavlm6(ROOT / f"data/snv_e1/{v}/{j}.wav", len(vm))[vm], "EMA": NN.ema_at(ROOT / f"data/snv_e1_feat/sparc/e1/{v}/{j}.npz", len(vm))[vm]}
        for t, (cm, sig) in cpcs.items():
            with torch.no_grad():
                out[f"c_{t}"] = cm.encode(torch.from_numpy(CP.x_of(u, sig))[None].to(DEV))[1][0].cpu().numpy().T[vm]
        return out

    pool, query = {}, {}
    for v in voices:
        parts = [reps(v, f"t0{j}") for j in (0, 1, 2)]
        pool[v] = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        query[v] = reps(v, "t09")
    names = list(pool[voices[0]])
    for v in voices:
        for k in names:
            mu = pool[v][k].mean(0)
            pool[v][k + "-c"] = pool[v][k] - mu
            query[v][k + "-c"] = query[v][k] - mu
    res: dict = {"chance_same_voice": round(1 / len(voices), 4), "chance_same_sex": {"F": round(len(fem) / len(voices), 4), "M": round(len(mal) / len(voices), 4)}, "by": {}}
    for k in names + [n + "-c" for n in names]:
        P, lab = [], []
        for i, v in enumerate(voices):
            x = pool[v][k]
            sel = rng.choice(len(x), min(PER, len(x)), replace=False)
            P.append(x[sel]), lab.append(np.full(len(sel), i))
        P, lab = np.concatenate(P), np.concatenate(lab)
        Pn = torch.from_numpy(P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-8)).float().to(DEV)
        same, sex = [], []
        for i, v in enumerate(voices):
            q = query[v][k][::SUB]
            qn = torch.from_numpy(q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)).float().to(DEV)
            nn = lab[(qn @ Pn.T).argmax(1).cpu().numpy()]
            same.append(float(np.mean(nn == i)))
            sex.append(float(np.mean([(voices[j] in fem) == (v in fem) for j in nn])))
        res["by"][k] = {"same_voice": round(float(np.mean(same)), 4), "same_sex": round(float(np.mean(sex)), 4)}
        print(k, json.dumps(res["by"][k]), flush=True)
    out = ROOT / f"results/snv_p2/voice_nn_{'_'.join(tags).replace('@', '-')}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("wrote", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
