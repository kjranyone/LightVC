"""段 P73 step a 補足 3(results/snv_p73/prereg.yaml・学習なし): EMA を使わない物差し = 最近傍で包絡を差し替えた音の CER。
源の有声のフレームごとに、源の話者自身の登録(先頭 tn 発話)の有声のフレームから表現で最も近いものを選び、包絡の形 c1〜c24 と周期性をそのフレームの値に差し替える。
強さ c0・高さ・有声 / 無声・無声のフレームは源のまま。出力部で描いて書き起こす。
表現 = copy(差し替えなし)・Yn(入力)・c(基準の符号の文脈)・E(EMA の最近傍 = 物差しそのもの)・rand(登録の有声のフレームから無作為 = 床)。
blur<k> = 包絡の形を時間に k フレームの移動平均でならす・nz<a> = 時間に 40 ms でならした正規雑音を、登録の有声のフレームの次元ごとの分散の a / 10 倍の大きさで足す(内容に依らない乱しの対照・step d 補足)。
pf=<フレーム数> = 登録の有声のフレームを先頭からその数に切る(参照の量を組の間で揃える)。行の pool = 参照の有声のフレームの数。
win=<フレーム数>(step d)= 参照を 2 通りにして両方を測る。in = 源の発話そのものの有声のフレーム(源のフレームから前後 win フレーム以内を除く)、out = 登録の有声のフレームを先頭から(源の有声のフレームの数 − 2 win − 1)に切ったもの。行の mode に残す。
CER の参照 = 源の音そのものの書き起こし(1 で切る)。書き起こしの文字は保存も表示もしない(数値だけ)。
    cd training && SNV_DEV=cuda OMP_NUM_THREADS=2 uv run --no-sync python snv_diag/p73_swap.py [sets=cv,valm,trm] [reps=copy,Yn,c,E,rand] [tn=3] [pf=0] [nmax=64] [out=results/snv_p73/swap.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import nvoc as N
import p18_align as AL
import p18_nnc as NN
import snv_p0_front as P
import snv_p1_eval as EV
import train_rvoc as TR
import train_snv_p3 as P3

ROOT = EV.ROOT
DEV = EV.DEV
SYS, CPC = "snv_p19_x", "snv_p30_dcpc"


def ema_lag(paths: list[str], us: list[dict]) -> list[torch.Tensor]:
    M = [torch.from_numpy(u["v"] > 0.5).to(DEV) for u in us]
    E = [AL.ema_at(p, u["y"].shape[1]) for p, u in zip(paths, us)]
    A = torch.cat([e[m] for e, m in zip(E, M)])
    return [AL.stack((e - A.mean(0)) / A.std(0).clamp(min=1e-6))[m] for e, m in zip(E, M)]


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[1:] if "=" in x)
    names, reps, tn, nmax = opt.get("sets", "cv,valm,trm").split(","), opt.get("reps", "copy,Yn,c,E,rand").split(","), int(opt.get("tn", 3)), int(opt.get("nmax", 64))
    pf, win = int(opt.get("pf", 0)), int(opt.get("win", 0))
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, kana, to16
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16" if DEV == "cuda" else "int8")
    tr = lambda w: kana("".join(s.text for s in asr.transcribe(to16(w.astype(np.float64), EV.SR).astype(np.float32), language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)[0]))
    gen = EV.load_gen()
    Pl = P3.Play(ROOT / "results" / CPC / "last.pt").to(DEV).eval()
    split = json.loads((ROOT / "results" / SYS / "split.json").read_text())
    sets = AL.build_sets(names, 64, split)

    @torch.no_grad()
    def render(u: dict, y: np.ndarray) -> np.ndarray:
        yt, lf, v = torch.from_numpy(y)[None].to(DEV), torch.from_numpy(u["lf"])[None].to(DEV), torch.from_numpy(u["v"])[None].to(DEV)
        cond = torch.cat([yt[:, :25], lf[:, None], v[:, None], yt[:, 25:29].clamp(0, 1.5) * v[:, None]], 1)
        f0 = torch.from_numpy(u["f0"])[None].to(DEV)
        return gen(cond, TR.excitation(f0, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:].astype(np.float32)

    def ref_of(it: dict) -> str:
        if "ref" in it:
            return it["ref"]
        path = it.get("path") or str(ROOT / "data/snv_e1" / it["src"].split("|")[0] / f"{it['src'].split('|')[1]}.wav")
        return tr(librosa.load(path, sr=P.SR, mono=True)[0].astype(np.float32))

    rows = []
    for nm, its in sets.items():
        for it in [x for x in its if x["valid"]][:nmax]:
            us = [it["u"]] + it["enr"][:tn]
            enr = list(range(1, len(us)))
            M = [u["v"] > 0.5 for u in us]
            Yp = np.concatenate([u["y"][:, m] for u, m in zip(us[1:], M[1:])], 1)
            if pf and Yp.shape[1] < pf:
                continue
            Yp = Yp[:, :pf] if pf else Yp
            if win:
                n0 = int(M[0].sum()) - 2 * win - 1
                if n0 < 4 * win or Yp.shape[1] < n0:
                    continue
                ref, t0, Y0 = ref_of(it), torch.from_numpy(np.flatnonzero(M[0])).to(DEV), it["u"]["y"][:, M[0]]
                for rep in reps:
                    xs = None if rep == "copy" else [c[torch.from_numpy(m).to(DEV)] for c, m in zip(NN.code(Pl, us, enr, rep), M)]
                    for mode in ("in", "out"):
                        y = it["u"]["y"].copy()
                        if rep != "copy" and mode == "in":
                            d = torch.cdist(xs[0], xs[0]).masked_fill((t0[:, None] - t0[None]).abs() <= win, float("inf"))
                            y[1:29][:, M[0]] = Y0[1:29][:, d.argmin(1).cpu().numpy()]
                        elif rep != "copy":
                            y[1:29][:, M[0]] = Yp[1:29][:, AL.nn(xs[0], torch.cat(xs[1:])[:n0]).cpu().numpy()]
                        if rep != "copy" or mode == "in":
                            rows.append({"set": nm, "spk": it["spk"], "src": it["src"], "rep": rep, "mode": mode, "pool": n0, "cer": round(min(cer(ref, tr(render(it["u"], y))), 1.0), 4)})
                continue
            ref = ref_of(it)
            for rep in reps:
                y = it["u"]["y"].copy()
                if rep.startswith("blur"):
                    k = int(rep[4:])
                    y[1:29][:, M[0]] = np.stack([np.convolve(np.pad(r, k // 2, mode="edge"), np.ones(k) / k, "valid") for r in y[1:29]])[:, M[0]]
                elif rep.startswith("nz"):
                    z = np.random.default_rng(len(rows)).standard_normal((28, y.shape[1] + 7))
                    z = np.stack([np.convolve(r, np.ones(8) / 8, "valid") for r in z])
                    y[1:29][:, M[0]] += (np.sqrt(int(rep[2:]) / 10) * Yp[1:29].std(1, keepdims=True) * z / z.std(1, keepdims=True))[:, M[0]].astype(y.dtype)
                elif rep != "copy":
                    if rep == "rand":
                        j = np.random.default_rng(len(rows)).integers(0, Yp.shape[1], int(M[0].sum()))
                    else:
                        xs = ema_lag(it["ema"][:len(us)], us) if rep == "E" else [c[torch.from_numpy(m).to(DEV)] for c, m in zip(NN.code(Pl, us, enr, rep), M)]
                        j = AL.nn(xs[0], torch.cat(xs[1:])[:Yp.shape[1]]).cpu().numpy()
                    y[1:29][:, M[0]] = Yp[1:29][:, j]
                rows.append({"set": nm, "spk": it["spk"], "src": it["src"], "rep": rep, "pool": int(Yp.shape[1]), "cer": round(min(cer(ref, tr(render(it["u"], y))), 1.0), 4)})
        ms = ("in", "out") if win else (None,)
        print(nm, {f"{rep}{'/' + m if m else ''}": round(float(np.mean([r["cer"] for r in rows if r["set"] == nm and r["rep"] == rep and r.get("mode") == m])), 4) for rep in reps for m in ms if any(r["set"] == nm and r["rep"] == rep and r.get("mode") == m for r in rows)}, len({r["src"] for r in rows if r["set"] == nm}), flush=True)
    out = ROOT / opt.get("out", "results/snv_p73/swap.json")
    out.write_text(json.dumps({"prereg": "results/snv_p73/prereg.yaml(step_a_supplement_3)", "sys": SYS, "cpc": CPC, "tn": tn, "pf": pf, "win": win, "rows": rows}, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
