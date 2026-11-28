from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import librosa
import numpy as np
import pyworld
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import p98_probe as PB
import p107_edges as E
import p119_copy as C9
import pae as PA
import snv_p0_front as P
import snv_p1_eval as EV

ROOT = Path(__file__).resolve().parent.parent.parent
VOICES = ("ef100", "ef032", "ef300")
EAR_VOICES = ("ef100", "ef032")


def center_smooth(e: np.ndarray) -> np.ndarray:
    q = np.pad(e, ((0, 0), (1, 1)), mode="edge")
    return (0.25 * q[:, :-2] + 0.5 * q[:, 1:-1] + 0.25 * q[:, 2:]).astype(np.float32)


def causal_smooth(e: np.ndarray) -> np.ndarray:
    q = np.pad(e, ((0, 0), (2, 0)), mode="edge")
    return (0.25 * q[:, :-2] + 0.5 * q[:, 1:-1] + 0.25 * q[:, 2:]).astype(np.float32)


def causal_lp_c0(c0: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    out = np.empty_like(c0, dtype=np.float32)
    out[0] = float(c0[0])
    for t in range(1, len(c0)):
        out[t] = out[t - 1] + alpha * (float(c0[t]) - out[t - 1])
    return out


def pack_cond(env25: np.ndarray, f0: np.ndarray, per4: np.ndarray) -> torch.Tensor:
    v = (f0 > 0).astype(np.float32)[None]
    lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)[None]
    c = np.concatenate([env25.astype(np.float32), lf, v, np.clip(per4.astype(np.float32), 0.0, 1.5) * v], 0)
    return torch.from_numpy(c)[None].to(EV.DEV)


def align_lm(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    best_v, best_s = 1e9, 0
    for s in range(-120, 361, 60):
        a = x[: len(x) - max(0, -s)] if s < 0 else x[: min(len(x), len(y) - s)]
        b = y[max(0, s) : max(0, s) + len(a)] if s >= 0 else y[: len(a)]
        if s < 0:
            a = x[-s : -s + min(len(x) + s, len(y))]
            b = y[: len(a)]
        v = C9.lm(a.astype(np.float32), b.astype(np.float32))
        if v < best_v:
            best_v, best_s = v, s
    return round(best_v, 3), best_s


def load_renderer(ck: str) -> torch.nn.Module:
    os.environ.pop("SNV_RCK", None)
    if ck:
        os.environ["SNV_RCK"] = ck
    return EV.load_gen()


def build_html() -> str:
    sec1_rows = [
        ("orig", "元の音", ""),
        ("B", "B: 雑音が消えた写し(基準)", "今の描画器 + 正解の高さ + 中心の包絡 + 中心の平滑(先読みあり)"),
        ("T1", "T1: 包絡だけ左寄せ", "B から包絡・周期性の計算だけ製品の左寄せに替えた音"),
        ("T2", "T2: 学び足し S + 因果の平滑", "T1 から描画器を S(30k 歩)・平滑を因果(遅れなし)に替えた音"),
        ("T3", "T3: 学び足し Y(平滑なし)", "T2 から平滑を外し、前回使った描画器 Y に替えた音"),
        ("T", "T: 高さも製品の前段(前回の T)", "T3 から高さと有声の判定を製品の前段 v3.1 に戻した音"),
    ]
    sec2_rows = [
        ("B", "B: 目標の写し(雑音なしの基準)", ""),
        ("K_Y", "K_Y: 前回の変換 K(参考)", "製品の前段の高さ + 描画器 Y(平滑なし)"),
        ("K_B", "K_B: B の土台で変換 K を描く", "K3G の出力すべて + 正解の高さ + 中心の 3 点平滑 + 今の描画器"),
        ("K_Bc0s", "K_Bc0s: 音量の揺れをさらに落とす", "K_B の上で、音量 c0 に因果の低域通過を重ねた音"),
        ("K_BTc0", "K_BTc0: 音量だけ正解に戻す", "K_B の上で、音量 c0 だけ B の正解に替えた音(声の形と息の量は K3G のまま)"),
    ]
    parts = [
        '<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow"><title>雑音のはしご P121</title>',
        "<style>body{font-family:system-ui,sans-serif;max-width:840px;margin:0 auto;padding:16px;line-height:1.6;background:#fafafa;color:#222}h1{font-size:1.3rem}h2{font-size:1.05rem;margin:0 0 6px}h3{font-size:.98rem;margin:10px 0 4px}section{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px 14px;margin:12px 0}.r{display:flex;flex-wrap:wrap;align-items:center;gap:4px 10px;margin:6px 0}.r span{width:17em;font-weight:600}audio{flex:1;min-width:220px}i{flex-basis:100%;font-style:normal;font-size:.8rem;color:#666}@media(prefers-color-scheme:dark){body{background:#161616;color:#eee}section{background:#222;border-color:#444}i{color:#aaa}}</style>",
        "<h1>写し T の雑音と、変換 K の音量 c0 の雑音の切り分け</h1>",
        "<p>段 P121。前回の「1-&gt; たぶんL。Tにもある」を受け、次の学習を 1 つに絞るための 2 つの確認です。<br>",
        "<b>知りたいこと 1(写し B → T)</b>: B(雑音なし)から T1 → T2 → T3 → T と 1 つずつ製品側に戻したとき、どこから雑音が入りますか。<br>",
        "<b>知りたいこと 2(変換 K_B 〜 K_BTc0)</b>: 写しの雑音が無い B の描き方の上で変換を鳴らしたとき、K_B・K_Bc0s・K_BTc0 のどれで雑音が消えますか(K_BTc0 でも残りますか)。</p>",
    ]
    for v in EAR_VOICES:
        parts.append(f"<section><h2>{v} — 1. 写しのはしご(B → T1 → T2 → T3 → T)</h2>")
        for key, label, note in sec1_rows:
            parts.append(f'<div class=r><span>{label}</span><audio controls preload=none src="{v}_{key}.wav"></audio><i>{note}</i></div>')
        parts.append(f"<h3>{v} — 2. 雑音なし B の描き方の上での変換(男声の版 → 元の声)</h3>")
        for key, label, note in sec2_rows:
            parts.append(f'<div class=r><span>{label}</span><audio controls preload=none src="{v}_{key}.wav"></audio><i>{note}</i></div>')
        parts.append("</section>")
    parts.append("</html>")
    return "\n".join(parts)


def main() -> None:
    out = ROOT / "results/snv_p121/ear"
    out.mkdir(parents=True, exist_ok=True)
    gen_A = load_renderer("")
    gen_S = load_renderer("results/snv_p97_s240/snap/ema_30k.pt")
    gen_Y = load_renderer("results/snv_p97_y240/snap/ema_30k.pt")
    S_ens = EV.Ens(E.SYS, gen_Y)
    lw = lambda p: librosa.load(p, sr=EV.SR, mono=True)[0].astype(np.float32)

    summary: dict = {"lm_raw": {}, "lm_aligned": {}, "best_shift": {}}
    for v in VOICES:
        refs = [PB.utt(lw(ROOT / f"data/snv_e1/{v}/t0{j}.wav")) for j in (0, 1, 2)]
        st = S_ens.structure("t" + v, refs)
        tgt_u = PB.utt(lw(ROOT / f"data/snv_e1/{v}/t05.wav"))
        src_u = PB.utt(lw(ROOT / f"data/snv_p_wav_x16/e1/{v}/t0{5}.wav"))
        enr = [PB.utt(lw(ROOT / f"data/snv_p_wav_x16/e1/{v}/t0{j}.wav")) for j in (0, 1, 2)]
        med = lambda us: float(np.median(np.concatenate([q["f0"][q["f0"] > 0] for q in us])))
        k_semi = float(12 * np.log2(med(refs) / med(enr)))
        _, yo = S_ens.convert(src_u, enr, st, k_semi, with_y=True)

        x0 = tgt_u["wav"].astype(np.float64)
        n = min(yo.shape[-1], tgt_u["y"].shape[1])
        x0 = x0[: n * P.H]
        f0_p = tgt_u["f0"][:n].astype(np.float32)
        f0_h_raw, t_h = pyworld.harvest(x0, EV.SR, frame_period=5.0)
        f0_h = pyworld.stonemask(x0, f0_h_raw, t_h, EV.SR)[:n].astype(np.float32)

        xa = np.concatenate([np.zeros(PA.A), x0, np.zeros(PA.A)])
        env_b_raw = PA.envelope(xa, f0_h)
        env_b = np.concatenate([((env_b_raw[0] - PA.C0_SIL) / 10)[None], env_b_raw[1:25] / 10], 0).astype(np.float32)
        per_b = PA.periodicity(torch.from_numpy(xa)[None].float(), torch.from_numpy(f0_h)[None]).numpy()[0]

        env_p_h_raw = P.envelope(x0, f0_h)
        env_p_h = np.concatenate([((env_p_h_raw[0] - PA.C0_SIL) / 10)[None], env_p_h_raw[1:25] / 10], 0).astype(np.float32)
        per_p_h = P.periodicity(torch.from_numpy(x0.astype(np.float32)), torch.from_numpy(f0_h)).numpy()

        env_p_p = tgt_u["y"][:25, :n].astype(np.float32)
        per_p_p = tgt_u["y"][25:29, :n].astype(np.float32)

        K = yo[0, :29, :n].float().cpu().numpy()
        K_env25 = K[:25]
        K_per4 = K[25:29]
        K_env25_lp = np.concatenate([causal_lp_c0(K_env25[0], 0.25)[None], K_env25[1:25]], 0)
        K_env25_tc0 = np.concatenate([env_b[:1], K_env25[1:25]], 0)

        arms = {
            "B": (gen_A, center_smooth(env_b), f0_h, per_b),
            "T1": (gen_A, center_smooth(env_p_h), f0_h, per_p_h),
            "T2": (gen_S, causal_smooth(env_p_h), f0_h, per_p_h),
            "T3": (gen_Y, env_p_h, f0_h, per_p_h),
            "T": (gen_Y, env_p_p, f0_p, per_p_p),
            "K_B": (gen_A, center_smooth(K_env25), f0_h, K_per4),
            "K_Bc0s": (gen_A, center_smooth(K_env25_lp), f0_h, K_per4),
            "K_BTc0": (gen_A, center_smooth(K_env25_tc0), f0_h, K_per4),
            "K_Y": (gen_Y, K_env25, f0_p, K_per4),
        }

        sf.write(out / f"{v}_orig.wav", x0.astype(np.float32), EV.SR, subtype="PCM_16")
        rms0 = float(np.sqrt(np.mean(x0 ** 2)) + 1e-9)
        for name, (g, e25, f0, pr) in arms.items():
            w = np.asarray(EV.draw(g, pack_cond(e25, f0, pr), torch.from_numpy(f0)[None].to(EV.DEV), None), np.float32)
            if name == "B":
                rmsb = float(np.sqrt(np.mean(w.astype(np.float64) ** 2)) + 1e-9)
                assert abs(20 * np.log10(rmsb / rms0)) < 3.0, (v, rmsb, rms0)
            sf.write(out / f"{v}_{name}.wav", np.clip(w, -0.99, 0.99), EV.SR, subtype="PCM_16")
            summary["lm_raw"].setdefault(name, []).append(round(C9.lm(x0.astype(np.float32), w), 3))
            alm, sh = align_lm(x0.astype(np.float32), w)
            summary["lm_aligned"].setdefault(name, []).append(alm)
            summary["best_shift"].setdefault(name, []).append(sh)

    for k in summary["lm_raw"]:
        print(k.ljust(8), "raw", summary["lm_raw"][k], round(float(np.mean(summary["lm_raw"][k])), 3), "| aligned", summary["lm_aligned"][k], round(float(np.mean(summary["lm_aligned"][k])), 3), "sh", summary["best_shift"][k])
    (ROOT / "results/snv_p121/ladder.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))

    tok = (ROOT / "results/snv_p104/ear_site.token").read_text().strip()
    site = ROOT / "results/snv_p104/ear_site" / tok / "p121"
    site.mkdir(parents=True, exist_ok=True)
    for v in EAR_VOICES:
        for p in out.glob(f"{v}_*.wav"):
            shutil.copy2(p, site / p.name)
    (site / "index.html").write_text(build_html())


if __name__ == "__main__":
    with torch.no_grad():
        main()
