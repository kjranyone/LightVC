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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pae as PA
import snv_f0v4 as V4
import snv_out_guard as OG
import snv_p0_front as P
import snv_p1_eval as EV
import train_snv_p1 as T
from p97_probe import IN, SPS

ROOT = EV.ROOT
EAR_VOICES = ("ef100", "ef032")
G_BASE = "+mask:snv_p45_gate_t80+fb:anch"
G_C0BG = "+mask:snv_p45_gate_t80+fb:c0bg0.45_0.15,anch"
SYS_BASE = "ens:" + "|".join(f"{t}{G_BASE}" for t in ("snv_p24_c", "snv_p30_c", "snv_p91e_c"))
SYS_C0BG = "ens:" + "|".join(f"{t}{G_C0BG}" for t in ("snv_p24_c", "snv_p30_c", "snv_p91e_c"))


def center_smooth(e: np.ndarray) -> np.ndarray:
    q = np.pad(e, ((0, 0), (1, 1)), mode="edge")
    return (0.25 * q[:, :-2] + 0.5 * q[:, 1:-1] + 0.25 * q[:, 2:]).astype(np.float32)


def pack_cond(env25: np.ndarray, f0: np.ndarray, per4: np.ndarray) -> torch.Tensor:
    v = (f0 > 0).astype(np.float32)[None]
    lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)[None]
    c = np.concatenate([env25.astype(np.float32), lf, v, np.clip(per4.astype(np.float32), 0.0, 1.5) * v], 0)
    return torch.from_numpy(c)[None].to(EV.DEV)


def load_gen_ck(ck: str) -> torch.nn.Module:
    os.environ.pop("SNV_RCK", None)
    if ck:
        os.environ["SNV_RCK"] = ck
    return EV.load_gen()


def raw_utt(x: np.ndarray) -> dict:
    x = x[: P.n_frames(len(x)) * P.H].astype(np.float32)
    d = P.analyze(x)
    f0, env, per = d["f0"].astype(np.float32), d["env"].astype(np.float32), d["per"].astype(np.float32)
    s, n = T.voiced_mean(env, f0)
    lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0).astype(np.float32)
    y = T.to_y(env, per, f0)
    m = y.shape[1]
    yc = np.concatenate([T.to_y(P.fixed_env(x, m), np.zeros((4, m), np.float32), np.ones(m, np.float32))[:25], y[25:]], 0).astype(np.float32)
    return {"y": y, "lf": lf, "v": (f0 > 0).astype(np.float32), "f0": f0, "vs": s.astype(np.float32), "vn": n, "yc": yc, "wav": x}


def build_html(win_label: str) -> str:
    sec1 = [
        ("orig", "元の音", "Irodori 目標音声 (48 kHz)"),
        ("B", "B: 合格基準の写し (先読みあり)", "Harvest + 中心包絡 + 中心平滑 + rvoc3hi (PESQ 3.55)"),
        ("old_T", "old_T: 前回の因果写し T", "前段 v3.1 + snv_p97_y240 (PESQ 2.79)"),
        ("new_T", "new_T: 今回の因果写し T (先読み 0)", "f0_voc v4 + snv_p123_s (PESQ 3.54・am_db +0.34 dB)"),
    ]
    sec2 = [
        ("src", "入力の男声", "同じ内容の男声版"),
        ("old_K_Y", "old_K: 前回の変換 K", "K3G + 前段 v3.1 + snv_p97_y240"),
        ("B0_oldA", "B0: 現行製品の変換", "K3G + 前段 v3.1 + rvoc3hi"),
        ("p124_K", f"P124: 今回の変換 ({win_label})", "K3G + f0_net/f0_voc v4 + c0bg + snv_p123_s"),
    ]
    parts = [
        '<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex,nofollow"><title>段 P124 耳の関門</title>',
        "<style>body{font-family:system-ui,sans-serif;max-width:860px;margin:0 auto;padding:16px;line-height:1.6;background:#fafafa;color:#222}h1{font-size:1.3rem}h2{font-size:1.05rem;margin:0 0 6px}h3{font-size:.98rem;margin:10px 0 4px}section{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px 14px;margin:12px 0}.r{display:flex;flex-wrap:wrap;align-items:center;gap:4px 10px;margin:6px 0}.r span{width:18em;font-weight:600}audio{flex:1;min-width:220px}i{flex-basis:100%;font-style:normal;font-size:.8rem;color:#666}@media(prefers-color-scheme:dark){body{background:#161616;color:#eee}section{background:#222;border-color:#444}i{color:#aaa}}</style>",
        "<h1>段 P124 耳の関門 — 因果写し T の回復と K3G 変換雑音の解消</h1>",
        "<p>1. 製品経路の因果写し T が合格基準 B 水準(PESQ 3.54)まで回復したことの確認。<br>2. 同じ土台の上での変換(合成男声および本物の男声 +0 / +12 半音: 現行製品 B0 対 P124 候補)。</p>",
    ]
    for v in EAR_VOICES:
        parts.append(f"<section><h2>{v} — 1. 写し合成の比較 (orig / B / old_T / new_T)</h2>")
        for key, label, note in sec1:
            parts.append(f'<div class=r><span>{label}</span><audio controls preload=none src="{v}_{key}.wav"></audio><i>{note}</i></div>')
        parts.append(f"<h3>{v} — 2. 男声からの変換 (old_K / 現行製品 B0 / 今回 P124)</h3>")
        for key, label, note in sec2:
            parts.append(f'<div class=r><span>{label}</span><audio controls preload=none src="{v}_{key}.wav"></audio><i>{note}</i></div>')
        parts.append("</section>")
    parts.append("<section><h2>3. 本物の男声 4 人 (現行製品 B0 対 今回 P124・+0 / +12 半音)</h2>")
    for idx, sp in enumerate(SPS, 1):
        parts.append(f"<h3>話者 {idx} ({sp})</h3>")
        rows_sp = [
            (f"spk{idx}_src.wav", "入力 (本物の男声)", ""),
            (f"spk{idx}_k0_product.wav", "現行製品 B0 (+0 半音)", "K3G + 前段 v3.1 + rvoc3hi"),
            (f"spk{idx}_k0_p124.wav", f"今回 P124 (+0 半音・{win_label})", "K3G + v4 + snv_p123_s"),
            (f"spk{idx}_k12_product.wav", "現行製品 B0 (+12 半音)", "K3G + 前段 v3.1 + rvoc3hi"),
            (f"spk{idx}_k12_p124.wav", f"今回 P124 (+12 半音・{win_label})", "K3G + v4 + snv_p123_s"),
        ]
        for fn, label, note in rows_sp:
            parts.append(f'<div class=r><span>{label}</span><audio controls preload=none src="{fn}"></audio><i>{note}</i></div>')
    parts.append("</section></html>")
    return "\n".join(parts)


@torch.no_grad()
def main() -> None:
    win = sys.argv[1] if len(sys.argv) > 1 else "R3"
    sys_win = SYS_C0BG if win == "R3" else SYS_BASE
    out = ROOT / "results/snv_p124/ear"
    out.mkdir(parents=True, exist_ok=True)
    lw = lambda p: librosa.load(p, sr=EV.SR, mono=True)[0].astype(np.float32)

    gen_A = load_gen_ck("")
    gen_Y = load_gen_ck("results/snv_p97_y240/snap/ema_30k.pt")
    gen_S = load_gen_ck("results/snv_p123_s/snap/ema_20k.pt")

    os.environ.pop("SNV_F0V4", None)
    ens_Y = EV.Ens(SYS_BASE, gen_Y)
    ens_B0 = EV.Ens(SYS_BASE, gen_A)

    for v in EAR_VOICES:
        x0 = lw(ROOT / f"data/snv_e1/{v}/t05.wav")
        Tn = P.n_frames(len(x0))
        x0 = x0[: Tn * P.H]
        sf.write(out / f"{v}_orig.wav", x0, EV.SR, subtype="PCM_16")

        x64 = x0.astype(np.float64)
        f0_hr, t_h = pyworld.harvest(x64, EV.SR, frame_period=5.0)
        f0_h = pyworld.stonemask(x64, f0_hr, t_h, EV.SR)[:Tn].astype(np.float32)
        xa = np.concatenate([np.zeros(PA.A), x64, np.zeros(PA.A)])
        eb_r = PA.envelope(xa, f0_h)
        eb = np.concatenate([((eb_r[0] - PA.C0_SIL) / 10)[None], eb_r[1:25] / 10], 0).astype(np.float32)
        pb = PA.periodicity(torch.from_numpy(xa)[None].float(), torch.from_numpy(f0_h)[None]).numpy()[0]
        w_B = EV.draw(gen_A, pack_cond(center_smooth(eb), f0_h, pb), torch.from_numpy(f0_h)[None].to(EV.DEV), None)
        sf.write(out / f"{v}_B.wav", np.clip(w_B, -0.99, 0.99), EV.SR, subtype="PCM_16")

        u_t = raw_utt(x0)
        w_oldT = EV.draw(gen_Y, pack_cond(u_t["y"][:25], u_t["f0"], u_t["y"][25:29]), torch.from_numpy(u_t["f0"])[None].to(EV.DEV), None)
        sf.write(out / f"{v}_old_T.wav", np.clip(w_oldT, -0.99, 0.99), EV.SR, subtype="PCM_16")

        _, fv_t, _ = V4.f0_dual_track(torch.from_numpy(x0))
        fv_np = fv_t.numpy().astype(np.float32)
        ev_n = P.envelope(x0, fv_np.astype(np.float64))
        pr_n = P.periodicity(torch.from_numpy(x0), fv_t).numpy()
        yn_t = T.to_y(ev_n, pr_n, fv_np)
        sm_nt = EV.smooth_env(gen_S, torch.from_numpy(yn_t[:25])[None].to(EV.DEV))[0].cpu().numpy()
        w_newT = EV.draw(gen_S, pack_cond(sm_nt, fv_np, yn_t[25:29]), torch.from_numpy(fv_np)[None].to(EV.DEV), None)
        sf.write(out / f"{v}_new_T.wav", np.clip(w_newT, -0.99, 0.99), EV.SR, subtype="PCM_16")

        x_src = lw(ROOT / f"data/snv_p_wav_x16/e1/{v}/t05.wav")
        x_src = x_src[: P.n_frames(len(x_src)) * P.H]
        sf.write(out / f"{v}_src.wav", x_src, EV.SR, subtype="PCM_16")
        refs = [raw_utt(lw(ROOT / f"data/snv_e1/{v}/t0{j}.wav")) for j in (0, 1, 2)]
        enr = [raw_utt(lw(ROOT / f"data/snv_p_wav_x16/e1/{v}/t0{j}.wav")) for j in (0, 1, 2)]
        u_s = raw_utt(x_src)
        med = lambda us: float(np.median(np.concatenate([q["f0"][q["f0"] > 0] for q in us])))

        os.environ.pop("SNV_F0V4", None)
        st_Y = ens_Y.structure("t" + v, refs)
        k_v3 = float(12.0 * np.log2(med(refs) / med([ens_Y.parts[0].prep(q) for q in enr])))
        w_kY = ens_Y.convert(u_s, enr, st_Y, k_v3)
        sf.write(out / f"{v}_old_K_Y.wav", np.clip(w_kY, -0.99, 0.99), EV.SR, subtype="PCM_16")

        st_B0 = ens_B0.structure("t" + v, refs)
        w_kB0 = ens_B0.convert(u_s, enr, st_B0, k_v3)
        sf.write(out / f"{v}_B0_oldA.wav", np.clip(w_kB0, -0.99, 0.99), EV.SR, subtype="PCM_16")

        os.environ["SNV_F0V4"] = "1"
        ens_W = EV.Ens(sys_win, gen_S)
        st_W = ens_W.structure("t" + v, refs)
        k_v4 = float(12.0 * np.log2(med(refs) / med([ens_W.parts[0].prep(q) for q in enr])))
        w_kW = ens_W.convert(u_s, enr, st_W, k_v4)
        sf.write(out / f"{v}_p124_K.wav", np.clip(w_kW, -0.99, 0.99), EV.SR, subtype="PCM_16")

    rd = lambda p: np.fromfile(p, np.float32)
    tg = [raw_utt(rd(p)) for p in sorted((IN / "tg").glob("*.f32"))]
    p_out = OG.speech_level([u["wav"] for u in tg])

    os.environ.pop("SNV_F0V4", None)
    ens_B0p = EV.Ens(SYS_BASE, gen_A)
    st_B0p = ens_B0p.structure("tg_p", tg)

    os.environ["SNV_F0V4"] = "1"
    ens_Wp = EV.Ens(sys_win, gen_S)
    st_Wp = ens_Wp.structure("tg_p", tg)

    for idx, sp in enumerate(SPS, 1):
        x = rd(IN / sp / "in.f32")
        u = raw_utt(x)
        enr = [raw_utt(rd(p)) for p in sorted((IN / sp / "op").glob("*.f32"))]
        p_in = OG.speech_level([e["wav"] for e in enr])
        sf.write(out / f"spk{idx}_src.wav", np.clip(u["wav"], -0.99, 0.99), EV.SR, subtype="PCM_16")
        for ks in (0, 12):
            os.environ.pop("SNV_F0V4", None)
            wb0 = OG.apply(u["wav"], ens_B0p.convert(u, enr, st_B0p, ks), p_in, p_out)
            sf.write(out / f"spk{idx}_k{ks}_product.wav", np.clip(wb0, -0.99, 0.99), EV.SR, subtype="PCM_16")
            os.environ["SNV_F0V4"] = "1"
            ww = OG.apply(u["wav"], ens_Wp.convert(u, enr, st_Wp, ks), p_in, p_out)
            sf.write(out / f"spk{idx}_k{ks}_p124.wav", np.clip(ww, -0.99, 0.99), EV.SR, subtype="PCM_16")
        for p in ens_B0p.parts:
            p.pcache.clear()
        for p in ens_Wp.parts:
            p.pcache.clear()

    html = build_html(win)
    (out / "index.html").write_text(html)
    tok_p = ROOT / "results/snv_p104/ear_site.token"
    if tok_p.exists():
        tok = tok_p.read_text().strip()
        site = ROOT / "results/snv_p104/ear_site" / tok / "p124"
        site.mkdir(parents=True, exist_ok=True)
        for p in out.glob("*.wav"):
            shutil.copy2(p, site / p.name)
        (site / "index.html").write_text(html)
    print("wrote ear package ->", out, flush=True)


if __name__ == "__main__":
    main()
