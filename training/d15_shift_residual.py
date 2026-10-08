"""D15(c5_1 レビューの G-1): 本物の残差を内容とずらして足したら同一性は上がるか(学習なし・自己変換・包絡だけ替える・参照のフレームは使わない)。
Ri = 同じ発話の実の包絡 − 同じ発話の単位平均(R_ins 型)・Rt = 実の包絡 − 参照の表の引き。
TAB_RiS05 / TAB_RiS10 = 表の引き + Ri を 0.5 / 1.0 s 循環シフト・TAB_RiB05 = Ri を 0.5 s ブロックで入れ替え・TAB_RtS05 = 表の引き + Rt を 0.5 s 循環シフト・TAB_Ri0 = 表 + Ri(ずらさない・内容に揃った上限)。
旧: D11: 自己変換の包絡の分解(D10 から派生)。包絡だけを替え、他(c0・f0・周期性)は実の分析。
COPY = 実の包絡・LP4 = 実の包絡を 4Hz で低域通過(単位の中の動きを消す)・INS = 同じ発話の中の単位平均(C1 の単位・推定のノイズなし・因果 5 平均)・INS_RAW = 同じ(平均の平滑なし)・TAB = 参照 20s の表(製品)。
旧: D10: 自己変換の自然さの梯子(学習なし)。目標自身の別発話を、写し合成の条件(実の分析)と変換の条件(製品経路)の組み合わせで出力部に通す。
条件の部品: env(実の CheapTrick 包絡 / 表の引き = C1 の硬い単位 + 因果 5 平均)・c0(実 / F2)・f0(教師 / F1)・per(実 / 話者平均の定数)。
COPY = 全部 実・TAB = 全部 製品経路・ONLY_x = x だけ製品経路(残りは実)。フレーム周期の変調の線(eval_nvoc.amline)を出力ごとに測り、
d4b(学習外の LDA)で同一性を測るための wav(cen_XX.wav・pYYY_tXX_<cond>.wav)を書く。
    uv run python d10_natural_ladder.py --out_dir <scratchpad>/d10 --out ../results/conv_p0/d10_natural.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
from d6_unit_cov import TTS_LEAK, load48

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_utt", type=int, default=3)
    ap.add_argument("--pool", default="held", choices=["held", "lda_out"], help="lda_out = 評価外の女声のうち d4b の LDA 学習話者でない話者(検出力のため多数)")
    ap.add_argument("--n_tg", type=int, default=60)
    a = ap.parse_args()
    import c1_content as CC
    import conv_c0 as C0
    import eval_nvoc as E
    import f0est as FE
    import f0hi as H
    import lvl as LV
    import nvoc as N
    import pae as PA
    import rvoc as R
    import train_c1 as T1
    import train_c3 as C3
    import train_f0est as TF
    import train_rvoc as TR
    dev = "cuda"
    od = Path(a.out_dir); od.mkdir(parents=True, exist_ok=True)
    sr_ = torch.load(ROOT / "results/rvoc3hi/snap/ema_330k.pt", map_location="cpu", weights_only=False)
    gen = R.RVoc(ch=sr_["cfg"]["ch"], kernels=tuple(sr_["cfg"]["kernels"]), dils=tuple(sr_["cfg"]["dils"]), d_cond=sr_["cfg"]["d_cond"]).to(dev)
    gen.load_state_dict(sr_["ema"]); gen.eval().requires_grad_(False)
    sm_w = TR.ckpt_env_smooth(sr_, None)
    sf1 = torch.load(ROOT / "results/f0est3/last.pt", map_location="cpu", weights_only=False)
    f1 = FE.F0Est(dils=(1, 2, 4, 8, 16, 32, 1, 2, 4, 8, 16, 32)).to(dev); f1.load_state_dict(sf1["net"]); f1.eval()
    f1front = FE.Front().to(dev)
    sf2 = torch.load(ROOT / "results/f2_2/last.pt", map_location="cpu", weights_only=False)
    f2 = LV.Lvl(sf2["cfg"]["ch"], tuple(sf2["cfg"]["dils"])).to(dev); f2.load_state_dict(sf2["net"]); f2.eval()
    st1 = torch.load(ROOT / "results/c1_1/last.pt", map_location="cpu", weights_only=False)
    c1 = CC.C1(st1["cfg"]["k"], st1["cfg"]["ch"], tuple(st1["cfg"]["dils"])).to(dev); c1.load_state_dict(st1["net"]); c1.eval()
    mfront = CC.MelFront().to(dev)
    Cb = torch.load(ROOT / "results/c1_1/codebook.pt").cpu().numpy()
    K = Cb.shape[0]

    def sm3(Ev):
        e = np.pad(Ev, ((0, 0), (1, 1)), mode="edge")
        return sm_w * e[:, :-2] + (1 - 2 * sm_w) * e[:, 1:-1] + sm_w * e[:, 2:]

    def analyse(x, f0):
        xa = np.concatenate([np.zeros(PA.A, np.float32), x, np.zeros(PA.A, np.float32)])
        env = sm3(PA.envelope(xa, f0))
        per = PA.periodicity(torch.from_numpy(xa)[None], torch.from_numpy(f0)[None])[0].numpy()
        return env, per

    @torch.no_grad()
    def post(x):
        n = len(x) // N.HOP
        return c1(mfront(torch.from_numpy(T1.prime(x))[None].to(dev)))[..., -n:].softmax(1)[0].T.cpu().numpy()

    @torch.no_grad()
    def render(c0, env24, f0, per):
        cc = np.concatenate([c0[None] - PA.C0_SIL, env24], 0)
        lf = np.where(f0 > 0, np.log(np.maximum(f0, 1.0) / 200.0), 0.0)[None]
        v = (f0 > 0).astype(np.float32)[None]
        cond = torch.from_numpy(np.concatenate([cc / 10, lf, v, per * v], 0).astype(np.float32))[None].to(dev)
        fo = torch.from_numpy(f0.astype(np.float32))[None].to(dev)
        y = gen(cond, TR.excitation(fo, cond.shape[-1] * N.HOP, torch.Generator().manual_seed(0)))[0].cpu().numpy()[N.DELAY:]
        return y / (np.abs(y).max() + 1e-9) * 0.9

    hs = TR.held_speakers()
    held = [s for s in sorted(hs) if s != "unknown" and s not in TTS_LEAK and (ROOT / "female-dataset" / s).is_dir()]
    if a.pool == "lda_out":
        allspk = sorted(p.name for p in (ROOT / "female-dataset").iterdir() if p.is_dir() and p.name not in (hs | TTS_LEAK))
        lda_tr = {allspk[i] for i in np.random.RandomState(0).permutation(len(allspk))[:1200]}
        cand = [s_ for s_ in allspk if s_ not in lda_tr]
        held = [cand[i] for i in np.random.RandomState(5).permutation(len(cand))[: a.n_tg * 2]]
    conds = ["COPY", "TAB", "TAB_Ri0", "TAB_RiS05", "TAB_RiS10", "TAB_RiB05", "TAB_RtS05"]
    am: dict = {c: [] for c in conds}
    ti = 0
    pi = 0
    for s in held:
        ws = sorted((ROOT / "female-dataset" / s).glob("*.wav"))
        ref, tot = [], 0.0
        for w in ws:
            if tot >= 20.0:
                break
            ref.append(w); tot += sf.info(str(w)).duration
        rest = [w for w in ws if w not in ref]
        if len(rest) < a.n_utt + 2:
            continue
        xr = np.concatenate([load48(w) for w in ref]); xr = xr[: len(xr) // N.HOP * N.HOP]
        f0r, _ = H.teacher_f0(xr, len(xr) // N.HOP)
        envr, perr = analyse(xr, f0r)
        pr = post(xr)
        T = C0.table(envr[1:25, : len(pr)], pr.argmax(1), K, Cb)
        vo_r = f0r > 0
        per_mean = perr[:, vo_r].mean(1)
        cen = np.concatenate([load48(w) for w in rest[a.n_utt:]])[: 25 * 48000]
        sf.write(od / f"cen_{ti:02d}.wav", cen, 48000)
        for w in rest[: a.n_utt]:
            x = load48(w)[: 8 * 48000]
            n = len(x) // N.HOP
            x = x[: n * N.HOP]
            if n < 200:
                continue
            f0t, _ = H.teacher_f0(x, n)
            env, per = analyse(x, f0t)
            with torch.no_grad():
                f0p = TF.infer(f1front, f1, x, dev)
                f0p = np.pad(f0p, (0, max(0, n - len(f0p))))[:n].astype(np.float32)
                c0p = f2(mfront(torch.cat([torch.zeros(1, N.WIN - N.HOP, device=dev), torch.from_numpy(x)[None].to(dev)], 1)))[0].cpu().numpy()[:n] * 10 + PA.C0_SIL
            u = post(x).argmax(1)
            u = np.pad(u, (0, max(0, n - len(u))), mode="edge")[:n]
            env_tab = C3.causal_avg(torch.from_numpy(T[:, u].astype(np.float32))[None], C3.W_SM)[0].numpy()
            per_c = np.repeat(per_mean[:, None], n, 1)
            real = {"env": env[1:25, :n], "c0": env[0, :n], "f0": f0t[:n], "per": per[:, :n]}
            prod = {"env": env_tab, "c0": c0p, "f0": f0p, "per": per_c}
            er = env[1:25, :n]
            ins_t = C0.table(er, u, K, Cb)
            ri = er - ins_t[:, u]
            rt = er - T[:, u]
            g_ = np.random.default_rng(pi)
            nb = max(1, n // 100)
            order = g_.permutation(nb)
            blk = np.concatenate([ri[:, b * 100:(b + 1) * 100] for b in order] + ([ri[:, nb * 100:]] if n > nb * 100 else []), 1)[:, :n]
            envs = {"COPY": er, "TAB": env_tab, "TAB_Ri0": env_tab + ri, "TAB_RiS05": env_tab + np.roll(ri, 100, 1), "TAB_RiS10": env_tab + np.roll(ri, 200, 1),
                    "TAB_RiB05": env_tab + blk, "TAB_RtS05": env_tab + np.roll(rt, 100, 1)}
            for c in conds:
                pick = dict(real); pick["env"] = envs[c]
                y = render(pick["c0"], pick["env"], pick["f0"], pick["per"])
                sf.write(od / f"p{pi:03d}_t{ti:02d}_{c}.wav", y.astype(np.float32), 48000)
                am[c].append(E.amline(y))
            sf.write(od / f"p{pi:03d}_t{ti:02d}_SRC.wav", x, 48000)
            pi += 1
        ti += 1
        if a.pool == "lda_out" and ti >= a.n_tg:
            break
        print(s, {c: round(float(np.mean(v)), 3) for c, v in am.items()}, flush=True)
    rep = {"n_targets": ti, "n_utts": pi, "am_line_db": {c: round(float(np.mean(v)), 3) for c, v in am.items()}}
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    Path(a.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
