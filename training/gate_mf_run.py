"""男声 → 女声の関門(results/conv_p0/gate_mf_v1_draft.yaml)を新しい実男声の録音に一度だけ当てる実行役。
入力 = --in <dir>/<話者>/*.wav(任意の標本化周波数・1 話者 5 分以上が目安)。話者ごとに 名前順でつないで 先頭 22 s = 操作者の参照(登録・前置き)・残りから 8 s の区間を最大 --per 本。
文の形(data/real_male_eval/README_録音のお願い.md の t00〜t39.wav + free.wav)なら、dev の各組と同じ 3 文をつないで同じ目標(dev)へ = TTS 男声(dev)・TTS 女声(devf)の同じ文・同じ目標の変換と内容をそろえて比べられる(参照 = free.wav)。
目標 = dev2 の 30 人(女声 → 女声の基準 devff2 と同じ目標)を区間ごとに巡回。製品の設定(--ckpt・操作者の遅い揺らぎ)で定常状態(参照の末尾 12 s の前置き)に描画し、
(N) 本物 150 対 [男声入力の変換・devff2 の女声入力の変換(同じ設定・D467 の書き出し)] の判別器 1 つの v1⊥ の決定値の差(男 − 女)を目標でまとめたブートストラップ・
(G) d109 v2 の 4 軸(入力そのものも並べる)・(I) 識別の余白の kNN-VC との差(中心 = dev2/cen_reg)・(C) CER(入力の文字起こし基準・女声入力の変換の CER と比べる)を出す。
    uv run python gate_mf_run.py --in <録音の置き場> --name devmale1 --ckpt ../results/diag_ppgvoc43/product_002500_m0.pt --out ../results/conv_p0/gate_mf_devmale1
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).parent))
import nvoc as N
import train_ppgvoc as P
from d273_data import BIN, job
from d274_evaldump import COM
from d6_unit_cov import load48

T_ = Path(__file__).parent
PRE = 12.0


def run(cmd: list) -> str:
    r = subprocess.run([str(c) for c in cmd], cwd=P.ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"FAIL {cmd[:3]} {r.stderr[-800:]}")
    return r.stdout


def build(src: Path, S: Path, per: int, tset: str, per_spk: int = 0) -> list:
    """<src>/<話者>/*.wav → S(d56・d56/_work・d85_work・d37/_work・devset.json)= dev の組と同じ構成。"""
    for d in ("d37/_work", "d56/_work", "d85_work"):
        (S / d).mkdir(parents=True, exist_ok=True)
    D = P.H / tset
    for f in sorted((D / "d37/_work").glob("ref_*.wav")):
        if not (S / "d37/_work" / f.name).exists():
            shutil.copy(f, S / "d37/_work" / f.name)
    for f in sorted((D / "d37").glob("cen_*.wav")):
        if not (S / "d37" / f.name).exists():
            shutil.copy(f, S / "d37" / f.name)
    for e in sorted((D / "d85_work").glob("en_*")):
        if not (S / "d85_work" / e.name).exists():
            shutil.copytree(e, S / "d85_work" / e.name)
    targets = json.loads((D / "devset.json").read_text())["targets"]
    log, pi, ti = [], 0, 0
    devp = json.loads((P.H / "dev" / "devset.json").read_text())["pairs"]
    for si, spk in enumerate(sorted(d for d in src.iterdir() if d.is_dir())):
        if (spk / "t00.wav").exists() and (spk / "free.wav").exists():
            ref = load48(spk / "free.wav").astype(np.float32)
            if len(ref) < 20 * N.SR:
                ref = np.concatenate([ref] + [load48(spk / f"t{j:02d}.wav") for j in range(35, 40) if (spk / f"t{j:02d}.wav").exists()]).astype(np.float32)
            gap = np.zeros(int(0.15 * N.SR), np.float32)
            en0 = None
            rows = [devp[(si * per_spk + j) % len(devp)] for j in range(per_spk)] if per_spk else devp
            for r in rows:
                ws = [spk / f"{u}.wav" for u in r["used"]]
                if not all(w.exists() for w in ws):
                    continue
                sgm = np.concatenate(sum(([load48(w), gap] for w in ws), [])).astype(np.float32)[: 8 * N.SR]
                pid = si * 100 + r["pi"]
                k = f"p{pid:04d}_t{r['ti']:03d}"
                sf.write(S / "d56" / f"{k}_SRC.wav", sgm, N.SR)
                opw = S / "d56/_work" / f"op_{pid:04d}.wav"
                sf.write(opw, ref, N.SR)
                opd = S / "d85_work" / f"op_{pid:04d}"
                if not (opd / "enroll.json").exists():
                    if en0 is None:
                        run([BIN, "artic-enroll", "-r", opw, "-o", opd])
                        en0 = opd
                    else:
                        shutil.copytree(en0, opd)
                log.append({"pi": pid, "ti": r["ti"], "kind": "MR", "spk": spk.name, "used": r["used"], "dev_pi": r["pi"], "target": targets[r["ti"]]})
            continue
        x = np.concatenate([load48(w) for w in sorted(spk.glob("*.wav"))]).astype(np.float32)
        nref = int(22 * N.SR)
        if len(x) < nref + 8 * N.SR:
            print("短すぎる話者を除く", spk.name, round(len(x) / N.SR, 1), flush=True)
            continue
        ref, rest = x[:nref], x[nref:]
        segs = [rest[i:i + 8 * N.SR] for i in range(0, len(rest) - 8 * N.SR + 1, 8 * N.SR)][:per]
        en0 = None
        for sgm in segs:
            if np.sqrt((sgm ** 2).mean()) < 1e-3:
                continue
            k = f"p{pi:04d}_t{ti:03d}"
            sf.write(S / "d56" / f"{k}_SRC.wav", sgm, N.SR)
            opw = S / "d56/_work" / f"op_{pi:04d}.wav"
            sf.write(opw, ref, N.SR)
            opd = S / "d85_work" / f"op_{pi:04d}"
            if not (opd / "enroll.json").exists():
                if en0 is None:
                    run([BIN, "artic-enroll", "-r", opw, "-o", opd])
                    en0 = opd
                else:
                    shutil.copytree(en0, opd)
            log.append({"pi": pi, "ti": ti, "kind": "MR", "spk": spk.name, "target": targets[ti]})
            pi, ti = pi + 1, (ti + 1) % len(targets)
    (S / "devset.json").write_text(json.dumps({"targets": targets, "pairs": log, "kind": "real_male"}, ensure_ascii=False, indent=1))
    return log


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per", type=int, default=30)
    ap.add_argument("--per_spk", type=int, default=0, help="文の形で話者ごとに dev の組から使う数(0 = 全部・巡回で割り当て)")
    ap.add_argument("--flags", default="", help="Rust の書き出しの追加の引数(+ で区切る・例 --voicing-hyst+0.7,0.7)")
    ap.add_argument("--stop_after_render", action="store_true", help="描画まで(関門の軸は gate_mf_v2_run.sh で)")
    ap.add_argument("--c0shift", action="store_true", help="音量の条件の水準ずらし(D558・4.40 − 前置き 12 s の発声中の c0n 平均・無音には足さない・Rust --c0-ppg-ref と同じ)")
    ap.add_argument("--opgain", type=float, default=1.5, help="操作者の遅い揺らぎの利得(製品 1.5)")
    ap.add_argument("--ffref", default=str(P.ROOT / "results/conv_p0/d467/devff2/OP15"), help="女声入力の変換(同じ設定・定常状態・目標 = dev2)")
    a = ap.parse_args()
    dev = "cuda"
    S, O = P.H / a.name, Path(a.out)
    O.mkdir(parents=True, exist_ok=True)
    sentence_mode = any((d / "t00.wav").exists() for d in Path(a.src).iterdir() if d.is_dir())
    log = build(Path(a.src), S, a.per, "dev" if sentence_mode else "dev2", a.per_spk)
    print("pairs", len(log), "speakers", len({r["spk"] for r in log}), flush=True)
    ed = P.H / "e2e" / f"eval_{a.name}_pre"
    ed.mkdir(parents=True, exist_ok=True)
    n0 = int(PRE * N.SR)
    jobs = []
    for r in log:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        w = S / "d56/_work" / f"pre_{k}.wav"
        if not w.exists():
            pre = load48(S / "d56/_work" / f"op_{r['pi']:04d}.wav")[-n0:]
            sf.write(w, np.concatenate([np.pad(pre, (n0 - len(pre), 0)), load48(S / "d56" / f"{k}_SRC.wav")]).astype(np.float32), N.SR)
        jobs.append(([BIN, "artic", "--enroll", S / "d85_work" / f"en_{r['ti']:03d}", "--operator-enroll", S / "d85_work" / f"op_{r['pi']:04d}", "--med-from-operator"] + COM + [x for x in a.flags.split("+") if x] + ["-i", w], ed / f"{k}.cond", ed / f"{k}.front"))
    with Pool(3) as pl:
        list(pl.imap_unordered(job, jobs))
    st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    st["slow_sigma"], st["slow_causal"], st["opslow_tau"], st["opslow_gain"] = 1.5, True, 4.0, a.opgain
    if st.get("cvin"):
        P.cvin_setup(st["cvin"] if Path(st["cvin"]).is_absolute() else str(Path(__file__).parent / st["cvin"]), dev)
    P.PPG = st.get("ppg", "soft")
    P.CTX.update(on=bool(st.get("ctx", False)), ch=int(st.get("ctx_ch", 192)), adapt=bool(st.get("adapt", False)))
    P.slow_from_ckpt(st)
    g = P.init_gen(dev, st.get("spk", "table"))
    g.load_state_dict(st["ema"])
    g.eval()
    cvd = S / "gate" / "CV"
    cvd.mkdir(parents=True, exist_ok=True)
    for r in log:
        k = f"p{r['pi']:04d}_t{r['ti']:03d}"
        c, lg = P.load_dump(ed, k)
        if a.c0shift:
            pre = c[:2400, 0]
            mu = float(pre[pre > 0.5].mean()) if (pre > 0.5).sum() > 50 else 4.40
            c = c.copy()
            c[:, 0] = np.where(c[:, 0] > 0.05, c[:, 0] + (4.40 - mu), c[:, 0])
        cvf = None
        if P.CVIN["on"]:
            import train_c1 as TC
            cvf = P.cvin_frames(torch.from_numpy(TC.prime(load48(S / "d56/_work" / f"pre_{k}.wav").astype(np.float32)))[None].to(dev), len(lg))
        y = P.render(g, c, lg, P.load_enroll(S / "d85_work" / f"en_{r['ti']:03d}"), dev, T_op=P.op_tab(S, k), cvf=cvf)[n0:n0 + 8 * N.SR]
        sf.write(cvd / f"{k}_CV.wav", (y / (np.abs(y).max() + 1e-9) * 0.9).astype(np.float32), N.SR)
    print("rendered", len(log), flush=True)
    if a.stop_after_render:
        return 0
    rep: dict = {"pairs": len(log), "speakers": sorted({r["spk"] for r in log})}
    nd = O / "N"
    for nm, src_, pat in (("MALE", cvd, "*_CV.wav"), ("FEMALE", Path(a.ffref), "*.wav")):
        (nd / nm).mkdir(parents=True, exist_ok=True)
        for f in sorted(src_.glob(pat)):
            dst = nd / nm / f"{f.name[:10]}_{nm}.wav"
            if not dst.exists():
                dst.symlink_to(f.resolve())
    run([sys.executable, T_ / "gate_pooled_v1.py", "--dir", nd, "--a", "MALE", "--b", "FEMALE", "--out", O / "N.json"])
    rep["N"] = json.loads((O / "N.json").read_text())
    run([sys.executable, T_ / "d109_axis_gate.py", "--s", S, "--no_base", "--tag", f"gate_mf_{a.name}", "--extra", "gate/CV:CV", "d56:SRC"])
    j = json.loads((P.ROOT / "results/conv_p0" / f"gate_mf_{a.name}.json").read_text())
    rep["G"] = {k: {"axes": v["axes"], "fail": {s_: x["median"] for s_, x in v["stats"].items() if not x["pass"]}} for k, v in j.items() if isinstance(v, dict) and "axes" in v}
    run([sys.executable, T_ / "d135_knnvc_calib.py", "--s", S, "--real_only"])
    slot = (lambda pat: f"{pat}|/nonexistent/*.wav|/nonexistent/*.wav") if sentence_mode else (lambda pat: f"/nonexistent/*.wav|{pat}|/nonexistent/*.wav")
    run([sys.executable, T_ / "d394_idmargin.py", "--only", "NOGROUP", "--extra", "CV=" + slot(f"{cvd}/p*_CV.wav"), "KNN=" + slot(f"{S}/d135_real/p*_KNN.wav"), "--vs", "KNN", "--out", O / "I.json"])
    rep["I"] = json.loads((O / "I.json").read_text()).get("paired", {})
    cd = O / "cer"
    cd.mkdir(parents=True, exist_ok=True)
    for f in cvd.glob("p*_CV.wav"):
        dst = cd / f.name
        if not dst.exists():
            dst.symlink_to(f.resolve())
    for f in (S / "d56").glob("p*_SRC.wav"):
        dst = cd / f.name
        if not dst.exists():
            dst.symlink_to(f)
    run([sys.executable, T_ / "d23_cer.py", "--dirs", cd, "--names", a.name, "--out", O / "C.json", "--device", "cuda"])
    rep["C"] = {k: v for k, v in json.loads((O / "C.json").read_text())[a.name].items() if k != "examples" and k != "per"}
    (O / "summary.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps(rep, ensure_ascii=False)[:3000], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
