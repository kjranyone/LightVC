"""段 P93 の耳の関門の材料(results/snv_p93/prereg.yaml の ear_material): 今の製品の経路(snv_p11_r)と候補 K3G を、同じ源・同じ目標の声・同じ半音で変換し、順を伏せて並べる。
源 = 開発の組 devB の男声(候補の確認の組)・JVNV の男声・JNV の音素の無い発声。条件 = 足さない・伴奏 10 dB・手拍子・笑い + 手拍子。目標 = A0F の架空の女声。
出力 = <out>/item<NN>_{ref,src,A,B}.wav(音量を揃える)・key.json(A / B の中身・聴く前に開かない)・index.html(聴き比べと答えの控え)。
    cd training && SNV_ENVC=1 SNV_DEV=cuda uv run --no-sync python snv_ear_set.py [out=results/snv_p93/ear] [k=7]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent / "snv_diag"))
import i10_jvnv_probe as JP
import i13_bgm_dev_probe as I13
import snv_dev_eval as DE
import snv_p0_front as P
import snv_p1_eval as EV
import train_snv_sep as SEP

ROOT = EV.ROOT
G = "+mask:snv_p45_gate_t80+fb:anch"
SYS = {"p11": "snv_p11_r", "k3g": f"ens:snv_p24_c{G}|snv_p30_c{G}|snv_p91e_c{G}"}
LVL = 0.05


def norm(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, np.float64)
    w = w * (LVL / max(float(np.sqrt((w ** 2).mean())), 1e-6))
    return (w / max(1.0, float(np.abs(w).max()) / 0.95)).astype(np.float32)


def main() -> int:
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    out, k = ROOT / opt.get("out", "results/snv_p93/ear"), int(opt.get("k", 7))
    out.mkdir(parents=True, exist_ok=True)
    load = lambda p: librosa.load(str(p), sr=P.SR, mono=True)[0].astype(np.float64)
    tg4 = json.loads((EV.A1 / "convert_meta.json").read_text())["targets"][:4]
    trefs = {t: [EV.utt(f"a0/{t}_{j}") for j in (0, 1, 2)] for t in tg4}
    cv = DE.cv_items("devB", 14)
    spk = {}
    for it in cv:
        spk.setdefault(it["spk"], it)
    cvs = list(spk.values())
    music = SEP.bg_split()["dev"]["music"]
    claps = json.loads((ROOT / "data/snv_sep/fsd50k_eval_sources.json").read_text())["pools"]["clap"]
    jv_enr = {s: [JP.feats(f"{s}_{e}_regular_02", "jvnv", None) for e in JP.EMOS] for s in ("M1", "M2")}
    spec = []
    for i, it in enumerate(cvs[:8]):
        spec.append({"kind": "本物の男声(devB)", "cond": "足さない", "u": it["u"], "x": load(it["u"]["wav_path"]), "enr": it["enr"]})
    for i, it in enumerate(cvs[8:11]):
        x = load(it["u"]["wav_path"])
        xm = I13.with_music(x, music[i % len(music)], 100 + i, False, 10.0)[1]
        spec.append({"kind": "本物の男声(devB)", "cond": "伴奏 10 dB", "u": I13.utt_of(xm), "x": xm, "enr": it["enr"]})
    for i, it in enumerate(cvs[11:13]):
        x = load(it["u"]["wav_path"])
        xm = I13.with_events(x, claps, 200 + i, False, 0.0, 4)[1]
        spec.append({"kind": "本物の男声(devB)", "cond": "手拍子", "u": I13.utt_of(xm), "x": xm, "enr": it["enr"]})
    for s, e in (("M1", "happy"), ("M2", "sad")):
        u = JP.feats(f"{s}_{e}_regular_01", "jvnv", None)
        spec.append({"kind": "スタジオの男声(JVNV)", "cond": "足さない", "u": u, "x": load(u["wav_path"]), "enr": jv_enr[s]})
    nv = [("M1", "M1_happy_01_R"), ("M2", "M2_happy_01_R"), ("M1", "M1_surprise_01_R")]
    for s, nm in nv:
        p = ROOT / "data/jnv/JNV" / s / f"{nm}.wav"
        if p.exists():
            x = load(p)
            spec.append({"kind": "音素の無い発声(JNV)", "cond": "足さない", "u": I13.utt_of(x), "x": x, "enr": jv_enr[s]})
    for i, (s, nm) in enumerate(nv[:2]):
        p = ROOT / "data/jnv/JNV" / s / f"{nm}.wav"
        if p.exists():
            xm = I13.with_events(load(p), claps, 300 + i, False, 0.0, 4)[1]
            spec.append({"kind": "音素の無い発声(JNV)", "cond": "笑い + 手拍子", "u": I13.utt_of(xm), "x": xm, "enr": jv_enr[s]})
    gen = EV.load_gen()
    sy = {n: EV.System(t, gen) for n, t in SYS.items()}
    g = np.random.default_rng(93)
    key, rows = {}, []
    for i, sp in enumerate(spec):
        tgt = tg4[i % 4]
        w = {n: norm(s.convert(sp["u"], sp["enr"], s.structure(f"t|{tgt}", trefs[tgt]), k)) for n, s in sy.items()}
        order = ["p11", "k3g"] if g.random() < 0.5 else ["k3g", "p11"]
        nm = f"item{i + 1:02d}"
        sf.write(out / f"{nm}_ref.wav", norm(load(EV.wav_path(f"a0/{tgt}_0"))), P.SR)
        sf.write(out / f"{nm}_src.wav", norm(sp["x"]), P.SR)
        for lab, n in zip("AB", order):
            sf.write(out / f"{nm}_{lab}.wav", w[n], P.SR)
        key[nm] = {"A": order[0], "B": order[1], "kind": sp["kind"], "cond": sp["cond"], "tgt": tgt, "k": k}
        rows.append((nm, sp["kind"], sp["cond"]))
        for s in sy.values():
            for q in (s.parts if hasattr(s, "parts") else [s]):
                q.scache.clear(), q.pcache.clear()
        print(nm, sp["kind"], sp["cond"], flush=True)
    (out / "key.json").write_text(json.dumps({"systems": SYS, "items": key}, ensure_ascii=False, indent=1))
    au = lambda f: f'<audio controls preload="none" src="{f}"></audio>'
    qs = (("near", "目標の声(基準)に近いのは"), ("clear", "言葉・発声が聞き取りやすいのは"), ("nice", "聴いていて心地よいのは"))
    body = "".join(
        f'<section><h2>{nm} <small>{kind}・{cond}</small></h2><div class="r"><span>基準(目標の声)</span>{au(nm + "_ref.wav")}</div><div class="r"><span>源(入力)</span>{au(nm + "_src.wav")}</div>'
        f'<div class="r"><span>A</span>{au(nm + "_A.wav")}</div><div class="r"><span>B</span>{au(nm + "_B.wav")}</div>'
        + "".join(f'<div class="q">{t}: ' + "".join(f'<label><input type="radio" name="{nm}_{q}" value="{v}">{v}</label>' for v in ("A", "B", "同じ")) + "</div>" for q, t in qs) + "</section>"
        for nm, kind, cond in rows)
    html = ('<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>聴き比べ K3G</title>'
            "<style>body{font-family:sans-serif;max-width:760px;margin:0 auto;padding:16px;background:#fafafa;color:#222}section{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px;margin:14px 0}"
            "h2{font-size:1rem;margin:0 0 8px}small{color:#666;font-weight:normal}.r{display:flex;align-items:center;gap:10px;margin:4px 0}.r span{width:9em}.q{margin:6px 0}label{margin-right:12px}"
            "textarea{width:100%;height:9em}button{padding:8px 16px;margin:8px 0}</style>"
            "<h1>聴き比べ(順は伏せてある)</h1><p>各項目で、まず基準(目標の声)を聴き、次に A と B を聴いて、3 つの問いに答える。音量は揃えてある。数項目で傾向がはっきりしたら、途中で止めてよい。"
            "A / B の中身は key.json に在る(答え終わるまで開かない)。</p>" + body +
            '<button onclick="save()">答えを書き出す</button><textarea id="o" readonly></textarea>'
            "<script>function save(){const r={};document.querySelectorAll('input[type=radio]:checked').forEach(e=>r[e.name]=e.value);document.getElementById('o').value=JSON.stringify(r)}</script></html>")
    (out / "index.html").write_text(html)
    print("items", len(rows), "→", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
