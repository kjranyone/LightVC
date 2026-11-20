"""段 P56(paper_snv rev131・§9 O20 のオーナーの選択 (b)・評価の入力だけ): 本物の伴奏の上の本物の男声の語りの開発の物差し。
材料 = YODAS2(espnet/yodas2・ja000 = 投稿者の字幕つきの日本語・YouTube の CC BY の動画・24 kHz・data/yodas2_ja)。学習・目標にしない。話者の特定を試みない(動画の番号は束ねる鍵だけ)。字幕と書き起こしは保存するが出力に出さない。
screen = 字幕の区間(2〜15 s)ごとに AudioSet の分類器(MIT/ast-finetuned-audioset-10-10-0.4593・16 kHz)の Music・Speech・Singing・男の声・女の声の確率を測り、Speech ≥ 0.5 の区間から均等に 30 個まで前段 v3 の有声の高さの中央値を測る(data/yodas2_ja/screen.json)。
select = 動画の語り手が男声(男の声の確率の平均 > 女の声の確率の平均・高さの中央値 < 175 Hz)で、伴奏の区間(Music ≥ 0.3・Speech ≥ 0.5・Singing < 0.1・3〜12 s・その区間でも男の声 > 女の声)が 2 つ以上、他の区間(男の声 > 女の声)が 20 s 以上ある動画。
         伴奏の区間を large-v3 で書き起こし、字幕との仮名の CER ≤ 0.2 の区間を源にする(動画あたり 3 つまで)。登録 = 同じ動画の他の区間(長い順に合わせて 20 s 以上・8 つまで)。台帳 = data/yodas2_ja/ruler.json。
run = 系ごとに、源(伴奏の上の語りそのもの・足す音なし)と登録を 48 kHz で前段に通し、成績表と同じ目標(A0F の先頭 4 声から 2 声・k 0 と 7)へ変換し、源の書き起こしに対する CER を測る(results/snv_p56/yodas.json・数値だけ)。
    cd training && uv run --no-sync python snv_diag/p56_yodas.py screen [shards=00000000,00000001,00000002]
    cd training && uv run --no-sync python snv_diag/p56_yodas.py select
    cd training && OMP_NUM_THREADS=2 uv run --no-sync python snv_diag/p56_yodas.py run snv_p30_c snv_p30_c+mask:snv_p45_gate_t80 [rate=1.0] [out=results/snv_p56/yodas.json]
"""
from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

import librosa
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parents[2]
YD = ROOT / "data/yodas2_ja"
SR_Y = 24000
AST = "MIT/ast-finetuned-audioset-10-10-0.4593"
LABELS = {"music": "Music", "speech": "Speech", "singing": "Singing", "male": "Male speech, man speaking", "female": "Female speech, woman speaking"}
KS = (0, 7)


def segs_of(texts: dict) -> list[dict]:
    out = []
    for k in sorted(texts):
        parts = k.rsplit("-", 3)
        st, en = int(parts[2]) / 100.0, int(parts[3]) / 100.0
        out.append({"id": k, "st": st, "en": en, "dur": round(en - st, 2)})
    return out


def audio_of(tf: tarfile.TarFile, member: tarfile.TarInfo) -> np.ndarray:
    import soundfile as sf
    data = tf.extractfile(member).read()
    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32")
    except Exception:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=Path(member.name).suffix) as f:
            f.write(data)
            f.flush()
            x, sr = librosa.load(f.name, sr=None, mono=True)
    x = x.mean(1) if x.ndim > 1 else x
    return librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=SR_Y) if sr != SR_Y else x.astype(np.float32)


def wav_path(aid: str) -> Path:
    return YD / "wav" / f"{aid}.wav"


def peak_rate(y: np.ndarray) -> float:
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks
    import i13_bgm_dev_probe as B
    u = B.utt_of(librosa.resample(y.astype(np.float32), orig_sr=SR_Y, target_sr=B.SR).astype(np.float64))
    c0, v = u["y"][0] * 10, u["v"] > 0.5
    pk, _ = find_peaks(np.where(v, uniform_filter1d(c0, 5), -1e9), prominence=3.0, distance=12)
    return float(len(pk) / (v.sum() / 200.0 + 1e-9))


def screen(shards: list[str], out_name: str = "screen.json") -> int:
    import soundfile as sf
    import torch
    from transformers import ASTFeatureExtractor, ASTForAudioClassification
    import snv_p0_front as P
    fe = ASTFeatureExtractor.from_pretrained(AST)
    m = ASTForAudioClassification.from_pretrained(AST).to("cuda").eval()
    idx = {k: m.config.label2id[v] for k, v in LABELS.items()}
    out_p = YD / out_name
    res = json.loads(out_p.read_text()) if out_p.exists() else {}
    (YD / "wav").mkdir(parents=True, exist_ok=True)
    for sh in shards:
        texts = {d["audio_id"]: d["text"] for d in json.loads((YD / "text" / f"{sh}.json").read_text())}
        with tarfile.open(YD / "audio" / f"{sh}.tar.gz", "r:gz") as tf:
            for mem in tf:
                if not mem.isfile():
                    continue
                aid = Path(mem.name).stem
                if aid not in texts or aid in res:
                    continue
                x = audio_of(tf, mem)
                sf.write(wav_path(aid), x, SR_Y, subtype="PCM_16")
                rows = []
                for s in segs_of(texts[aid]):
                    if not 2.0 <= s["dur"] <= 15.0:
                        continue
                    y = x[int(s["st"] * SR_Y):int(s["en"] * SR_Y)]
                    if len(y) < SR_Y:
                        continue
                    y16 = librosa.resample(y, orig_sr=SR_Y, target_sr=16000)
                    with torch.no_grad():
                        pr = torch.sigmoid(m(fe([y16], sampling_rate=16000, return_tensors="pt")["input_values"].to("cuda")).logits)[0].float().cpu().numpy()
                    rows.append(s | {k: round(float(pr[i]), 4) for k, i in idx.items()} | {"f0": 0.0, "vfrac": 0.0})
                sp = [r for r in rows if r["speech"] >= 0.5]
                for r in sp[:: max(1, len(sp) // 30)][:30]:
                    y48 = librosa.resample(x[int(r["st"] * SR_Y):int(r["en"] * SR_Y)], orig_sr=SR_Y, target_sr=P.SR)
                    n = P.n_frames(len(y48))
                    f0 = P.analyze(y48[: n * P.H].astype(np.float32))["f0"]
                    v = f0 > 0
                    r["f0"], r["vfrac"] = (round(float(np.median(f0[v])), 1) if v.any() else 0.0), round(float(v.mean()), 3)
                res[aid] = {"shard": sh, "dur": round(len(x) / SR_Y, 2), "segs": rows}
                print("screened", len(res), "segs", len(rows), flush=True)
                out_p.write_text(json.dumps(res))
    out_p.write_text(json.dumps(res))
    print("done", len(res), flush=True)
    return 0


def select() -> int:
    import soundfile as sf
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, kana, to16
    sc = {}
    for q in sorted(YD.glob("screen*.json")):
        sc |= json.loads(q.read_text())
    texts = {}
    for p in sorted((YD / "text").glob("*.json")):
        texts |= {d["audio_id"]: d["text"] for d in json.loads(p.read_text())}
    asr = WhisperModel("large-v3", device="cuda", compute_type="float16")
    out, stat = [], {"videos": len(sc), "male": 0, "bgm2": 0, "enroll20": 0, "kept": 0, "src_tried": 0, "src_valid": 0}
    for aid in sorted(sc):
        segs = sc[aid]["segs"]
        if not segs:
            continue
        male = np.mean([s["male"] for s in segs]) > np.mean([s["female"] for s in segs])
        f0s = [s["f0"] for s in segs if s["f0"] > 0]
        if not (male and f0s and float(np.median(f0s)) < 175.0):
            continue
        stat["male"] += 1
        bgm = [s for s in segs if s["music"] >= 0.3 and s["speech"] >= 0.5 and s["singing"] < 0.1 and 3.0 <= s["dur"] <= 12.0 and s["male"] > s["female"]]
        if len(bgm) < 2:
            continue
        stat["bgm2"] += 1
        bgm = sorted(bgm, key=lambda s: -s["music"])
        x = sf.read(str(wav_path(aid)), dtype="float32")[0]
        src = []
        for s in bgm:
            if len(src) == 3:
                break
            y = x[int(s["st"] * SR_Y):int(s["en"] * SR_Y)].astype(np.float64)
            hyp = kana("".join(t.text for t in asr.transcribe(to16(y, SR_Y).astype(np.float32), language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)[0]))
            stat["src_tried"] += 1
            c = float(cer(kana(texts[aid][s["id"]]), hyp))
            if c <= 0.2 and len(hyp) >= 8:
                src.append(s | {"orig": hyp, "cap_cer": round(c, 4), "rate": round(peak_rate(y), 2)})
        stat["src_valid"] += len(src)
        used = {s["id"] for s in src}
        rest = sorted([s for s in segs if s["id"] not in used and s["dur"] >= 2.0 and s["male"] > s["female"]], key=lambda s: -s["dur"])
        enr, tot = [], 0.0
        for s in rest:
            if len(enr) == 8 or tot >= 20.0:
                break
            enr.append(s)
            tot += s["dur"]
        if tot < 20.0:
            continue
        stat["enroll20"] += 1
        if len(src) < 2:
            continue
        stat["kept"] += 1
        out.append({"video": aid, "src": src, "enroll": [{k: s[k] for k in ("id", "st", "en", "dur", "music")} for s in enr]})
    (YD / "ruler.json").write_text(json.dumps({"material": "YODAS2 ja000(espnet/yodas2・CC BY 3.0・評価の入力だけ)", "stat": stat, "videos": out}, ensure_ascii=False))
    print(json.dumps(stat, ensure_ascii=False), "源", sum(len(v["src"]) for v in out), flush=True)
    return 0


def run(tags: list[str], opt: dict) -> int:
    import soundfile as sf
    import i13_bgm_dev_probe as B
    import snv_p1_eval as EV
    from faster_whisper import WhisperModel
    from snv_a0_measure import cer, kana, to16
    rate = float(opt.get("rate", 1.0))
    if rate != 1.0:
        import i15_rate_probe as RP
    rl = json.loads((YD / "ruler.json").read_text())["videos"]
    asr = WhisperModel("large-v3", device=EV.DEV, compute_type="float16" if EV.DEV == "cuda" else "int8")

    def tr(w: np.ndarray) -> str:
        segs, _ = asr.transcribe(to16(w.astype(np.float64), EV.SR).astype(np.float32), language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(x.text for x in segs))

    gen = EV.load_gen()
    tgts = json.loads((EV.A1 / "convert_meta.json").read_text())["targets"][:4]
    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tgts}
    systems = [EV.System(t, gen) for t in tags]
    rows = []
    for i, vd in enumerate(rl):
        x = sf.read(str(wav_path(vd["video"])), dtype="float32")[0]
        cut = lambda s: librosa.resample(x[int(s["st"] * SR_Y):int(s["en"] * SR_Y)], orig_sr=SR_Y, target_sr=B.SR).astype(np.float64)
        fast = (lambda y: RP.wsola(y, rate)) if rate != 1.0 else (lambda y: y)
        enr = [B.utt_of(fast(cut(s))) for s in vd["enroll"]]
        tg = [tgts[i % 4], tgts[(i + 2) % 4]]
        for s in vd["src"]:
            xs = fast(cut(s))
            ref = s["orig"]
            u = B.utt_of(xs)
            for sy in systems:
                for v in tg:
                    for k in KS:
                        w = sy.convert(u, enr, sy.structure(f"t|{v}", trefs[v]), k)
                        rows.append({"video": vd["video"], "idx": i, "seg": s["id"], "music": s["music"], "sys": sy.tag, "tgt": v, "k": k, "cer": round(cer(ref, tr(w)), 4)})
        print("video", i, {t: round(float(np.mean([min(r["cer"], 1.0) for r in rows if r["idx"] == i and r["sys"] == t])), 3) for t in tags}, flush=True)
    out = ROOT / opt.get("out", "results/snv_p56/yodas.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"material": "YODAS2 ja000(CC BY 3.0・評価の入力だけ)", "rate": rate, "tags": tags, "n_videos": len(rl), "rows": rows}, ensure_ascii=False, indent=1))
    return 0


def main() -> int:
    opt = dict(x.split("=", 1) for x in sys.argv[2:] if "=" in x)
    mode = sys.argv[1]
    if mode == "screen":
        return screen(opt.get("shards", "00000000,00000001,00000002").split(","), opt.get("out", "screen.json"))
    if mode == "select":
        return select()
    return run([x for x in sys.argv[2:] if "=" not in x], opt)


if __name__ == "__main__":
    raise SystemExit(main())
