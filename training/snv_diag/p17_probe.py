"""段 P17(results/snv_p17/prereg.yaml・学習なし): 開発の組の録音の形式(約 11 kHz の帯域の切り・MP3 32 kHz 48 kb/s)が、白くした演奏の声を跨いだ比べを壊すかの差し替えの探り。
乱し: bw11・bw8 = 位相を動かさない帯域の切り(切りの幅 400 Hz)/ mp3 = ffmpeg の libmp3lame で 32 kHz・48 kb/s に符号化して戻す(ずれは相互相関で合わせる)/ id = 何もしない。
乱した音は 16 bit に丸める(開発の組の wav と同じ = 切った帯域には量子化の雑音の平らな床が残る・開発の組の実測は 1〜4 kHz 比 −65〜−80 dB)。
条件:
  族の中に足す = org_g0_{id,bw11,bw8,mp3}: data/snv_d5 の対照の声 g0_c1(源 t01・t02・登録 = 残りの 2 発話)の源と登録を乱す。自己(k 0・tgt なし)と他の声の構造(k 7・割り当ての目標 2 声)。
               jv_{bw11,mp3}: JVNV の男声 2 人(源 regular_01・登録 regular_02 の 6 感情)の源と登録を乱す(k 7)。
  鍵を合わせる = 名前の末尾 _k: 目標の構造の鍵だけを、同じ乱しを通した目標の参照の演奏から作る(値は加工しない Y)。org_g0_{bw11,bw8,mp3}_k・jv_{bw11,mp3}_k が道具の較正。
               kbw11・kmp3: devA の源と登録はそのまま・鍵を bw11 / mp3 に。bw8 = devA の源と登録を bw8 に(鍵はそのまま)。bw8_k8 = 源・登録・鍵を全て bw8 に。
CER = large-v3 の仮名で、源の元の音の書き起こし 対 出力。M = 出力の ECAPA + LDA が A0F の 24 声の中で目標を 1 位に当てるか。基準の行は results/snv_p15/rows.json(org_g0・jv6・base4)。
    cd training && OMP_NUM_THREADS=1 uv run --no-sync python snv_diag/p17_probe.py [conds=...] [n=64] [sys=snv_p8_wst] [out=results/snv_p17/rows.json] [redo=1] [selftest]
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p15_probe as Q
import snv_p0_front as P
import train_snv_p1 as T

ROOT = Path(__file__).resolve().parents[2]
SR = P.SR
ORG = {"org_g0_id": ("id", None), "org_g0_bw11": ("bw11", None), "org_g0_bw8": ("bw8", None), "org_g0_mp3": ("mp3", None),
       "org_g0_bw11_k": ("bw11", "bw11"), "org_g0_bw8_k": ("bw8", "bw8"), "org_g0_mp3_k": ("mp3", "mp3")}
JV = {"jv_bw11": ("bw11", None), "jv_mp3": ("mp3", None), "jv_bw11_k": ("bw11", "bw11"), "jv_mp3_k": ("mp3", "mp3")}
CV = {"kbw11": (None, "bw11"), "kmp3": (None, "mp3"), "bw8": ("bw8", None), "bw8_k8": ("bw8", "bw8")}
CONDS = tuple(ORG) + tuple(JV) + tuple(CV)
BW = {"bw11": 11000.0, "bw8": 8000.0}


def brick(x: np.ndarray, fc: float, tw: float = 400.0) -> np.ndarray:
    n = len(x)
    f = np.fft.rfftfreq(n, 1.0 / SR)
    H = 0.5 - 0.5 * np.cos(np.pi * np.clip((fc + tw / 2 - f) / tw, 0.0, 1.0))
    return np.fft.irfft(np.fft.rfft(x.astype(np.float64)) * H, n).astype(np.float32)


def mp3(x: np.ndarray) -> np.ndarray:
    import librosa
    import soundfile as sf
    with tempfile.TemporaryDirectory() as d:
        sf.write(f"{d}/a.wav", np.clip(x, -1.0, 1.0), SR, subtype="PCM_16")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{d}/a.wav", "-ar", "32000", "-ac", "1", "-c:a", "libmp3lame", "-b:a", "48k", f"{d}/a.mp3"], check=True)
        y = librosa.load(f"{d}/a.mp3", sr=SR, mono=True)[0].astype(np.float64)
    n = len(x)
    N = 1 << int(np.ceil(np.log2(n + len(y))))
    c = np.fft.irfft(np.fft.rfft(y, N) * np.conj(np.fft.rfft(brick(x, 10000.0).astype(np.float64), N)), N)
    lags = np.concatenate([np.arange(0, 4801), np.arange(-4800, 0)])
    lag = int(lags[np.argmax(c[lags])])
    y = y[lag:] if lag >= 0 else np.concatenate([np.zeros(-lag), y])
    return np.concatenate([y, np.zeros(max(0, n - len(y)))])[:n].astype(np.float32)


def degrade(x: np.ndarray, deg: str | None) -> np.ndarray:
    if deg in (None, "id"):
        return x
    y = mp3(x) if deg == "mp3" else brick(x, BW[deg])
    return (np.round(np.clip(y, -1.0, 1.0) * 32767.0) / 32767.0).astype(np.float32)


def deg_job(job: tuple) -> tuple:
    import librosa
    key, path, deg = job
    torch.set_num_threads(1)
    return (deg, key), Q.feat(degrade(librosa.load(path, sr=SR, mono=True)[0].astype(np.float32), deg))


def selftest() -> None:
    import librosa
    from scipy.signal import welch
    x = librosa.load(str(sorted((ROOT / "data/snv_d5").glob("*_g0_c1/t01.wav"))[0]), sr=SR, mono=True)[0].astype(np.float32)
    assert np.abs(brick(x, 30000.0) - x).max() < 1e-5
    lv = lambda y, lo, hi: (lambda f, p: float(10 * np.log10(p[(f >= lo) & (f < hi)].mean() + 1e-20) - 10 * np.log10(p[(f >= 1000) & (f < 4000)].mean() + 1e-20)))(*welch(y, SR, nperseg=4096, noverlap=2048))
    for d in ("id", "bw11", "bw8", "mp3"):
        y = degrade(x, d)
        assert len(y) == len(x)
        lo = brick(x, 6000.0)
        r = float(np.corrcoef(brick(y, 6000.0), lo)[0, 1])
        print(d, "6〜8 kHz", round(lv(y, 6000, 8000), 1), "8.5〜10.5", round(lv(y, 8500, 10500), 1), "11.5〜13", round(lv(y, 11500, 13000), 1), "13〜16", round(lv(y, 13000, 16000), 1), "6 kHz 未満の相関", round(r, 4), flush=True)
        assert r > (0.9 if d == "mp3" else 0.999), (d, r)
    a, b = Q.feat(x), Q.feat(degrade(x, "mp3"))
    assert a["y"].shape == b["y"].shape
    print("selftest ok", "フレーム", a["y"].shape[1], "有声の一致(mp3)", round(float(np.mean(a["v"] == b["v"])), 3))


def main() -> int:
    if "selftest" in sys.argv[1:]:
        selftest()
        return 0
    opt = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    n, tag = int(opt.get("n", 64)), opt.get("sys", "snv_p8_wst")
    out = ROOT / opt.get("out", "results/snv_p17/rows.json")
    res = json.loads(out.read_text()) if out.exists() else {"prereg": "results/snv_p17/prereg.yaml", "sys": tag, "n": n, "rows": {}, "summary": {}}
    assert res["sys"] == tag, (res["sys"], tag)
    conds = [c for c in opt.get("conds", ",".join(CONDS)).split(",") if c not in res["rows"] or opt.get("redo") == "1"]
    assert all(c in CONDS for c in conds), conds
    if not conds:
        print(json.dumps(res["summary"], ensure_ascii=False, indent=1))
        return 0
    import snv_p1_eval as EV
    dv = json.loads((ROOT / "results/snv_d5/dev_rows_devA.json").read_text())
    man = json.loads((ROOT / "data/cv22_ja/dev_manifest.json").read_text())
    dm = json.loads((ROOT / "data/snv_d5/manifest.json").read_text())
    all_t = json.loads((EV.A1 / "convert_meta.json").read_text())["targets"]
    tg4 = all_t[:4]
    ld = lambda sp, c: T.load_utt(T.FEAT / f"cv/{sp}_{c['name']}.npz")
    cv = []
    for i, s in enumerate([x for x in man["speakers"] if x["split"] == "devA"][:n]):
        for c in s["src"][:2]:
            src = f"{s['speaker']}_{c['name']}"
            cv.append({"set": "cv", "spk": s["speaker"], "idx": i, "src": src, "ref": dv["ref"][src]["orig"], "names": [src] + [f"{s['speaker']}_{e['name']}" for e in s["enroll"]],
                       "u": ld(s["speaker"], c), "enr": [ld(s["speaker"], e) for e in s["enroll"]]})
    vs = [v for v in dm["voices"] if (v["type"], v["cap"]) == ("g0", "c1")]
    jpath = lambda nm: str(ROOT / f"data/jvnv/jvnv_v1/{nm.split('_')[0]}/{nm.split('_')[1]}/regular/{nm}.wav")
    jobs: set = set()
    for c in conds:
        sd, kd = (ORG | JV | CV)[c]
        if c in ORG:
            jobs |= {(f"{v['voice']}|{t}", str(ROOT / f"data/snv_d5/{v['voice']}/{t}.wav"), sd) for v in vs for t in ("t00", "t01", "t02")}
        if c in JV:
            jobs |= {(nm, jpath(nm), sd) for sp in ("M1", "M2") for e in Q.EMOS for nm in (f"{sp}_{e}_regular_01", f"{sp}_{e}_regular_02")}
        if c in CV and sd is not None:
            jobs |= {(k, str(T.FEAT / f"cv/{k}.wav"), sd) for it in cv for k in it["names"]}
        if kd is not None:
            jobs |= {(f"{v}_{j}", str(EV.A0 / f"wav/{v}_{j}.wav"), kd) for v in tg4 for j in (0, 1, 2)}
    with ProcessPoolExecutor(11) as ex:
        F = dict(ex.map(deg_job, sorted(jobs), chunksize=4))
    print("乱した特徴", len(F), flush=True)

    from faster_whisper import WhisperModel
    from snv_a0_measure import kana, load, to16
    from snv_a0b_measure import proj
    from snv_n0_measure import trim
    from speechbrain.inference.speaker import EncoderClassifier
    import librosa
    DEV = EV.DEV
    asr = WhisperModel("large-v3", device=DEV, compute_type="float16" if DEV == "cuda" else "int8")
    w16 = lambda w: to16(w.astype(np.float64), SR).astype(np.float32)

    def tr(y16: np.ndarray) -> str:
        segs, _ = asr.transcribe(y16, language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return kana("".join(x.text for x in segs))

    ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": DEV})
    Tz = np.load(ROOT / "data/snv_e1_feat/ecapa_lda_T.npz")

    def emb(y16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = ec.encode_batch(torch.from_numpy(y16).float()[None].to(DEV)).reshape(-1)
        return proj(torch.nn.functional.normalize(e, dim=0).cpu().numpy(), (Tz["mu"], Tz["scalings"]))

    Et = {v: emb(np.concatenate([to16(*load(EV.A0 / f"wav/{v}_{j}.wav")).astype(np.float32) for j in (0, 1, 2)])) for v in all_t}
    hit = lambda y16, v: int(max(all_t, key=lambda u: float(emb(trim(y16)) @ Et[u])) == v)
    trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in tg4}
    s = EV.System(tag, EV.load_gen())
    assert s.p3, "この探りは P3 / P8 の型(演奏 = Pl)の系だけ"
    rcache: dict = {}

    def ref_of(path: str) -> str:
        if path not in rcache:
            rcache[path] = tr(w16(librosa.load(path, sr=SR, mono=True)[0]))
        return rcache[path]

    @torch.no_grad()
    def struct(v: str, kd: str | None) -> dict:
        if kd is None:
            return s.structure(f"t|{v}|3", trefs[v])
        key = f"tk|{v}|{kd}"
        if key not in s.cache:
            deg = [F[(kd, f"{v}_{j}")] for j in (0, 1, 2)]
            assert all(a["y"].shape == b["y"].shape for a, b in zip(trefs[v], deg))
            ry, _, rm, _ = EV.struct_inputs(trefs[v])
            dy, dvv, dmk, _ = EV.struct_inputs(deg)
            s.cache[key] = s.C.structure(ry, s.Pl(dy, dvv, s.Pl.mean_of(dy, dvv, dmk)), rm)
        return s.cache[key]

    def trial(cond: str, it: dict, u: dict, enr: list, v: str | None, S: dict, k: int) -> dict:
        y16 = w16(s.convert(u, enr, S, k))
        o = tr(y16)
        sb, dl, ins = Q.ops(it["ref"], o)
        r = {"cond": cond, "set": it["set"], "spk": it["spk"], "src": it["src"], "tgt": v, "k": k, "cer": round((sb + dl + ins) / max(len(it["ref"]), 1), 4), "sub": sb, "del": dl, "ins": ins, "n_ref": len(it["ref"])}
        if v is not None:
            r["m"] = hit(y16, v)
        return r | {"out": o} if it["set"] == "cv" else r

    for cond in conds:
        sd, kd = (ORG | JV | CV)[cond]
        if cond in ORG:
            its = [{"set": cond, "spk": v["voice"], "idx": i, "src": f"{v['voice']}|{t}", "u": F[(sd, f"{v['voice']}|{t}")], "enr": [F[(sd, f"{v['voice']}|{o}")] for o in ("t00", "t01", "t02") if o != t],
                    "ref": ref_of(str(ROOT / f"data/snv_d5/{v['voice']}/{t}.wav"))} for i, v in enumerate(vs) for t in ("t01", "t02")]
        elif cond in JV:
            its = [{"set": cond, "spk": sp, "idx": i, "src": f"{sp}_{e}_regular_01", "u": F[(sd, f"{sp}_{e}_regular_01")], "enr": [F[(sd, f"{sp}_{x}_regular_02")] for x in Q.EMOS],
                    "ref": ref_of(jpath(f"{sp}_{e}_regular_01"))} for i, sp in enumerate(("M1", "M2")) for e in Q.EMOS]
        else:
            its = [it | ({"u": F[(sd, it["names"][0])], "enr": [F[(sd, k)] for k in it["names"][1:]]} if sd is not None else {}) for it in cv]
        rows = []
        for it in its:
            if cond in ORG and kd is None:
                rows.append(trial(cond, it, it["u"], it["enr"], None, s.structure(f"self|{cond}|{it['spk']}|{it['src']}", it["enr"]), 0))
            for v in (tg4[it["idx"] % 4], tg4[(it["idx"] + 2) % 4]):
                rows.append(trial(cond, it, it["u"], it["enr"], v, struct(v, kd), 7))
        res["rows"][cond] = rows
        X = [r for r in rows if r["tgt"] is not None]
        res["summary"][cond] = Q.summarize(X) | ({"self": Q.summarize([r for r in rows if r["tgt"] is None])["all"]} if len(X) < len(rows) else {})
        print(cond, json.dumps(res["summary"][cond], ensure_ascii=False), flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
