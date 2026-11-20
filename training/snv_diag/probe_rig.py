"""差し替えの探りの共通の道具(段 P18〜・学習なし): 系(snv_p1_eval.System)・large-v3 の仮名の書き起こし・ECAPA + LDA の目標の判定・A0F の目標 4 声の参照と、
源の組(devA の先頭 n 人 × s0・s1 / data/snv_d5 の生成した声の模型 / JVNV の男声 2 人)。p15_probe.py・p17_probe.py と同じ物差し。
各々の源の ema = 外の調音の物差し(SPARC の EMA・snv_diag/p18_sparc_feats.py)の npz の道 [源, 登録…](鎖を通した模型は、通す前の音の EMA)。
"""
from __future__ import annotations

import json
import sys
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
ORG = {"g0": ("g0", "c1", False), "w": ("w", "c2", False), "g0_ch": ("g0", "c1", True), "w_ch": ("w", "c2", True)}
SP = ROOT / "data/snv_e1_feat/sparc"


def cv_items(n: int = 64) -> list[dict]:
    dv = json.loads((ROOT / "results/snv_d5/dev_rows_devA.json").read_text())
    man = json.loads((ROOT / "data/cv22_ja/dev_manifest.json").read_text())
    ld = lambda sp, c: T.load_utt(T.FEAT / f"cv/{sp}_{c['name']}.npz")
    out = []
    for i, s in enumerate([x for x in man["speakers"] if x["split"] == "devA"][:n]):
        enr = [ld(s["speaker"], c) for c in s["enroll"]]
        for c in s["src"][:2]:
            src = f"{s['speaker']}_{c['name']}"
            out.append({"set": "cv", "spk": s["speaker"], "idx": i, "src": src, "u": ld(s["speaker"], c), "enr": enr, "ref": dv["ref"][src]["orig"], "valid": dv["ref"][src]["text_vs_orig"] <= 0.1,
                        "ema": [str(SP / f"cv/{src}.npz")] + [str(SP / f"cv/{s['speaker']}_{e['name']}.npz") for e in s["enroll"]]})
    return out


def org_items(kind: str) -> list[dict]:
    typ, cap, chain = ORG[kind]
    dm = json.loads((ROOT / "data/snv_d5/manifest.json").read_text())
    vs = [v for v in dm["voices"] if (v["type"], v["cap"]) == (typ, cap)]
    jobs = [(f"{v['voice']}|{t}", str(ROOT / f"data/snv_d5/{v['voice']}/{t}.wav"), v["voice"] if chain else None) for v in vs for t in ("t00", "t01", "t02")]
    with ProcessPoolExecutor(11) as ex:
        F = dict(ex.map(Q.wav_job, jobs, chunksize=2))
    return [{"set": f"org_{kind}", "spk": v["voice"], "idx": i, "src": f"{v['voice']}|{t}", "u": F[f"{v['voice']}|{t}"], "enr": [F[f"{v['voice']}|{o}"] for o in ("t00", "t01", "t02") if o != t],
             "path": str(ROOT / f"data/snv_d5/{v['voice']}/{t}.wav"), "valid": True, "ema": [str(SP / f"d5/{v['voice']}_{t}.npz")] + [str(SP / f"d5/{v['voice']}_{o}.npz") for o in ("t00", "t01", "t02") if o != t]}
            for i, v in enumerate(vs) for t in ("t01", "t02")]


def jv_items() -> list[dict]:
    jpath = lambda nm: str(ROOT / f"data/jvnv/jvnv_v1/{nm.split('_')[0]}/{nm.split('_')[1]}/regular/{nm}.wav")
    jobs = [(nm, jpath(nm), None) for sp in ("M1", "M2") for e in Q.EMOS for nm in (f"{sp}_{e}_regular_01", f"{sp}_{e}_regular_02")]
    with ProcessPoolExecutor(11) as ex:
        F = dict(ex.map(Q.wav_job, jobs, chunksize=2))
    return [{"set": "jv", "spk": sp, "idx": i, "src": f"{sp}_{e}_regular_01", "u": F[f"{sp}_{e}_regular_01"], "enr": [F[f"{sp}_{x}_regular_02"] for x in Q.EMOS], "path": jpath(f"{sp}_{e}_regular_01"), "valid": True,
             "ema": [str(SP / f"jvnv/{sp}_{e}_regular_01.npz")] + [str(SP / f"jvnv/{sp}_{x}_regular_02.npz") for x in Q.EMOS]} for i, sp in enumerate(("M1", "M2")) for e in Q.EMOS]


def ext_items(name: str) -> list[dict]:
    """段 P73: data/snv_p73/sets.json の組(声ごとに源と登録の音と EMA の道)。"""
    man = json.loads((ROOT / "data/snv_p73/sets.json").read_text())[name]
    jobs = sorted({(w["wav"], str(ROOT / w["wav"]), None) for v in man for w in v["src"] + v["enr"]})
    with ProcessPoolExecutor(11) as ex:
        F = dict(ex.map(Q.wav_job, jobs, chunksize=2))
    return [{"set": f"x:{name}", "spk": v["voice"], "idx": i, "src": w["wav"], "u": F[w["wav"]], "enr": [F[e["wav"]] for e in v["enr"]], "path": str(ROOT / w["wav"]), "valid": True,
             "ema": [str(ROOT / w["ema"])] + [str(ROOT / e["ema"]) for e in v["enr"]]} for i, v in enumerate(man) for w in v["src"]]


def targets() -> tuple[list, list]:
    import snv_p1_eval as EV
    all_t = json.loads((EV.A1 / "convert_meta.json").read_text())["targets"]
    return all_t, all_t[:4]


class Rig:
    def __init__(self, tag: str) -> None:
        import snv_p1_eval as EV
        from faster_whisper import WhisperModel
        from snv_a0_measure import kana, load, to16
        from snv_a0b_measure import proj
        from snv_n0_measure import trim
        from speechbrain.inference.speaker import EncoderClassifier
        self.EV, self.kana, self.trim, self.proj, self.to16 = EV, kana, trim, proj, to16
        self.asr = WhisperModel("large-v3", device=EV.DEV, compute_type="float16" if EV.DEV == "cuda" else "int8")
        self.ec = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir=str(ROOT / "training/hf_models/spkrec-ecapa"), run_opts={"device": EV.DEV})
        self.Tz = np.load(ROOT / "data/snv_e1_feat/ecapa_lda_T.npz")
        self.all_t, self.tg4 = targets()
        self.Et = {v: self.emb(np.concatenate([to16(*load(EV.A0 / f"wav/{v}_{j}.wav")).astype(np.float32) for j in (0, 1, 2)])) for v in self.all_t}
        self.trefs = {v: [EV.utt(f"a0/{v}_{j}") for j in (0, 1, 2)] for v in self.tg4}
        self.s = EV.System(tag, EV.load_gen())
        self.rcache: dict = {}

    def w16(self, w: np.ndarray) -> np.ndarray:
        return self.to16(w.astype(np.float64), SR).astype(np.float32)

    def tr(self, y16: np.ndarray) -> str:
        segs, _ = self.asr.transcribe(y16, language="ja", beam_size=5, temperature=0.0, condition_on_previous_text=False, vad_filter=False)
        return self.kana("".join(x.text for x in segs))

    def emb(self, y16: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = self.ec.encode_batch(torch.from_numpy(y16).float()[None].to(self.EV.DEV)).reshape(-1)
        return self.proj(torch.nn.functional.normalize(e, dim=0).cpu().numpy(), (self.Tz["mu"], self.Tz["scalings"]))

    def top(self, y16: np.ndarray) -> str:
        e = self.emb(self.trim(y16))
        return max(self.all_t, key=lambda u: float(e @ self.Et[u]))

    def ref_of(self, it: dict) -> str:
        if "ref" not in it:
            if it["path"] not in self.rcache:
                import librosa
                self.rcache[it["path"]] = self.tr(self.w16(librosa.load(it["path"], sr=SR, mono=True)[0]))
            it["ref"] = self.rcache[it["path"]]
        return it["ref"]

    def assigned(self, it: dict) -> tuple:
        return self.tg4[it["idx"] % 4], self.tg4[(it["idx"] + 2) % 4]

    def row(self, it: dict, w: np.ndarray, **extra) -> dict:
        y16 = self.w16(w)
        o = self.tr(y16)
        ref = self.ref_of(it)
        sb, dl, ins = Q.ops(ref, o)
        r = {"set": it["set"], "spk": it["spk"], "src": it["src"], "valid": it["valid"], "cer": round((sb + dl + ins) / max(len(ref), 1), 4), "sub": sb, "del": dl, "ins": ins, "n_ref": len(ref), "top": self.top(y16)} | extra
        return r | {"out": o} if it["set"] == "cv" else r
