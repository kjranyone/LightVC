"""E1 eval — intelligibility (Whisper CER) + speaker leak (ECAPA) + render.

CER: transcribe gt with Whisper (reference), transcribe e1<mode> output, CER =
char edit distance / len(ref). This is the proper intelligibility gate (does NOT
diverge from the ear like harmonic/valley proxies). ceiling = freebig(mel_of gt)
gives the vocoder's own CER floor.

Leak: cross-VC — content from held speaker A, timbre z_spk from a DIFFERENT
speaker B. ECAPA cos(out, A) = source leak (want LOW with scrub); cos(out, B) =
target match. Higher scrub should push cos(out,A) down.

render -> results/e1/ (gt / ceiling / e1<mode>, self-recon held-out).
"""
from __future__ import annotations

import sys
import argparse
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import librosa
import jiwer

sys.path.insert(0, str(Path(__file__).parent))
from train_m1 import mel_of, SR, HOP, DEV
from train_m2 import TimbreEncoder, ContentScrub, TIMBRE_DIM, ART_DIM
from mel_gen import MelGen
from train_z1 import load_freebig

CER_TRANS = jiwer.Compose([jiwer.RemoveWhiteSpace(replace_by_space=False)])


def load_wav(path):
    w, _ = sf.read(path, dtype="float32")
    return w.mean(1) if w.ndim > 1 else w


def build_cond(d):
    content = d["content"].float(); f0 = d["f0"].float(); energy = d["energy"].float()
    tmel = f0.shape[0]
    c = F.interpolate(content.t().unsqueeze(0), size=tmel, mode="linear",
                      align_corners=False).squeeze(0)
    logf0 = torch.log(f0.clamp(min=1.0)) / 7.0
    eng = torch.log(energy.clamp(min=1e-4)) * 0.2
    return torch.cat([c, logf0.unsqueeze(0), eng.unsqueeze(0)], dim=0), tmel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/e1_none/last.pt")
    ap.add_argument("--feat", default="../data/rcav_feat")
    ap.add_argument("--freebig", default="checkpoints/freebig/foundation_bigvgan_parity.pt")
    ap.add_argument("--out", default="../results/e1")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--n-leak", type=int, default=12)
    ap.add_argument("--whisper", default="large-v3")
    ap.add_argument("--write-audio", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    ck = torch.load(args.ckpt, map_location=DEV, weights_only=False)
    ga = ck.get("args", {"dim": 384, "layers": 6})
    mode = ck.get("scrub_mode", "none")
    g = MelGen(cond_dim=768 + 2, dim=ga["dim"], n_layers=ga["layers"],
               timbre_dim=TIMBRE_DIM, art_dim=ART_DIM).to(DEV)
    g.load_state_dict(ck["g"]); g.eval()
    t = TimbreEncoder().to(DEV); t.load_state_dict(ck["t"]); t.eval()
    scrub = None
    if "scrub" in ck:
        scrub = ContentScrub().to(DEV); scrub.load_state_dict(ck["scrub"]); scrub.eval()
    voc, voc_step = load_freebig(args.freebig)

    held = [Path(x) for x in (Path(args.ckpt).parent / "heldout.txt").read_text().split("\n") if x]
    random.shuffle(held)
    print(f"E1 eval | mode {mode} | step {ck.get('step')} | held {len(held)} | whisper {args.whisper}", flush=True)

    from faster_whisper import WhisperModel
    wm = WhisperModel(args.whisper, device="cuda", compute_type="float16")

    def transcribe(w44):
        w16 = librosa.resample(w44, orig_sr=SR, target_sr=16000)
        segs, _ = wm.transcribe(w16.astype(np.float32), language="ja", beam_size=1)
        return "".join(s.text for s in segs).strip()

    @torch.no_grad()
    def synth(d, z_spk):
        cond, tmel = build_cond(d); cond = cond.unsqueeze(0).to(DEV)
        cc = scrub(cond[:, :768]) if scrub is not None else cond[:, :768]
        cond2 = torch.cat([cc, cond[:, 768:]], dim=1)
        return voc(g(cond2, z_spk, None))[0].cpu().numpy(), tmel

    @torch.no_grad()
    def zsp(wav):
        return t(mel_of(torch.from_numpy(np.ascontiguousarray(wav)).float().unsqueeze(0).to(DEV)))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cers_out, cers_ceil = [], []
    written = 0
    for f in held:
        if len(cers_out) >= args.n:
            break
        d = torch.load(f, weights_only=False)
        if "f0" not in d or "energy" not in d:
            continue
        y = load_wav(d["path"])
        t_gt = transcribe(y)
        if len(t_gt) < 2:
            continue
        with torch.no_grad():
            z = zsp(y[: int(3 * SR)])
            w_out, tmel = synth(d, z)
            y_c = y[: tmel * HOP]
            w_ceil = voc(mel_of(torch.from_numpy(np.ascontiguousarray(y_c)).float().unsqueeze(0).to(DEV)))[0].cpu().numpy()
        t_out = transcribe(w_out); t_ceil = transcribe(w_ceil)
        cers_out.append(jiwer.cer(CER_TRANS(t_gt), CER_TRANS(t_out)))
        cers_ceil.append(jiwer.cer(CER_TRANS(t_gt), CER_TRANS(t_ceil)))
        if written < args.write_audio:
            stem = f"{written:02d}_{f.parent.name}_{f.stem}"
            sf.write(out / f"{stem}_gt.wav", np.clip(y_c, -1, 1), SR, subtype="PCM_16")
            sf.write(out / f"{stem}_ceiling.wav", np.clip(w_ceil, -1, 1), SR, subtype="PCM_16")
            sf.write(out / f"{stem}_e1{mode}.wav", np.clip(w_out, -1, 1), SR, subtype="PCM_16")
            written += 1

    # speaker leak (cross-VC): content A, timbre B
    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                           savedir="hf_models/spkrec-ecapa",
                                           run_opts={"device": "cpu"})

    def emb(w44):
        w16 = librosa.resample(w44, orig_sr=SR, target_sr=16000)
        e = ecapa.encode_batch(torch.from_numpy(w16).float().unsqueeze(0)).squeeze().detach()
        return e / (e.norm() + 1e-6)

    by_spk = defaultdict(list)
    for f in held:
        by_spk[f.parent.name].append(f)
    spks = list(by_spk)
    leakA, matchB = [], []
    for f in held:
        if len(leakA) >= args.n_leak:
            break
        spkA = f.parent.name
        others = [s for s in spks if s != spkA and by_spk[s]]
        if not others:
            continue
        dA = torch.load(f, weights_only=False)
        if "f0" not in dA or "energy" not in dA:
            continue
        fB = random.choice(by_spk[random.choice(others)])
        dB = torch.load(fB, weights_only=False)
        wB = load_wav(dB["path"])
        with torch.no_grad():
            zB = zsp(wB[: int(3 * SR)])
            w_out, _ = synth(dA, zB)
        eo, ea, eb = emb(w_out), emb(load_wav(dA["path"])), emb(wB)
        leakA.append(float((eo * ea).sum())); matchB.append(float((eo * eb).sum()))

    print(f"\n=== E1 {mode} (step {ck.get('step')}) | {len(cers_out)} CER utts, {len(leakA)} leak pairs ===")
    print(f"  CER  ceiling {np.mean(cers_ceil):.3f}   e1{mode} {np.mean(cers_out):.3f}   (lower=more intelligible)")
    print(f"  leak(cross): cos(out, SOURCE-A) {np.mean(leakA):+.3f}  cos(out, TARGET-B) {np.mean(matchB):+.3f}  (want A low / B high)")
    print(f"  render -> {out} ({written} trios)", flush=True)


if __name__ == "__main__":
    main()
