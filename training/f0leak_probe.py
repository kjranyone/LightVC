"""Single-hypothesis probe: is the cross-VC speaker leak driven by SOURCE F0
register (not by content)? Re-synth each cross pair two ways on the SAME
e1_none G: (i) source-A F0 (baseline), (ii) A's logF0 median-shifted to B's
register. ECAPA cos(out,A)=source leak, cos(out,B)=target match. If (ii) drops
leakA / raises matchB -> F0 register is the leak channel (=> srcshift). If flat
-> timbre encoder too weak (=> output-side ECAPA/CIPT). Anchors: raw cos(A,B)
between-speaker floor, cos(A,A') same-speaker ceiling.
"""
from __future__ import annotations
import sys, random, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np, torch, torch.nn.functional as F, soundfile as sf, librosa

sys.path.insert(0, str(Path(__file__).parent))
from train_m1 import mel_of, SR, HOP, DEV
from train_m2 import TimbreEncoder, TIMBRE_DIM, ART_DIM
from mel_gen import MelGen
from train_z1 import load_freebig


def load_wav(p):
    w, _ = sf.read(p, dtype="float32")
    return w.mean(1) if w.ndim > 1 else w


def vmedian_logf0(f0):
    v = f0[f0 > 50.0]
    return float(np.log(np.clip(v, 1.0, None)).mean()) if v.size else np.log(150.0)


def build_cond(d, logf0_shift=0.0):
    content = d["content"].float(); f0 = d["f0"].float(); energy = d["energy"].float()
    tmel = f0.shape[0]
    c = F.interpolate(content.t().unsqueeze(0), size=tmel, mode="linear",
                      align_corners=False).squeeze(0)
    lf = torch.log(f0.clamp(min=1.0))
    voiced = f0 > 50.0
    lf = lf + logf0_shift * voiced.float()          # shift register (voiced only)
    logf0 = (lf / 7.0)
    eng = torch.log(energy.clamp(min=1e-4)) * 0.2
    return torch.cat([c, logf0.unsqueeze(0), eng.unsqueeze(0)], dim=0), tmel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/e1_none/last.pt")
    ap.add_argument("--freebig", default="checkpoints/freebig/foundation_bigvgan_parity.pt")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    random.seed(args.seed)

    ck = torch.load(args.ckpt, map_location=DEV, weights_only=False)
    ga = ck.get("args", {"dim": 384, "layers": 6})
    g = MelGen(cond_dim=770, dim=ga["dim"], n_layers=ga["layers"],
               timbre_dim=TIMBRE_DIM, art_dim=ART_DIM).to(DEV)
    g.load_state_dict(ck["g"]); g.eval()
    t = TimbreEncoder().to(DEV); t.load_state_dict(ck["t"]); t.eval()
    voc, _ = load_freebig(args.freebig)

    held = [Path(x) for x in (Path(args.ckpt).parent / "heldout.txt").read_text().split("\n") if x]
    held = [f for f in held if f.exists()]
    by_spk = defaultdict(list)
    for f in held:
        by_spk[f.parent.name].append(f)
    spks = [s for s in by_spk if by_spk[s]]

    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                           savedir="hf_models/spkrec-ecapa",
                                           run_opts={"device": "cpu"})

    def emb(w44):
        w16 = librosa.resample(w44, orig_sr=SR, target_sr=16000)
        e = ecapa.encode_batch(torch.from_numpy(w16).float().unsqueeze(0)).squeeze().detach()
        return e / (e.norm() + 1e-6)

    @torch.no_grad()
    def zsp(w):
        return t(mel_of(torch.from_numpy(np.ascontiguousarray(w)).float().unsqueeze(0).to(DEV)))

    @torch.no_grad()
    def synth(dA, zB, shift):
        cond, _ = build_cond(dA, shift); cond = cond.unsqueeze(0).to(DEV)
        return voc(g(cond, zB, None))[0].cpu().numpy()

    random.shuffle(held)
    base = {"leakA": [], "matchB": []}
    shft = {"leakA": [], "matchB": []}
    raw_AB, raw_AA = [], []
    dreg = []
    for f in held:
        if len(base["leakA"]) >= args.n:
            break
        spkA = f.parent.name
        others = [s for s in spks if s != spkA]
        if not others:
            continue
        dA = torch.load(f, weights_only=False)
        if "f0" not in dA:
            continue
        fB = random.choice(by_spk[random.choice(others)])
        dB = torch.load(fB, weights_only=False)
        if "f0" not in dB:
            continue
        wA, wB = load_wav(dA["path"]), load_wav(dB["path"])
        medA, medB = vmedian_logf0(dA["f0"].numpy()), vmedian_logf0(dB["f0"].numpy())
        shift = medB - medA
        dreg.append(shift * 12 / np.log(2))          # semitones
        zB = zsp(wB[: int(3 * SR)])
        eA, eB = emb(wA), emb(wB)
        for tag, sh in (("base", 0.0), ("shift", shift)):
            wo = synth(dA, zB, sh); eo = emb(wo)
            dd = base if tag == "base" else shft
            dd["leakA"].append(float((eo * eA).sum())); dd["matchB"].append(float((eo * eB).sum()))
        raw_AB.append(float((eA * eB).sum()))
        pool = [x for x in by_spk[spkA] if x != f]
        if pool:
            eA2 = emb(load_wav(torch.load(random.choice(pool), weights_only=False)["path"]))
            raw_AA.append(float((eA * eA2).sum()))

    m = lambda x: float(np.mean(x)) if x else float("nan")
    print(f"\n=== F0-register leak probe | e1_none step {ck.get('step')} | n={len(base['leakA'])} ===")
    print(f"  anchors: raw cos(A,B) between-spk {m(raw_AB):+.3f} | cos(A,A') same-spk {m(raw_AA):+.3f} "
          f"| median register shift {m(dreg):+.1f} semitones")
    print(f"  BASE  (source F0):   leakA {m(base['leakA']):+.3f}  matchB {m(base['matchB']):+.3f}  "
          f"margin(A-B) {m(base['leakA'])-m(base['matchB']):+.3f}")
    print(f"  SHIFT (F0->B reg):   leakA {m(shft['leakA']):+.3f}  matchB {m(shft['matchB']):+.3f}  "
          f"margin(A-B) {m(shft['leakA'])-m(shft['matchB']):+.3f}")
    print(f"  => dLeakA {m(shft['leakA'])-m(base['leakA']):+.3f}  dMatchB {m(shft['matchB'])-m(base['matchB']):+.3f} "
          f"(F0 leak channel if dLeakA<0 & dMatchB>0)", flush=True)


if __name__ == "__main__":
    main()
