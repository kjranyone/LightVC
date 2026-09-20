"""H2 probe: does the timbre AdaIN inject the TARGET's identity, or a generic one?
Same A-content, two timbre refs on the SAME e1_none G:
  out_B  = G(A_content, timbre=B)   (B = different speaker)
  out_A2 = G(A_content, timbre=A')  (A' = another utt of A)
If timbre carries target identity, swapping the ref moves output identity toward
that ref:  dTowardB = cos(out_B,B) - cos(out_A2,B) > 0
           dTowardA = cos(out_A2,A) - cos(out_B,A) > 0
~0 => AdaIN injects generic identity (timbre encoder is the weak link, => CIPT
output-side identity supervision). Anchors: raw between/same-speaker cos.
"""
from __future__ import annotations
import sys, random, argparse
from pathlib import Path
from collections import defaultdict
import numpy as np, torch, soundfile as sf, librosa

sys.path.insert(0, str(Path(__file__).parent))
from train_m1 import mel_of, SR, HOP, DEV
from train_m2 import TimbreEncoder, TIMBRE_DIM, ART_DIM
from mel_gen import MelGen
from train_z1 import load_freebig
from f0leak_probe import load_wav, build_cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/e1_none/last.pt")
    ap.add_argument("--freebig", default="checkpoints/freebig/foundation_bigvgan_parity.pt")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2)
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
    spks = [s for s in by_spk if len(by_spk[s]) >= 2]

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
    def synth(dA, zref):
        cond, _ = build_cond(dA, 0.0); cond = cond.unsqueeze(0).to(DEV)
        return voc(g(cond, zref, None))[0].cpu().numpy()

    random.shuffle(held)
    oB_B, oB_A, oA2_B, oA2_A, rawAB, rawAA = [], [], [], [], [], []
    for f in held:
        if len(oB_B) >= args.n:
            break
        spkA = f.parent.name
        if len(by_spk[spkA]) < 2:
            continue
        others = [s for s in spks if s != spkA]
        if not others:
            continue
        dA = torch.load(f, weights_only=False)
        if "f0" not in dA:
            continue
        fA2 = random.choice([x for x in by_spk[spkA] if x != f])
        fB = random.choice(by_spk[random.choice(others)])
        dA2, dB = torch.load(fA2, weights_only=False), torch.load(fB, weights_only=False)
        wA, wA2, wB = load_wav(dA["path"]), load_wav(dA2["path"]), load_wav(dB["path"])
        zB, zA2 = zsp(wB[: int(3 * SR)]), zsp(wA2[: int(3 * SR)])
        eA, eB = emb(wA), emb(wB)
        eoB, eoA2 = emb(synth(dA, zB)), emb(synth(dA, zA2))
        oB_B.append(float((eoB * eB).sum())); oB_A.append(float((eoB * eA).sum()))
        oA2_B.append(float((eoA2 * eB).sum())); oA2_A.append(float((eoA2 * eA).sum()))
        rawAB.append(float((eA * eB).sum())); rawAA.append(float((emb(wA2) * eA).sum()))

    m = lambda x: float(np.mean(x)) if x else float("nan")
    print(f"\n=== H2 timbre-injection probe | e1_none step {ck.get('step')} | n={len(oB_B)} ===")
    print(f"  anchors: raw between-spk cos(A,B) {m(rawAB):+.3f} | same-spk cos(A,A') {m(rawAA):+.3f}")
    print(f"  out(timbre=B):  cos(,B) {m(oB_B):+.3f}  cos(,A) {m(oB_A):+.3f}")
    print(f"  out(timbre=A'): cos(,B) {m(oA2_B):+.3f}  cos(,A) {m(oA2_A):+.3f}")
    dB_ = m(oB_B) - m(oA2_B); dA_ = m(oA2_A) - m(oB_A)
    print(f"  => dTowardB {dB_:+.3f}  dTowardA {dA_:+.3f}  "
          f"(timbre carries target id if both >0 and sizeable; ~0 => generic)")
    print(f"  swap-sensitivity (how much output id moves when ref swaps): {0.5*(dB_+dA_):+.3f} "
          f"vs available same-vs-diff gap {m(rawAA)-m(rawAB):+.3f}", flush=True)


if __name__ == "__main__":
    main()
