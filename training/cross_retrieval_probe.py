"""Cross-speaker retrieval = the actual VC test. Source A (content), pool = a
DIFFERENT target speaker B (content+mel). kNN-match A's ContentVec against B's
pool -> retrieve B's REAL mel frames -> freebig = "B's voice saying A's content".
Measures: harmonic sharpness (should stay ~self-retrieval), ECAPA identity
(cos to B vs A -> did it take on B?), saves wavs. Runs female->female (mechanism)
and male->female (the actual バ美声 task, cross-gender content match)."""
from __future__ import annotations
import sys, random
from pathlib import Path
from collections import defaultdict
import numpy as np, torch, torch.nn.functional as F, soundfile as sf, librosa
sys.path.insert(0, str(Path(__file__).parent))
from train_m1 import SR, HOP, DEV
from train_z1 import load_freebig
from f0leak_probe import load_wav
from retrieval_probe import frames, sharp
import torchaudio.functional as AF


def main():
    random.seed(0)
    voc, _ = load_freebig("checkpoints/freebig/foundation_bigvgan_parity.pt")
    from speechbrain.inference.speaker import EncoderClassifier
    ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                           savedir="hf_models/spkrec-ecapa", run_opts={"device": "cpu"})

    def emb(y):
        w16 = librosa.resample(np.asarray(y, np.float32), orig_sr=SR, target_sr=16000)
        e = ecapa.encode_batch(torch.from_numpy(w16).unsqueeze(0)).squeeze().detach()
        return e / (e.norm() + 1e-6)

    fem = defaultdict(list)
    for f in sorted(Path(x.strip()) for x in open("/tmp/e5eval_held.txt") if x.strip()):
        fem[f.parent.name].append(f)
    fem = {s: v for s, v in fem.items() if len(v) >= 3}
    male = defaultdict(list)
    for f in sorted(Path("../data/male_feat").rglob("*.pt")):
        male[f.parent.name].append(f)
    male = {s: v for s, v in male.items() if len(v) >= 3}
    out = Path("../results/xretr"); out.mkdir(parents=True, exist_ok=True)

    def retr(src_f, pool_fs, k=1):
        ds = torch.load(src_f, weights_only=False)
        cs, ms, ws = frames(ds)
        pc, pm = [], []
        for pf in pool_fs:
            cf, mf, _ = frames(torch.load(pf, weights_only=False)); pc.append(cf); pm.append(mf)
        pc = torch.cat(pc, 0); pm = torch.cat(pm, 0)
        sim = F.normalize(cs, dim=-1) @ F.normalize(pc, dim=-1).t()
        with torch.no_grad():
            rm = pm[sim.topk(k, dim=-1).indices].mean(1)
            y = voc(rm.t().unsqueeze(0))[0].cpu().numpy()
        return y, ws

    fs, ms_ = list(fem.keys()), list(male.keys())
    tasks = [("F->F", random.choice(fs), random.choice(fs)) for _ in range(3)]
    tasks += [("M->F", random.choice(ms_), random.choice(fs)) for _ in range(3)]
    print(f"{'task':6s} {'sharp':>6s} {'cos(out,B)':>11s} {'cos(out,A)':>11s}  (B=target,want高 / A=source,want低)")
    agg = defaultdict(lambda: [[], [], []])
    for tag, A, B in tasks:
        if A == B:
            continue
        srcpool = male if tag.startswith("M") else fem
        src_f = srcpool[A][0]; pool_fs = fem[B]
        y, _ = retr(src_f, pool_fs, k=1)
        eB = emb(load_wav(torch.load(fem[B][1], weights_only=False)["path"]))
        eA = emb(load_wav(torch.load(srcpool[A][0], weights_only=False)["path"]))
        eo = emb(y)
        sh = sharp(y); cB = float((eo * eB).sum()); cA = float((eo * eA).sum())
        agg[tag][0].append(sh); agg[tag][1].append(cB); agg[tag][2].append(cA)
        sf.write(out / f"{tag}_{A[:6]}_to_{B[:6]}.wav", np.clip(y, -1, 1), SR, subtype="PCM_16")
    for tag in ["F->F", "M->F"]:
        a = agg[tag]
        if a[0]:
            print(f"{tag:6s} {np.mean(a[0]):6.2f} {np.mean(a[1]):11.3f} {np.mean(a[2]):11.3f}")
    print("wavs ->", out, "| 参照: self-retr sharp 3.19, gt 3.47")


if __name__ == "__main__":
    main()
