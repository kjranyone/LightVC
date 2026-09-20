"""Validate the gender-invariant content encoder on M->F retrieval intelligibility.
Match in z=adapter(ContentVec) space instead of raw ContentVec. If z is gender-
invariant, a RAW male source (no pitch hack) retrieves the right female frames ->
low CER. Compares raw-ContentVec (0.96) and z-space (target: << 0.96)."""
from __future__ import annotations
import sys, argparse, random
from pathlib import Path
from collections import defaultdict
import numpy as np, torch, torch.nn.functional as F, soundfile as sf, librosa
sys.path.insert(0, str(Path(__file__).parent))
from train_m1 import mel_of, SR, DEV
from train_z1 import load_freebig
from f0leak_probe import load_wav
from train_content_enc import Adapter
from transformers import HubertModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/content_enc/last.pt")
    ap.add_argument("--pool-spk", type=int, default=0); ap.add_argument("--pool-n", type=int, default=60)
    args = ap.parse_args()
    voc, _ = load_freebig("checkpoints/freebig/foundation_bigvgan_parity.pt")
    cv = HubertModel.from_pretrained("lengyue233/content-vec-best").to(DEV).eval()
    enc = Adapter().to(DEV); enc.load_state_dict(torch.load(args.ckpt, map_location=DEV)["enc"]); enc.eval()
    from faster_whisper import WhisperModel
    wm = WhisperModel("large-v3", device="cuda", compute_type="float16"); import jiwer

    def tx(y):
        s, _ = wm.transcribe(librosa.resample(np.asarray(y, np.float32), orig_sr=SR, target_sr=16000).astype(np.float32), language="ja", beam_size=1)
        return "".join(x.text for x in s).strip()

    @torch.no_grad()
    def feat(w44, use_z):
        w16 = librosa.resample(np.asarray(w44, np.float32), orig_sr=SR, target_sr=16000)
        c = cv(torch.from_numpy(np.ascontiguousarray(w16)).float().view(1, -1).to(DEV)).last_hidden_state.transpose(1, 2)  # [1,768,Tc]
        z = enc(c) if use_z else F.normalize(c, dim=1)   # [1,d,Tc]
        m = mel_of(torch.from_numpy(np.ascontiguousarray(w44)).float().unsqueeze(0).to(DEV))[0]; T = m.shape[-1]
        zf = F.interpolate(z, size=T, mode="linear", align_corners=False)[0].t()  # [T,d]
        return zf, m.t()

    ftt = defaultdict(list)
    for f in sorted(Path("../data/female_tts_feat").rglob("*.pt")):
        ftt[f.parent.name].append(f)
    ftt = {s: v for s, v in ftt.items() if len(v) >= 40}
    male = defaultdict(list)
    for f in sorted(Path("../data/male_feat").rglob("*.pt")):
        male[f.parent.name].append(f)
    Bspk = list(ftt)[args.pool_spk]; pool_fs = ftt[Bspk][:args.pool_n]

    def run(use_z):
        pc, pm = [], []
        for pf in pool_fs:
            zf, mf = feat(load_wav(torch.load(pf, weights_only=False)["path"]), use_z); pc.append(zf); pm.append(mf)
        pc = torch.cat(pc, 0); pmn = torch.cat(pm, 0).cpu().numpy()
        pcn = F.normalize(pc, dim=-1)
        dm = torch.load(male[list(male)[0]][0], weights_only=False); srcw = load_wav(dm["path"]); ref = tx(np.asarray(srcw))
        cs, _ = feat(srcw, use_z)
        sim = (F.normalize(cs, dim=-1) @ pcn.t()).cpu().numpy()
        cand = np.argsort(-sim, 1)[:, :16]; idx = np.zeros(len(sim), int); prevm = pmn[cand[0, 0]]; idx[0] = cand[0, 0]
        for t in range(1, len(sim)):
            c = cand[t]; d = np.linalg.norm(pmn[c] - prevm, 1); j = c[np.argmin(d)]; idx[t] = j; prevm = pmn[j]
        with torch.no_grad():
            y = voc(torch.tensor(pmn[idx]).t().unsqueeze(0).to(DEV))[0].cpu().numpy()
        return jiwer.cer(ref.replace(" ", ""), tx(y).replace(" ", "")), ref, y

    print(f"pool {Bspk[:8]} ({len(pool_fs)} utts) | ckpt step {torch.load(args.ckpt, map_location='cpu')['step']}")
    for use_z, tag in [(False, "raw-ContentVec"), (True, "z (gender-inv)")]:
        cer, ref, y = run(use_z)
        sf.write(f"../results/spdcheck/MtoF_{'z' if use_z else 'raw'}.wav", np.clip(y, -1, 1), SR, subtype="PCM_16")
        print(f"  {tag:16s} M->F CER {cer:.2f}")
    print("  (raw基準0.96, +12st hack 0.35) z が下がれば gender-inv encoder 成功")


if __name__ == "__main__":
    main()
