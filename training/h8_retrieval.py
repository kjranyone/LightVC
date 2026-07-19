"""
h8_retrieval.py — WavLM-space kNN-VC baseline (review hypothesis H8).

Question: oracle margin (~0.53) >> the learned adapter's realized margin (~0.27),
and the adapter's output is muffled because it adds a large linear delta that lands
OFF the target manifold. Can we recover the headroom AND avoid the muffle by
assembling the output purely from REAL target frames via content retrieval?

Method (proper kNN-VC, unlike knncv_baseline.py which matches in DAC space):
  - content space = WavLM-base last_hidden_state (same as _feat.pt source features)
  - for each source frame: cosine-kNN match its WavLM to a TARGET-speaker pool of
    (WavLM, DAC) frames built from that speaker's OTHER-text VCTK utterances
  - z_gen frame = softmax-weighted mean of the retrieved REAL target DAC frames
  - decode z_gen with the (finetuned) DAC decoder

Because z_gen is built from in-manifold target latents, it should decode CLEAN
(no 8kHz muffle) and carry the target speaker — testing whether retrieval beats
the large-delta generator.

Reports margin vs adapter(0.27)/oracle(0.53) and muffle proxies; can export
source/oracle/knn trios for listen_gui.py.

Usage:
  cd training
  uv run python h8_retrieval.py --n 24 --pool-utts 8 --k 4 --export-audio 4
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
import librosa

sys.path.insert(0, str(Path(__file__).parent))
from train_phase3b import (DEVICE, DAC_SR, load_dac, load_ecapa, resample_16k,
                           ecapa_embed, hard_quantize_all, hard_rvq_requantize)
from gate0_codec import load_finetuned_decoder
import kansei_proxies as kp

EVAL = Path("../data/phase3_10k/eval")
VCTK = Path("../data/vctk_200")
SR16 = 16000


def load_wavlm():
    from transformers import WavLMModel
    m = WavLMModel.from_pretrained("microsoft/wavlm-base").to(DEVICE).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def wavlm_feat(model, wav44):
    w16 = librosa.resample(wav44.astype(np.float32), orig_sr=DAC_SR, target_sr=SR16)
    t = torch.from_numpy(w16).float().unsqueeze(0).to(DEVICE)
    return model(t).last_hidden_state.squeeze(0)   # [T', 768]


def interp_T(x_td, T):
    """x_td [t, C] -> [T, C] via linear interpolation over time."""
    x = x_td.transpose(0, 1).unsqueeze(0)          # [1, C, t]
    y = F.interpolate(x, size=T, mode="linear", align_corners=False)
    return y.squeeze(0).transpose(0, 1)            # [T, C]


@torch.no_grad()
def dac_encode(dac, wav44):
    x = torch.from_numpy(wav44).float().view(1, 1, -1).to(DEVICE)
    return dac.encoder(x).squeeze(0)               # [1024, T]


def build_pool(dac, wavlm, spk, exclude_texts, pool_utts):
    """(WavLM@DAC-rate, DAC) frames from spk's VCTK utts, excluding given texts."""
    wavs = sorted((VCTK / spk).glob(f"{spk}_*.wav"))
    picked, dfr, wfr = [], [], []
    for w in wavs:
        tid = w.stem.split("_")[1]
        if tid in exclude_texts:
            continue
        try:
            audio, sr = sf.read(str(w), dtype="float32")
            if audio.ndim > 1:
                audio = audio[:, 0]
            if sr != DAC_SR:
                audio = librosa.resample(audio.astype(np.float64), orig_sr=sr, target_sr=DAC_SR).astype(np.float32)
            if len(audio) < DAC_SR // 2:
                continue
            z = dac_encode(dac, audio)             # [1024, T]
            wl = wavlm_feat(wavlm, audio)          # [T', 768]
            wl = interp_T(wl, z.shape[1])          # [T, 768]
            dfr.append(z.transpose(0, 1))          # [T,1024]
            wfr.append(wl)                          # [T,768]
            picked.append(tid)
        except Exception as e:
            print(f"    pool skip {w.name}: {e}")
        if len(picked) >= pool_utts:
            break
    if not dfr:
        return None
    return (torch.cat(dfr, 0), F.normalize(torch.cat(wfr, 0), dim=-1), picked)  # dac[N,1024], wl[N,768]


@torch.no_grad()
def knn_assemble(src_wl_T, pool_wl, pool_dac, k, temp=5.0):
    q = F.normalize(src_wl_T, dim=-1)              # [T,768]
    sim = q @ pool_wl.t()                          # [T,N]
    tk_sim, tk_idx = sim.topk(k, dim=-1)           # [T,k]
    w = F.softmax(tk_sim * temp, dim=-1).unsqueeze(-1)   # [T,k,1]
    gathered = pool_dac[tk_idx]                    # [T,k,1024]
    z = (gathered * w).sum(1)                      # [T,1024]
    return z.transpose(0, 1).contiguous()          # [1024,T]


def save_clip(y, out_dir, name):
    a, b = int(0.3 * DAC_SR), int(3.3 * DAC_SR)
    seg = y[a:b] if b <= len(y) else y
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(out_dir / name, np.clip(seg, -1, 1).astype(np.float32), DAC_SR, subtype="PCM_16")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--pool-utts", type=int, default=8)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--min-pool-pairs", type=int, default=3, help="only eval speakers with >= this many eval pairs")
    ap.add_argument("--export-audio", type=int, default=0)
    ap.add_argument("--export-dir", default="../results/diag_h8_knn")
    ap.add_argument("--out", default="../results/h8_retrieval.json")
    args = ap.parse_args()

    dac = load_finetuned_decoder()
    wavlm = load_wavlm()
    ecapa = load_ecapa()

    pairs = sorted([p for p in EVAL.glob("pair_*.pt") if not p.stem.endswith("_feat")])
    # group by tgt_spk, and collect that spk's eval target texts (to exclude from pool)
    by_spk = defaultdict(list)
    tgt_texts = defaultdict(set)
    meta = {}
    for p in pairs:
        d = torch.load(p, map_location="cpu")
        by_spk[d["tgt_spk"]].append(p)
        tgt_texts[d["tgt_spk"]].add(d["text_id"])
        meta[p] = (d["tgt_spk"], d["text_id"])
    eval_spks = [s for s, v in by_spk.items() if len(v) >= args.min_pool_pairs]
    eval_spks.sort(key=lambda s: -len(by_spk[s]))
    print(f"H8 retrieval | k={args.k} pool_utts={args.pool_utts} | speakers={eval_spks[:6]}...")

    pool_cache = {}
    res = {"knn": defaultdict(list), "adapter_ref": {"margin": 0.27, "oracle": 0.53}}
    export_dir = Path(args.export_dir)
    done = 0

    for spk in eval_spks:
        if done >= args.n:
            break
        pool = pool_cache.get(spk)
        if pool is None:
            pool = build_pool(dac, wavlm, spk, tgt_texts[spk], args.pool_utts)
            pool_cache[spk] = pool
        if pool is None:
            continue
        pool_dac, pool_wl, _ = pool
        for pp in by_spk[spk]:
            if done >= args.n:
                break
            d = torch.load(pp, map_location="cpu")
            feat_p = pp.with_name(pp.stem + "_feat.pt")
            if not feat_p.exists():
                continue
            f = torch.load(feat_p, map_location="cpu")
            z_s = d["z_s"].float().to(DEVICE)
            q0_s = d["q0_s"].float().to(DEVICE)
            z_t = d["z_t_aligned"].float().to(DEVICE)
            tgt_emb = d["timbre"].float().to(DEVICE).squeeze()
            src_wl = f["wavlm"].float().to(DEVICE)                     # [172,768]
            T = z_s.shape[1]
            src_wl_T = interp_T(src_wl, T)                             # [T,768]

            z_gen = knn_assemble(src_wl_T, pool_wl, pool_dac, args.k)  # [1024,T]
            with torch.no_grad():
                gen = dac.decoder(z_gen.unsqueeze(0)).squeeze().cpu().numpy().astype(np.float32)
                src = dac.decoder(hard_quantize_all(dac, z_s.unsqueeze(0))).squeeze().cpu().numpy().astype(np.float32)
                ora = dac.decoder(hard_rvq_requantize(dac, q0_s.unsqueeze(0), z_t.unsqueeze(0))).squeeze().cpu().numpy().astype(np.float32)
                eg = ecapa_embed(ecapa, resample_16k(torch.from_numpy(gen).unsqueeze(0).to(DEVICE))).squeeze()
                es = ecapa_embed(ecapa, resample_16k(torch.from_numpy(src).unsqueeze(0).to(DEVICE))).squeeze()

            tsim = F.cosine_similarity(eg, tgt_emb, dim=0).item()
            ssim = F.cosine_similarity(eg, es, dim=0).item()
            m = kp.analyze(gen, full=False)
            res["knn"]["target_sim"].append(tsim)
            res["knn"]["source_sim"].append(ssim)
            res["knn"]["margin"].append(tsim - ssim)
            res["knn"]["hf_ratio"].append(m["hf_ratio"])
            res["knn"]["cliff"].append(m["eight_k_cliff"])

            if done < args.export_audio:
                save_clip(src, export_dir, f"diag{done}_source.wav")
                save_clip(ora, export_dir, f"diag{done}_oracle.wav")
                save_clip(gen, export_dir, f"diag{done}_knn.wav")
            done += 1
            print(f"  [{done}] {spk} t{meta[pp][1]:>4}  margin={tsim-ssim:+.3f} (tgt={tsim:.3f} src={ssim:.3f})  hf={m['hf_ratio']:.4f} cliff={m['eight_k_cliff']:.3f}", flush=True)

    summ = {k: round(float(np.mean(v)), 4) for k, v in res["knn"].items() if v}
    report = {"n": done, "k": args.k, "pool_utts": args.pool_utts, "knn_mean": summ,
              "reference": {"adapter_realized_margin": 0.27, "oracle_margin": 0.53,
                            "adapter_gen_hf_ratio": 0.009, "oracle_hf_ratio": 0.017,
                            "adapter_gen_cliff": 0.13, "oracle_cliff": 0.31}}
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n=== H8 kNN-VC (WavLM-space retrieval) ===")
    print(f"  n={done}")
    print(f"  margin   knn={summ.get('margin')}   (adapter realized 0.27, oracle 0.53)")
    print(f"  target_sim={summ.get('target_sim')}  source_sim={summ.get('source_sim')}")
    print(f"  muffle   knn hf_ratio={summ.get('hf_ratio')} cliff={summ.get('cliff')}   (adapter gen 0.009/0.13, oracle 0.017/0.31)")
    print(f"wrote {args.out}")
    if args.export_audio:
        print(f"listen: uv run python listen_gui.py --dir {export_dir}")


if __name__ == "__main__":
    main()
