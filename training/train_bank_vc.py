"""
Timbre Bank VC: cross-attention retrieval + causal residual

Architecture:
  target_ref -> B_t = {(k_i, v_i)}
  q_k = Q_theta(source_mcep_k)
  p_k = Attn(q_k, K_psi(B_t), V_psi(B_t))
  mcep_hat_k = p_k + R_theta(source_mcep_k, p_k, f0_k)

Loss:
  L1(mcep_hat, mc_tgt_dtw) + lambda * L1(residual, 0)

Zero-init residual: starts as pure retrieval, gradually learns correction.
"""
import sys, json, math, random, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

MC_DIM = 25
MC_CACHE = Path("data/mc_cache")
SF_PAIRS = Path("data/sf_pairs")


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

    def forward(self, x):
        h = self.conv1(x)
        h = h.transpose(1, 2)
        h = self.norm1(h)
        h = F.gelu(h)
        h = h.transpose(1, 2)
        h = self.conv2(h)
        h = h.transpose(1, 2)
        h = self.norm2(h)
        h = F.gelu(h)
        h = h.transpose(1, 2)
        return x + h


class TimbreBankVC(nn.Module):
    def __init__(self, mc_dim=MC_DIM, attn_dim=128, n_heads=4,
                 n_res_blocks=4, res_hidden=128, kernel_size=7,
                 lambda_retr=0.5, lambda_res=0.5):
        super().__init__()
        self.mc_dim = mc_dim
        self.attn_dim = attn_dim
        self.lambda_retr = lambda_retr
        self.lambda_res = lambda_res

        self.q_proj = nn.Linear(mc_dim, attn_dim)
        self.k_proj = nn.Linear(mc_dim, attn_dim)
        self.v_proj = nn.Linear(mc_dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, mc_dim)

        nn.init.eye_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.eye_(self.k_proj.weight)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.eye_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.eye_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        for p in self.q_proj.parameters():
            p.requires_grad = False
        for p in self.k_proj.parameters():
            p.requires_grad = False
        for p in self.v_proj.parameters():
            p.requires_grad = False
        for p in self.out_proj.parameters():
            p.requires_grad = False

        self.n_heads = n_heads
        self.head_dim = attn_dim // n_heads
        self.log_temp = nn.Parameter(torch.tensor(0.0))
        self.log_temp.requires_grad = False

        res_in = mc_dim * 2 + 1
        self.res_in_proj = nn.Linear(res_in, res_hidden)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(res_hidden, kernel_size, dilation=2**i)
            for i in range(n_res_blocks)
        ])
        self.res_out = nn.Linear(res_hidden, mc_dim)
        nn.init.zeros_(self.res_out.weight)
        nn.init.zeros_(self.res_out.bias)

    def forward(self, mc_src, bank_mc, f0_src, return_components=False):
        B, T_src, _ = mc_src.shape
        T_bank = bank_mc.shape[1]

        src_mean = mc_src.mean(dim=1, keepdim=True)
        bank_mean = bank_mc.mean(dim=1, keepdim=True)

        q = self.q_proj(mc_src - src_mean)
        k = self.k_proj(bank_mc - bank_mean)
        v = self.v_proj(bank_mc)

        q = q.view(B, T_src, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T_bank, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T_bank, self.n_heads, self.head_dim).transpose(1, 2)

        q_norm = (q ** 2).sum(dim=-1, keepdim=True)
        k_norm = (k ** 2).sum(dim=-1, keepdim=True)
        dot = torch.matmul(q, k.transpose(-2, -1))
        dist = q_norm + k_norm.transpose(-2, -1) - 2 * dot
        dist = dist.clamp(min=0)
        temp = torch.exp(self.log_temp) + 0.01
        attn = F.softmax(-dist / temp, dim=-1)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T_src, self.attn_dim)
        retrieved = self.out_proj(out)

        res_in = torch.cat([mc_src, retrieved, f0_src.unsqueeze(-1)], dim=-1)
        h = self.res_in_proj(res_in).transpose(1, 2)
        for block in self.res_blocks:
            h = block(h)
        h = h.transpose(1, 2)
        residual = self.res_out(h)

        mcep_hat = retrieved + residual

        if return_components:
            return mcep_hat, retrieved, residual, attn
        return mcep_hat


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, speaker_banks, max_bank_frames=1024):
        self.pairs = pairs
        self.banks = speaker_banks
        self.max_bank_frames = max_bank_frames

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        d = np.load(p)

        mc_src = d["mc_src"].astype(np.float32)
        mc_tgt = d["mc_tgt"].astype(np.float32)
        f0_src = d["f0_src"].astype(np.float32)

        T = min(len(mc_src), len(mc_tgt), len(f0_src))
        mc_src = mc_src[:T]
        mc_tgt = mc_tgt[:T]
        f0_src = f0_src[:T]

        spk_tgt = str(d["spk_tgt"])
        bank_mc = self.banks[spk_tgt]
        if len(bank_mc) > self.max_bank_frames:
            start = random.randint(0, len(bank_mc) - self.max_bank_frames)
            bank_mc = bank_mc[start:start + self.max_bank_frames]
        elif len(bank_mc) < 100:
            indices = np.random.choice(len(bank_mc), 100, replace=True)
            bank_mc = bank_mc[indices]

        return {
            "mc_src": torch.from_numpy(mc_src),
            "mc_tgt": torch.from_numpy(mc_tgt),
            "f0_src": torch.from_numpy(f0_src),
            "bank_mc": torch.from_numpy(bank_mc.astype(np.float32)),
        }


def collate_fn(batch):
    T = min(b["mc_src"].shape[0] for b in batch)
    T_bank = min(b["bank_mc"].shape[0] for b in batch)

    return {
        "mc_src": torch.stack([b["mc_src"][:T] for b in batch]),
        "mc_tgt": torch.stack([b["mc_tgt"][:T] for b in batch]),
        "f0_src": torch.stack([b["f0_src"][:T] for b in batch]),
        "bank_mc": torch.stack([b["bank_mc"][:T_bank] for b in batch]),
    }


def load_speaker_banks(n_utts=20):
    banks = {}
    for spk_dir in sorted(MC_CACHE.iterdir()):
        if not spk_dir.is_dir(): continue
        spk_id = spk_dir.name
        files = sorted(spk_dir.glob("*.npz"))[:n_utts]
        if len(files) < 5:
            continue
        all_mc = []
        for f in files:
            d = np.load(f)
            all_mc.append(d["mc"])
        banks[spk_id] = np.concatenate(all_mc, axis=0).astype(np.float32)
    return banks


def load_pairs():
    pairs = sorted(SF_PAIRS.glob("pair_*.npz"))
    print(f"  Loaded {len(pairs)} pre-computed pairs")
    return pairs


def train(args):
    print("=== Timbre Bank VC Training ===\n")

    print("Loading speaker banks...")
    banks = load_speaker_banks(n_utts=args.n_bank_utts)
    print(f"  {len(banks)} speakers")
    bank_sizes = [len(v) for v in banks.values()]
    print(f"  Bank frames: mean={np.mean(bank_sizes):.0f}, min={min(bank_sizes)}, max={max(bank_sizes)}")

    print("Loading pairs...")
    pairs = load_pairs()

    dataset = PairDataset(pairs, banks, max_bank_frames=args.max_bank_frames)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, drop_last=True,
    )

    device = torch.device("cuda")
    model = TimbreBankVC(
        attn_dim=args.attn_dim, n_heads=args.n_heads,
        n_res_blocks=args.n_res_blocks, res_hidden=args.res_hidden,
        lambda_retr=args.lambda_retr, lambda_res=args.lambda_res,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} ({n_params/1e6:.2f}M)\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    step0_path = output_dir / "step_000000.pt"
    torch.save({
        "model": model.state_dict(),
        "config": {
            "attn_dim": args.attn_dim,
            "n_heads": args.n_heads,
            "n_res_blocks": args.n_res_blocks,
            "res_hidden": args.res_hidden,
            "mc_dim": MC_DIM,
            "lambda_retr": args.lambda_retr,
            "lambda_res": args.lambda_res,
        },
        "step": 0,
        "loss": 0.0,
    }, step0_path)
    print(f"  Saved step-0 checkpoint: {step0_path}")

    step = 0
    log_interval = 200
    save_interval = args.save_every
    t0 = time.time()

    print(f"Training {args.max_steps} steps (B={args.batch_size})...\n")

    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps: break
            mc_src = batch["mc_src"].to(device)
            mc_tgt = batch["mc_tgt"].to(device)
            f0_src = batch["f0_src"].to(device)
            bank_mc = batch["bank_mc"].to(device)

            mcep_hat, retrieved, residual, attn = model(
                mc_src, bank_mc, f0_src, return_components=True)
            loss_l1 = F.l1_loss(mcep_hat, mc_tgt)
            loss_retr = F.l1_loss(retrieved, mc_tgt)
            loss_res = F.l1_loss(residual, torch.zeros_like(residual))

            loss = loss_l1 + model.lambda_retr * loss_retr + model.lambda_res * loss_res

            if step % log_interval == 0:
                retr_l1 = loss_retr.item()
                res_l1 = loss_res.item()
                tgt_l1 = F.l1_loss(mc_tgt, torch.zeros_like(mc_tgt)).item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % log_interval == 0:
                elapsed = time.time() - t0
                speed = step / elapsed
                eta = (args.max_steps - step) / speed
                print(f"step {step}/{args.max_steps} | l1={loss_l1.item():.4f} "
                      f"retr={retr_l1:.4f} res={res_l1:.4f} tgt_scale={tgt_l1:.4f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e} | "
                      f"{speed:.1f}step/s ETA {eta:.0f}s", flush=True)

            if step % save_interval == 0 or step == args.max_steps:
                ckpt_path = output_dir / f"step_{step:06d}.pt"
                torch.save({
                    "model": model.state_dict(),
                    "config": {
                        "attn_dim": args.attn_dim,
                        "n_heads": args.n_heads,
                        "n_res_blocks": args.n_res_blocks,
                        "res_hidden": args.res_hidden,
                        "mc_dim": MC_DIM,
                        "lambda_retr": args.lambda_retr,
                        "lambda_res": args.lambda_res,
                    },
                    "step": step,
                    "loss": loss_l1.item(),
                }, ckpt_path)
                latest = output_dir / "latest.pt"
                torch.save({
                    "model": model.state_dict(),
                    "config": {
                        "attn_dim": args.attn_dim,
                        "n_heads": args.n_heads,
                        "n_res_blocks": args.n_res_blocks,
                        "res_hidden": args.res_hidden,
                        "mc_dim": MC_DIM,
                        "lambda_retr": args.lambda_retr,
                        "lambda_res": args.lambda_res,
                    },
                    "step": step,
                    "loss": loss_l1.item(),
                }, latest)
                print(f"  Saved: {ckpt_path}", flush=True)

    print(f"\nTraining complete. Final L1: {loss_l1.item():.4f}")


def eval_model(args):
    import soundfile as sf
    import pyworld as world
    import pysptk as sptk
    import librosa
    from speechbrain.inference.speaker import EncoderClassifier

    SR = 16000
    FRAME_PERIOD = 5.0
    FFTL = 2048
    ALPHA = 0.410
    MC_ORDER = 24
    VCTK_WAV = Path("../data/vctk_200")

    device = torch.device("cuda")

    print("=== Timbre Bank VC Evaluation ===\n")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = TimbreBankVC(
        mc_dim=cfg["mc_dim"], attn_dim=cfg["attn_dim"], n_heads=cfg["n_heads"],
        n_res_blocks=cfg["n_res_blocks"], res_hidden=cfg["res_hidden"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    secs_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="hf_models/spkrec-ecapa",
        run_opts={"device": str(device)},
    )

    print("Loading speaker banks...")
    banks = load_speaker_banks(n_utts=args.n_bank_utts)

    def analyze_wav(wav_path):
        wav, sr = sf.read(str(wav_path), dtype="float32")
        if wav.ndim > 1: wav = wav[:, 0]
        if sr != SR:
            wav = librosa.resample(wav.astype(np.float64), orig_sr=sr, target_sr=SR)
        wav = wav.astype(np.float64)
        f0, t = world.dio(wav, SR, frame_period=FRAME_PERIOD)
        f0 = world.stonemask(wav, f0, t, SR)
        sp = world.cheaptrick(wav, f0, t, SR, fft_size=FFTL)
        ap = world.d4c(wav, f0, t, SR, fft_size=FFTL)
        mc = sptk.sp2mc(sp, MC_ORDER, ALPHA)
        return {"f0": f0.astype(np.float32), "mc": mc.astype(np.float32), "ap": ap}

    def synth(f0, mc, ap):
        mc64 = np.ascontiguousarray(mc, dtype=np.float64)
        sp = sptk.mc2sp(mc64, ALPHA, FFTL)
        ap64 = np.ascontiguousarray(ap, dtype=np.float64)
        f064 = np.ascontiguousarray(f0, dtype=np.float64)
        return world.synthesize(f064, sp, ap64, SR, frame_period=FRAME_PERIOD).astype(np.float32)

    def shift_f0(f0, tgt_mean):
        voiced = f0[f0 > 0]
        if len(voiced) == 0: return f0.astype(np.float64)
        src_mean = float(np.exp(np.mean(np.log(voiced))))
        return np.where(f0 > 0, f0 * tgt_mean / src_mean, 0).astype(np.float64)

    groups = defaultdict(list)
    for d in sorted(VCTK_WAV.iterdir()):
        if not d.is_dir(): continue
        for w in d.glob("*.wav"):
            parts = w.stem.split("_")
            if len(parts) >= 2:
                groups[parts[1]].append((d.name, str(w)))

    test_pairs = []
    used = set()
    for tid, utts in sorted(groups.items()):
        if len(utts) < 2: continue
        for i in range(len(utts)):
            for j in range(i+1, len(utts)):
                sa, wa = utts[i]; sb, wb = utts[j]
                if sa == sb or sa in used or sb in used: continue
                test_pairs.append({"src": sa, "src_wav": wa, "tgt": sb, "tgt_wav": wb, "text": tid})
                used.add(sa); used.add(sb)
                if len(test_pairs) >= args.n_eval: break
        if len(test_pairs) >= args.n_eval: break

    print(f"Evaluating on {len(test_pairs)} pairs\n")

    model_scores = []
    retrieval_scores = []
    oracle_scores = []

    for idx, p in enumerate(test_pairs):
        feat_s = analyze_wav(p["src_wav"])
        feat_t = analyze_wav(p["tgt_wav"])

        mc_s = feat_s["mc"]
        f0_s = feat_s["f0"]
        ap_s = feat_s["ap"]
        mc_t = feat_t["mc"]
        f0_t = feat_t["f0"]
        T = len(mc_s)

        tgt_voiced = f0_t[f0_t > 0]
        tgt_mean_f0 = float(np.exp(np.mean(np.log(tgt_voiced)))) if len(tgt_voiced) > 0 else 200.0
        f0_shifted = shift_f0(f0_s, tgt_mean_f0)

        bank_mc = banks[p["tgt"]]
        if len(bank_mc) > 512:
            bank_mc = bank_mc[:512]

        mc_src_t = torch.from_numpy(mc_s[:T]).float().unsqueeze(0).to(device)
        f0_src_t = torch.from_numpy(f0_s[:T]).float().unsqueeze(0).to(device)
        bank_t = torch.from_numpy(bank_mc).float().unsqueeze(0).to(device)

        with torch.no_grad():
            mcep_hat, retrieved, residual, attn = model(
                mc_src_t, bank_t, f0_src_t, return_components=True)

        mc_pred = mcep_hat.squeeze(0).cpu().numpy()
        mc_retr = retrieved.squeeze(0).cpu().numpy()

        wav_model = synth(f0_shifted[:T], mc_pred[:T], ap_s[:T])
        wav_retr = synth(f0_shifted[:T], mc_retr[:T], ap_s[:T])

        from fastdtw import fastdtw
        dist, path = fastdtw(mc_s, mc_t, radius=30)
        src_map = np.zeros(T, dtype=int)
        for s, t in path:
            if s < T: src_map[s] = min(t, len(mc_t)-1)
        for i in range(1, T):
            if src_map[i] == 0: src_map[i] = src_map[i-1]
        mc_t_aligned = mc_t[src_map]
        wav_oracle = synth(f0_shifted[:T], mc_t_aligned[:T], ap_s[:T])

        wav_tgt, sr = sf.read(p["tgt_wav"], dtype="float32")
        if sr != SR: wav_tgt = librosa.resample(wav_tgt.astype(np.float32), orig_sr=sr, target_sr=SR)

        with torch.no_grad():
            def emb(w): return secs_model.encode_batch(
                torch.from_numpy(w.astype(np.float32)).unsqueeze(0).to(device)).squeeze(0)
            e_tgt = emb(wav_tgt)
            m_sim = F.cosine_similarity(e_tgt, emb(wav_model), dim=-1).item()
            r_sim = F.cosine_similarity(e_tgt, emb(wav_retr), dim=-1).item()
            o_sim = F.cosine_similarity(e_tgt, emb(wav_oracle), dim=-1).item()

        model_scores.append(m_sim)
        retrieval_scores.append(r_sim)
        oracle_scores.append(o_sim)
        print(f"  [{idx+1}/{len(test_pairs)}] {p['src']}→{p['tgt']}: "
              f"model={m_sim:.3f} retr={r_sim:.3f} oracle={o_sim:.3f}", flush=True)

    m = np.array(model_scores)
    r = np.array(retrieval_scores)
    o = np.array(oracle_scores)
    print(f"\n=== 結果 ===")
    print(f"モデル(retrieval+residual): {m.mean():.4f} ± {m.std():.4f}")
    print(f"純retrieval(attention):     {r.mean():.4f} ± {r.std():.4f}")
    print(f"Oracle(DTW):                {o.mean():.4f} ± {o.std():.4f}")
    print(f"モデル/Oracle:              {m.mean()/o.mean():.1%}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    train_p = sub.add_parser("train")
    train_p.add_argument("--output", default="checkpoints/bank_vc")
    train_p.add_argument("--batch_size", type=int, default=8)
    train_p.add_argument("--lr", type=float, default=5e-4)
    train_p.add_argument("--max_steps", type=int, default=30000)
    train_p.add_argument("--save_every", type=int, default=5000)
    train_p.add_argument("--attn_dim", type=int, default=128)
    train_p.add_argument("--n_heads", type=int, default=4)
    train_p.add_argument("--n_res_blocks", type=int, default=4)
    train_p.add_argument("--res_hidden", type=int, default=128)
    train_p.add_argument("--n_bank_utts", type=int, default=20)
    train_p.add_argument("--max_bank_frames", type=int, default=512)
    train_p.add_argument("--lambda_retr", type=float, default=0.5)
    train_p.add_argument("--lambda_res", type=float, default=0.5)

    eval_p = sub.add_parser("eval")
    eval_p.add_argument("--checkpoint", required=True)
    eval_p.add_argument("--n_eval", type=int, default=20)
    eval_p.add_argument("--n_bank_utts", type=int, default=20)

    args = parser.parse_args()
    if args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        eval_model(args)
