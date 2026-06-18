import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn.functional as F
from converter import FlowConverter, ConverterConfig
from train_flow import load_latent_corpus, sample_flow_batch

device = torch.device("cuda")

speakers = load_latent_corpus("data/vctk_latents_200/", max_frames=200)
spk_list = list(speakers.keys())

cfg = {
    "latent_dim": 1024, "hidden_dim": 1024, "n_conv_blocks": 4,
    "speaker_embed_dim": 256, "enable_timbre": True, "n_timbre_tokens": 32,
    "n_attn_heads": 8, "bottleneck_dim": 256, "time_embed_dim": 128,
    "n_depth_groups": 0,
}
config = ConverterConfig(**cfg)
model = FlowConverter(config).to(device)

ckpt = torch.load("checkpoints/phase_b_utte_1024/best.pt", map_location=device, weights_only=False)
missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
print(f"Loaded warm-start: missing={len(missing)}, unexpected={len(unexpected)}")

model.eval()

with torch.no_grad():
    z_src, z_tgt, z_ref, src_spk = sample_flow_batch(speakers, 8, 200, device)
    
    v_target = z_tgt - z_src
    t = torch.ones(8, device=device)
    v_pred = model.forward_velocity(z_src, t, z_ref)
    
    print(f"\n=== Velocity stats ===")
    print(f"v_target: mean={v_target.mean():.4f} std={v_target.std():.4f} abs_mean={v_target.abs().mean():.4f}")
    print(f"v_pred:   mean={v_pred.mean():.4f} std={v_pred.std():.4f} abs_mean={v_pred.abs().mean():.4f}")
    
    fm_zero = F.mse_loss(torch.zeros_like(v_target), v_target)
    fm_raw = F.mse_loss(v_pred, v_target)
    print(f"\nfm loss (v_pred=0):    {fm_zero:.4f}")
    print(f"fm loss (v_pred=cold): {fm_raw:.4f}")
    
    flat_pred = v_pred.flatten(1)
    flat_tgt = v_target.flatten(1)
    cos_sim = F.cosine_similarity(flat_pred, flat_tgt, dim=-1).mean()
    print(f"cos_sim(v_pred, v_target): {cos_sim:.6f}")
    
    print(f"\n=== Per-sample correlation ===")
    for b in range(min(4, 8)):
        c = torch.corrcoef(torch.stack([v_pred[b].flatten(), v_target[b].flatten()]))[0, 1]
        print(f"  batch {b} (spk {src_spk[b]}): corr={c:.4f}")
    
    print(f"\n=== Latent stats ===")
    print(f"z_src: mean={z_src.mean():.4f} std={z_src.std():.4f} min={z_src.min():.4f} max={z_src.max():.4f}")
    print(f"z_tgt: mean={z_tgt.mean():.4f} std={z_tgt.std():.4f} min={z_tgt.min():.4f} max={z_tgt.max():.4f}")
    print(f"z_ref: mean={z_ref.mean():.4f} std={z_ref.std():.4f}")
    
    print(f"\n=== Bottleneck info ===")
    content_src = model.bottleneck(z_src)
    content_tgt = model.bottleneck(z_tgt)
    print(f"content_src: std={content_src.std():.4f}")
    print(f"content_tgt: std={content_tgt.std():.4f}")
    print(f"content difference: {(content_src - content_tgt).abs().mean():.4f}")
    
    print(f"\n=== Speaker embedding ===")
    emb_src = model.speaker_encoder(z_src)
    emb_tgt = model.speaker_encoder(z_tgt)
    emb_ref = model.speaker_encoder(z_ref)
    print(f"emb_src: norm={emb_src.norm(dim=-1).mean():.4f}")
    print(f"emb_tgt: norm={emb_tgt.norm(dim=-1).mean():.4f}")
    print(f"emb_ref: norm={emb_ref.norm(dim=-1).mean():.4f}")
    print(f"cos_sim(src,tgt): {F.cosine_similarity(emb_src, emb_tgt, dim=-1).mean():.4f}")
    print(f"cos_sim(ref,tgt): {F.cosine_similarity(emb_ref, emb_tgt, dim=-1).mean():.4f}")
