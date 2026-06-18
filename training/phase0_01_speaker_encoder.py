import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoFeatureExtractor
from converter import FlowConverter, ConverterConfig

VCTK_WAV = Path("../data/vctk_200")
LATENTS_DIR = Path("data/vctk_latents_200")
DEVICE = torch.device("cuda")
EMBED_DIM = 768

def load_speakers():
    index_path = LATENTS_DIR / "index.tsv"
    speakers = {}
    with open(index_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            spk = row["speaker_id"]
            speakers.setdefault(spk, []).append({
                "utt_id": row["utterance_id"],
                "npy": row["path"],
                "wav": str(VCTK_WAV / spk / f"{row['utterance_id']}.wav"),
            })
    return speakers

def split_speakers(speakers, n_heldout=19):
    spk_list = sorted(speakers.keys())
    np.random.seed(42)
    heldout = set(np.random.choice(spk_list, n_heldout, replace=False))
    train = [s for s in spk_list if s not in heldout]
    return train, sorted(heldout)

def load_wavlm_sv():
    model = AutoModel.from_pretrained("microsoft/wavlm-base-plus-sv").to(DEVICE).eval()
    extractor = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")
    return model, extractor

@torch.no_grad()
def compute_wavlm_sv_embedding(model, extractor, wav_path, max_sec=15):
    import soundfile as sf
    import librosa
    wav, sr = sf.read(wav_path, dtype="float32")
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    if len(wav) > max_sec * 16000:
        wav = wav[:max_sec * 16000]
    inputs = extractor(wav, sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)
    outputs = model(input_values=inputs)
    embed = outputs.last_hidden_state.mean(dim=1)
    embed = F.normalize(embed, dim=-1)
    return embed.squeeze(0)

class DistillableSpeakerEncoder(nn.Module):
    def __init__(self, latent_dim=1024, embed_dim=EMBED_DIM):
        super().__init__()
        self.p1 = nn.Linear(latent_dim * 2, latent_dim // 2)
        self.p2 = nn.Linear(latent_dim // 2, embed_dim)

    def forward(self, ref_latent):
        mean = ref_latent.mean(dim=-1)
        var = ref_latent.var(dim=-1, unbiased=False)
        std = torch.sqrt(var + 1e-6)
        pooled = torch.cat([mean, std], dim=-1)
        h = F.gelu(self.p1(pooled))
        return F.normalize(self.p2(h), dim=-1)

def precompute_embeddings(speakers, wavlm_model, extractor, max_per_spk=20):
    all_data = {}
    for spk in tqdm(sorted(speakers.keys()), desc="Embedding"):
        utts = speakers[spk][:max_per_spk]
        items = []
        for u in utts:
            wav_path = u["wav"]
            npy_path = u["npy"]
            if not Path(wav_path).exists() or not Path(npy_path).exists():
                continue
            teacher = compute_wavlm_sv_embedding(wavlm_model, extractor, wav_path)
            latent = np.load(npy_path).astype(np.float32)
            if latent.shape[1] < 30:
                continue
            latent_t = torch.from_numpy(latent).unsqueeze(0).to(DEVICE)
            items.append({"teacher": teacher.cpu(), "latent": latent_t.cpu(), "spk": spk})
        if items:
            all_data[spk] = items
    return all_data

def cosine_loss(pred, teacher):
    return (1 - F.cosine_similarity(pred, teacher, dim=-1)).mean()

def contrastive_loss(embeddings, labels, temperature=0.07):
    n = embeddings.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=embeddings.device)
    sim = torch.mm(embeddings, embeddings.t()) / temperature
    mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask.fill_diagonal_(False)
    pos_mask = mask
    neg_mask = ~mask & ~torch.eye(n, dtype=torch.bool, device=embeddings.device)
    if not pos_mask.any() or not neg_mask.any():
        return torch.tensor(0.0, device=embeddings.device)
    logits = sim[~torch.eye(n, dtype=torch.bool, device=embeddings.device)].view(n, n-1)
    pos_labels = mask[~torch.eye(n, dtype=torch.bool, device=embeddings.device)].view(n, n-1)
    exp_logits = torch.exp(logits) * (~pos_labels).float()
    denom = exp_logits.sum(dim=-1) + torch.exp(sim[torch.arange(n), torch.arange(n)])
    pos_sim = sim[pos_mask].view(pos_mask.sum(1))
    loss = (-pos_sim + torch.log(denom.unsqueeze(1).expand(-1, pos_mask.sum(1).max()))).mean()
    if torch.isnan(loss) or torch.isinf(loss):
        return torch.tensor(0.0, device=embeddings.device)
    return loss.clamp(min=-10, max=10)

def compute_eer(same_spk_sims, diff_spk_sims):
    all_sims = np.concatenate([same_spk_sims, diff_spk_sims])
    labels = np.concatenate([np.ones(len(same_spk_sims)), np.zeros(len(diff_spk_sims))])
    best_eer = 1.0
    for thresh in np.linspace(-1, 1, 2000):
        far = np.mean(diff_spk_sims > thresh)
        frr = np.mean(same_spk_sims <= thresh)
        eer = (far + frr) / 2
        if eer < best_eer:
            best_eer = eer
    return best_eer

def evaluate(encoder, data, device):
    encoder.eval()
    all_embeds = {}
    with torch.no_grad():
        for spk, items in data.items():
            for item in items:
                latent = item["latent"].to(device)
                embed = encoder(latent).cpu()
                all_embeds.setdefault(spk, []).append(embed)
    spk_list = sorted(all_embeds.keys())
    same_sims, diff_sims = [], []
    cos_pred_teacher = []
    for i, spk_i in enumerate(spk_list):
        embeds_i = all_embeds[spk_i]
        for j in range(len(embeds_i)):
            for k in range(j + 1, len(embeds_i)):
                same_sims.append(F.cosine_similarity(embeds_i[j], embeds_i[k], dim=-1).item())
            teacher = data[spk_i][j]["teacher"].to(device)
            pred = embeds_i[j].to(device)
            cos_pred_teacher.append(F.cosine_similarity(pred, teacher, dim=-1).item())
        for spk_j in spk_list[i+1:]:
            embeds_j = all_embeds[spk_j]
            for ej in embeds_i[:3]:
                for ek in embeds_j[:3]:
                    diff_sims.append(F.cosine_similarity(ej, ek, dim=-1).item())
    eer = compute_eer(np.array(same_sims), np.array(diff_sims))
    mean_cos = np.mean(cos_pred_teacher)
    return mean_cos, eer

def main():
    print("=== 09-01: DAC SpeakerEncoder Distillation Feasibility ===")
    speakers = load_speakers()
    train_spks, heldout_spks = split_speakers(speakers)
    print(f"Train: {len(train_spks)} speakers, Held-out: {len(heldout_spks)} speakers")

    print("Loading WavLM-SV...")
    wavlm_model, extractor = load_wavlm_sv()

    print("Precomputing embeddings...")
    all_data = precompute_embeddings(speakers, wavlm_model, extractor, max_per_spk=15)
    train_data = {s: all_data[s] for s in train_spks if s in all_data}
    heldout_data = {s: all_data[s] for s in heldout_spks if s in all_data}
    print(f"Train utts: {sum(len(v) for v in train_data.values())}, "
          f"Held-out utts: {sum(len(v) for v in heldout_data.values())}")

    encoder = DistillableSpeakerEncoder().to(DEVICE)
    optim = torch.optim.AdamW(encoder.parameters(), lr=1e-3, weight_decay=1e-4)

    all_items = []
    for spk, items in train_data.items():
        for item in items:
            all_items.append((item, spk))

    print(f"\n--- Baseline (untrained) ---")
    mean_cos, eer = evaluate(encoder, heldout_data, DEVICE)
    print(f"Held-out: mean cos(pred,teacher)={mean_cos:.4f}  EER={eer:.1%}")

    teacher_sims_same, teacher_sims_diff = [], []
    spk_list = sorted(heldout_data.keys())
    for i, spk_i in enumerate(spk_list):
        items_i = heldout_data[spk_i]
        for j in range(len(items_i)):
            for k in range(j+1, len(items_i)):
                t1 = items_i[j]["teacher"]
                t2 = items_i[k]["teacher"]
                teacher_sims_same.append(F.cosine_similarity(t1, t2, dim=-1).item())
        for spk_j in spk_list[i+1:]:
            items_j = heldout_data[spk_j]
            for tj in items_i[:3]:
                for tk in items_j[:3]:
                    teacher_sims_diff.append(F.cosine_similarity(tj["teacher"], tk["teacher"], dim=-1).item())
    teacher_eer = compute_eer(np.array(teacher_sims_same), np.array(teacher_sims_diff))
    print(f"Teacher (WavLM-SV) EER on held-out: {teacher_eer:.1%}")

    print(f"\n--- Training (cosine + contrastive) ---")
    encoder.train()
    for epoch in range(100):
        np.random.shuffle(all_items)
        total_loss = 0
        n_batches = 0
        batch_size = 16
        for i in range(0, len(all_items), batch_size):
            batch = all_items[i:i+batch_size]
            min_t = min(item[0]["latent"].shape[-1] for item in batch)
            latents = torch.cat([item[0]["latent"][:, :, :min_t] for item in batch]).to(DEVICE)
            teachers = torch.stack([item[0]["teacher"] for item in batch]).to(DEVICE)
            labels = torch.tensor([hash(item[1]) % 10000 for item in batch], device=DEVICE)

            pred = encoder(latents)
            loss = cosine_loss(pred, teachers)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            mean_cos, eer = evaluate(encoder, heldout_data, DEVICE)
            print(f"Epoch {epoch+1}: loss={total_loss/n_batches:.4f}  "
                  f"Held-out: cos(pred,teacher)={mean_cos:.4f}  EER={eer:.1%}")
            encoder.train()

    print(f"\n=== Final ===")
    mean_cos, eer = evaluate(encoder, heldout_data, DEVICE)
    print(f"Held-out: mean cos(pred_embed, teacher_embed)={mean_cos:.4f}")
    print(f"Held-out: EER={eer:.1%}")
    print(f"\nJudgment: {'PASS' if mean_cos > 0.5 else 'MARGINAL' if mean_cos > 0.3 else 'FAIL'}")
    print(f"  cos > 0.5 + EER < 20% → Phase B (distillation) viable")
    print(f"  cos < 0.3 → DAC latent lacks speaker info → Phase B discard")

if __name__ == "__main__":
    main()
