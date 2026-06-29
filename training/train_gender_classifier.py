"""
C2-2 Phase 1: Gender Classifier for source leakage evaluation.

Trains a binary classifier on ECAPA embeddings to distinguish male/female.
Used as:
  1. Evaluation metric: P(male) in converted output
  2. Training loss: gender leakage penalty in B1 adapter training

Usage:
  cd training
  uv run python train_gender_classifier.py
"""
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
import librosa

sys.path.insert(0, str(Path(__file__).parent))
from train_phase3b import DEVICE, DAC_SR, load_ecapa, resample_16k, ecapa_embed

VCTK_WAV = Path("../data/vctk_200")
SPEAKER_INFO = Path("../data/vctk/VCTK-Corpus/VCTK-Corpus/speaker-info.txt")
OUT_DIR = Path("checkpoints/gender_classifier")


def load_gender_labels():
    labels = {}
    if not SPEAKER_INFO.exists():
        print(f"WARNING: speaker-info.txt not found at {SPEAKER_INFO}")
        return labels
    for line in SPEAKER_INFO.read_text().strip().split("\n")[1:]:
        parts = line.split()
        if len(parts) >= 3:
            spk_id = parts[0]
            gender = parts[2].upper()
            if gender in ("F", "M"):
                labels[f"p{spk_id}"] = 0.0 if gender == "F" else 1.0
    return labels


def extract_all_embeddings(ecapa, max_per_speaker=5):
    speaker_dirs = sorted([d for d in VCTK_WAV.iterdir() if d.is_dir()])
    embeddings = {}
    
    for si, sd in enumerate(speaker_dirs):
        wavs = sorted(sd.glob("*.wav"))[:max_per_speaker]
        embs = []
        for w in wavs:
            audio, sr = sf.read(str(w), dtype="float32")
            if audio.ndim > 1:
                audio = audio[:, 0]
            if sr != DAC_SR:
                audio = librosa.resample(audio.astype(np.float64), orig_sr=sr, target_sr=DAC_SR).astype(np.float32)
            if len(audio) < 8000:
                continue
            with torch.no_grad():
                audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(DEVICE)
                audio_16k = resample_16k(audio_t)
                emb = ecapa_embed(ecapa, audio_16k).squeeze(0).cpu()
            embs.append(emb)
        if embs:
            embeddings[sd.name] = torch.stack(embs).mean(0)
        if (si + 1) % 20 == 0:
            print(f"  [{si+1}/{len(speaker_dirs)}] speakers", flush=True)
    
    return embeddings


def train_classifier(embeddings, labels):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, LeaveOneOut
    
    spk_ids = sorted(embeddings.keys())
    X = []
    y = []
    for spk in spk_ids:
        if spk in labels:
            X.append(embeddings[spk].numpy())
            y.append(labels[spk])
    
    X = np.array(X)
    y = np.array(y)
    
    n_female = int((y == 0).sum())
    n_male = int((y == 1).sum())
    print(f"Dataset: {len(X)} speakers (F={n_female} M={n_male})")
    
    loo = LeaveOneOut()
    clf = LogisticRegression(C=1.0, max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=loo, scoring="accuracy")
    print(f"Leave-one-out accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    
    clf.fit(X, y)
    
    train_acc = clf.score(X, y)
    print(f"Full-data train accuracy: {train_acc:.3f}")
    
    y_prob = clf.predict_proba(X)[:, 1]
    for i in range(len(X)):
        spk = [s for s in spk_ids if s in labels][i]
        if y_prob[i] > 0.5 and y[i] == 0 or y_prob[i] <= 0.5 and y[i] == 1:
            print(f"  MISCLASSIFIED: {spk} gender={'M' if y[i]==1 else 'F'} P(male)={y_prob[i]:.3f}")
    
    return clf, X, y


def save_classifier(clf, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    W = torch.from_numpy(clf.coef_).float()  # [1, 192]
    b = torch.from_numpy(clf.intercept_).float()  # [1]
    
    torch.save({
        "weight": W,
        "bias": b,
        "classes": clf.classes_.tolist(),
    }, output_dir / "gender_classifier.pt")
    
    np.save(str(output_dir / "gender_embeddings.npy"), None, allow_pickle=True)
    print(f"Saved: {output_dir / 'gender_classifier.pt'}")
    print(f"  weight: {W.shape}  bias: {b.shape}")
    
    coeff_norm = W.norm().item()
    print(f"  weight L2 norm: {coeff_norm:.4f}")


def main():
    print("=== C2-2 Phase 1: Gender Classifier ===\n")
    
    labels = load_gender_labels()
    print(f"Gender labels: {len(labels)} speakers from speaker-info.txt")
    n_f = sum(1 for v in labels.values() if v == 0.0)
    n_m = sum(1 for v in labels.values() if v == 1.0)
    print(f"  Female: {n_f}, Male: {n_m}\n")
    
    print("Loading ECAPA...")
    ecapa = load_ecapa()
    
    print("Extracting embeddings...")
    embeddings = extract_all_embeddings(ecapa, max_per_speaker=5)
    print(f"Extracted: {len(embeddings)} speakers\n")
    
    print("Training classifier...")
    clf, X, y = train_classifier(embeddings, labels)
    
    save_classifier(clf, OUT_DIR)
    
    print("\nDone. Usage in B1 training:")
    print("  gender_prob = sigmoid(ECAPA(output_audio) @ W + b)")
    print("  L_gender = gender_prob  # minimize P(male)")
    print(f"  Load: torch.load('{OUT_DIR / 'gender_classifier.pt'}')")


if __name__ == "__main__":
    main()
