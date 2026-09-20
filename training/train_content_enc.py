"""Gender-invariant content encoder (Z8 critical path, full-data). Diagnosis:
ContentVec is speaker- but NOT gender-invariant -> male content mismatches female
frames -> M->F retrieval garbles (CER 0.96). Fix: an adapter z=E(ContentVec(x))
trained CONTRASTIVELY so z(x_t) == z(gender-perturbed x_t) (invariance to the
vocal-tract/pitch axis) while staying discriminative across frames (phonetic
content). Perturbation = GPU frequency-axis warp (STFT -> warp freq by r ->
ISTFT), which shifts formants+pitch together (the gender axis), duration-
preserving and fast (done on GPU in the loop; dataloader only loads audio).
Positives = same frame under warp; InfoNCE negatives = other frames. FULL data.
"""
from __future__ import annotations
import sys, argparse, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
from transformers import HubertModel

sys.path.insert(0, str(Path(__file__).parent))
from train_m1 import DEV
CV_SR, SEG = 16000, 2.0
_WIN = {}


def freq_warp(w, r):
    """w [B,L] 16k audio, r [B] warp factors -> formant+pitch shifted, same length."""
    nfft, hop = 512, 128
    if w.device not in _WIN:
        _WIN[w.device] = torch.hann_window(nfft, device=w.device)
    win = _WIN[w.device]
    S = torch.stft(w, nfft, hop, window=win, return_complex=True)      # [B,F,T]
    mag, ph = S.abs(), S.angle()
    B, Fb, T = mag.shape
    f = torch.arange(Fb, device=w.device).view(1, Fb, 1).float()
    gy = ((f / r.view(B, 1, 1)) / (Fb - 1)) * 2 - 1                    # source freq, normalized
    gx = torch.linspace(-1, 1, T, device=w.device).view(1, 1, T).expand(B, Fb, T)
    grid = torch.stack([gx, gy.expand(B, Fb, T)], dim=-1)
    magw = F.grid_sample(mag.unsqueeze(1), grid, align_corners=True, padding_mode="border")[:, 0]
    y = torch.istft(magw * torch.exp(1j * ph), nfft, hop, window=win, length=w.shape[-1])
    return y


class SegSet(Dataset):
    def __init__(self, roots):
        self.files = []
        for r in roots:
            self.files += list(Path(r).rglob("*.pt"))
        random.shuffle(self.files)
        self.n = int(SEG * CV_SR)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        try:
            y, _ = librosa.load(torch.load(self.files[i], weights_only=False)["path"], sr=CV_SR, mono=True)
            if len(y) < self.n:
                y = np.pad(y, (0, self.n - len(y)))
            s = random.randint(0, len(y) - self.n)
            return torch.from_numpy(y[s:s + self.n]).float()
        except Exception:
            return self.__getitem__((i + 1) % len(self.files))


class Adapter(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(768, 512, 5, padding=2), nn.GELU(),
            nn.Conv1d(512, 512, 5, padding=2), nn.GELU(),
            nn.Conv1d(512, d, 1))

    def forward(self, cv):
        return F.normalize(self.net(cv), dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", default=["../data/female_real_feat", "../data/female_tts_feat", "../data/male_feat"])
    ap.add_argument("--out", default="checkpoints/content_enc"); ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--tau", type=float, default=0.1); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=2000)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cv = HubertModel.from_pretrained("lengyue233/content-vec-best").to(DEV).eval()
    for p in cv.parameters():
        p.requires_grad_(False)
    enc = Adapter().to(DEV)
    opt = torch.optim.AdamW(enc.parameters(), args.lr, betas=(0.8, 0.99))
    dl = DataLoader(SegSet(args.roots), batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    drop_last=True, persistent_workers=args.workers > 0)
    print(f"content-enc | files {len(dl.dataset)} | GPU freq-warp InfoNCE | steps {args.steps}", flush=True)

    @torch.no_grad()
    def cvf(w):
        return cv(w.to(DEV)).last_hidden_state.transpose(1, 2)

    step = 0
    while step < args.steps:
        for y in dl:
            y = y.to(DEV)
            r = torch.empty(y.shape[0], device=DEV).uniform_(0.78, 1.28)
            with torch.no_grad():
                yp = freq_warp(y, r)
            zo, zp = enc(cvf(y)), enc(cvf(yp))
            T = min(zo.shape[-1], zp.shape[-1])
            a = zo[..., :T].permute(0, 2, 1).reshape(-1, zo.shape[1])
            b = zp[..., :T].permute(0, 2, 1).reshape(-1, zp.shape[1])
            logits = (a @ b.t()) / args.tau
            lbl = torch.arange(a.shape[0], device=DEV)
            loss = 0.5 * (F.cross_entropy(logits, lbl) + F.cross_entropy(logits.t(), lbl))
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 5.0); opt.step()
            if step % 100 == 0:
                acc = (logits.argmax(1) == lbl).float().mean().item()
                print(f"step {step} nce {loss.item():.3f} pos-acc {acc:.3f}", flush=True)
            if step % args.save_every == 0 and step > 0:
                torch.save({"enc": enc.state_dict(), "step": step}, out / "last.pt")
            step += 1
            if step >= args.steps:
                break
    torch.save({"enc": enc.state_dict(), "step": step}, out / "last.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
