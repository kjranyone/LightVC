"""
Phase 3 model: Target Latent Generator (TLG)

Input: z_s [B, T, 1024], f0 [B, T], energy [B, T], timbre [B, 192]
Output: z_t_like [B, T, 1024]

Architecture:
  - Content projection + prosody injection
  - FiLM conditioning with target timbre
  - Causal Transformer encoder
  - Output projection to 1024-dim latent space
"""
import torch
import torch.nn as nn


class TLG(nn.Module):
    def __init__(
        self,
        content_dim=1024,
        hidden_dim=512,
        timbre_dim=192,
        n_heads=8,
        n_layers=6,
        max_len=2048,
        causal=True,
    ):
        super().__init__()
        self.causal = causal
        self.hidden_dim = hidden_dim

        self.content_proj = nn.Linear(content_dim, hidden_dim)
        self.f0_proj = nn.Linear(1, hidden_dim)
        self.energy_proj = nn.Linear(1, hidden_dim)

        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        self.film_gamma = nn.Linear(timbre_dim, hidden_dim)
        self.film_beta = nn.Linear(timbre_dim, hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, content_dim)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_s, f0, energy, timbre):
        """
        z_s:     [B, T, 1024]
        f0:      [B, T]
        energy:  [B, T]
        timbre:  [B, 192]
        Returns: z_t_like [B, T, 1024]
        """
        B, T, _ = z_s.shape

        h = self.content_proj(z_s)
        h = h + self.f0_proj(f0.unsqueeze(-1))
        h = h + self.energy_proj(energy.unsqueeze(-1))

        pos = torch.arange(T, device=z_s.device).unsqueeze(0)
        h = h + self.pos_emb(pos)

        gamma = self.film_gamma(timbre).unsqueeze(1)
        beta = self.film_beta(timbre).unsqueeze(1)
        h = h * (1 + gamma) + beta

        if self.causal:
            mask = torch.triu(
                torch.ones(T, T, device=z_s.device, dtype=torch.bool),
                diagonal=1,
            )
            h = self.transformer(h, mask=mask)
        else:
            h = self.transformer(h)

        h = self.norm(h)
        z_delta = self.out_proj(h)
        z_t_like = z_s + z_delta
        return z_t_like


class TLG_Codec(nn.Module):
    """Predicts discrete RVQ codes for depths 1-8 directly.

    Avoids the quantization cascade sensitivity of continuous latent prediction.
    CE loss on codebook indices is the primary training signal.
    """

    def __init__(
        self,
        content_dim=1024,
        hidden_dim=512,
        timbre_dim=192,
        n_heads=8,
        n_layers=6,
        n_depths=8,
        codebook_size=1024,
        max_len=2048,
    ):
        super().__init__()
        self.n_depths = n_depths

        self.content_proj = nn.Linear(content_dim, hidden_dim)
        self.f0_proj = nn.Linear(1, hidden_dim)
        self.energy_proj = nn.Linear(1, hidden_dim)

        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        self.film_gamma = nn.Linear(timbre_dim, hidden_dim)
        self.film_beta = nn.Linear(timbre_dim, hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        self.depth_emb = nn.Embedding(n_depths, hidden_dim)
        nn.init.normal_(self.depth_emb.weight, std=0.02)
        self.code_head = nn.Linear(hidden_dim, codebook_size)

    def forward(self, z_s, f0, energy, timbre):
        """
        z_s:     [B, T, 1024]
        f0:      [B, T]
        energy:  [B, T]
        timbre:  [B, 192]
        Returns: logits [B, n_depths, T, codebook_size]
        """
        B, T, _ = z_s.shape

        h = self.content_proj(z_s)
        h = h + self.f0_proj(f0.unsqueeze(-1))
        h = h + self.energy_proj(energy.unsqueeze(-1))

        pos = torch.arange(T, device=z_s.device).unsqueeze(0)
        h = h + self.pos_emb(pos)

        gamma = self.film_gamma(timbre).unsqueeze(1)
        beta = self.film_beta(timbre).unsqueeze(1)
        h = h * (1 + gamma) + beta

        h = self.transformer(h)
        h = self.norm(h)

        depths = torch.arange(self.n_depths, device=z_s.device)
        depth_emb = self.depth_emb(depths)
        h_expanded = h.unsqueeze(1) + depth_emb.reshape(1, self.n_depths, 1, -1)

        logits = self.code_head(h_expanded)
        return logits


class TLG_Embed(nn.Module):
    """Predicts codebook embeddings (8-dim per depth) instead of discrete codes.

    Regression on 8-dim vectors is far easier than 1024-class CE.
    At inference: nearest-neighbor quantization recovers discrete codes.
    """

    def __init__(
        self,
        content_dim=1024,
        hidden_dim=512,
        timbre_dim=192,
        n_heads=8,
        n_layers=6,
        n_depths=8,
        codebook_dim=8,
        max_len=2048,
    ):
        super().__init__()
        self.n_depths = n_depths

        self.content_proj = nn.Linear(content_dim, hidden_dim)
        self.f0_proj = nn.Linear(1, hidden_dim)
        self.energy_proj = nn.Linear(1, hidden_dim)

        self.pos_emb = nn.Embedding(max_len, hidden_dim)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

        self.film_gamma = nn.Linear(timbre_dim, hidden_dim)
        self.film_beta = nn.Linear(timbre_dim, hidden_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        self.depth_emb = nn.Embedding(n_depths, hidden_dim)
        nn.init.normal_(self.depth_emb.weight, std=0.02)
        self.embed_head = nn.Linear(hidden_dim, codebook_dim)

    def forward(self, z_s, f0, energy, timbre):
        """
        Returns: predicted codebook embeddings [B, n_depths, T, codebook_dim]
        """
        B, T, _ = z_s.shape

        h = self.content_proj(z_s)
        h = h + self.f0_proj(f0.unsqueeze(-1))
        h = h + self.energy_proj(energy.unsqueeze(-1))

        pos = torch.arange(T, device=z_s.device).unsqueeze(0)
        h = h + self.pos_emb(pos)

        gamma = self.film_gamma(timbre).unsqueeze(1)
        beta = self.film_beta(timbre).unsqueeze(1)
        h = h * (1 + gamma) + beta

        h = self.transformer(h)
        h = self.norm(h)

        depths = torch.arange(self.n_depths, device=z_s.device)
        depth_emb = self.depth_emb(depths)
        h_expanded = h.unsqueeze(1) + depth_emb.reshape(1, self.n_depths, 1, -1)

        return self.embed_head(h_expanded)
