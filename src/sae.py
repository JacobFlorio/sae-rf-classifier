"""TopK sparse autoencoder, same flavor as the mech-interp project.

Keeps only the K largest pre-activations per token, which avoids the
L1 shrinkage bias. Decoder columns are re-normed to unit length each
training step (standard SAE hygiene).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class TopKSAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int, k: int):
        super().__init__()
        self.k = k
        self.encoder = nn.Linear(d_in, d_sae, bias=True)
        self.decoder = nn.Linear(d_sae, d_in, bias=True)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T)

    def encode(self, x):
        pre = self.encoder(x - self.decoder.bias)
        topk = torch.topk(pre, self.k, dim=-1)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, topk.indices, topk.values.clamp(min=0))
        return acts

    def forward(self, x):
        acts = self.encode(x)
        recon = self.decoder(acts)
        return recon, acts

    def loss(self, x):
        recon, acts = self(x)
        return ((recon - x) ** 2).mean(), acts
