"""Denoising molecular autoencoder; frozen embeddings for supervised branches."""
from __future__ import annotations

import torch
from torch import nn


class MolecularAutoencoder(nn.Module):
    def __init__(self, fingerprint_dim: int = 3072, descriptor_dim: int = 20,
                 latent_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.fingerprint_encoder = nn.Sequential(nn.Linear(fingerprint_dim, 512), nn.LayerNorm(512),
                                                 nn.GELU(), nn.Dropout(dropout), nn.Linear(512, 128), nn.GELU())
        self.descriptor_encoder = nn.Sequential(nn.Linear(descriptor_dim, 32), nn.GELU())
        self.fusion = nn.Sequential(nn.Linear(160, latent_dim), nn.LayerNorm(latent_dim), nn.GELU())
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 512), nn.GELU(), nn.Linear(512, fingerprint_dim))
        self.descriptor_decoder = nn.Sequential(nn.Linear(latent_dim, 64), nn.GELU(), nn.Linear(64, descriptor_dim))

    def encode(self, fingerprints: torch.Tensor, descriptors: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat([self.fingerprint_encoder(fingerprints), self.descriptor_encoder(descriptors)], dim=-1))

    def forward(self, fingerprints: torch.Tensor, descriptors: torch.Tensor):
        embedding = self.encode(fingerprints, descriptors)
        return self.decoder(embedding), self.descriptor_decoder(embedding)

    def freeze(self) -> MolecularAutoencoder:
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self


def corrupt_bits(bits: torch.Tensor, mask_probability: float, bit_noise: float) -> torch.Tensor:
    masked = bits * (torch.rand_like(bits) >= mask_probability)
    flips = torch.rand_like(bits) < bit_noise
    return torch.where(flips, 1.0 - masked, masked)
