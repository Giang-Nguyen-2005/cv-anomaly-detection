from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import nn

from .convlstm import ConvLSTM


class ConvLSTMAutoEncoder(nn.Module):
    """
    Simple sequence-to-sequence ConvLSTM autoencoder.

    Input : (B, T, C, H, W)
    Output: (B, T, C, H, W)
    """

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: Sequence[int] = (64, 32, 32),
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.encoder = ConvLSTM(
            input_dim=input_dim, hidden_dims=hidden_dims, kernel_size=kernel_size
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], hidden_dims[-1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dims[-1], input_dim, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"Expected input shape (B, T, C, H, W), got {x.shape}")

        layer_outputs, _ = self.encoder(x)
        latent_sequence = layer_outputs[-1]  # (B, T, hidden, H, W)

        reconstructions = []
        for t in range(latent_sequence.size(1)):
            rec_t = self.decoder(latent_sequence[:, t])
            reconstructions.append(rec_t)

        return torch.stack(reconstructions, dim=1)

    @staticmethod
    def reconstruction_loss(
        reconstructed: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return nn.functional.mse_loss(reconstructed, target)


if __name__ == "__main__":
    model = ConvLSTMAutoEncoder(input_dim=1, hidden_dims=(64, 32, 32), kernel_size=3)
    dummy = torch.randn(2, 10, 1, 64, 64)
    out = model(dummy)
    print(f"Input shape : {dummy.shape}")
    print(f"Output shape: {out.shape}")
