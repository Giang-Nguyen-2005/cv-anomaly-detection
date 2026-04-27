from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim

        self.gates = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(
        self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x_t, h_prev], dim=1)
        gates = self.gates(combined)
        i, f, o, g = torch.chunk(gates, chunks=4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t

    def init_hidden(
        self, batch_size: int, spatial_size: Tuple[int, int], device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        height, width = spatial_size
        h = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        return h, c


class ConvLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dims: Sequence[int] = (64, 32, 32),
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer.")

        self.hidden_dims = list(hidden_dims)
        cells: List[ConvLSTMCell] = []
        current_input_dim = input_dim

        for hidden_dim in self.hidden_dims:
            cells.append(
                ConvLSTMCell(
                    input_dim=current_input_dim,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                )
            )
            current_input_dim = hidden_dim

        self.cells = nn.ModuleList(cells)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Tensor with shape (B, T, C, H, W)

        Returns:
            layer_outputs: list tensors, each shape (B, T, hidden_dim, H, W)
            last_states: list[(h, c)] for each layer
        """
        if x.dim() != 5:
            raise ValueError(f"Expected input shape (B, T, C, H, W), got {x.shape}")

        batch_size, seq_len, _, height, width = x.shape
        device = x.device

        layer_input = x
        layer_outputs: List[torch.Tensor] = []
        last_states: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for cell in self.cells:
            h_t, c_t = cell.init_hidden(
                batch_size=batch_size, spatial_size=(height, width), device=device
            )
            outputs = []
            for t in range(seq_len):
                h_t, c_t = cell(layer_input[:, t], h_t, c_t)
                outputs.append(h_t)

            output_tensor = torch.stack(outputs, dim=1)
            layer_outputs.append(output_tensor)
            last_states.append((h_t, c_t))
            layer_input = output_tensor

        return layer_outputs, last_states
