from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .low_precision import Linear4Bit, block_dequantize_4bit


class QLoRALinear(Linear4Bit):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        group_size: int = 16,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias, group_size)
        self.requires_grad_(False)

        # The adapters are created after requires_grad_(False) above, so they stay trainable.
        # They are kept in float32 for stable training.
        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False, dtype=torch.float32)
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False, dtype=torch.float32)
        # lora_a keeps nn.Linear's default random init; lora_b starts at zero so the adapter adds
        # nothing until it is trained (output initially matches BigNet4Bit exactly).
        torch.nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear4Bit.forward runs entirely under no_grad, which would cut the gradient path back through
        # this layer to the adapters of earlier layers. Instead, only the dequantization (a constant) is
        # done under no_grad, and the linear op runs with autograd so gradients flow through x.
        with torch.no_grad():
            weight = block_dequantize_4bit(self.weight_q4, self.weight_norm).view(self._shape)
        return torch.nn.functional.linear(x, weight, self.bias) + self.lora_b(self.lora_a(x))


class QLoRABigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels, lora_dim, group_size):
            super().__init__()
            self.model = torch.nn.Sequential(
                QLoRALinear(channels, channels, lora_dim, group_size),
                torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim, group_size),
                torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim, group_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32, group_size: int = 16):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim, group_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> QLoRABigNet:
    net = QLoRABigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
