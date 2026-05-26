from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from chess_engine.moves import MOVE_ENCODER


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = channels // reduction
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        s = self.pool(x).view(b, c)
        s = torch.relu(self.fc1(s))
        s = self.fc2(s)
        gamma, beta = s.chunk(2, dim=1)
        gamma = torch.sigmoid(gamma).view(b, c, 1, 1)
        beta = beta.view(b, c, 1, 1)
        return x * gamma + beta


class ResidualSEBlock(nn.Module):
    def __init__(self, channels: int, se_reduction: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SEBlock(channels, se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return torch.relu(out + residual)


class ChessNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 18,
        channels: int = 128,
        num_blocks: int = 10,
        se_reduction: int = 4,
        policy_channels: int = 32,
        value_hidden: int = 256,
        num_policy_moves: int | None = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        num_policy = num_policy_moves or MOVE_ENCODER.num_moves

        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[
                ResidualSEBlock(channels, se_reduction)
                for _ in range(num_blocks)
            ]
        )
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, policy_channels, 1, bias=False),
            nn.BatchNorm2d(policy_channels),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(policy_channels * 64, num_policy)

        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(64, value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(value_hidden, 1),
            nn.Tanh(),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.input_conv(x)
        h = self.blocks(h)
        p = self.policy_conv(h).reshape(x.size(0), -1)
        policy_logits = self.policy_fc(p)
        v = self.value_conv(h).reshape(x.size(0), -1)
        value = self.value_fc(v)
        return policy_logits, value

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model_from_config(cfg: dict) -> ChessNet:
    m = cfg.get("model", cfg)
    return ChessNet(
        input_channels=m.get("input_channels", 18),
        channels=m.get("channels", 128),
        num_blocks=m.get("num_blocks", 10),
        se_reduction=m.get("se_reduction", 4),
        policy_channels=m.get("policy_channels", 32),
        value_hidden=m.get("value_hidden", 256),
    )
