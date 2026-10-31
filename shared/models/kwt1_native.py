"""Evaluation port of Arm's post-norm, one-head KWT-1 (64 channels, 12 blocks).

TensorFlow weights are transposed by experiments/export.py and checked against
native logits. Output classes keep the order used during source training.
Architecture source: https://github.com/ARM-software/keyword-transformer
"""
import torch
from torch import nn
from torch.nn import functional as F


class NativeKWTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = nn.Linear(64, 64, bias=False)
        self.k = nn.Linear(64, 64, bias=False)
        self.v = nn.Linear(64, 64, bias=False)
        self.proj = nn.Linear(64, 64)
        self.ff1, self.ff2 = nn.Linear(64, 256), nn.Linear(256, 64)
        self.ln1, self.ln2 = nn.LayerNorm(64, eps=1e-6), nn.LayerNorm(64, eps=1e-6)

    def forward(self, x):
        attention = (self.q(x) @ self.k(x).transpose(-1, -2) / 8.).softmax(-1)
        x = self.ln1(x + self.proj(attention @ self.v(x)))
        return self.ln2(x + self.ff2(F.gelu(self.ff1(x), approximate='none')))


class NativeKWT(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, 99, 64))
        self.cls = nn.Parameter(torch.zeros(1, 1, 64))
        self.patch = nn.Linear(40, 64)
        self.blocks = nn.ModuleList([NativeKWTBlock() for _ in range(12)])
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.cat([self.cls.expand(len(x), -1, -1), self.patch(x)], 1) + self.pos
        for block in self.blocks:
            x = block(x)
        return self.head(x[:, 0])
