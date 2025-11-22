import torch, torch.nn as nn, torch.nn.functional as F
from .base import SRModel

class _FSRCNN(nn.Module):
    def __init__(self, d=56, s=12, m=4):
        super().__init__()
        self.feature = nn.Conv2d(3, d, 5, padding=2)
        self.prelu_feature = nn.PReLU(num_parameters=1, init=0.25)

        self.shrink  = nn.Conv2d(d, s, 1)
        self.prelu_shrink = nn.PReLU(num_parameters=1, init=0.25)

        # mapping layers
        blocks = []
        for _ in range(m):
            blocks.append(nn.Conv2d(s, s, 3, padding=1))
            blocks.append(nn.PReLU(num_parameters=1, init=0.25))
        self.mapping = nn.Sequential(*blocks)

        self.expand  = nn.Conv2d(s, d, 1)
        self.prelu_expand = nn.PReLU(num_parameters=1, init=0.25)

        self.out     = nn.Conv2d(d, 3, 3, padding=1)

    def forward(self, x):
        x = self.prelu_feature(self.feature(x))
        x = self.prelu_shrink(self.shrink(x))
        x = self.mapping(x)
        x = self.prelu_expand(self.expand(x))
        return torch.clamp(self.out(x), 0, 1)

class FSRCNN(SRModel):
    name = "fsrcnn"
    def __init__(self, device="cuda"):
        super().__init__(device)
        self._m = _FSRCNN().to(device)
    def module(self): return self._m
    def forward_tensor(self, lr_up): return self._m(lr_up)
