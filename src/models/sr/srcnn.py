import torch, torch.nn as nn, torch.nn.functional as F
from .base import SRModel

class _SRCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,64,9,padding=4)
        self.conv2 = nn.Conv2d(64,32,5,padding=2)
        self.conv3 = nn.Conv2d(32,3,5,padding=2)
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return torch.clamp(self.conv3(x),0,1)

class SRCNN(SRModel):
    name = "srcnn"
    def __init__(self, device="cuda"):
        super().__init__(device)
        self._m = _SRCNN().to(device)
    def module(self): return self._m
    def forward_tensor(self, lr_up): return self._m(lr_up)
