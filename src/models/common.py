import torch
import torch.nn as nn

def get_device(pref: str = "cuda") -> str:
    return "cuda" if (pref == "cuda" and torch.cuda.is_available()) else "cpu"

def init_kaiming(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def set_trainable(model: nn.Module, flag: bool = True) -> nn.Module:
    for p in model.parameters():
        p.requires_grad = flag
    return model

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
