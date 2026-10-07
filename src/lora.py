import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """y = W x + scale * (alpha / r) * B A x; W is frozen and B is initialized to 0, so the output initially equals the base model"""
    # the weight update is constrained to a low-rank factorization ΔW = B·A, with B of shape (out, r) and A of shape (r, in)
    def __init__(self, base, r, alpha):
        super().__init__()
        self.base, self.factor, self.scale = base, alpha / r, 1.0  # scale: LoRA strength at inference
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + self.scale * self.factor * (x @ self.A.T @ self.B.T)


def apply_lora(model, r, alpha=None, targets=("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")):
    """freeze the whole model, then replace every nn.Linear whose name ends with one of targets by LoRALinear (only A and B are trainable)"""
    model.requires_grad_(False)
    for name, m in list(model.named_modules()):
        for child_name, child in list(m.named_children()):
            if isinstance(child, nn.Linear) and f"{name}.{child_name}".endswith(targets):
                setattr(m, child_name, LoRALinear(child, r, alpha or r))
    return model


def lora_state_dict(model):
    return {k: v for k, v in model.state_dict().items() if k.endswith((".A", ".B"))}


def set_lora_scale(model, scale):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.scale = scale
