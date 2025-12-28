import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    return (x.round() - x).detach() + x


class QuantLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        group_size: int = 128,
        bit: int = 4,
        gamma_dist: Optional[nn.Parameter] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.bit = bit

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.requires_grad = False

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            self.bias.requires_grad = False
        else:
            self.register_parameter("bias", None)

        self.gamma_dist = gamma_dist
        if self.gamma_dist is not None:
            self.gamma_dist.requires_grad = True

        group_count = in_features // group_size
        self.gamma_task = nn.Parameter(torch.ones(out_features, group_count, 1))

    def _group_reshape(self, weight: torch.Tensor) -> torch.Tensor:
        group_count = weight.shape[1] // self.group_size
        return weight.view(weight.shape[0], group_count, self.group_size)

    def compute_group_scale(self, weight: torch.Tensor) -> torch.Tensor:
        weight_grouped = self._group_reshape(weight)
        max_val = weight_grouped.abs().amax(dim=2, keepdim=True)
        scale = max_val / (2 ** (self.bit - 1) - 1)
        return scale.clamp_min(1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        gamma_dist = None
        if self.gamma_dist is not None:
            gamma_dist = F.softplus(self.gamma_dist)
            weight = weight * gamma_dist

        scale = self.compute_group_scale(weight)
        weight_grouped = self._group_reshape(weight)
        qmin = -(2 ** (self.bit - 1))
        qmax = 2 ** (self.bit - 1) - 1
        weight_int = _ste_round(weight_grouped / scale).clamp(qmin, qmax)
        weight_dequant = weight_int * scale
        weight_dequant = weight_dequant.view_as(weight)

        if gamma_dist is not None:
            weight_dequant = weight_dequant / gamma_dist

        weight_dequant = weight_dequant * self.gamma_task.view_as(weight)

        return F.linear(x, weight_dequant, self.bias)
