import torch
import torch.nn as nn

from quant_layer import QuantLinear
from model_utils import _iter_decoder_layers


def _expand_group_param(param: torch.Tensor, group_size: int) -> torch.Tensor:
    if param.dim() == 2:
        return param
    out, group_count, _ = param.shape
    return param.view(out, group_count * group_size)


def fuse_and_export(model: nn.Module) -> nn.Module:
    with torch.no_grad():
        for layer in _iter_decoder_layers(model):
            attn = layer.self_attn
            mlp = layer.mlp

            shared_gamma_dist = None
            for proj_name in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
                module = getattr(attn, proj_name, None) or getattr(mlp, proj_name, None)
                if not isinstance(module, QuantLinear):
                    continue
                if shared_gamma_dist is None and module.gamma_dist is not None:
                    shared_gamma_dist = torch.nn.functional.softplus(module.gamma_dist)

            if shared_gamma_dist is not None:
                layer.input_layernorm.weight.mul_(shared_gamma_dist.squeeze(0))

            o_proj = attn.o_proj
            if isinstance(o_proj, QuantLinear) and o_proj.gamma_dist is not None:
                o_gamma = torch.nn.functional.softplus(o_proj.gamma_dist)
                attn.v_proj.weight.mul_(o_gamma.squeeze(0))

            for module in (
                attn.q_proj,
                attn.k_proj,
                attn.v_proj,
                attn.o_proj,
                mlp.gate_proj,
                mlp.up_proj,
                mlp.down_proj,
            ):
                if not isinstance(module, QuantLinear):
                    continue
                weight = module.weight
                gamma_dist = (
                    torch.nn.functional.softplus(module.gamma_dist)
                    if module.gamma_dist is not None
                    else None
                )
                if gamma_dist is not None:
                    weight = weight * gamma_dist
                scale_group = module.compute_group_scale(weight)
                gamma_task = module.gamma_task
                scale_group = scale_group * gamma_task
                scale_full = _expand_group_param(scale_group, module.group_size)
                if gamma_dist is not None:
                    scale_full = scale_full / gamma_dist
                qmin = -(2 ** (module.bit - 1))
                qmax = 2 ** (module.bit - 1) - 1
                weight_int = (module.weight / scale_full).round().clamp(qmin, qmax)
                module.weight.data.copy_(weight_int * scale_full)
                module.register_buffer("fused_scale", scale_full)
                module.gamma_dist = None
                module.gamma_task = None

    return model
