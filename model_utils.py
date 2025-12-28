from typing import Iterable, Optional

import torch
import torch.nn as nn

from quant_layer import QuantLinear


def _iter_decoder_layers(model: nn.Module) -> Iterable[nn.Module]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError("Could not find decoder layers on the model.")


def _replace_linear(parent: nn.Module, name: str, module: nn.Linear, quant: QuantLinear) -> None:
    quant.weight.data.copy_(module.weight.data)
    if module.bias is not None and quant.bias is not None:
        quant.bias.data.copy_(module.bias.data)
    setattr(parent, name, quant)


def inject_peft_model(model: nn.Module, group_size: int = 128, bit: int = 4) -> nn.Module:
    for layer in _iter_decoder_layers(model):
        attn = layer.self_attn
        mlp = layer.mlp

        shared_gamma_dist: Optional[nn.Parameter] = None
        for proj_name in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
            module = getattr(attn, proj_name, None) or getattr(mlp, proj_name, None)
            if module is None:
                continue
            if shared_gamma_dist is None:
                shared_gamma_dist = nn.Parameter(torch.ones(1, module.in_features))
            quant = QuantLinear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                group_size=group_size,
                bit=bit,
                gamma_dist=shared_gamma_dist,
            )
            _replace_linear(attn if hasattr(attn, proj_name) else mlp, proj_name, module, quant)

        o_proj = attn.o_proj
        o_gamma_dist = nn.Parameter(torch.ones(1, o_proj.in_features))
        o_quant = QuantLinear(
            o_proj.in_features,
            o_proj.out_features,
            bias=o_proj.bias is not None,
            group_size=group_size,
            bit=bit,
            gamma_dist=o_gamma_dist,
        )
        _replace_linear(attn, "o_proj", o_proj, o_quant)

        down_proj = mlp.down_proj
        down_quant = QuantLinear(
            down_proj.in_features,
            down_proj.out_features,
            bias=down_proj.bias is not None,
            group_size=group_size,
            bit=bit,
            gamma_dist=None,
        )
        _replace_linear(mlp, "down_proj", down_proj, down_quant)

    return model
