import torch
import torch.nn as nn

from quant_layer import QuantLinear


def build_toy_model() -> nn.Module:
    return nn.Sequential(
        QuantLinear(256, 256, group_size=128, bit=4, gamma_dist=None),
        nn.GELU(),
        QuantLinear(256, 128, group_size=128, bit=4, gamma_dist=None),
    )


def main() -> None:
    torch.manual_seed(0)
    model = build_toy_model()

    for module in model.modules():
        if isinstance(module, QuantLinear):
            module.weight.requires_grad = False
            if module.bias is not None:
                module.bias.requires_grad = False

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
    )

    for step in range(3):
        inputs = torch.randn(8, 256)
        targets = torch.randn(8, 128)
        outputs = model(inputs)
        loss = nn.functional.mse_loss(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"step={step} loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
