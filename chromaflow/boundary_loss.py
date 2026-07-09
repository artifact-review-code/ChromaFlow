"""L_boundary — geometric touching-boundary supervision."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def compute_boundary_loss(
    boundary_logits: torch.Tensor,
    clu: torch.Tensor,
    d_to_clu1: torch.Tensor,
    d_to_clu1_2nd: torch.Tensor,
    sigma: float = 3.0,
) -> torch.Tensor:
    if boundary_logits.ndim == 3:
        bl = boundary_logits.unsqueeze(1)
    else:
        bl = boundary_logits
    d1 = d_to_clu1.unsqueeze(1) if d_to_clu1.ndim == 3 else d_to_clu1
    d2 = d_to_clu1_2nd.unsqueeze(1) if d_to_clu1_2nd.ndim == 3 else d_to_clu1_2nd
    d1 = d1.to(dtype=bl.dtype)
    d2 = d2.to(dtype=bl.dtype)
    target = torch.exp(-d1 / float(sigma)) * torch.exp(-d2 / float(sigma))
    target = target.clamp(0.0, 1.0)
    return F.binary_cross_entropy_with_logits(bl, target, reduction="mean")


if __name__ == "__main__":
    torch.manual_seed(0)
    bl = torch.randn(2, 1, 32, 32, requires_grad=True)
    clu = torch.randint(0, 3, (2, 32, 32))
    d1 = torch.rand(2, 32, 32) * 30
    d2 = d1 + torch.rand(2, 32, 32) * 10
    loss = compute_boundary_loss(bl, clu, d1, d2)
    loss.backward()
    print(f"loss={loss.item():.4f}  grad_norm={bl.grad.norm().item():.5f}")
