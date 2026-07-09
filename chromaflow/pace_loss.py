"""PACE — Point-Anchored Chromophore-stable Encoding (train-only encoder reg).

Two stain-augmented views of the same crop share point annotations. Each point's
core prototype (encoder layer2 feature at the point) is pulled together across
views (positive) and pushed apart from neighboring points' prototypes (negatives).
H/E only re-weights hard negatives (competing pairs with an H-valley between them),
never used as a foreground/background label. Projection head is train-only.

NOTE (preflight): same-point cross-stain cos is already 0.929 (encoder already
stain-robust via stain_aug), so the lever here is the DISCRIMINATION term, not
stain-invariance. eta>0 (H-valley hard-neg) focuses negatives on competing pairs.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
# skimage Ruifrok-Johnston H&E&DAB stain matrix
RGB_FROM_HED = torch.tensor([[0.65, 0.70, 0.29],
                             [0.07, 0.99, 0.11],
                             [0.27, 0.57, 0.78]], dtype=torch.float32)


def hed_jitter_tensor(x_norm: torch.Tensor, sigma: float = 0.08, bias: float = 0.0) -> torch.Tensor:
    """Differentiable-ish HED-space stain jitter on an ImageNet-normalized RGB batch.
    Returns a re-normalized RGB batch (same shape). Used to make view B."""
    dev, dt = x_norm.device, torch.float32
    mean = IMAGENET_MEAN.to(dev, dt); std = IMAGENET_STD.to(dev, dt)
    rgb = (x_norm.to(dt) * std + mean).clamp(1e-4, 1.0)
    rgb_from_hed = RGB_FROM_HED.to(dev, dt)
    hed_from_rgb = torch.linalg.inv(rgb_from_hed)
    od = -torch.log(rgb)                                   # (B,3,H,W)
    hed = torch.einsum("bchw,ck->bkhw", od, hed_from_rgb)
    B = x_norm.shape[0]
    a = 1.0 + (torch.rand(B, 3, 1, 1, device=dev, dtype=dt) * 2 - 1) * sigma
    b = (torch.rand(B, 3, 1, 1, device=dev, dtype=dt) * 2 - 1) * bias
    hed = hed * a + b
    od2 = torch.einsum("bchw,ck->bkhw", hed, rgb_from_hed)
    rgb2 = torch.exp(-od2).clamp(0.0, 1.0)
    return ((rgb2 - mean) / std).to(x_norm.dtype)


def _norm01_img(x):  # (H,W)
    return (x - x.min()) / (x.max() - x.min() + 1e-6)


def compute_pace_loss(proj_a, proj_b, point_map, stain_h,
                      tau: float = 0.2, r_n: float = 48.0, sigma_n: float = 24.0,
                      eta: float = 0.0, use_he: bool = False, stride: int = 8) -> torch.Tensor:
    B, D, Hf, Wf = proj_a.shape
    dev = proj_a.device
    total = proj_a.new_tensor(0.0)
    n_used = 0
    for b in range(B):
        pm = point_map[b]
        if pm.ndim == 3:
            pm = pm[0]
        ys, xs = torch.nonzero(pm > 0.5, as_tuple=True)
        if ys.numel() < 2:
            continue
        pts = torch.stack([ys, xs], 1).float()                    # (N,2) image coords
        fy = (ys // stride).clamp(0, Hf - 1).long()
        fx = (xs // stride).clamp(0, Wf - 1).long()
        pa = proj_a[b, :, fy, fx].t()                              # (N,D) normalized
        pb = proj_b[b, :, fy, fx].t()
        pa = F.normalize(pa, dim=1); pb = F.normalize(pb, dim=1)
        sim_pos = (pa * pb).sum(1) / tau                           # (N,)
        sim_neg = (pa @ pb.t()) / tau                              # (N,N)
        d2 = torch.cdist(pts, pts) ** 2                            # (N,N)
        eye = torch.eye(pts.shape[0], device=dev, dtype=torch.bool)
        nb = (d2 < (r_n * r_n)) & (~eye)
        w = torch.exp(-d2 / (2 * sigma_n * sigma_n)) * nb.float()
        if use_he and eta > 0:
            Hn = _norm01_img(stain_h[b] if stain_h[b].ndim == 2 else stain_h[b, 0])
            mid = ((pts[:, None, :] + pts[None, :, :]) / 2).long()
            my = mid[..., 0].clamp(0, Hn.shape[0] - 1)
            mx = mid[..., 1].clamp(0, Hn.shape[1] - 1)
            V = 1.0 - Hn[my, mx]                                   # (N,N) valley score
            w = w * (1.0 + eta * V)
        # InfoNCE: -log( e^pos / (e^pos + sum_j w_ij e^neg_ij) )
        neg_term = (w * torch.exp(sim_neg)).sum(1)                 # (N,)
        denom = torch.exp(sim_pos) + neg_term + 1e-8
        loss_i = -(sim_pos - torch.log(denom))
        # keep only points that actually have neighbors
        has_nb = nb.any(1)
        if has_nb.any():
            total = total + loss_i[has_nb].mean()
            n_used += 1
    if n_used == 0:
        return proj_a.new_tensor(0.0)
    return total / n_used


if __name__ == "__main__":
    torch.manual_seed(0)
    pa = F.normalize(torch.randn(2, 64, 28, 28), dim=1)
    pb = F.normalize(torch.randn(2, 64, 28, 28), dim=1)
    pm = torch.zeros(2, 224, 224); pm[:, ::20, ::20] = 1
    sh = torch.rand(2, 224, 224)
    x = torch.randn(2, 3, 224, 224)
    xb = hed_jitter_tensor(x, 0.08)
    print("jitter out shape", tuple(xb.shape), "finite", bool(torch.isfinite(xb).all()))
    for he in (False, True):
        l = compute_pace_loss(pa, pb, pm, sh, eta=0.3, use_he=he)
        print(f"pace loss (use_he={he}): {l.item():.4f}")
