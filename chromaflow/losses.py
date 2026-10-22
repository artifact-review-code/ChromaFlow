from __future__ import annotations

import torch
import torch.nn.functional as F

import math
from typing import Optional

from chromaflow.affinity_loss import PathIndex, affinity_loss


# ---- D8 Evidential Uncertainty utilities -----------------------------------
def evidential_mse(alpha: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:
    """Sensoy MSE: E[(y - p)^2] = (y - p)^2 + p(1-p)/(S+1), per pixel.

    Args:
        alpha:         (B, K, H, W)  Dirichlet concentrations, all >= 1
        target_onehot: (B, K, H, W)  one-hot {0,1}, same K as alpha
    Returns: (B, 1, H, W) per-pixel evidential MSE.
    """
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S
    mse = (target_onehot - p) ** 2
    var = p * (1.0 - p) / (S + 1.0)
    return (mse + var).sum(dim=1, keepdim=True)


def evidential_kl(alpha: torch.Tensor, target_onehot: torch.Tensor) -> torch.Tensor:
    """Sensoy KL: KL[ Dir(alpha_tilde) || Dir(1,1,...) ].

    alpha_tilde = (1 - y) * alpha + y * 1   keeps correct-class evidence,
    zeros out wrong-class evidence. KL pulls wrong-class alpha → 1 (no evidence).
    """
    K = alpha.shape[1]
    alpha_tilde = (1.0 - target_onehot) * alpha + target_onehot
    S_tilde = alpha_tilde.sum(dim=1, keepdim=True)
    # Use float32 for lgamma/digamma stability under AMP.
    a32 = alpha_tilde.float()
    S32 = S_tilde.float()
    K_t = torch.tensor(float(K), device=a32.device)
    kl = (torch.lgamma(S32)
          - torch.lgamma(K_t)
          - torch.lgamma(a32).sum(dim=1, keepdim=True)
          + ((a32 - 1.0) * (torch.digamma(a32) - torch.digamma(S32))).sum(dim=1, keepdim=True))
    return kl.to(alpha.dtype)
# ------------------------------------------------------------------------------

_PATH_INDEX_CACHE = {}
_PATH_BUFFERS_CACHE = {}

def _get_path_index(radius: float, crop_size: int, device):
    key = (radius, crop_size)
    if key not in _PATH_INDEX_CACHE:
        _PATH_INDEX_CACHE[key] = PathIndex(radius=radius, default_size=(crop_size, crop_size))
    pi = _PATH_INDEX_CACHE[key]
    dev_key = (radius, crop_size, str(device))
    if dev_key not in _PATH_BUFFERS_CACHE:
        import torch
        buffers = [torch.from_numpy(p).long().to(device) for p in pi.path_indices]
        src_buf = torch.from_numpy(pi.src_indices).long().to(device)
        dst_buf = torch.from_numpy(pi.dst_indices).long().to(device)
        _PATH_BUFFERS_CACHE[dev_key] = (buffers, src_buf, dst_buf)
    return pi, *_PATH_BUFFERS_CACHE[dev_key]



def _safe_denominator(mask: torch.Tensor) -> torch.Tensor:
    return mask.sum().clamp(min=1.0)


def masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * mask).sum() / _safe_denominator(mask)


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if mask.shape[1] == 1 and pred.shape[1] != 1:
        mask = mask.expand(-1, pred.shape[1], -1, -1)
    loss = (pred - target).pow(2)
    return (loss * mask).sum() / _safe_denominator(mask)


def positive_only_bce(logits: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
    if pos_mask.shape[1] != 1:
        pos_mask = pos_mask.unsqueeze(1)
    if pos_mask.sum() <= 0:
        return logits.new_tensor(0.0)
    target = torch.ones_like(logits)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * pos_mask).sum() / _safe_denominator(pos_mask)


def point_anchor_loss(fg_logits: torch.Tensor, point_map: torch.Tensor) -> torch.Tensor:
    if point_map.ndim == 3:
        point_map = point_map.unsqueeze(1)
    point_mask = (point_map > 0).to(dtype=fg_logits.dtype)
    return positive_only_bce(fg_logits, point_mask)


def _central_diff_x(value: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(value)
    out[:, :, :, 1:-1] = 0.5 * (value[:, :, :, 2:] - value[:, :, :, :-2])
    out[:, :, :, 0] = value[:, :, :, 1] - value[:, :, :, 0]
    out[:, :, :, -1] = value[:, :, :, -1] - value[:, :, :, -2]
    return out


def _central_diff_y(value: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(value)
    out[:, :, 1:-1, :] = 0.5 * (value[:, :, 2:, :] - value[:, :, :-2, :])
    out[:, :, 0, :] = value[:, :, 1, :] - value[:, :, 0, :]
    out[:, :, -1, :] = value[:, :, -1, :] - value[:, :, -2, :]
    return out


def divergence(flow: torch.Tensor) -> torch.Tensor:
    """Compute divergence of dx,dy flow as d(dx)/dx + d(dy)/dy."""
    return _central_diff_x(flow[:, 0:1]) + _central_diff_y(flow[:, 1:2])


def gaussian_point_density(
    point_map: torch.Tensor,
    sigma: float,
    mode: str = "sum",
) -> torch.Tensor:
    """Gaussian density from sparse points.
    mode="sum"  (legacy behavior): rho(x) = sum_i exp(-||x-p_i||^2/(2 sigma^2)).
                Convolution-based; overlapping targets can exceed 1. Retained
                for backward compatibility.
    mode="max": rho(x) = exp(-d_min^2/(2 sigma^2)) using the nearest point.
                The target remains in [0, 1] and is used by both L_div and the
                centerness regression in the canonical ChromaFlow objective.
    Returns (B, 1, H, W).
    """
    if point_map.ndim == 3:
        point_map = point_map.unsqueeze(1)
    pm = (point_map > 0).to(dtype=torch.float32)
    if sigma <= 0:
        return pm
    if mode == "sum":
        radius = max(1, int(round(3.0 * sigma)))
        coords = torch.arange(-radius, radius + 1, device=pm.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(xx.pow(2) + yy.pow(2)) / (2.0 * sigma * sigma))
        kernel = kernel.to(dtype=pm.dtype).view(1, 1, kernel.shape[0], kernel.shape[1])
        return F.conv2d(pm, kernel, padding=radius)
    if mode != "max":
        raise ValueError(f"Unsupported mode: {mode}")
    B, _, H, W = pm.shape
    out = torch.zeros_like(pm)
    sigma2_2 = 2.0 * sigma * sigma
    yy = torch.arange(H, device=pm.device, dtype=torch.float32)
    xx = torch.arange(W, device=pm.device, dtype=torch.float32)
    for b in range(B):
        coords = pm[b, 0].nonzero(as_tuple=False).float()  # (N, 2) [y, x]
        if coords.numel() == 0:
            continue
        dy2 = (yy.view(1, H, 1) - coords[:, 0].view(-1, 1, 1)).pow(2)
        dx2 = (xx.view(1, 1, W) - coords[:, 1].view(-1, 1, 1)).pow(2)
        d2_min = (dy2 + dx2).min(dim=0).values
        out[b, 0] = torch.exp(-d2_min / sigma2_2)
    return out


def voronoi_ridge_target(point_map: torch.Tensor,
                          stride: int = 8,
                          tau: float = 0.3,
                          r_max: float = 15.0) -> torch.Tensor:
    """Point-derivable Voronoi ridge target for HU-BIM aux supervision.

    For each pixel x at the down-sampled grid (resolution H/stride x W/stride):
        d_1 = nearest GT point distance
        d_2 = second-nearest GT point distance
        target(x) = 1 if d_2 - d_1 < tau * min(d_1, r_max), else 0
    i.e. pixels that are roughly equidistant to two nearby points -> they
    lie on the Voronoi ridge between adjacent nuclei. tau controls
    "fuzziness" of the ridge band.

    Args:
        point_map: (B, H, W) or (B, 1, H, W) full-resolution binary point map
        stride:    output resolution divisor (default 8, matching s2 / HU-BIM)
        tau:       ridge half-width as fraction of d_1
        r_max:     full-res radius cap (auto-scaled by 1/stride internally)
    Returns:
        (B, 1, H//stride, W//stride) float32 in {0., 1.}
    """
    if point_map.ndim == 3:
        point_map = point_map.unsqueeze(1)
    pm = (point_map > 0).to(dtype=torch.float32)
    B, _, H, W = pm.shape
    Hs, Ws = H // stride, W // stride
    r_max_s = float(r_max) / float(stride)
    out = torch.zeros(B, 1, Hs, Ws, device=pm.device, dtype=pm.dtype)
    yy = torch.arange(Hs, device=pm.device, dtype=torch.float32)
    xx = torch.arange(Ws, device=pm.device, dtype=torch.float32)
    for b in range(B):
        coords = pm[b, 0].nonzero(as_tuple=False).float() / float(stride)
        if coords.shape[0] < 2:
            continue
        dy2 = (yy.view(1, Hs, 1) - coords[:, 0].view(-1, 1, 1)).pow(2)
        dx2 = (xx.view(1, 1, Ws) - coords[:, 1].view(-1, 1, 1)).pow(2)
        d = (dy2 + dx2).sqrt()                     # (N, Hs, Ws)
        sd, _ = d.sort(dim=0)
        d1 = sd[0]
        d2 = sd[1]
        margin = d2 - d1
        thresh = tau * torch.minimum(d1, torch.tensor(r_max_s,
                                                          device=d1.device))
        out[b, 0] = (margin < thresh).float()
    return out


def voronoi_ridge_soft_target(point_map: torch.Tensor,
                                  stride: int = 8,
                                  sigma_b: float = 5.0,
                                  r_core: float = 0.0) -> torch.Tensor:
    """Soft Gaussian ridge target.

    For each pixel x at the down-sampled grid:
        d_1 = nearest GT point distance
        d_2 = second-nearest GT point distance
        b*(x) = exp(-(d_2 - d_1)^2 / (2 * sigma_b^2)) * I[d_1 > r_core]
    The I[d_1 > r_core] mask EXCLUDES the central region near each point
    (where the network is supposed to predict the center), so ridge signal
    only affects the inter-nucleus boundary band, not the core.

    sigma_b and r_core are in DOWN-SAMPLED units (i.e. at stride-pixel scale).
    Returns: (B, 1, H//stride, W//stride) float32 in [0, 1].
    """
    if point_map.ndim == 3:
        point_map = point_map.unsqueeze(1)
    pm = (point_map > 0).to(dtype=torch.float32)
    B, _, H, W = pm.shape
    Hs, Ws = H // stride, W // stride
    sigma_b_s = float(sigma_b)
    r_core_s = float(r_core)
    out = torch.zeros(B, 1, Hs, Ws, device=pm.device, dtype=pm.dtype)
    yy = torch.arange(Hs, device=pm.device, dtype=torch.float32)
    xx = torch.arange(Ws, device=pm.device, dtype=torch.float32)
    for b in range(B):
        coords = pm[b, 0].nonzero(as_tuple=False).float() / float(stride)
        if coords.shape[0] < 2:
            continue
        dy2 = (yy.view(1, Hs, 1) - coords[:, 0].view(-1, 1, 1)).pow(2)
        dx2 = (xx.view(1, 1, Ws) - coords[:, 1].view(-1, 1, 1)).pow(2)
        d = (dy2 + dx2).sqrt()
        sd, _ = d.sort(dim=0)
        d1 = sd[0]
        d2 = sd[1]
        diff = d2 - d1
        soft = torch.exp(-diff.pow(2) / (2.0 * sigma_b_s * sigma_b_s))
        if r_core_s > 0:
            soft = soft * (d1 > r_core_s).float()
        out[b, 0] = soft
    return out


def ridge_smoothl1_loss(ridge_logits: torch.Tensor,
                          target: torch.Tensor,
                          fg_mask_1_stride: torch.Tensor) -> torch.Tensor:
    """SmoothL1 between sigmoid(ridge_logits) and soft target b* in [0,1],
    masked to foreground pixels at downsampled resolution.
    """
    pred = torch.sigmoid(ridge_logits)
    sl1 = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
    denom = fg_mask_1_stride.sum().clamp(min=1.0)
    return (sl1 * fg_mask_1_stride).sum() / denom


def boundary_inference_loss(b_logits: torch.Tensor,
                              target: torch.Tensor,
                              fg_mask_1_stride: torch.Tensor) -> torch.Tensor:
    """BCE loss for HU-BIM aux boundary head at downsampled resolution.

    b_logits:        (B, 1, Hs, Ws) raw logits
    target:          (B, 1, Hs, Ws) binary in {0,1}
    fg_mask_1_stride: (B, 1, Hs, Ws) float weight (only supervise fg pixels)
    """
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        b_logits, target, reduction="none")
    denom = fg_mask_1_stride.sum().clamp(min=1.0)
    return (bce * fg_mask_1_stride).sum() / denom


def hucaf_distill_loss(aux_fg_logits: torch.Tensor,
                       main_fg_logits: torch.Tensor,
                       fg_mask: torch.Tensor | None = None) -> torch.Tensor:
    """HU-CAF distillation: MSE on sigmoid space between UNI-only fg head
    (student) and fused main fg head (teacher, detached). Pulls UNI projection
    to be nuclei-locating.
    """
    teacher = torch.sigmoid(main_fg_logits.detach())
    student = torch.sigmoid(aux_fg_logits)
    se = (student - teacher).pow(2)
    if fg_mask is None:
        return se.mean()
    if fg_mask.ndim == 3:
        fg_mask = fg_mask.unsqueeze(1)
    fg_mask = fg_mask.to(dtype=se.dtype)
    denom = fg_mask.sum().clamp(min=1.0)
    return (se * fg_mask).sum() / denom


def adaptive_max_sink(
    point_map: torch.Tensor,
    alpha: float = 0.25,
    sigma_min: float = 3.0,
    sigma_max: float = 8.0,
) -> torch.Tensor:
    """Per-point density-adaptive max-union Gaussian sink.

    For each point p_i in each batch element, compute
        d_NN(p_i) = min_{j != i} ||p_i - p_j||
        sigma_i  = clip(alpha * d_NN(p_i), sigma_min, sigma_max)
    then form
        rho_i(x) = exp(-||x - p_i||^2 / (2 * sigma_i^2))
        rho_sink(x) = max_i rho_i(x).

    For a singleton point (N==1) we fall back to sigma_max.
    Returns (B, 1, H, W) float32.
    """
    if point_map.ndim == 3:
        point_map = point_map.unsqueeze(1)
    pm = (point_map > 0).to(dtype=torch.float32)
    B, _, H, W = pm.shape
    out = torch.zeros_like(pm)
    yy = torch.arange(H, device=pm.device, dtype=torch.float32)
    xx = torch.arange(W, device=pm.device, dtype=torch.float32)
    for b in range(B):
        coords = pm[b, 0].nonzero(as_tuple=False).float()  # (N, 2) [y, x]
        N = coords.shape[0]
        if N == 0:
            continue
        if N == 1:
            sigma_per = coords.new_full((1,), float(sigma_max))
        else:
            d = torch.cdist(coords, coords)  # (N, N)
            d.fill_diagonal_(float("inf"))
            d_nn = d.min(dim=1).values  # (N,)
            sigma_per = (alpha * d_nn).clamp(sigma_min, sigma_max)
        sigma2_2 = 2.0 * sigma_per.pow(2)  # (N,)
        dy2 = (yy.view(1, H, 1) - coords[:, 0].view(-1, 1, 1)).pow(2)  # (N, H, 1)
        dx2 = (xx.view(1, 1, W) - coords[:, 1].view(-1, 1, 1)).pow(2)  # (N, 1, W)
        d2 = dy2 + dx2
        rho_per = torch.exp(-d2 / sigma2_2.view(-1, 1, 1))  # (N, H, W)
        out[b, 0] = rho_per.max(dim=0).values
    return out


def divergence_sink_loss(
    flow: torch.Tensor,
    point_map: torch.Tensor,
    fg_mask: torch.Tensor,
    conductance: torch.Tensor | None = None,
    sink_sigma: float = 3.0,
    sink_adaptive: bool = False,
    sink_alpha: float = 0.25,
    sink_min: float = 3.0,
    sink_max: float = 8.0,
    phi: torch.Tensor | None = None,
    screen_kappa: float = 0.0,
    source_density: torch.Tensor | None = None,
) -> torch.Tensor:
    """Point-sink divergence loss.

    Screened-Poisson (U1): if phi is given and screen_kappa>0, the constraint
    becomes (Delta - kappa^2) phi = rho_sink, i.e. residual = div(flow) +
    rho_sink + kappa^2 * phi. This adds a length scale 1/kappa (~nucleus radius)
    so a point's flow influence decays (Yukawa kernel) instead of smearing across
    the tile (plain Poisson, log-kernel) — a locality prior direct flow cannot express.

    If sink_adaptive=False (legacy): uses fixed sink_sigma in max-union mode.
    If sink_adaptive=True: per-point sigma_i = clip(sink_alpha * d_NN(p_i),
        sink_min, sink_max). Equivalent to scaling sink width with local point
        spacing — wider sinks in sparse regions, narrower in dense regions to
        avoid cross-instance interference.
    """
    if fg_mask.ndim == 3:
        fg_mask = fg_mask.unsqueeze(1)
    fg_mask = fg_mask.to(dtype=flow.dtype)
    if conductance is not None:
        if conductance.ndim == 3:
            conductance = conductance.unsqueeze(1)
        flow = flow * conductance.to(dtype=flow.dtype)
    if source_density is not None and not sink_adaptive:
        rho_sink = source_density.to(dtype=flow.dtype)
        if rho_sink.ndim == 3:
            rho_sink = rho_sink.unsqueeze(1)
    elif sink_adaptive:
        rho_sink = adaptive_max_sink(point_map, alpha=sink_alpha,
                                       sigma_min=sink_min, sigma_max=sink_max
                                       ).to(dtype=flow.dtype)
    else:
        rho_sink = gaussian_point_density(point_map, sink_sigma, mode="max"
                                            ).to(dtype=flow.dtype)
    residual = divergence(flow) + rho_sink
    if phi is not None and screen_kappa > 0.0:
        if phi.ndim == 3:
            phi = phi.unsqueeze(1)
        residual = residual + (screen_kappa ** 2) * phi.to(dtype=flow.dtype)
    return (residual.pow(2) * fg_mask).sum() / _safe_denominator(fg_mask)



def focal_mse_loss(pred_logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """Focal-weighted MSE for Gaussian heatmap regression (centerness).
    weight = 1 + gamma * target emphasizes peak pixels; mean over all pixels.
    Note: a pos/neg-balanced variant was tried in R1.0 and made downstream NMS
    reconcile WORSE (test PQ -0.05 vs Phase4 baseline) by producing narrower
    centerness lobes. Reverted; the original unmasked focal-MSE is kept.
    """
    pred = torch.sigmoid(pred_logits)
    if target.ndim == 3:
        target = target.unsqueeze(1)
    weight = 1.0 + gamma * target
    return (weight * (pred - target).pow(2)).mean()


def psic_loss(
    z: torch.Tensor,
    point_map: torch.Tensor,
    clu: torch.Tensor,
    r_pos: float = 10.0,
    beta: float = 0.6,
    r_neg: float = 64.0,
    k_min: int = 4,
    k_anchor: int = 8,
    tau: float = 0.1,
    # v2 additions:
    adaptive_radius: bool = False,   # v2A/B/C: r_i = min(r_pos_max, alpha * d_NN(p_i))
    r_pos_max: float = 8.0,
    alpha_dnn: float = 0.25,
    top_k_neg: int = 0,              # v2B/C: top-K hard negatives by similarity (0 = all)
    use_margin: bool = False,        # v2C: replace InfoNCE with margin contrast
    margin_value: float = 0.15,
    neg_weight: float = 0.5,
) -> torch.Tensor:
    """Vectorized Point-Sink Instance Contrastive loss.

    Per-image (b):
      1. cdist(grid, points) -> (HW, N) distance
      2. d1, i_star = topk(min, k=1)
      3. d2 = topk(min, k=2)
      4. safe_mask = (d1 < r_pos) & (d1 < beta * d2) & (clu != 2)
      5. scatter mean -> per-nucleus prototype (N, C)
      6. sample k_anchor anchors per valid nucleus
      7. InfoNCE: pos = proto[i_star], negs = proto in neighbor mask
    All inner loops are over batch elements only.
    """
    if point_map.ndim == 4:
        point_map = point_map.squeeze(1)
    if clu.ndim == 4:
        clu = clu.squeeze(1)
    B, C, H, W = z.shape
    device = z.device

    # Speed-up: downsample to /4 resolution. Contrast features at full /1 res are
    # redundant for instance-level discrimination, and /4 still gives many anchor
    # pixels per nucleus (radius r_pos=10 -> ~3 px at /4, ~28 anchors per nucleus).
    scale = 4
    z_lo = torch.nn.functional.avg_pool2d(z, scale)  # (B, C, H/4, W/4)
    z_lo = torch.nn.functional.normalize(z_lo, dim=1)
    point_map_lo = torch.nn.functional.max_pool2d(point_map.unsqueeze(1).float(), scale).squeeze(1)
    clu_lo = torch.nn.functional.avg_pool2d(clu.unsqueeze(1).float(), scale).squeeze(1)
    # clu_lo: 0 if all bg, 1 if all fg, 2 if includes ignore. Convert back to 3-class.
    clu_lo = torch.where(clu_lo > 1.5, torch.tensor(2.0, device=device),
                         torch.where(clu_lo > 0.5, torch.tensor(1.0, device=device),
                                     torch.tensor(0.0, device=device)))
    z = z_lo
    point_map = point_map_lo
    clu = clu_lo
    B, C, H, W = z.shape  # update to /4 dims
    # Scale spatial hyperparams to /4 resolution
    r_pos = r_pos / scale
    r_neg = r_neg / scale

    # Pre-compute pixel grid once
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    grid = torch.stack([yy.flatten(), xx.flatten()], dim=1)  # (HW, 2)
    HW = H * W

    losses = []
    for b in range(B):
        pts = (point_map[b] > 0).nonzero(as_tuple=False).float()  # (N, 2)
        n = pts.shape[0]
        if n < 2:
            continue
        # (HW, N) — optimized CUDA kernel
        dist = torch.cdist(grid, pts)
        # nearest 2
        d_sorted, idx_sorted = dist.topk(min(2, n), dim=1, largest=False)  # (HW, 2)
        d1 = d_sorted[:, 0]
        i_star = idx_sorted[:, 0]
        d2 = d_sorted[:, 1] if n >= 2 else d1 + r_neg * 2

        clu_b = clu[b].flatten()
        # Adaptive positive core (v2): per-nucleus radius r_i = min(r_pos_max, alpha * d_NN(p_i))
        if adaptive_radius:
            pp_d_all = torch.cdist(pts, pts)
            pp_d_all.fill_diagonal_(float("inf"))
            d_NN = pp_d_all.min(dim=1).values  # (N,)
            r_per_nuc = torch.clamp(alpha_dnn * d_NN, max=r_pos_max)  # (N,)
            # per-pixel r_i lookup
            r_pix = r_per_nuc[i_star]  # (HW,)
            safe = (d1 < r_pix) & (d1 < beta * d2) & (clu_b != 2)
        else:
            safe = (d1 < r_pos) & (d1 < beta * d2) & (clu_b != 2)
        if safe.sum() < k_min:
            continue

        # extract z of safe pixels
        zb_flat = z[b].permute(1, 2, 0).reshape(HW, C)  # (HW, C)
        z_safe = zb_flat[safe]  # (M, C)
        i_safe = i_star[safe]   # (M,) nucleus id
        if z_safe.shape[0] < 2:
            continue

        # per-nucleus prototype via scatter mean
        proto_sum = torch.zeros((n, C), device=device, dtype=z.dtype)
        proto_cnt = torch.zeros(n, device=device, dtype=z.dtype)
        proto_sum.index_add_(0, i_safe, z_safe)
        ones = torch.ones(z_safe.shape[0], device=device, dtype=z.dtype)
        proto_cnt.index_add_(0, i_safe, ones)
        valid = proto_cnt >= float(k_min)
        if int(valid.sum().item()) < 2:
            continue
        proto = proto_sum / proto_cnt.clamp(min=1.0).unsqueeze(1)
        proto = torch.nn.functional.normalize(proto, dim=1).detach()

        # nucleus-nucleus distance for hard negative mask: (N, N)
        pp_d = torch.cdist(pts, pts)
        pp_d.fill_diagonal_(float("inf"))
        neg_mask = (pp_d < r_neg) & valid.unsqueeze(0)  # (N, N) bool, row i = negatives for nucleus i

        # Sample up to k_anchor anchors per valid nucleus.
        # We do this by sorting safe pixels by i_safe and picking first k_anchor of each group,
        # but easier: random global permutation then dedup-by-nucleus with a counter.
        # Even simpler: just take ALL safe pixels (no subsample) — k_anchor cap helps memory only.
        # If too many, take random k_anchor per nucleus via stratified sampling.
        # For simplicity here, cap per nucleus by taking modulo positions.
        # (Per-image safe pixels are O(N * pi*r_pos^2) ~ N * 300, with N~50, 15000 pixels.
        #  All fit easily; skip explicit subsampling.)
        # If M_total > k_anchor*N_valid, randomly drop:
        max_M = int(k_anchor * valid.sum().item())
        if z_safe.shape[0] > max_M:
            perm = torch.randperm(z_safe.shape[0], device=device)[:max_M]
            z_safe = z_safe[perm]
            i_safe = i_safe[perm]

        # filter anchors whose nucleus has no negative neighbors
        anchor_has_neg = neg_mask[i_safe].any(dim=1)  # (M,)
        if int(anchor_has_neg.sum().item()) == 0:
            continue
        z_safe = z_safe[anchor_has_neg]
        i_safe = i_safe[anchor_has_neg]

        # Similarities: z_safe @ proto.T  (M, N), cosine since both L2-normalized
        m_anchors = z_safe.shape[0]
        sim_cos = z_safe @ proto.t()                  # (M, N) in [-1, 1]
        # Each anchor's nucleus index for positive
        pos_cos = sim_cos.gather(1, i_safe.unsqueeze(1)).squeeze(1)  # (M,)
        # Per-anchor negative mask: only valid neighbor nuclei (not self)
        neg_per_anchor = neg_mask[i_safe].clone()  # (M, N) bool, True where j is neg for i_safe
        neg_per_anchor[torch.arange(m_anchors, device=device), i_safe] = False  # exclude self

        # v2B/C: keep only top-K hard negatives (most similar to anchor)
        if top_k_neg > 0:
            # mask non-neg as very-negative so they're not picked
            scores = sim_cos.masked_fill(~neg_per_anchor, float("-inf"))
            k_eff = min(top_k_neg, neg_per_anchor.sum(dim=1).max().item() if neg_per_anchor.any() else 0)
            if k_eff == 0:
                continue
            top_vals, top_idx = scores.topk(k_eff, dim=1, largest=True)  # (M, K)
            # Build a fresh mask containing only top-K negatives
            new_neg = torch.zeros_like(neg_per_anchor)
            row_idx = torch.arange(m_anchors, device=device).unsqueeze(1).expand_as(top_idx)
            valid_topk = top_vals > float("-inf")
            new_neg[row_idx[valid_topk], top_idx[valid_topk]] = True
            neg_per_anchor = new_neg

        if use_margin:
            # v2C: margin contrastive — positive pulls cos→1, negatives pushed below margin
            # L_pos = mean (1 - z^T q_pos)
            l_pos = (1.0 - pos_cos).mean()
            # L_neg = mean (max(0, z^T q_neg - margin))^2 over top-K negatives
            neg_scores = sim_cos.masked_fill(~neg_per_anchor, 0.0)  # zero out non-neg (won't contribute)
            margin_violation = torch.relu(neg_scores - margin_value)
            margin_violation = margin_violation * neg_per_anchor.float()
            n_neg_per_anchor = neg_per_anchor.float().sum(dim=1).clamp(min=1.0)
            l_neg = ((margin_violation.pow(2).sum(dim=1)) / n_neg_per_anchor).mean()
            loss_b = l_pos + neg_weight * l_neg
        else:
            # v1/v2A/v2B: InfoNCE
            sim_logits = sim_cos / tau
            denom_mask = neg_per_anchor.clone()
            denom_mask[torch.arange(m_anchors, device=device), i_safe] = True  # include positive
            sim_masked = sim_logits.masked_fill(~denom_mask, float("-inf"))
            pos_logit = sim_logits.gather(1, i_safe.unsqueeze(1)).squeeze(1)
            log_denom = torch.logsumexp(sim_masked, dim=1)
            loss_b = (log_denom - pos_logit).mean()
        losses.append(loss_b)

    if not losses:
        return z.new_tensor(0.0)
    return torch.stack(losses).mean()


def compute_instance_mask_loss(outputs):
    """Per-instance BCE + Dice loss on point-prompted mask decoder output.

    Expects outputs dict to contain:
      - 'mask_logits':  (B, N, H, W) per-query mask logits
      - 'mask_targets': (B, N, H, W) binary instance masks
      - 'mask_valid':   (B, N) bool, True for real instances (not padded)

    Returns (loss_mask_bce, loss_mask_dice). Padding queries contribute zero.
    """
    import torch
    import torch.nn.functional as F
    logits = outputs["mask_logits"]
    targets = outputs["mask_targets"]
    valid = outputs["mask_valid"]
    B, N, H, W = logits.shape

    valid_f = valid.float().view(B, N, 1, 1)  # (B, N, 1, 1)
    # BCE per pixel, sum over spatial, average over valid queries
    bce_pp = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # (B, N, H, W)
    bce_per_query = (bce_pp * valid_f).flatten(2).mean(dim=2)  # (B, N)
    denom = valid.float().sum().clamp(min=1.0)
    loss_bce = (bce_per_query * valid.float()).sum() / denom

    # Dice per query
    probs = torch.sigmoid(logits)
    inter = (probs * targets * valid_f).flatten(2).sum(dim=2)  # (B, N)
    p_sum = (probs * valid_f).flatten(2).sum(dim=2)
    t_sum = (targets * valid_f).flatten(2).sum(dim=2)
    dice = (2.0 * inter + 1.0) / (p_sum + t_sum + 1.0)
    dice_loss = (1.0 - dice) * valid.float()
    loss_dice = dice_loss.sum() / denom

    return loss_bce, loss_dice


def compute_mask_iou_loss(outputs):
    """L1 loss between predicted IoU score and the actual IoU of the predicted mask
    vs ground-truth instance mask. Trains the IoU head as a quality estimator.
    """
    import torch
    logits = outputs["mask_logits"]
    targets = outputs["mask_targets"]
    valid = outputs["mask_valid"]
    iou_pred = outputs["mask_iou_pred"]  # (B, N)
    B, N, H, W = logits.shape

    with torch.no_grad():
        probs = torch.sigmoid(logits) > 0.5
        gt_b = targets > 0.5
        inter = (probs & gt_b).flatten(2).sum(dim=2).float()
        union = (probs | gt_b).flatten(2).sum(dim=2).float().clamp(min=1.0)
        iou_gt = inter / union  # (B, N)
        iou_gt = iou_gt * valid.float()

    diff = (iou_pred - iou_gt).abs() * valid.float()
    denom = valid.float().sum().clamp(min=1.0)
    return diff.sum() / denom


def compute_hv_loss(outputs, batch, stain_alpha: float = 1.0):
    """Stain-weighted MSE loss between predicted hv-map and the per-instance
    normalized distance target computed from batch['instance_id'].

    The pixelwise weight is fg_mask * (1 + stain_alpha * stain_h), so pixels
    with strong hematoxylin staining contribute more to the loss. This is the
    'stain-weighted' part of HSDR.
    """
    import torch
    from chromaflow.hv_utils import compute_hv_target_torch
    hv_pred = outputs["hv_map"]
    instance_id = batch["instance_id"].long()
    hv_target, fg_mask = compute_hv_target_torch(instance_id)
    hv_target = hv_target.to(dtype=hv_pred.dtype)
    fg_mask = fg_mask.to(dtype=hv_pred.dtype)
    stain_h = batch.get("stain_h", None)
    if stain_h is not None:
        stain_h = stain_h.to(dtype=hv_pred.dtype)
        if stain_h.ndim == 3:
            stain_h = stain_h.unsqueeze(1)
        weight = fg_mask * (1.0 + stain_alpha * stain_h)
    else:
        weight = fg_mask
    diff_sq = (hv_pred - hv_target).pow(2) * weight
    denom = weight.sum().clamp(min=1.0)
    return diff_sq.sum() / denom


def compute_stain_weighted_dice(fg_logits, fg_target, fg_mask, stain_h, alpha: float = 0.5):
    """B7 SCR-interior: Stain-Weighted Dice loss on fg head.

    Pixels with strong hematoxylin staining are biologically near-certain to be
    nucleus interior. We amplify their contribution to the Dice numerator and
    denominator by (1 + alpha * stain_h), so high-H pixels exert stronger pull
    on fg predictions. Together with HCA (attention end) and HGBL (boundary
    supervision), this is the third manifestation of the hematoxylin prior:
    interior supervision.

    Args:
        fg_logits: (B, 1, H, W) raw logits from fg_head
        fg_target: (B, 1, H, W) binary fg mask (1=fg, 0=bg)
        fg_mask:   (B, 1, H, W) valid mask (1=keep, 0=ignore voronoi-boundary px)
        stain_h:   (B, H, W) or (B, 1, H, W) hematoxylin density in [~0, 1]
        alpha:     stain weight (0=uniform Dice, 0.5=mild bias, 1.0=strong bias)

    Returns:
        Scalar Dice loss (1 - dice) averaged over batch.
    """
    import torch
    probs = torch.sigmoid(fg_logits)
    if stain_h.ndim == 3:
        stain_h = stain_h.unsqueeze(1)
    stain_h = stain_h.to(dtype=fg_logits.dtype)
    # Per-image min-max normalize stain_h so alpha has consistent meaning across batches.
    s_min = stain_h.flatten(2).amin(dim=2).view(-1, 1, 1, 1)
    s_max = stain_h.flatten(2).amax(dim=2).view(-1, 1, 1, 1)
    s_norm = (stain_h - s_min) / (s_max - s_min + 1e-6)
    weight = (1.0 + alpha * s_norm) * fg_mask
    inter = (probs * fg_target * weight).flatten(1).sum(dim=1)
    p_sum = (probs * weight).flatten(1).sum(dim=1)
    t_sum = (fg_target * weight).flatten(1).sum(dim=1)
    dice = (2.0 * inter + 1.0) / (p_sum + t_sum + 1.0)
    return (1.0 - dice).mean()


# ---- E1 PCGS: Probabilistic Chromophore-Guided Supervision -----------------
# Hard-threshold L_stain_soft + L_eosin_neg are degenerate special cases.
# PCGS treats each clu==2 pixel's fg/bg label as a latent variable observed
# through (H, E, distance-to-nearest-point); the BCE target is the posterior
# p(y=1 | H, E, d) under a Gaussian likelihood model whose (μ, σ, prior)
# parameters are estimated by EM across epochs.

def _distance_transform_to_points(point_map: torch.Tensor, max_dist: float = 30.0) -> torch.Tensor:
    """Per-image Euclidean distance from each pixel to the nearest GT point.

    Args:
        point_map: (B, H, W) or (B, 1, H, W) binary point indicator (0/1)
        max_dist:  clamp upper bound (pixels)
    Returns:
        (B, 1, H, W) float distance in [0, max_dist] on same device as input.
    """
    from scipy.ndimage import distance_transform_edt
    if point_map.ndim == 3:
        pm = point_map.unsqueeze(1)
    else:
        pm = point_map
    device = pm.device
    pm_np = (pm.detach().cpu().numpy() > 0).astype("uint8")  # (B,1,H,W)
    B, _, H, W = pm_np.shape
    out = torch.empty((B, 1, H, W), dtype=torch.float32)
    for b in range(B):
        if pm_np[b, 0].sum() == 0:
            out[b, 0].fill_(float(max_dist))
        else:
            d = distance_transform_edt(1 - pm_np[b, 0]).astype("float32")
            out[b, 0] = torch.from_numpy(d)
    out = out.clamp_(0.0, float(max_dist))
    return out.to(device)


def _gaussian_pdf(x: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:
    """Univariate Gaussian PDF, broadcast-safe; returns same dtype as x."""
    sigma = max(float(sigma), 1e-3)
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * (2.0 * math.pi) ** 0.5)


def _per_image_gaussian_fit(values: torch.Tensor, mask: torch.Tensor,
                              default_mu: float = 0.5, default_sigma: float = 0.15,
                              sigma_min: float = 0.05, sigma_max: float = 0.40):
    """Vectorized per-image (μ, σ) from mask-selected pixels.

    Args:
        values: (B, 1, H, W) values in [0, 1]
        mask:   (B, 1, H, W) bool / float anchor
    Returns:
        (μ, σ): tensors of shape (B, 1, 1, 1) each
    """
    m = mask.to(dtype=values.dtype)
    denom = m.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
    mu = (values * m).sum(dim=(2, 3), keepdim=True) / denom
    var = ((values - mu) ** 2 * m).sum(dim=(2, 3), keepdim=True) / denom
    sigma = var.clamp(min=sigma_min ** 2).sqrt().clamp(max=sigma_max)
    # Per-image fallback: if mask has <16 pixels, use defaults.
    too_few = (m.sum(dim=(2, 3), keepdim=True) < 16.0).to(dtype=values.dtype)
    mu = torch.where(too_few > 0, values.new_full(mu.shape, default_mu), mu)
    sigma = torch.where(too_few > 0, values.new_full(sigma.shape, default_sigma), sigma)
    return mu, sigma


def _gauss_pdf_tensor(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Gaussian PDF with broadcastable mu/sigma tensors (per-image params)."""
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * (2.0 * math.pi) ** 0.5)


# ============================ PCGS-v2 helpers ===============================

def _per_image_bivariate_gauss_fit(
    x1: torch.Tensor,           # (B,1,H,W) channel 1 values, e.g. h_norm
    x2: torch.Tensor,           # (B,1,H,W) channel 2 values, e.g. e_comp
    mask: torch.Tensor,         # (B,1,H,W) bool anchor
    default_mu: tuple = (0.5, 0.5),
    default_var: tuple = (0.04, 0.04),         # diag covariance fallback (σ=0.2 each)
    var_min: float = 0.05 ** 2,                # min var → σ_min=0.05
    var_max: float = 0.40 ** 2,                # max var → σ_max=0.40
    corr_max: float = 0.95,                    # max |correlation| to keep Σ pos-def
):
    """Fit per-image bivariate (μ, Σ) on (x1, x2) over masked pixels.

    Returns:
        mu: (B, 2, 1, 1) — per-image 2D mean
        Sigma_inv: (B, 2, 2, 1, 1) — per-image inverse covariance
        log_det: (B, 1, 1, 1) — log det of Σ (for log-PDF normalization)
    """
    m = mask.to(dtype=x1.dtype)
    n = m.sum(dim=(2, 3), keepdim=True)         # (B,1,1,1)
    denom = n.clamp(min=1.0)

    # Means
    mu1 = (x1 * m).sum(dim=(2, 3), keepdim=True) / denom
    mu2 = (x2 * m).sum(dim=(2, 3), keepdim=True) / denom

    # Centered values
    d1 = (x1 - mu1) * m
    d2 = (x2 - mu2) * m

    # Variances + covariance
    var1 = (d1 * d1).sum(dim=(2, 3), keepdim=True) / denom
    var2 = (d2 * d2).sum(dim=(2, 3), keepdim=True) / denom
    cov12 = (d1 * d2).sum(dim=(2, 3), keepdim=True) / denom

    # Clamp variances
    var1 = var1.clamp(min=var_min, max=var_max)
    var2 = var2.clamp(min=var_min, max=var_max)

    # Clamp correlation to keep Σ positive definite
    sigma1 = var1.sqrt()
    sigma2 = var2.sqrt()
    corr = cov12 / (sigma1 * sigma2 + 1e-8)
    corr = corr.clamp(min=-corr_max, max=corr_max)
    cov12 = corr * sigma1 * sigma2

    # Per-image fallback: if mask has <16 pixels, use diag default.
    too_few = (n < 16.0).to(dtype=x1.dtype)
    mu1 = torch.where(too_few > 0, x1.new_full(mu1.shape, default_mu[0]), mu1)
    mu2 = torch.where(too_few > 0, x2.new_full(mu2.shape, default_mu[1]), mu2)
    var1 = torch.where(too_few > 0, x1.new_full(var1.shape, default_var[0]), var1)
    var2 = torch.where(too_few > 0, x2.new_full(var2.shape, default_var[1]), var2)
    cov12 = torch.where(too_few > 0, torch.zeros_like(cov12), cov12)

    # det(Σ) = var1·var2 - cov12²
    det_S = (var1 * var2 - cov12 * cov12).clamp(min=1e-8)
    log_det = torch.log(det_S)                                          # (B,1,1,1)

    # Pack (μ, Σ⁻¹) for fast per-pixel eval
    # Σ⁻¹ = (1/det) · [[var2, -cov12], [-cov12, var1]]
    inv_det = 1.0 / det_S
    s_inv_11 = var2 * inv_det           # (B,1,1,1)
    s_inv_22 = var1 * inv_det
    s_inv_12 = -cov12 * inv_det

    return mu1, mu2, s_inv_11, s_inv_12, s_inv_22, log_det


def _bivariate_log_lik(
    x1: torch.Tensor, x2: torch.Tensor,
    mu1: torch.Tensor, mu2: torch.Tensor,
    s_inv_11: torch.Tensor, s_inv_12: torch.Tensor, s_inv_22: torch.Tensor,
    log_det: torch.Tensor,
) -> torch.Tensor:
    """Per-pixel log Gaussian likelihood for bivariate (x1, x2)."""
    d1 = x1 - mu1
    d2 = x2 - mu2
    quad = d1 * d1 * s_inv_11 + 2.0 * d1 * d2 * s_inv_12 + d2 * d2 * s_inv_22
    log_norm = -0.5 * (2.0 * math.log(2.0 * math.pi) + log_det)
    return log_norm - 0.5 * quad


def compute_pcgs_v2_loss(
    fg_logits: torch.Tensor,         # (B, 1, H, W)
    clu: torch.Tensor,               # (B, H, W) {0,1,2}
    stain_h: torch.Tensor,           # (B, H, W) or (B, 1, H, W)
    stain_e: torch.Tensor,           # same
    point_map: torch.Tensor,         # same
    bg_anchor_min_dist: float = 40.0,
    spatial_sigma: float = 30.0,     # Gaussian decay σ for distance weighting
    prior_fg: float = 0.5,
    target_temperature: float = 1.0,
    eps: float = 1e-6,
    return_diag: bool = False,
):
    """PCGS-v2: bivariate Gaussian + spatial-weighted BCE on voronoi-gap.

    Improvements over compute_pcgs_anchor_loss:
        1. Bivariate (H, E_comp) Gaussian likelihood with covariance, instead of
           Naive Bayes product of marginals (fixes wrong independence assumption).
        2. Spatial confidence weight w(d) = exp(-d²/(2σ_d²)) on BCE, instead of
           hard cutoff `d <= max_dist`. Provides geometric inductive bias close to
           anchors while still supervising the full voronoi-gap (with low weight far
           away). Subsumes BoNuS's hand-crafted dist<30 gate.
    """
    if stain_h.ndim == 3:
        stain_h = stain_h.unsqueeze(1)
    if stain_e.ndim == 3:
        stain_e = stain_e.unsqueeze(1)
    if point_map.ndim == 3:
        point_map = point_map.unsqueeze(1)
    stain_h = stain_h.to(dtype=fg_logits.dtype)
    stain_e = stain_e.to(dtype=fg_logits.dtype)

    def _norm(x):
        lo = x.flatten(2).amin(dim=2).view(-1, 1, 1, 1)
        hi = x.flatten(2).amax(dim=2).view(-1, 1, 1, 1)
        return (x - lo) / (hi - lo + 1e-6)
    h_norm = _norm(stain_h)
    e_norm = _norm(stain_e)
    e_comp = 1.0 - e_norm

    # Distance field (clamp safe; we use it for both bg_anchor and spatial weight)
    max_d_field = max(bg_anchor_min_dist + 10.0, spatial_sigma * 3.0)
    d = _distance_transform_to_points(point_map, max_dist=max_d_field).to(
        dtype=fg_logits.dtype)

    # Anchors (same as v1)
    fg_anchor = (clu == 1).unsqueeze(1)
    bg_anchor = ((clu == 0).unsqueeze(1)) & (d > bg_anchor_min_dist)

    # ✦ v2 D1: bivariate Gaussian fit on (h_norm, e_comp) for fg, (h_norm, e_norm) for bg
    mu_Hf, mu_Ef, s_inv_HHf, s_inv_HEf, s_inv_EEf, log_det_f = (
        _per_image_bivariate_gauss_fit(h_norm, e_comp, fg_anchor,
                                        default_mu=(0.65, 0.60),
                                        default_var=(0.15**2, 0.18**2)))
    mu_Hb, mu_Eb, s_inv_HHb, s_inv_HEb, s_inv_EEb, log_det_b = (
        _per_image_bivariate_gauss_fit(h_norm, e_norm, bg_anchor,
                                        default_mu=(0.30, 0.55),
                                        default_var=(0.18**2, 0.20**2)))

    # Voronoi-gap region (clu==2). v2 supervises ALL of it with spatial weighting.
    vg_mask = (clu == 2).unsqueeze(1)
    if vg_mask.sum().item() < 1:
        loss = fg_logits.new_tensor(0.0)
        if return_diag:
            return loss, {}
        return loss

    # Per-pixel log-likelihoods
    log_lik_fg = _bivariate_log_lik(h_norm, e_comp,
                                      mu_Hf, mu_Ef,
                                      s_inv_HHf, s_inv_HEf, s_inv_EEf, log_det_f)
    log_lik_bg = _bivariate_log_lik(h_norm, e_norm,
                                      mu_Hb, mu_Eb,
                                      s_inv_HHb, s_inv_HEb, s_inv_EEb, log_det_b)

    # Posterior in log space (numerically stable)
    log_prior_fg = math.log(max(prior_fg, eps))
    log_prior_bg = math.log(max(1.0 - prior_fg, eps))
    log_num = log_prior_fg + log_lik_fg
    log_den = torch.stack([log_prior_fg + log_lik_fg,
                            log_prior_bg + log_lik_bg], dim=0)
    log_norm = torch.logsumexp(log_den, dim=0)
    log_posterior_fg = log_num - log_norm
    posterior = torch.exp(log_posterior_fg).clamp(0.01, 0.99).to(
        dtype=fg_logits.dtype)

    # Temperature sharpening (same as v1)
    if target_temperature != 1.0 and target_temperature > 0:
        logit_p = torch.log(posterior / (1.0 - posterior + eps))
        target = torch.sigmoid(target_temperature * logit_p).clamp(0.01, 0.99)
    else:
        target = posterior

    # ✦ v2 D2: spatial confidence weighting (replaces hard d<=max_dist cutoff)
    spatial_w = torch.exp(-(d * d) / (2.0 * spatial_sigma * spatial_sigma))
    # Combined mask: voronoi-gap × spatial weight
    full_w = vg_mask.to(dtype=fg_logits.dtype) * spatial_w

    bce_pp = F.binary_cross_entropy_with_logits(fg_logits, target, reduction="none")
    val = (bce_pp * full_w).sum() / full_w.sum().clamp(min=1.0)

    if return_diag:
        diag = {
            "mu_H_fg": float(mu_Hf.mean().item()),
            "mu_E_fg_comp": float(mu_Ef.mean().item()),
            "mu_H_bg": float(mu_Hb.mean().item()),
            "mu_E_bg": float(mu_Eb.mean().item()),
            "n_fg_anchor": float(fg_anchor.float().sum().item()),
            "n_bg_anchor": float(bg_anchor.float().sum().item()),
            "vg_mean_w": float(full_w.mean().item()),
        }
        return val, diag
    return val



def compute_pcgs_anchor_loss(
    fg_logits: torch.Tensor,         # (B, 1, H, W) raw logits
    clu: torch.Tensor,               # (B, H, W) long {0,1,2}
    stain_h: torch.Tensor,           # (B, H, W) or (B, 1, H, W)
    stain_e: torch.Tensor,           # (B, H, W) or (B, 1, H, W)
    point_map: torch.Tensor,         # (B, H, W) or (B, 1, H, W)
    bg_anchor_min_dist: float = 40.0,
    max_dist: float = 30.0,
    prior_fg: float = 0.5,
    target_temperature: float = 1.0,  # T>1 sharpens posterior toward binary; T=1 raw posterior
    eps: float = 1e-6,
    return_diag: bool = False,
):
    """PCGS-Anchor: stateless per-batch Bayesian posterior on voronoi-gap.

    Anchor regions (from BoNuS supervision structure, not from model output):
      - fg_anchor = (clu == 1)                              ← cluster disk
      - bg_anchor = (clu == 0) ∩ (dist_to_nearest_point > bg_anchor_min_dist)

    Per-image Gaussian fits on H_norm and (1-E_norm) for fg_anchor,
    on H_norm and E_norm for bg_anchor. The voronoi-gap (clu==2) posterior:
        p(y=1|H,E) ∝ prior_fg · N(H;μ_Hf,σ_Hf) · N(1-E;μ_Ef,σ_Ef)
        p(y=0|H,E) ∝ (1-prior_fg) · N(H;μ_Hb,σ_Hb) · N(E;μ_Eb,σ_Eb)
    is used as the BCE target.

    Spatial proximity gate: only clu==2 pixels within `max_dist` of any GT point
    are supervised (consistent with the L_stain_soft / L_eosin_neg gates).

    No EM, no state, no warmup. All distributions re-fit per batch.

    Returns: scalar BCE; if return_diag, also dict of {mu_Hf, sigma_Hf, ...} for logging.
    """
    if stain_h.ndim == 3:
        stain_h = stain_h.unsqueeze(1)
    if stain_e.ndim == 3:
        stain_e = stain_e.unsqueeze(1)
    if point_map.ndim == 3:
        point_map = point_map.unsqueeze(1)
    stain_h = stain_h.to(dtype=fg_logits.dtype)
    stain_e = stain_e.to(dtype=fg_logits.dtype)

    # Per-image min-max normalize H and E to [0,1].
    def _norm(x):
        lo = x.flatten(2).amin(dim=2).view(-1, 1, 1, 1)
        hi = x.flatten(2).amax(dim=2).view(-1, 1, 1, 1)
        return (x - lo) / (hi - lo + 1e-6)
    h_norm = _norm(stain_h)
    e_norm = _norm(stain_e)
    e_comp = 1.0 - e_norm  # fg likes high (1-E)

    # Distance field to nearest GT point (clamped for safety).
    d = _distance_transform_to_points(
        point_map, max_dist=max(bg_anchor_min_dist + 10.0, max_dist + 10.0)
    ).to(dtype=fg_logits.dtype)

    # Anchor masks.
    fg_anchor = (clu == 1).unsqueeze(1)                                    # cluster disk
    bg_anchor = ((clu == 0).unsqueeze(1)) & (d > bg_anchor_min_dist)

    # Per-image Gaussian fits.
    mu_Hf, sg_Hf = _per_image_gaussian_fit(h_norm, fg_anchor, default_mu=0.65, default_sigma=0.15)
    mu_Ef, sg_Ef = _per_image_gaussian_fit(e_comp, fg_anchor, default_mu=0.60, default_sigma=0.18)
    mu_Hb, sg_Hb = _per_image_gaussian_fit(h_norm, bg_anchor, default_mu=0.30, default_sigma=0.18)
    mu_Eb, sg_Eb = _per_image_gaussian_fit(e_norm, bg_anchor, default_mu=0.55, default_sigma=0.20)

    # Voronoi-gap mask with proximity gate (proximity gate consistent with
    # L_stain_soft / L_eosin_neg behaviour at max_dist=30).
    vg_mask = (clu == 2).unsqueeze(1)
    if max_dist > 0:
        vg_mask = vg_mask & (d <= max_dist)
    n_vg = vg_mask.sum()
    if n_vg.item() < 1:
        loss = fg_logits.new_tensor(0.0)
        if return_diag:
            return loss, {}
        return loss

    lik_fg = _gauss_pdf_tensor(h_norm, mu_Hf, sg_Hf) * _gauss_pdf_tensor(e_comp, mu_Ef, sg_Ef)
    lik_bg = _gauss_pdf_tensor(h_norm, mu_Hb, sg_Hb) * _gauss_pdf_tensor(e_norm, mu_Eb, sg_Eb)
    num = prior_fg * lik_fg
    den = num + (1.0 - prior_fg) * lik_bg
    posterior = (num / (den + eps)).clamp(0.01, 0.99).to(dtype=fg_logits.dtype)

    # E1-T (route C.1): temperature sharpening of posterior toward binary.
    # T=1: raw posterior (E1.6 baseline). T→∞: hard step at 0.5 (≈ C5).
    # Sharpens cross-domain robustness by encouraging more confident model outputs,
    # while preserving the per-image posterior ordering / calibration.
    if target_temperature != 1.0 and target_temperature > 0:
        logit_p = torch.log(posterior / (1.0 - posterior + eps))
        target = torch.sigmoid(target_temperature * logit_p).clamp(0.01, 0.99)
    else:
        target = posterior

    bce_pp = F.binary_cross_entropy_with_logits(fg_logits, target, reduction="none")
    m32 = vg_mask.to(dtype=bce_pp.dtype)
    val = (bce_pp * m32).sum() / m32.sum().clamp(min=1.0)

    if return_diag:
        diag = {
            "mu_H_fg": float(mu_Hf.mean().item()),
            "sigma_H_fg": float(sg_Hf.mean().item()),
            "mu_E_fg_comp": float(mu_Ef.mean().item()),
            "sigma_E_fg_comp": float(sg_Ef.mean().item()),
            "mu_H_bg": float(mu_Hb.mean().item()),
            "sigma_H_bg": float(sg_Hb.mean().item()),
            "mu_E_bg": float(mu_Eb.mean().item()),
            "sigma_E_bg": float(sg_Eb.mean().item()),
            "n_fg_anchor": float(fg_anchor.float().sum().item()),
            "n_bg_anchor": float(bg_anchor.float().sum().item()),
            "n_vg": float(n_vg.item()),
        }
        return val, diag
    return val



def _median_nearest_neighbor_distance(
    point_map: torch.Tensor,
    fallback: float = 40.0,
) -> torch.Tensor:
    """Per-image median point-to-nearest-point distance."""
    if point_map.ndim == 3:
        point_map = point_map.unsqueeze(1)
    pm = (point_map > 0)
    B = pm.shape[0]
    vals = []
    for b in range(B):
        coords = pm[b, 0].nonzero(as_tuple=False).to(dtype=torch.float32)
        if coords.shape[0] < 2:
            vals.append(torch.tensor(float(fallback), device=point_map.device, dtype=torch.float32))
            continue
        dist = torch.cdist(coords, coords)
        dist.fill_diagonal_(float("inf"))
        vals.append(dist.min(dim=1).values.median())
    return torch.stack(vals).view(B, 1, 1, 1)


def _norm_per_image_01(x: torch.Tensor) -> torch.Tensor:
    lo = x.flatten(2).amin(dim=2).view(-1, 1, 1, 1)
    hi = x.flatten(2).amax(dim=2).view(-1, 1, 1, 1)
    return ((x - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)


def point_stain_posterior_maps(
    stain_h: torch.Tensor,
    stain_e: torch.Tensor,
    point_map: torch.Tensor,
    mode: str = "posterior",
    adaptive_radius: bool = True,
    bg_min_dist: float = 45.0,
    prior_fg: float = 0.5,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Build BoNuS-free PSPS targets from H/E stains and point geometry only."""
    valid_modes = {"posterior", "h_only", "e_only", "no_distance", "shuffled_he"}
    if mode not in valid_modes:
        raise ValueError(f"Unsupported point-stain mode: {mode}")
    if stain_h.ndim == 3:
        stain_h = stain_h.unsqueeze(1)
    if stain_e.ndim == 3:
        stain_e = stain_e.unsqueeze(1)
    if point_map.ndim == 3:
        point_map = point_map.unsqueeze(1)

    dtype = stain_h.dtype
    device = stain_h.device
    B = stain_h.shape[0]
    stain_e = stain_e.to(device=device, dtype=dtype)
    point_map = point_map.to(device=device)

    h_norm = _norm_per_image_01(stain_h)
    e_norm = _norm_per_image_01(stain_e)
    if mode == "shuffled_he":
        Bsh, _, Hsh, Wsh = h_norm.shape
        h_flat = h_norm.flatten(2)
        e_flat = e_norm.flatten(2)
        h_shuf = torch.empty_like(h_flat)
        e_shuf = torch.empty_like(e_flat)
        for b in range(Bsh):
            perm = torch.randperm(Hsh * Wsh, device=h_norm.device)
            h_shuf[b, 0] = h_flat[b, 0, perm]
            e_shuf[b, 0] = e_flat[b, 0, perm]
        h_norm = h_shuf.view(Bsh, 1, Hsh, Wsh)
        e_norm = e_shuf.view(Bsh, 1, Hsh, Wsh)
    e_comp = 1.0 - e_norm

    if adaptive_radius:
        m = _median_nearest_neighbor_distance(point_map, fallback=40.0).to(device=device, dtype=dtype)
        r_core = (0.25 * m).clamp(6.0, 12.0)
        r_band = (0.65 * m).clamp(20.0, 40.0)
    else:
        m = stain_h.new_full((B, 1, 1, 1), 40.0)
        r_core = stain_h.new_full((B, 1, 1, 1), 8.0)
        r_band = stain_h.new_full((B, 1, 1, 1), 30.0)
    r_bg = torch.maximum(
        r_band + 10.0,
        stain_h.new_full((B, 1, 1, 1), float(bg_min_dist)),
    )

    max_dist = max(float(bg_min_dist) + 50.0, 95.0)
    d = _distance_transform_to_points(point_map, max_dist=max_dist).to(device=device, dtype=dtype)

    has_point = (point_map > 0).flatten(1).any(dim=1).view(B, 1, 1, 1)
    near_mask = (d <= r_band) & has_point
    far_mask = (d >= r_bg) & has_point
    fg_anchor = (d <= r_core) & has_point
    bg_anchor = far_mask

    mu_Hf, sg_Hf = _per_image_gaussian_fit(h_norm, fg_anchor, default_mu=0.65, default_sigma=0.15)
    mu_Ef, sg_Ef = _per_image_gaussian_fit(e_comp, fg_anchor, default_mu=0.60, default_sigma=0.18)
    mu_Hb, sg_Hb = _per_image_gaussian_fit(h_norm, bg_anchor, default_mu=0.30, default_sigma=0.18)
    mu_Eb, sg_Eb = _per_image_gaussian_fit(e_norm, bg_anchor, default_mu=0.55, default_sigma=0.20)

    if mode == "h_only":
        lik_fg = _gauss_pdf_tensor(h_norm, mu_Hf, sg_Hf)
        lik_bg = _gauss_pdf_tensor(h_norm, mu_Hb, sg_Hb)
    elif mode == "e_only":
        lik_fg = _gauss_pdf_tensor(e_comp, mu_Ef, sg_Ef)
        lik_bg = _gauss_pdf_tensor(e_norm, mu_Eb, sg_Eb)
    else:
        lik_fg = _gauss_pdf_tensor(h_norm, mu_Hf, sg_Hf) * _gauss_pdf_tensor(e_comp, mu_Ef, sg_Ef)
        lik_bg = _gauss_pdf_tensor(h_norm, mu_Hb, sg_Hb) * _gauss_pdf_tensor(e_norm, mu_Eb, sg_Eb)
    num = float(prior_fg) * lik_fg
    den = num + (1.0 - float(prior_fg)) * lik_bg
    stain_post = (num / (den + eps)).clamp(0.01, 0.99)

    sigma_d = (0.5 * r_band).clamp(min=1.0)
    dist_prior = torch.exp(-0.5 * (d / sigma_d).pow(2)).clamp(0.01, 0.99)
    if mode == "no_distance":
        target = stain_post
    else:
        logit_stain = torch.log(stain_post / (1.0 - stain_post + eps))
        logit_dist = torch.log(dist_prior / (1.0 - dist_prior + eps))
        target = torch.sigmoid(logit_stain + 0.5 * logit_dist).clamp(0.01, 0.99)

    posterior_confidence = (2.0 * (target - 0.5).abs()).clamp(0.0, 1.0)
    # Safety gate: near-point pixels are allowed to add positive stain expansion
    # only; negative PSPS supervision is restricted to far-background anchors.
    # This avoids turning weak-H nucleus rims into hard background.
    near_pos_weight = (2.0 * (target - 0.5)).clamp(min=0.0, max=1.0) * near_mask.to(dtype=dtype)
    far_neg_weight = (2.0 * (0.5 - target)).clamp(min=0.0, max=1.0) * far_mask.to(dtype=dtype)
    confidence = near_pos_weight + far_neg_weight
    return {
        "h_norm": h_norm,
        "e_norm": e_norm,
        "d": d,
        "m": m,
        "r_core": r_core,
        "r_band": r_band,
        "r_bg": r_bg,
        "fg_anchor": fg_anchor.to(dtype=dtype),
        "bg_anchor": bg_anchor.to(dtype=dtype),
        "band_mask": near_mask.to(dtype=dtype),
        "far_mask": far_mask.to(dtype=dtype),
        "supervision_mask": ((near_pos_weight + far_neg_weight) > 0).to(dtype=dtype),
        "target": target,
        "confidence": confidence,
        "posterior_confidence": posterior_confidence,
        "stain_post": stain_post,
        "dist_prior": dist_prior,
    }


def compute_point_stain_posterior_loss(
    fg_logits: torch.Tensor,
    stain_h: torch.Tensor,
    stain_e: torch.Tensor,
    point_map: torch.Tensor,
    mode: str = "posterior",
    adaptive_radius: bool = True,
    bg_min_dist: float = 45.0,
    return_diag: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """Point-Stain Posterior Supervision loss.

    The interface deliberately has no `clu` argument: all anchors are derived
    from `point_map`, `stain_h`, and `stain_e`.
    """
    if mode != "posterior":
        valid_modes = {"posterior", "h_only", "e_only", "no_distance", "shuffled_he"}
        if mode not in valid_modes:
            raise ValueError(f"Unsupported point-stain mode: {mode}")
    maps = point_stain_posterior_maps(
        stain_h.to(dtype=fg_logits.dtype),
        stain_e.to(dtype=fg_logits.dtype),
        point_map,
        mode=mode,
        adaptive_radius=adaptive_radius,
        bg_min_dist=bg_min_dist,
    )
    target = maps["target"].detach().to(dtype=fg_logits.dtype)
    weight = maps["confidence"].detach().to(dtype=fg_logits.dtype)
    denom = weight.sum()
    if denom.detach().item() < 1.0:
        loss = fg_logits.new_tensor(0.0)
    else:
        bce = F.binary_cross_entropy_with_logits(fg_logits, target, reduction="none")
        loss = (bce * weight).sum() / denom.clamp(min=1.0)
    if not return_diag:
        return loss
    with torch.no_grad():
        t = maps["target"].detach().clamp(1e-6, 1.0 - 1e-6)
        band = maps["supervision_mask"] > 0
        ent = -(t * torch.log(t) + (1.0 - t) * torch.log(1.0 - t))
        diag = {
            "weight_sum": float(weight.sum().item()),
            "target_entropy": float(ent[band].mean().item()) if band.any() else 0.0,
            "mean_confidence": float(weight[band].mean().item()) if band.any() else 0.0,
            "mean_r_core": float(maps["r_core"].mean().item()),
            "mean_r_band": float(maps["r_band"].mean().item()),
            "mean_r_bg": float(maps["r_bg"].mean().item()),
            "n_fg_anchor": float(maps["fg_anchor"].sum().item()),
            "n_bg_anchor": float(maps["bg_anchor"].sum().item()),
            "n_band": float(maps["band_mask"].sum().item()),
            "n_far": float(maps["far_mask"].sum().item()),
            "n_supervised": float(maps["supervision_mask"].sum().item()),
        }
    return loss, diag



def compute_he_quadrant_loss(
    fg_logits: torch.Tensor,        # (B, 1, H, W) raw logits
    clu: torch.Tensor,              # (B, H, W) with values in {0, 1, 2}
    stain_h: torch.Tensor,          # (B, H, W) or (B, 1, H, W)
    stain_e: torch.Tensor,          # (B, H, W) or (B, 1, H, W)
    h_lo: float = 0.5,              # lower bound (exclusive) for H_norm
    h_hi: float = 1.01,             # upper bound (exclusive) for H_norm
    e_lo: float = 0.5,              # lower bound (exclusive) for E_norm
    e_hi: float = 1.01,             # upper bound (exclusive) for E_norm
    target: float = 1.0,            # BCE target (1.0 = strong fg, 0.0 = strong bg)
    point_map=None,                 # (B, 1, H, W) for distance gating
    max_dist: float = 30.0,         # proximity gate radius
):
    """E5 DCSS: H/E quadrant-based stain supervision on BoNuS ignore zone.

    Oracle MO-val diagnostic showed:
      * BoNuS L_eosin_neg trigger (H<0.5 AND E>0.5) fires on 74% FG-truth pixels
        but labels them as BG — directionally INVERTED.
      * Reversed trigger (H>0.5 AND E>0.5) fires on 85% FG-truth pixels →
        STRONG FG supervision (this loss with target=1.0).
      * Low-stain corner (H<0.5 AND E<0.15) fires on 99% BG-truth pixels →
        STRONG BG supervision (this loss with target=0.0).

    By choosing (h_lo, h_hi, e_lo, e_hi) we select an H/E quadrant;  sets
    the supervision direction. Calling this twice with two configs gives the full
    Direction-Corrected Stain Supervision (DCSS) replacement of L_eosin_neg.

    Args:
        fg_logits: (B,1,H,W) raw logits
        clu: (B,H,W) cluster labels in {0,1,2}; supervision restricted to clu==2
        stain_h, stain_e: H and E channels in [~0,1] (per-image min-max normalized inside)
        h_lo, h_hi, e_lo, e_hi: quadrant bounds in [0,1]
        target: BCE target value for pixels in the selected quadrant
        point_map, max_dist: optional proximity gate
    """
    import torch.nn.functional as F
    if stain_h.ndim == 3:
        stain_h = stain_h.unsqueeze(1)
    if stain_e.ndim == 3:
        stain_e = stain_e.unsqueeze(1)
    stain_h = stain_h.to(dtype=fg_logits.dtype)
    stain_e = stain_e.to(dtype=fg_logits.dtype)

    # Per-image min-max normalize (consistent with stain_soft/eosin_neg).
    def _norm(x):
        x_min = x.flatten(2).amin(dim=2).view(-1, 1, 1, 1)
        x_max = x.flatten(2).amax(dim=2).view(-1, 1, 1, 1)
        return ((x - x_min) / (x_max - x_min + 1e-6)).clamp(0.0, 1.0)

    h = _norm(stain_h)
    e = _norm(stain_e)

    # Quadrant trigger
    h_in = (h > h_lo) & (h <= h_hi)
    e_in = (e > e_lo) & (e <= e_hi)
    quad = h_in & e_in  # (B,1,H,W)

    mask = (clu == 2).unsqueeze(1) & quad
    mask = mask.to(dtype=fg_logits.dtype)

    if point_map is not None and max_dist > 0:
        d = _distance_transform_to_points(point_map, max_dist=max_dist).to(dtype=fg_logits.dtype)
        if d.ndim == 3:
            d = d.unsqueeze(1)
        mask = mask * (d <= max_dist).to(dtype=fg_logits.dtype)

    n_eff = mask.sum().clamp_min(1.0)
    target_tensor = torch.full_like(fg_logits, float(target))
    loss_px = F.binary_cross_entropy_with_logits(
        fg_logits, target_tensor.detach(), reduction='none'
    )
    return (loss_px * mask).sum() / n_eff


def compute_anchor_cdf_loss(
    fg_logits: torch.Tensor,        # (B, 1, H, W) raw logits
    clu: torch.Tensor,              # (B, H, W) long with values in {0, 1, 2}
    stain_h: torch.Tensor,          # (B, H, W) or (B, 1, H, W) H density in ~[0,1]
    stain_e=None,                   # (B, H, W) or (B, 1, H, W) E density in ~[0,1]
    point_map=None,                 # (B, 1, H, W) point map for proximity gating
    max_dist: float = 30.0,         # proximity gate radius (0 = disabled)
    n_bins: int = 64,               # histogram resolution
    smooth_eps: float = 1.0,        # Laplace smoothing count (added to each bin)
    use_e: bool = True,             # combine E posterior via log-odds
    target_clip: float = 0.05,      # clip target to [eps, 1-eps] for BCE stability
):
    """E4 Anchor-CDF: non-parametric per-image empirical Bayesian posterior
    supervision on the BoNuS ignore zone (clu==2).

    Replaces hand-crafted thresholds in L_stain_soft (H > 0.5) and L_eosin_neg
    (E > 0.5 ∧ H < 0.5) with a principled adaptive formulation:

    For each image, we estimate empirical H (and optionally E) densities from
    BoNuS anchor regions provided for free by the dataset:
        P_fg(v) = (Laplace-smoothed) histogram of stain v in clu == 1
        P_bg(v) = (Laplace-smoothed) histogram of stain v in clu == 0
    The Bayesian posterior under uniform prior is then
        P(fg | v) = P_fg(v) / (P_fg(v) + P_bg(v))
    serving as soft BCE supervision for ignore-zone pixels.

    Key properties:
      * Non-parametric: makes no Gaussian assumption (cf. PCGS-v2, which used
        a bivariate Gauss and lost -9.2pp; H/E in clu==2 are bimodal/heavy-tailed).
      * Adapts to per-image staining variance by construction (no manual θ).
      * H and E posteriors can be combined via additive log-odds, assuming
        conditional independence given fg/bg.
      * Distance-to-point proximity gating reuses _distance_transform_to_points.

    The output is a scalar BCE loss averaged over clu==2 pixels (within max_dist).
    """
    import torch.nn.functional as F
    if stain_h.ndim == 3:
        stain_h = stain_h.unsqueeze(1)
    stain_h = stain_h.to(dtype=fg_logits.dtype)
    if stain_e is not None:
        if stain_e.ndim == 3:
            stain_e = stain_e.unsqueeze(1)
        stain_e = stain_e.to(dtype=fg_logits.dtype)

    B = fg_logits.shape[0]
    Hpx, Wpx = fg_logits.shape[-2], fg_logits.shape[-1]
    N = Hpx * Wpx
    device = fg_logits.device
    dtype = fg_logits.dtype

    # Per-image min-max normalize (consistent with stain_soft / eosin_neg).
    def _norm_per_image(s):
        s_min = s.flatten(2).amin(dim=2).view(B, 1, 1, 1)
        s_max = s.flatten(2).amax(dim=2).view(B, 1, 1, 1)
        return ((s - s_min) / (s_max - s_min + 1e-6)).clamp(0.0, 1.0)

    h_norm = _norm_per_image(stain_h)                       # (B,1,H,W) in [0,1]
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1, device=device, dtype=dtype)
    h_bin = torch.bucketize(h_norm.detach(), bin_edges).clamp(1, n_bins) - 1  # ∈[0,n_bins-1]
    h_bin_flat = h_bin.view(B, N)
    clu_flat = clu.view(B, N).long()

    def _per_image_posterior(bin_flat):
        # Returns (B, n_bins) tensor of P(fg | bin)
        post_list = []
        for b in range(B):
            vals = bin_flat[b]
            clu_b = clu_flat[b]
            fg_m = (clu_b == 1)
            bg_m = (clu_b == 0)
            if fg_m.sum() == 0 or bg_m.sum() == 0:
                post_list.append(torch.full((n_bins,), 0.5, device=device, dtype=dtype))
                continue
            hist_fg = torch.bincount(vals[fg_m], minlength=n_bins).to(dtype) + smooth_eps
            hist_bg = torch.bincount(vals[bg_m], minlength=n_bins).to(dtype) + smooth_eps
            pdf_fg = hist_fg / hist_fg.sum()
            pdf_bg = hist_bg / hist_bg.sum()
            post = pdf_fg / (pdf_fg + pdf_bg + 1e-8)
            post_list.append(post.detach())
        return torch.stack(post_list, dim=0)  # (B, n_bins)

    post_h = _per_image_posterior(h_bin_flat)                 # (B, n_bins)
    target_h = torch.gather(post_h, 1, h_bin_flat).view(B, 1, Hpx, Wpx)

    if use_e and stain_e is not None:
        e_norm = _norm_per_image(stain_e)
        e_bin = torch.bucketize(e_norm.detach(), bin_edges).clamp(1, n_bins) - 1
        e_bin_flat = e_bin.view(B, N)
        post_e = _per_image_posterior(e_bin_flat)
        target_e = torch.gather(post_e, 1, e_bin_flat).view(B, 1, Hpx, Wpx)
        # Combine via additive log-odds (conditional independence + uniform prior)
        eps = 1e-4
        th = target_h.clamp(eps, 1.0 - eps)
        te = target_e.clamp(eps, 1.0 - eps)
        log_odds = torch.log(th / (1.0 - th)) + torch.log(te / (1.0 - te))
        target = torch.sigmoid(log_odds)
    else:
        target = target_h

    # Numerical guard
    target = target.clamp(target_clip, 1.0 - target_clip)

    # Mask: clu == 2 (BoNuS ignore zone), optionally distance-gated.
    mask = (clu == 2).unsqueeze(1).to(dtype=dtype)
    if point_map is not None and max_dist > 0:
        d = _distance_transform_to_points(point_map, max_dist=max_dist).to(dtype=dtype)
        if d.ndim == 3:
            d = d.unsqueeze(1)
        prox = (d <= max_dist).to(dtype=dtype)
        mask = mask * prox

    n_eff = mask.sum().clamp_min(1.0)
    loss_px = F.binary_cross_entropy_with_logits(
        fg_logits, target.detach(), reduction='none'
    )
    loss = (loss_px * mask).sum() / n_eff
    return loss


def compute_pcgs_loss(
    fg_logits: torch.Tensor,        # (B, 1, H, W) raw logits
    clu: torch.Tensor,              # (B, H, W) long {0,1,2}
    stain_h: torch.Tensor,          # (B, H, W) or (B, 1, H, W) raw H density
    stain_e: torch.Tensor,          # (B, H, W) or (B, 1, H, W) raw E density
    point_map: torch.Tensor,        # (B, H, W) or (B, 1, H, W) binary
    pcgs_stats: dict,
    max_dist: float = 30.0,
    eps: float = 1e-6,
    return_em_inputs: bool = False,
):
    """PCGS BCE on voronoi-gap (clu==2) pixels.

    target = clamp( p(y=1 | H, E, d) , 0.01, 0.99 )

    Posterior:
      lik_fg = N(H; μ_Hf, σ_Hf) · N(1-E; μ_Ef, σ_Ef)
      lik_bg = N(H; μ_Hb, σ_Hb) · N(E;   μ_Eb, σ_Eb)
      prox   = exp(-d² / (2 σ_d²))
      num    = prior_fg · lik_fg · prox
      den    = num + (1 - prior_fg) · lik_bg · (1 - prox)

    Returns: scalar BCE (mean over voronoi-gap pixels).
             If return_em_inputs, also returns dict
             {h_norm, e_norm, d, mask, posterior} for the EM accumulator.
    """
    if stain_h.ndim == 3:
        stain_h = stain_h.unsqueeze(1)
    if stain_e.ndim == 3:
        stain_e = stain_e.unsqueeze(1)
    if point_map.ndim == 3:
        point_map = point_map.unsqueeze(1)

    stain_h = stain_h.to(dtype=fg_logits.dtype)
    stain_e = stain_e.to(dtype=fg_logits.dtype)

    # Per-image min-max normalize H and E to [0,1] (same convention as B9 / B12).
    def _norm(x):
        lo = x.flatten(2).amin(dim=2).view(-1, 1, 1, 1)
        hi = x.flatten(2).amax(dim=2).view(-1, 1, 1, 1)
        return (x - lo) / (hi - lo + 1e-6)
    h_norm = _norm(stain_h)
    e_norm = _norm(stain_e)

    # Distance field (one-shot per batch, ~5ms on CPU at 224²).
    d = _distance_transform_to_points(point_map, max_dist=max_dist).to(dtype=fg_logits.dtype)

    # Voronoi gap mask: only supervise clu==2 pixels.
    mask = (clu == 2).unsqueeze(1)
    n = mask.sum()
    if n.item() < 1:
        loss = fg_logits.new_tensor(0.0)
        if return_em_inputs:
            return loss, {"h_norm": h_norm, "e_norm": e_norm, "d": d, "mask": mask, "posterior": None}
        return loss

    # Read EM parameters (kept as Python floats inside pcgs_stats dict).
    mu_Hf, sg_Hf = float(pcgs_stats["mu_H_fg"]), float(pcgs_stats["sigma_H_fg"])
    mu_Ef, sg_Ef = float(pcgs_stats["mu_E_fg"]), float(pcgs_stats["sigma_E_fg"])  # on (1-E)
    mu_Hb, sg_Hb = float(pcgs_stats["mu_H_bg"]), float(pcgs_stats["sigma_H_bg"])
    mu_Eb, sg_Eb = float(pcgs_stats["mu_E_bg"]), float(pcgs_stats["sigma_E_bg"])
    sigma_d = max(float(pcgs_stats.get("sigma_d", 8.0)), 1e-3)
    prior_fg = float(pcgs_stats.get("prior_fg", 0.35))
    prior_fg = min(max(prior_fg, 1e-3), 1.0 - 1e-3)

    lik_fg = _gaussian_pdf(h_norm, mu_Hf, sg_Hf) * _gaussian_pdf(1.0 - e_norm, mu_Ef, sg_Ef)
    lik_bg = _gaussian_pdf(h_norm, mu_Hb, sg_Hb) * _gaussian_pdf(e_norm,       mu_Eb, sg_Eb)
    prox   = torch.exp(-(d * d) / (2.0 * sigma_d * sigma_d))

    num = prior_fg * lik_fg * prox
    den = num + (1.0 - prior_fg) * lik_bg * (1.0 - prox + 1e-6)
    posterior = (num / (den + eps)).clamp(0.01, 0.99).to(dtype=fg_logits.dtype)

    bce_pp = F.binary_cross_entropy_with_logits(fg_logits, posterior, reduction="none")
    m32 = mask.to(dtype=bce_pp.dtype)
    val = (bce_pp * m32).sum() / m32.sum().clamp(min=1.0)

    if return_em_inputs:
        return val, {"h_norm": h_norm, "e_norm": e_norm, "d": d, "mask": mask, "posterior": posterior}
    return val


@torch.no_grad()
def pcgs_em_accumulate(
    fg_sigmoid: torch.Tensor,       # (B, 1, H, W) detached sigmoid(fg_logits)
    em_inputs: dict,                # output of compute_pcgs_loss(return_em_inputs=True)
) -> dict:
    """Accumulate sufficient statistics for the EM M-step.

    Uses the current model's detached fg probability as the soft
    responsibility r(x); fg/bg statistics weighted by r and 1-r.

    Returns dict of running sums (CPU floats):
      n_fg, n_bg, sum_H_fg, sum_H2_fg, sum_Ec_fg, sum_Ec2_fg,
      sum_H_bg, sum_H2_bg, sum_E_bg, sum_E2_bg,
      n_d_fg, sum_d2_fg
    """
    h = em_inputs["h_norm"]
    e = em_inputs["e_norm"]
    d = em_inputs["d"]
    mask = em_inputs["mask"].to(dtype=fg_sigmoid.dtype)
    r = fg_sigmoid.to(dtype=h.dtype) * mask           # responsibility for fg
    rb = (1.0 - fg_sigmoid).to(dtype=h.dtype) * mask  # responsibility for bg
    ec = 1.0 - e                                       # complement of E (for fg)

    def _s(x):
        return float(x.detach().cpu().item())

    out = {
        "n_fg":      _s(r.sum()),
        "n_bg":      _s(rb.sum()),
        "sum_H_fg":  _s((r * h).sum()),
        "sum_H2_fg": _s((r * h * h).sum()),
        "sum_Ec_fg": _s((r * ec).sum()),
        "sum_Ec2_fg":_s((r * ec * ec).sum()),
        "sum_H_bg":  _s((rb * h).sum()),
        "sum_H2_bg": _s((rb * h * h).sum()),
        "sum_E_bg":  _s((rb * e).sum()),
        "sum_E2_bg": _s((rb * e * e).sum()),
    }
    # σ_d update: only over high-confidence fg pixels (r > 0.5).
    hi = (fg_sigmoid > 0.5).to(dtype=h.dtype) * mask
    out["n_d_fg"]    = _s(hi.sum())
    out["sum_d2_fg"] = _s((hi * d * d).sum())
    return out


def pcgs_em_update_stats(
    pcgs_stats: dict,
    accum: dict,
    ema: float = 0.9,
    sigma_clamp: tuple = (0.05, 0.40),
    prior_clamp: tuple = (0.10, 0.70),
    sigma_d_clamp: tuple = (2.0, 20.0),
) -> dict:
    """Apply M-step using accumulated sufficient stats; EMA-blend into pcgs_stats.

    Returns the updated pcgs_stats (in-place mutation).
    """
    nf = max(accum["n_fg"], 1e-6)
    nb = max(accum["n_bg"], 1e-6)
    new_mu_Hf  = accum["sum_H_fg"]  / nf
    new_var_Hf = max(accum["sum_H2_fg"] / nf - new_mu_Hf * new_mu_Hf, sigma_clamp[0] ** 2)
    new_mu_Ef  = accum["sum_Ec_fg"] / nf            # on (1-E)
    new_var_Ef = max(accum["sum_Ec2_fg"] / nf - new_mu_Ef * new_mu_Ef, sigma_clamp[0] ** 2)
    new_mu_Hb  = accum["sum_H_bg"]  / nb
    new_var_Hb = max(accum["sum_H2_bg"] / nb - new_mu_Hb * new_mu_Hb, sigma_clamp[0] ** 2)
    new_mu_Eb  = accum["sum_E_bg"]  / nb
    new_var_Eb = max(accum["sum_E2_bg"] / nb - new_mu_Eb * new_mu_Eb, sigma_clamp[0] ** 2)
    import math as _m

    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    new = {
        "mu_H_fg":    _clamp(new_mu_Hf, 0.0, 1.0),
        "sigma_H_fg": _clamp(_m.sqrt(new_var_Hf), *sigma_clamp),
        "mu_E_fg":    _clamp(new_mu_Ef, 0.0, 1.0),
        "sigma_E_fg": _clamp(_m.sqrt(new_var_Ef), *sigma_clamp),
        "mu_H_bg":    _clamp(new_mu_Hb, 0.0, 1.0),
        "sigma_H_bg": _clamp(_m.sqrt(new_var_Hb), *sigma_clamp),
        "mu_E_bg":    _clamp(new_mu_Eb, 0.0, 1.0),
        "sigma_E_bg": _clamp(_m.sqrt(new_var_Eb), *sigma_clamp),
        "prior_fg":   _clamp(nf / (nf + nb), *prior_clamp),
    }
    nd = max(accum["n_d_fg"], 1e-6)
    new_sigma_d = _m.sqrt(max(accum["sum_d2_fg"] / nd, sigma_d_clamp[0] ** 2))
    new["sigma_d"] = _clamp(new_sigma_d, *sigma_d_clamp)

    for k, v in new.items():
        prev = float(pcgs_stats.get(k, v))
        pcgs_stats[k] = ema * prev + (1.0 - ema) * v
    return pcgs_stats


def compute_eosin_anchored_neg_loss(
    fg_logits, clu, stain_e, stain_h, tau_e: float = 0.5, tau_h: float = 0.5,
    point_map=None, max_dist: int = 0, min_dist: int = 0,
    fg_confidence_gate: float = 0.0,
    uncertainty=None, uncertainty_pow: float = 1.0,
    use_cpgs: bool = False, cpgs_k: float = 4.0, cpgs_alpha: float = 1.0,
    adaptive_radius: bool = False,
    adapt_alpha: float = 0.6,
    adapt_r_min: int = 15,
    adapt_r_max: int = 50,
):
    """B12 MPS-negative: Hard bg supervision on the BoNuS ignore zone (clu==2)
    where eosin is strong AND hematoxylin is weak.

    Symmetric counterpart to L_stain_soft. Where L_stain_soft uses H (DNA marker)
    to soft-positive-anchor ignore-zone pixels likely to be nucleus interior,
    L_eosin_neg uses E (cytoplasm marker) + low H to hard-negative-anchor
    ignore-zone pixels likely to be pure cytoplasm/stroma.

    Theory: H and E are complementary chromophores in standard H&E staining.
    Together they recover the BoNuS-unsupervised pixels with biological certainty:
      - high H, low E in clu==2  → soft fg (L_stain_soft, target=alpha~0.7)
      - low H, high E in clu==2  → hard bg (L_eosin_neg,  target=0.0)
      - low H, low E in clu==2   → truly unsupervised (low-confidence region)

    Args:
        fg_logits: (B, 1, H, W)
        clu:       (B, H, W) long
        stain_e:   (B, H, W) eosin density in [0, 1]
        stain_h:   (B, H, W) hematoxylin density in [0, 1]
        tau_e:     min normalized E to qualify as cytoplasm
        tau_h:     max normalized H to qualify as cytoplasm

    Returns:
        Scalar BCE loss on (clu==2) ∩ (e_norm>tau_e) ∩ (h_norm<tau_h) pixels with target 0.
    """
    import torch
    import torch.nn.functional as F
    if stain_e.ndim == 3:
        stain_e = stain_e.unsqueeze(1)
    if stain_h.ndim == 3:
        stain_h = stain_h.unsqueeze(1)
    stain_e = stain_e.to(dtype=fg_logits.dtype)
    stain_h = stain_h.to(dtype=fg_logits.dtype)
    # Per-image min-max normalize.
    def _norm(x):
        lo = x.flatten(2).amin(dim=2).view(-1, 1, 1, 1)
        hi = x.flatten(2).amax(dim=2).view(-1, 1, 1, 1)
        return (x - lo) / (hi - lo + 1e-6)
    e_norm = _norm(stain_e)
    h_norm = _norm(stain_h)
    ignore = (clu == 2).unsqueeze(1)
    # CPGS: drop hard E/H thresholds; weight per-pixel by inverted polarization.
    if use_cpgs:
        hard_bg = ignore
        pol = (h_norm - e_norm) / (h_norm + e_norm + 1e-3)  # in [-1, 1]
        # Eosin-neg wants E-dominant pixels → negative pol → invert: sigmoid(-k·pol).
        cpgs_w = torch.sigmoid(-cpgs_k * pol.float()).clamp(min=1e-6).pow(cpgs_alpha)
        cpgs_w = cpgs_w.to(fg_logits.dtype)
    else:
        hard_bg = ignore & (e_norm > tau_e) & (h_norm < tau_h)
        cpgs_w = None
    # B13: symmetric proximity gate also for L_eosin_neg — restricts hard bg
    # supervision to neighborhood of GT points. Keeps the loss focused on
    # ignore-zone pixels that are biologically meaningful (adjacent to nuclei),
    # not arbitrary background stroma.
    if point_map is not None and max_dist > 0:
        if point_map.ndim == 3:
            pm = point_map.float().unsqueeze(1)
        else:
            pm = point_map.float()
        if adaptive_radius:
            outer, _ = _adaptive_proximity_mask(
                pm.squeeze(1), alpha=adapt_alpha, r_min=adapt_r_min, r_max=adapt_r_max,
            )
        else:
            k_out = 2 * int(max_dist) + 1
            outer = F.max_pool2d(pm, kernel_size=k_out, stride=1, padding=int(max_dist)) > 0
        if min_dist > 0:
            # B16: inner-radius gate — exclude immediate nucleus boundary
            # (dist < min_dist) so L_eosin_neg cannot erode the core.
            k_in = 2 * int(min_dist) + 1
            inner = F.max_pool2d(pm, kernel_size=k_in, stride=1, padding=int(min_dist)) > 0
            proximity = outer & ~inner
        else:
            proximity = outer
        hard_bg = hard_bg & proximity
    # C2: self-confidence gate — drop pixels the model already believes are fg.
    # Cuts the ~24.6% leak where eosin_neg trigger overlaps real nucleus
    # boundary/interior. Detached so the gate decision isn't part of the graph.
    # 0.0 = disabled (preserves B-series / C1 behavior).
    if fg_confidence_gate > 0.0:
        with torch.no_grad():
            fg_prob = torch.sigmoid(fg_logits)
        hard_bg = hard_bg & (fg_prob < fg_confidence_gate)
    n = hard_bg.sum()
    if n.item() < 1:
        return fg_logits.new_tensor(0.0)
    target = torch.zeros_like(fg_logits)
    bce_pp = F.binary_cross_entropy_with_logits(fg_logits, target, reduction="none")
    bce32 = bce_pp.float()
    hb32 = hard_bg.float()
    if uncertainty is not None:
        u = uncertainty.float()
        if u.shape != bce32.shape:
            u = u.expand_as(bce32)
        if uncertainty_pow != 1.0:
            u = u.clamp(min=1e-6).pow(uncertainty_pow)
        bce32 = bce32 * u
    if cpgs_w is not None:
        w32 = cpgs_w.float()
        weighted_mask = hb32 * w32
        denom = weighted_mask.sum().clamp(min=1.0)
        val = (bce32 * weighted_mask).sum() / denom
    else:
        val = (bce32 * hb32).sum() / n.float().clamp(min=1.0)
    return val.to(fg_logits.dtype)


def compute_eosin_source_div_loss(
    flow, clu, stain_e, stain_h, tau_e: float = 0.5, tau_h: float = 0.5,
    point_map=None, max_dist: int = 0, margin: float = 0.3,
):
    """L_src: eosin-anchored SOURCE ridge on the flow divergence.

    Same gate as L_eosin_neg (clu==2 AND e_norm>tau_e AND h_norm<tau_h, i.e.
    eosin-rich cytoplasm/stroma in the touching-nuclei gap, optionally within
    max_dist of points), but instead of pushing the fg head to bg, it supervises
    the FLOW FIELD to have POSITIVE divergence there: flow points AWAY from the
    gap toward both neighbouring nuclei -> a source ridge that separates them.

    Hinge: penalize div(flow) below `margin`. Upgrades the load-bearing E- signal
    from "shape the fg map" to "shape the field's differential topology". Cellpose
    has no supervised source ridge, so this is a supervision-level differentiator.
    """
    import torch
    import torch.nn.functional as F
    if stain_e.ndim == 3:
        stain_e = stain_e.unsqueeze(1)
    if stain_h.ndim == 3:
        stain_h = stain_h.unsqueeze(1)
    stain_e = stain_e.to(dtype=flow.dtype)
    stain_h = stain_h.to(dtype=flow.dtype)

    def _norm(x):
        lo = x.flatten(2).amin(dim=2).view(-1, 1, 1, 1)
        hi = x.flatten(2).amax(dim=2).view(-1, 1, 1, 1)
        return (x - lo) / (hi - lo + 1e-6)

    e_norm = _norm(stain_e)
    h_norm = _norm(stain_h)
    gate = (clu == 2).unsqueeze(1) & (e_norm > tau_e) & (h_norm < tau_h)
    if point_map is not None and max_dist > 0:
        pm = point_map.float().unsqueeze(1) if point_map.ndim == 3 else point_map.float()
        k_out = 2 * int(max_dist) + 1
        prox = F.max_pool2d(pm, kernel_size=k_out, stride=1, padding=int(max_dist)) > 0
        gate = gate & prox
    n = gate.sum()
    if n.item() < 1:
        return flow.new_tensor(0.0)
    div = divergence(flow)                      # (B, 1, H, W); want div >= margin on gate
    deficit = F.relu(margin - div).float()
    return (deficit * gate.float()).sum() / n.float().clamp(min=1.0)


def _adaptive_proximity_mask(point_map, alpha: float = 0.6,
                              r_min: int = 15, r_max: int = 50):
    """Per-image adaptive proximity mask. Radius = clip(alpha * median(NN_dist), r_min, r_max).
    point_map: (B, H, W) or (B, 1, H, W). Returns (B, 1, H, W) bool mask + list of radii used.
    """
    import torch
    import torch.nn.functional as F
    if point_map.ndim == 3:
        pm = point_map.float().unsqueeze(1)
    else:
        pm = point_map.float()
    B = pm.shape[0]
    masks = []
    radii = []
    for b in range(B):
        coords = torch.nonzero(pm[b, 0] > 0).float()  # (N, 2)
        n = coords.shape[0]
        if n < 2:
            r = r_min
        else:
            diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # (N, N, 2)
            dist = (diff.pow(2).sum(-1) + 1e-6).sqrt()
            dist = dist + torch.eye(n, device=dist.device) * 1e6  # mask diagonal
            nn_dist = dist.min(dim=1).values  # (N,)
            median_nn = nn_dist.median().item()
            r = max(r_min, min(r_max, int(alpha * median_nn)))
        k = 2 * r + 1
        m = F.max_pool2d(pm[b:b+1], kernel_size=k, stride=1, padding=r) > 0
        masks.append(m)
        radii.append(r)
    return torch.cat(masks, dim=0), radii


def compute_stain_anchored_soft_label_loss(
    fg_logits, clu, stain_h, alpha: float = 0.7, h_threshold: float = 0.5,
    point_map=None, max_dist: int = 0,
    target_mode: str = "binary",
    proximity_kernel: str = "square",
    trigger_opening: int = 0,
    adaptive_radius: bool = False,
    adapt_alpha: float = 0.6,
    adapt_r_min: int = 15,
    adapt_r_max: int = 50,
    uncertainty=None, uncertainty_pow: float = 1.0,
    use_cpgs: bool = False, cpgs_k: float = 4.0, cpgs_alpha: float = 1.0,
    stain_e=None,
):
    """B9 SCR-soft: Soft fg supervision on the BoNuS ignore zone (clu==2)
    where hematoxylin staining is strong.

    BoNuS supervision labels each pixel as:
      clu == 0: bg (forced)
      clu == 1: fg (cluster disk around point)
      clu == 2: ignore (voronoi boundary + ambiguous areas)

    Pixels in clu==2 with high H stain are biologically very likely to be
    nucleus interior — they're typically in the gap between two adjacent
    voronoi cells where the real nucleus body extends past the cluster disk
    but inside its voronoi cell. Supervising these pixels with a SOFT fg
    target (default alpha=0.7) directly closes the under-segmentation gap
    that BoNuS leaves unsupervised, without contradicting the original
    cluster-BCE/voronoi supervision.

    Args:
        fg_logits: (B, 1, H, W) raw logits
        clu:       (B, H, W) long tensor with values in {0, 1, 2}
        stain_h:   (B, H, W) or (B, 1, H, W) H density in [0, 1]
        alpha:     soft fg target value (typical 0.5-0.9)
        h_threshold: pixels with normalized H above this in the ignore zone
                     get the soft supervision

    Returns:
        Scalar BCE loss on the soft-anchored pixels, normalized by their count.
        Returns 0 tensor if no pixels qualify.
    """
    import torch
    import torch.nn.functional as F
    if stain_h.ndim == 3:
        stain_h = stain_h.unsqueeze(1)
    stain_h = stain_h.to(dtype=fg_logits.dtype)
    # Per-image min-max normalize stain_h so threshold is dataset-stable.
    s_min = stain_h.flatten(2).amin(dim=2).view(-1, 1, 1, 1)
    s_max = stain_h.flatten(2).amax(dim=2).view(-1, 1, 1, 1)
    s_norm = (stain_h - s_min) / (s_max - s_min + 1e-6)
    ignore = (clu == 2).unsqueeze(1)                    # (B, 1, H, W)
    # B14: soft_mask is either binary (H > h_threshold) or full ignore zone.
    # CPGS (D-series followup): drop H threshold; weight per-pixel by chromophore polarization.
    if use_cpgs:
        soft_mask = ignore
        # Compute polarization (need E channel).
        if stain_e is None:
            raise ValueError("use_cpgs=True requires stain_e")
        if stain_e.ndim == 3:
            stain_e = stain_e.unsqueeze(1)
        e = stain_e.to(dtype=fg_logits.dtype)
        e_min = e.flatten(2).amin(dim=2).view(-1, 1, 1, 1)
        e_max = e.flatten(2).amax(dim=2).view(-1, 1, 1, 1)
        e_norm = (e - e_min) / (e_max - e_min + 1e-6)
        pol = (s_norm - e_norm) / (s_norm + e_norm + 1e-3)  # in [-1, 1]
        cpgs_w = torch.sigmoid(cpgs_k * pol.float()).clamp(min=1e-6).pow(cpgs_alpha)
        cpgs_w = cpgs_w.to(fg_logits.dtype)
    elif target_mode == "binary":
        soft_mask = ignore & (s_norm > h_threshold)
        cpgs_w = None
    elif target_mode == "continuous":
        soft_mask = ignore
        cpgs_w = None
    else:
        raise ValueError(f"target_mode must be 'binary' or 'continuous', got {target_mode}")
    # Proximity gate — restrict to neighborhood of GT points.
    if point_map is not None and max_dist > 0:
        if point_map.ndim == 3:
            pm = point_map.float().unsqueeze(1)
        else:
            pm = point_map.float()
        if adaptive_radius:
            # C7: per-image adaptive radius based on median NN distance of GT points.
            proximity, _ = _adaptive_proximity_mask(
                pm.squeeze(1), alpha=adapt_alpha, r_min=adapt_r_min, r_max=adapt_r_max,
            )
        elif proximity_kernel == "disk":
            # B14a: circular kernel via 2D conv with disk mask. Truly Euclidean.
            md = int(max_dist)
            yy, xx = torch.meshgrid(
                torch.arange(-md, md + 1, device=pm.device),
                torch.arange(-md, md + 1, device=pm.device),
                indexing="ij",
            )
            disk = ((yy ** 2 + xx ** 2) <= md ** 2).to(dtype=pm.dtype).unsqueeze(0).unsqueeze(0)
            proximity = F.conv2d(pm, disk, padding=md) > 0
        else:  # square
            kernel = 2 * int(max_dist) + 1
            proximity = F.max_pool2d(pm, kernel_size=kernel, stride=1, padding=int(max_dist)) > 0
        soft_mask = soft_mask & proximity
    # C1: morphological opening on the trigger mask to remove isolated H-high
    # specks (typically stroma micro-structures) that would otherwise inject
    # spurious soft-fg supervision into the bg region. Default 0 = disabled
    # (B-series behavior preserved).
    if trigger_opening > 0:
        k = int(trigger_opening)
        kernel = 2 * k + 1
        soft_f = soft_mask.float()
        # Erosion = -maxpool(-x); Dilation = maxpool(x).
        eroded = -F.max_pool2d(-soft_f, kernel_size=kernel, stride=1, padding=k)
        opened = F.max_pool2d(eroded, kernel_size=kernel, stride=1, padding=k)
        soft_mask = opened > 0.5
    n_soft = soft_mask.sum()
    if n_soft.item() < 1:
        return fg_logits.new_tensor(0.0)
    target = torch.full_like(fg_logits, alpha)
    bce_pp = F.binary_cross_entropy_with_logits(fg_logits, target, reduction="none")
    bce32 = bce_pp.float()
    mask32 = soft_mask.float()
    if uncertainty is not None:
        u = uncertainty.float()
        if u.shape != bce32.shape:
            u = u.expand_as(bce32)
        if uncertainty_pow != 1.0:
            u = u.clamp(min=1e-6).pow(uncertainty_pow)
        bce32 = bce32 * u
    if cpgs_w is not None:
        # Per-pixel polarization weight; normalize by sum(mask * w).
        w32 = cpgs_w.float()
        weighted_mask = mask32 * w32
        denom = weighted_mask.sum().clamp(min=1.0)
        loss = (bce32 * weighted_mask).sum() / denom
    else:
        loss = (bce32 * mask32).sum() / n_soft.float().clamp(min=1.0)
    return loss.to(fg_logits.dtype)


def compute_h_gradient_boundary_loss(outputs, batch):
    """BCE loss on boundary_head supervised by normalized Sobel(stain_h).

    The H channel ([0, 1]) is brightest inside nuclei. |∇H| peaks at nucleus
    boundaries, giving a free stain-grounded boundary signal. Hematoxylin-derived
    boundaries are also smoother and more biologically meaningful than e.g.
    Voronoi or distance-transform boundaries.
    """
    import torch
    import torch.nn.functional as F
    b_logits = outputs["boundary_logits"]                  # (B, 1, H, W)
    if "stain_h" not in batch:
        return b_logits.new_tensor(0.0)
    stain_h = batch["stain_h"].to(dtype=b_logits.dtype)
    if stain_h.ndim == 3:
        stain_h = stain_h.unsqueeze(1)                     # (B, 1, H, W)

    # 3×3 Sobel filters.
    kx = torch.tensor([[-1.0, 0.0, 1.0],
                        [-2.0, 0.0, 2.0],
                        [-1.0, 0.0, 1.0]], dtype=b_logits.dtype, device=b_logits.device).view(1, 1, 3, 3) / 8.0
    ky = kx.transpose(-2, -1).contiguous()
    gx = F.conv2d(stain_h, kx, padding=1)
    gy = F.conv2d(stain_h, ky, padding=1)
    grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-6)        # (B, 1, H, W)
    grad_max = grad_mag.flatten(2).amax(dim=-1).clamp(min=1e-6).view(-1, 1, 1, 1)
    target = (grad_mag / grad_max).clamp(0.0, 1.0)

    # BCE with logits. We weight by fg-or-near-fg region only (avoid background
    # noise drowning the boundary signal).
    return F.binary_cross_entropy_with_logits(b_logits, target)


def compute_stain_confidence_loss(
    fg_logits, clu, stain_h, stain_e,
    sharpness: float = 8.0,
    point_map=None, max_dist: int = 0,
    confidence_floor: float = 1.0,
):
    """B17 SCMP - Stain-Confidence Modality Prior.

    Unified continuous replacement for L_stain_soft + L_eosin_neg. For each
    pixel in the BoNuS ignore zone (clu==2) within proximity of any GT point,
    derive both the pseudo-label and its sharpness from the H/E differential:
        diff   = norm(H) - norm(E)        in [-1, 1]
        target = sigmoid(k * diff)        in (0, 1)
    Decisive supervision at stain-clear pixels, gentle pull-toward-0.5 at
    chromophore-ambiguous boundaries. No threshold hyper-params.
    """
    import torch
    import torch.nn.functional as F
    if stain_e.ndim == 3:
        stain_e = stain_e.unsqueeze(1)
    if stain_h.ndim == 3:
        stain_h = stain_h.unsqueeze(1)
    stain_e = stain_e.to(dtype=fg_logits.dtype)
    stain_h = stain_h.to(dtype=fg_logits.dtype)
    def _norm(x):
        lo = x.flatten(2).amin(dim=2).view(-1, 1, 1, 1)
        hi = x.flatten(2).amax(dim=2).view(-1, 1, 1, 1)
        return (x - lo) / (hi - lo + 1e-6)
    e_norm = _norm(stain_e)
    h_norm = _norm(stain_h)
    diff = (h_norm - e_norm).float()
    target = torch.sigmoid(sharpness * diff).to(dtype=fg_logits.dtype)
    mask = (clu == 2).unsqueeze(1)
    if point_map is not None and max_dist > 0:
        if point_map.ndim == 3:
            pm = point_map.float().unsqueeze(1)
        else:
            pm = point_map.float()
        kernel = 2 * int(max_dist) + 1
        proximity = F.max_pool2d(pm, kernel_size=kernel, stride=1, padding=int(max_dist)) > 0
        mask = mask & proximity
    n = mask.sum()
    if n.item() < 1:
        return fg_logits.new_tensor(0.0)
    bce = F.binary_cross_entropy_with_logits(fg_logits, target, reduction="none")
    bce32 = bce.float()
    m32 = mask.float()
    if confidence_floor < 1.0:
        # B18: weight = max(|diff|, floor). |diff|=0 at boundary → weight=floor.
        # |diff|=1 at stain-clear → weight=1.
        w = torch.clamp(diff.abs(), min=confidence_floor).float()
        num = (bce32 * m32 * w).sum()
        den = (m32 * w).sum().clamp(min=1.0)
        val = num / den
    else:
        val = (bce32 * m32).sum() / n.float().clamp(min=1.0)
    return val.to(fg_logits.dtype)

def adaptive_proximity_mask(point_map, alpha: float = 0.6, r_min: int = 8, r_max: int = 30):
    """Per-point adaptive disk mask. radius_i = clip(alpha * d_NN_i, r_min, r_max);
    pixel is in-mask if within its NEAREST point's radius. Returns (B,1,H,W) float."""
    if point_map.ndim == 3:
        point_map = point_map.unsqueeze(1)
    B, _, H, W = point_map.shape
    dev = point_map.device
    out = torch.zeros((B, 1, H, W), device=dev)
    Y = torch.arange(H, device=dev).float()
    X = torch.arange(W, device=dev).float()
    for b in range(B):
        pts = (point_map[b, 0] > 0).nonzero(as_tuple=False).float()  # (N,2) [y,x]
        N = pts.shape[0]
        if N == 0:
            continue
        if N == 1:
            radii = torch.full((1,), float(r_max), device=dev)
        else:
            pd = torch.cdist(pts, pts) + torch.eye(N, device=dev) * 1e9
            dnn = pd.min(dim=1).values
            radii = (alpha * dnn).clamp(min=float(r_min), max=float(r_max))
        dy = Y.view(1, H, 1) - pts[:, 0].view(N, 1, 1)
        dx = X.view(1, 1, W) - pts[:, 1].view(N, 1, 1)
        d2 = dy * dy + dx * dx                 # (N,H,W)
        nidx = d2.argmin(dim=0)                # (H,W)
        dmin2 = d2.min(dim=0).values           # (H,W)
        rmap = radii[nidx]                     # (H,W)
        out[b, 0] = (dmin2 <= rmap * rmap).float()
    return out


def compute_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    lambda_fg: float = 1.0,
    lambda_flow: float = 2.0,
    lambda_point: float = 2.0,
    lambda_sam_fg: float = 0.0,
    lambda_sam_flow: float = 0.0,
    lambda_div: float = 0.5,
    div_conductance: str = "constant",
    screen_kappa: float = 0.1,
    sink_sigma: float = 3.0,
    sink_adaptive: bool = False,
    sink_alpha: float = 0.25,
    sink_min: float = 3.0,
    sink_max: float = 8.0,
    stain_alpha: float = 0.1,
    lambda_center: float = 0.75,
    center_sigma: float = 3.0,
    lambda_aff: float = 0.0,
    aff_radius: float = 5.0,
    lambda_psic: float = 0.0,
    psic_r_pos: float = 10.0,
    psic_beta: float = 0.6,
    psic_r_neg: float = 64.0,
    psic_k_min: int = 4,
    psic_k_anchor: int = 8,
    psic_tau: float = 0.1,
    # v2 additions
    psic_adaptive_radius: bool = False,
    psic_r_pos_max: float = 8.0,
    psic_alpha_dnn: float = 0.25,
    psic_top_k_neg: int = 0,
    psic_use_margin: bool = False,
    psic_margin_value: float = 0.15,
    psic_neg_weight: float = 0.5,
    # B6 HGBL (hematoxylin-gradient boundary loss)
    lambda_h_boundary: float = 0.0,
    # B7 SCR-interior (stain-weighted Dice on fg head)
    lambda_stain_dice: float = 0.0,
    stain_dice_alpha: float = 0.5,
    # B9 SCR-soft (stain-anchored soft fg label in BoNuS ignore zone clu==2)
    lambda_stain_soft: float = 1.0,
    stain_soft_alpha: float = 0.7,
    stain_soft_h_thresh: float = 0.5,
    # B13 proximity gate: only supervise ignore-zone pixels within max_dist of any GT point.
    # 0 = disabled (legacy B9/B12 behavior).
    stain_soft_max_dist: int = 0,
    # B14: target mode + proximity kernel for L_stain_soft
    stain_soft_target_mode: str = "binary",
    stain_soft_proximity_kernel: str = "square",
    # C1: morphological opening on stain_soft trigger mask (kernel radius, px).
    # 0 = disabled (preserves B-series). Removes isolated H-high specks.
    stain_soft_trigger_opening: int = 0,
    # B15: expand cluster disk via stain prior. When True, the L_stain_soft pixels
    # are promoted to HARD fg supervision (target=1.0) integrated into L_fg, instead
    # of soft target=alpha. This directly enlarges the fg supervision radius from
    # ~10px (BoNuS cluster disk) to H-stain-defined nucleus boundary (~15-18px).
    # When enabled, L_stain_soft is automatically zeroed (avoid double-counting).
    expand_cluster_via_stain: bool = False,
    expand_cluster_h_thresh: float = 0.5,
    expand_cluster_max_dist: int = 15,
    # B12 MPS-negative (eosin-anchored hard bg)
    lambda_stain_flow: float = 0.0,
    stain_flow_h_thresh: float = 0.5,
    stain_flow_max_dist: int = 30,
    stain_flow_adaptive: bool = False,
    stain_flow_adapt_alpha: float = 0.6,
    stain_flow_r_min: int = 8,
    stain_flow_r_max: int = 30,
    lambda_eosin_neg: float = 0.5,
    eosin_neg_tau_e: float = 0.5,
    eosin_neg_tau_h: float = 0.5,
    eosin_neg_max_dist: int = 0,
    eosin_neg_min_dist: int = 0,
    # L_src: eosin-anchored source-ridge on flow divergence (same gate as E-).
    lambda_src: float = 0.0,
    src_tau_e: float = 0.5,
    src_tau_h: float = 0.5,
    src_max_dist: int = 30,
    src_margin: float = 0.3,
    # E4 Anchor-CDF: non-parametric per-image empirical Bayesian posterior
    # supervision on BoNuS ignore zone (clu==2). Replaces hand-crafted
    # threshold heuristics (L_stain_soft + L_eosin_neg) with adaptive form.
    lambda_anchor_cdf: float = 0.0,
    anchor_cdf_n_bins: int = 64,
    anchor_cdf_smooth_eps: float = 1.0,
    anchor_cdf_use_e: bool = True,
    anchor_cdf_max_dist: float = 30.0,
    anchor_cdf_target_clip: float = 0.05,
    # PSPS: BoNuS-free point-stain posterior supervision.
    lambda_point_stain: float = 0.0,
    point_stain_mode: str = "posterior",
    point_stain_adaptive_radius: bool = True,
    point_stain_bg_min_dist: float = 45.0,
    # E5 DCSS: Direction-Corrected Stain Supervision (boundary-fg + gap-bg)
    lambda_he_boundary_fg: float = 0.0,
    he_boundary_tau_h: float = 0.5,
    he_boundary_tau_e: float = 0.5,
    lambda_he_gap_bg: float = 0.0,
    he_gap_tau_h: float = 0.5,
    he_gap_tau_e: float = 0.15,
    he_quad_max_dist: float = 30.0,
    # C2: self-confidence gate. Pixels with sigmoid(fg_logits) >= gate are
    # excluded from L_eosin_neg, cutting the ~24.6% leak onto real fg/boundary.
    # 0.0 = disabled.
    eosin_neg_fg_gate: float = 0.0,
    # C7: adaptive proximity gate (per-image radius from NN distance median)
    adaptive_proximity: bool = False,
    adapt_alpha: float = 0.6,
    adapt_r_min: int = 15,
    adapt_r_max: int = 50,
    lambda_scmp: float = 0.0,
    scmp_sharpness: float = 8.0,
    scmp_max_dist: int = 0,
    scmp_confidence_floor: float = 1.0,
    lambda_stain_recon: float = 0.0,
    # D8 evidential
    use_cpgs: bool = False,
    cpgs_k: float = 4.0,
    cpgs_alpha: float = 1.0,
    lambda_evidential_mse: float = 0.0,
    lambda_evidential_kl: float = 0.0,
    uncertainty_gate: float = 0.0,
    uncertainty_gate_pow: float = 1.0,
    # B5 HSDR (stain-weighted hv-map)
    lambda_hv: float = 0.0,
    hv_stain_alpha: float = 1.0,
    # B3 instance mask decoder
    lambda_mask: float = 0.0,
    lambda_mask_iou: float = 0.0,
    # HU-BIM (boundary inference)
    lambda_bim: float = 0.0,
    bim_tau: float = 0.3,
    bim_r_max: float = 15.0,
    # HU-CAF (cross-branch aux fg distill)
    lambda_hucaf: float = 0.0,
    # RCF-Head (Ridge-Conditioned Flow Head)
    lambda_ridge: float = 0.0,
    ridge_sigma_b: float = 5.0,
    ridge_r_core: float = 0.0,
    # SCS: Stain-Confidence Sampling on clu==1.
    # Per-pixel chromophore-polarization weight on L_fg / L_flow within the
    # cluster disk. Weight w(x) = clamp(H_norm - E_norm + 0.5, scs_min_weight, 1).
    # Pure-H pixels (DNA-rich nucleus interior) get weight=1, E-leaning pixels
    # (suspect eosinophilic stroma falling inside cluster disk) get min_weight.
    # Only changes the *weight*, not the target — orthogonal to SCMP/CPGS failures.
    use_scs: bool = False,
    scs_min_weight: float = 0.2,
    # E1 PCGS — Probabilistic Chromophore-Guided Supervision
    use_pcgs: bool = False,
    pcgs_stats: Optional[dict] = None,
    pcgs_lambda: float = 1.0,
    pcgs_max_dist: float = 30.0,
    pcgs_warmup_active: bool = False,
    pcgs_return_em: bool = False,
    pcgs_mode: str = "anchor",                  # "anchor" (default, stateless) or "em" (legacy)
    pcgs_bg_anchor_min_dist: float = 40.0,
    pcgs_prior_fg: float = 0.5,
    pcgs_additive: bool = False,                # True: keep L_stain_soft+L_eosin_neg active, PCGS is *additional* per-image calibration
    pcgs_target_temperature: float = 1.0,       # E1-T route C.1: T>1 sharpens posterior toward binary for cross-domain robustness
    # SPP (Stain Polarization Posterior) — replaces L_stain_soft + L_eosin_neg
    use_spp: bool = False,
    spp_state = None,
    lambda_spp: float = 0.0,
    spp_gamma: float = 1.0,
    # GAP (Geometric Anchor-Proximity Posterior)
    use_gap: bool = False,
    lambda_gap: float = 0.0,
    gap_R: float = 7.5,
    gap_tau: float = 3.5,
    gap_mode: str = "soft",
    gap_d_fg_max: float = 4.0,
    gap_d_bg_min: float = 12.0,
    gap_target_fg: float = 0.7,
    gap_target_bg: float = 0.0,
    # MSST (Multi-Signal Sparse-Hard Trigger): joint (d, smoothed-H) supervision
    use_msst: bool = False,
    lambda_msst: float = 0.0,
    msst_d_fg_max: float = 2.0,
    msst_d_bg_min: float = 10.0,
    msst_h_kernel: int = 5,
    msst_h_fg_min: float = 0.5,
    msst_h_bg_max: float = 0.3,
    msst_target_fg: float = 0.7,
    msst_target_bg: float = 0.0,
    # NICHE (Stratified Chromophore-Neutral Geometric Supervision)
    use_niche: bool = False,
    lambda_niche: float = 0.0,
    niche_rho_max: float = 0.2,
    niche_d_fg_max: float = 2.0,
    niche_d_bg_min: float = 12.0,
    niche_target_fg: float = 0.7,
    niche_target_bg: float = 0.0,
    # IDG (Inter-Disk Gap): touching-boundary bg supervision
    use_idg: bool = False,
    lambda_idg: float = 0.0,
    idg_sigma: float = 3.0,
    idg_h_alpha: float = 2.0,
    idg_trigger_q: float = 0.95,
    idg_target: float = 0.0,
    # Boundary head
    use_boundary: bool = False,
    lambda_boundary: float = 0.0,
    boundary_sigma: float = 3.0,
    use_boundary_h: bool = False,
    boundary_h_sigma: float = 1.5,
    boundary_h_floor: float = 0.0,
    use_csbb: bool = False,
    csbb_rb: float = 18.0,
    csbb_h_floor: float = 0.4,
    use_gcsbb: bool = False,
    gcsbb_fg_tau: float = 0.3,
    gcsbb_crowd_r: float = 18.0,
    # CGFS — Chromophore-Guided Flow Supervision
    use_dhs: bool = False,
    lambda_dhs: float = 0.0,
    dhs_sigma: float = 1.5,
    dhs_target_low: float = 0.0,
    dhs_target_high: float = 1.0,
    use_hgfa: bool = False,
    lambda_hgfa: float = 0.0,
    hgfa_sigma: float = 1.5,
    hgfa_d_max: float = 5.0,
    hgfa_grad_floor: float = 0.0,
    hgfa_weight_pow: float = 1.0,
    # CPR — Cross-Pixel Polarity Ranking (Module A, DSC paper).
    use_cpr: bool = False,
    lambda_cpr: float = 0.0,
    cpr_n_pairs: int = 64,
    cpr_rho_diff_min: float = 0.3,
    cpr_margin_init: float = 0.05,
    cpr_margin_final: float = 0.25,
    cpr_margin_warmup_epochs: int = 40,
    current_epoch: int = 0,
    # PEF — Polarity-Embedded Feature head (Module B, DSC paper).
    use_pef: bool = False,
    lambda_pef: float = 0.0,
) -> dict[str, torch.Tensor]:
    fg_logits = outputs["fg_logits"]
    flow_pred = outputs["flow"]
    clu = batch["clu"].long()

    target_fg = (clu == 1).to(dtype=fg_logits.dtype).unsqueeze(1)
    fg_mask = (clu != 2).to(dtype=fg_logits.dtype).unsqueeze(1)
    # B15: expand cluster disk via stain prior.
    # Pixels in clu==2 with high H-stain near GT centers → promote to HARD fg.
    # This effectively grows the fg supervision radius from 10px to ~15-18px,
    # directly fixing under-segmentation while preserving voronoi-line bg constraint.
    if expand_cluster_via_stain and "stain_h" in batch and "point_map" in batch:
        import torch.nn.functional as F
        sh = batch["stain_h"].to(dtype=fg_logits.dtype)
        if sh.ndim == 3:
            sh = sh.unsqueeze(1)
        s_min = sh.flatten(2).amin(dim=2).view(-1, 1, 1, 1)
        s_max = sh.flatten(2).amax(dim=2).view(-1, 1, 1, 1)
        s_norm = (sh - s_min) / (s_max - s_min + 1e-6)
        pm = batch["point_map"].float()
        if pm.ndim == 3:
            pm = pm.unsqueeze(1)
        kernel = 2 * int(expand_cluster_max_dist) + 1
        proximity = F.max_pool2d(pm, kernel_size=kernel, stride=1,
                                   padding=int(expand_cluster_max_dist)) > 0
        clu2 = (clu == 2).unsqueeze(1)
        promote = clu2 & (s_norm > expand_cluster_h_thresh) & proximity
        promote_f = promote.to(dtype=fg_logits.dtype)
        # Add promoted pixels as fg target & supervised (no longer ignored).
        target_fg = (target_fg + promote_f).clamp(0.0, 1.0)
        fg_mask = (fg_mask + promote_f).clamp(0.0, 1.0)

    # SCS: per-pixel polarization weight on clu==1 (cluster disk).
    # Keeps clu==0 (forced bg) weight=1, but down-weights cluster-disk pixels
    # whose chromophore signature is suspicious (E > H).
    scs_w = None
    if use_scs and "stain_h" in batch and "stain_e" in batch:
        import torch.nn.functional as F
        sh = batch["stain_h"].to(dtype=fg_logits.dtype)
        se = batch["stain_e"].to(dtype=fg_logits.dtype)
        if sh.ndim == 3:
            sh = sh.unsqueeze(1)
        if se.ndim == 3:
            se = se.unsqueeze(1)
        h_min = sh.flatten(2).amin(dim=2).view(-1, 1, 1, 1)
        h_max = sh.flatten(2).amax(dim=2).view(-1, 1, 1, 1)
        h_norm = (sh - h_min) / (h_max - h_min + 1e-6)
        e_min = se.flatten(2).amin(dim=2).view(-1, 1, 1, 1)
        e_max = se.flatten(2).amax(dim=2).view(-1, 1, 1, 1)
        e_norm = (se - e_min) / (e_max - e_min + 1e-6)
        # w(x) = clamp(H - E + 0.5, min_weight, 1.0). Pure H → 1.5 → clamp to 1.0;
        # pure E → -0.5 → clamp to min_weight; balanced → 0.5 → mid.
        scs_w = torch.clamp(h_norm - e_norm + 0.5, scs_min_weight, 1.0)

    flow_target = batch["offset"].to(dtype=flow_pred.dtype)
    basin_valid = batch["basin_valid"].to(dtype=flow_pred.dtype).unsqueeze(1)
    flow_mask = (clu == 1).to(dtype=flow_pred.dtype).unsqueeze(1)

    if scs_w is not None:
        # L_fg: apply SCS weight only on clu==1 pixels; clu==0 forced-bg keeps w=1.
        clu1_f = (clu == 1).to(dtype=fg_logits.dtype).unsqueeze(1)
        fg_weight = fg_mask * (clu1_f * scs_w + (1.0 - clu1_f))
        loss_fg = masked_bce_with_logits(fg_logits, target_fg, fg_weight)
        # L_flow: flow_mask is already clu==1 only; multiply by SCS weight.
        flow_weight = flow_mask * scs_w
        loss_flow = masked_mse(flow_pred, flow_target, flow_weight)
    else:
        loss_fg = masked_bce_with_logits(fg_logits, target_fg, fg_mask)
        loss_flow = masked_mse(flow_pred, flow_target, flow_mask)

    loss_point = point_anchor_loss(fg_logits, batch["point_map"])

    total = lambda_fg * loss_fg + lambda_flow * loss_flow + lambda_point * loss_point

    loss_stain_flow = flow_pred.new_tensor(0.0)
    if lambda_stain_flow != 0.0 and "stain_h" in batch:
        import torch.nn.functional as _F
        _sh = batch["stain_h"].to(dtype=flow_pred.dtype)
        if _sh.ndim == 3:
            _sh = _sh.unsqueeze(1)
        _smin = _sh.flatten(2).amin(dim=2).view(-1, 1, 1, 1)
        _smax = _sh.flatten(2).amax(dim=2).view(-1, 1, 1, 1)
        _snorm = (_sh - _smin) / (_smax - _smin + 1e-6)
        _mask = (clu == 2).to(dtype=flow_pred.dtype).unsqueeze(1) \
            * (_snorm > stain_flow_h_thresh).to(dtype=flow_pred.dtype) * basin_valid
        if stain_flow_adaptive and "point_map" in batch:
            _prox = adaptive_proximity_mask(batch["point_map"], stain_flow_adapt_alpha,
                                            stain_flow_r_min, stain_flow_r_max).to(dtype=flow_pred.dtype)
            _mask = _mask * _prox
        elif stain_flow_max_dist > 0 and "point_map" in batch:
            _pm = batch["point_map"].to(dtype=flow_pred.dtype)
            if _pm.ndim == 3:
                _pm = _pm.unsqueeze(1)
            _k = int(2 * stain_flow_max_dist + 1)
            _prox = _F.max_pool2d((_pm > 0).to(dtype=flow_pred.dtype), kernel_size=_k,
                                  stride=1, padding=int(stain_flow_max_dist))
            _mask = _mask * _prox
        loss_stain_flow = masked_mse(flow_pred, flow_target, _mask)
        total = total + lambda_stain_flow * loss_stain_flow

    loss_div = fg_logits.new_tensor(0.0)
    if lambda_div != 0.0:
        conductance = None
        if div_conductance == "stain":
            if "stain_h" not in batch:
                raise KeyError("div_conductance='stain' requires batch['stain_h']")
            stain_h = batch["stain_h"].to(dtype=flow_pred.dtype).unsqueeze(1)
            conductance = stain_alpha + (1.0 - stain_alpha) * stain_h
        elif div_conductance == "eosin":
            # Heterogeneous-medium Poisson: the eosin-rich Voronoi gap (cytoplasm/
            # stroma between touching nuclei) acts as a low-conductance barrier so
            # the potential flow is cut there. divergence_sink_loss applies
            # flow <- c * flow, hence residual = div(c*flow)+rho_sink solves
            # -div(c grad phi) = rho_sink, with c = alpha + (1-alpha)*(1 - E_norm).
            # Moves the eosin signal from a separate L_eosin_neg anchor into the
            # PDE medium itself (stain-as-conductance unification).
            if "stain_e" not in batch:
                raise KeyError("div_conductance='eosin' requires batch['stain_e']")
            stain_e = batch["stain_e"].to(dtype=flow_pred.dtype).unsqueeze(1)
            e_norm = _norm_per_image_01(stain_e)
            conductance = stain_alpha + (1.0 - stain_alpha) * (1.0 - e_norm)
        elif div_conductance != "constant":
            raise ValueError(f"Unsupported div_conductance: {div_conductance}")
        loss_div = divergence_sink_loss(
            flow_pred,
            batch["point_map"],
            (clu == 1),
            conductance=conductance,
            sink_sigma=sink_sigma,
            sink_adaptive=sink_adaptive,
            sink_alpha=sink_alpha,
            sink_min=sink_min,
            sink_max=sink_max,
            phi=outputs.get("phi"),
            screen_kappa=screen_kappa,
            source_density=batch.get("rho"),
        )
        total = total + lambda_div * loss_div

    loss_center = fg_logits.new_tensor(0.0)
    if lambda_center != 0.0 and "centerness_logits" in outputs:
        center_target = batch.get("center_target")
        if center_target is None:
            center_target = gaussian_point_density(batch["point_map"], center_sigma, mode="max")
        center_target = center_target.to(dtype=fg_logits.dtype)
        loss_center = focal_mse_loss(outputs["centerness_logits"], center_target, gamma=2.0)
        total = total + lambda_center * loss_center

    loss_aff = fg_logits.new_tensor(0.0)
    if lambda_aff != 0.0 and "boundary_logits" in outputs and "instance_id" in batch:
        boundary_logits = outputs["boundary_logits"]
        instance_id = batch["instance_id"]
        if instance_id.ndim == 4:
            instance_id = instance_id.squeeze(1)
        H, W = boundary_logits.shape[-2:]
        pi, path_buffers, src_buf, dst_buf = _get_path_index(aff_radius, H, boundary_logits.device)
        pos_l, neg_l = affinity_loss(boundary_logits, instance_id, pi, path_buffers, src_buf, dst_buf)
        loss_aff = pos_l + neg_l
        total = total + lambda_aff * loss_aff

    loss_psic = fg_logits.new_tensor(0.0)
    if lambda_psic != 0.0 and "contrast_z" in outputs:
        loss_psic = psic_loss(
            outputs["contrast_z"],
            batch["point_map"],
            clu,
            r_pos=psic_r_pos, beta=psic_beta, r_neg=psic_r_neg,
            k_min=psic_k_min, k_anchor=psic_k_anchor, tau=psic_tau,
            adaptive_radius=psic_adaptive_radius,
            r_pos_max=psic_r_pos_max,
            alpha_dnn=psic_alpha_dnn,
            top_k_neg=psic_top_k_neg,
            use_margin=psic_use_margin,
            margin_value=psic_margin_value,
            neg_weight=psic_neg_weight,
        )
        total = total + lambda_psic * loss_psic

    loss_sam_fg = fg_logits.new_tensor(0.0)
    if lambda_sam_fg != 0.0 and "sam_fg_mask" in batch:
        sam_fg_mask = batch["sam_fg_mask"].to(dtype=fg_logits.dtype).unsqueeze(1)
        loss_sam_fg = positive_only_bce(fg_logits, sam_fg_mask)
        total = total + lambda_sam_fg * loss_sam_fg

    loss_sam_flow = fg_logits.new_tensor(0.0)
    if lambda_sam_flow != 0.0 and "sam_flow" in batch and "sam_flow_mask" in batch:
        sam_flow = batch["sam_flow"].to(dtype=flow_pred.dtype)
        sam_flow_mask = batch["sam_flow_mask"].to(dtype=flow_pred.dtype).unsqueeze(1)
        loss_sam_flow = masked_mse(flow_pred, sam_flow, sam_flow_mask)
        total = total + lambda_sam_flow * loss_sam_flow

    # HU-BIM auxiliary boundary loss (point-derivable Voronoi ridge target @ 1/8)
    loss_bim = fg_logits.new_tensor(0.0)
    if lambda_bim != 0.0 and "aux_boundary_logits" in outputs:
        b_logits = outputs["aux_boundary_logits"]
        stride = max(1, fg_logits.shape[-1] // b_logits.shape[-1])
        target = voronoi_ridge_target(batch["point_map"], stride=stride,
                                         tau=bim_tau, r_max=bim_r_max
                                         ).to(dtype=b_logits.dtype)
        # fg_mask at 1/stride resolution (use clu==1 downsampled)
        fg_full = (clu == 1).to(dtype=b_logits.dtype).unsqueeze(1)
        fg_ds = torch.nn.functional.max_pool2d(fg_full, kernel_size=stride,
                                                  stride=stride)
        loss_bim = boundary_inference_loss(b_logits, target, fg_ds)
        total = total + lambda_bim * loss_bim

    # HU-CAF cross-branch aux fg distillation
    loss_hucaf = fg_logits.new_tensor(0.0)
    if lambda_hucaf != 0.0 and "aux_fg_logits_uni" in outputs:
        loss_hucaf = hucaf_distill_loss(
            outputs["aux_fg_logits_uni"], fg_logits,
            fg_mask=(clu == 1).to(dtype=fg_logits.dtype).unsqueeze(1),
        )
        total = total + lambda_hucaf * loss_hucaf

    # RCF-Head soft ridge SmoothL1 loss
    loss_ridge = fg_logits.new_tensor(0.0)
    if lambda_ridge != 0.0 and "aux_ridge_logits" in outputs:
        r_logits = outputs["aux_ridge_logits"]
        stride = max(1, fg_logits.shape[-1] // r_logits.shape[-1])
        target_r = voronoi_ridge_soft_target(batch["point_map"], stride=stride,
                                                sigma_b=ridge_sigma_b,
                                                r_core=ridge_r_core
                                                ).to(dtype=r_logits.dtype)
        fg_full = (clu == 1).to(dtype=r_logits.dtype).unsqueeze(1)
        fg_ds = torch.nn.functional.max_pool2d(fg_full, kernel_size=stride,
                                                  stride=stride)
        loss_ridge = ridge_smoothl1_loss(r_logits, target_r, fg_ds)
        total = total + lambda_ridge * loss_ridge

    # B5 HSDR: stain-weighted hv-map regression
    loss_hv = fg_logits.new_tensor(0.0)
    if lambda_hv != 0.0 and "hv_map" in outputs and "instance_id" in batch:
        loss_hv = compute_hv_loss(outputs, batch, stain_alpha=hv_stain_alpha)
        total = total + lambda_hv * loss_hv

    # B3 instance mask decoder loss
    loss_mask_bce = fg_logits.new_tensor(0.0)
    loss_mask_dice = fg_logits.new_tensor(0.0)
    loss_mask_iou = fg_logits.new_tensor(0.0)
    if lambda_mask != 0.0 and "mask_logits" in outputs and "mask_targets" in outputs:
        loss_mask_bce, loss_mask_dice = compute_instance_mask_loss(outputs)
        total = total + lambda_mask * (loss_mask_bce + loss_mask_dice)
        if lambda_mask_iou != 0.0 and "mask_iou_pred" in outputs:
            loss_mask_iou = compute_mask_iou_loss(outputs)
            total = total + lambda_mask_iou * loss_mask_iou

    # B6 HGBL: hematoxylin-gradient boundary loss
    loss_h_boundary = fg_logits.new_tensor(0.0)
    if lambda_h_boundary != 0.0 and "boundary_logits" in outputs and "stain_h" in batch:
        loss_h_boundary = compute_h_gradient_boundary_loss(outputs, batch)
        total = total + lambda_h_boundary * loss_h_boundary

    # B7 SCR-interior: stain-weighted Dice on fg head
    loss_stain_dice = fg_logits.new_tensor(0.0)
    if lambda_stain_dice != 0.0 and "stain_h" in batch:
        loss_stain_dice = compute_stain_weighted_dice(
            fg_logits, target_fg, fg_mask, batch["stain_h"], alpha=stain_dice_alpha,
        )
        total = total + lambda_stain_dice * loss_stain_dice

    # E1 PCGS gate: when PCGS is active (post-warmup) AND not in additive mode,
    # force the hard-threshold voronoi-gap supervisors to zero so PCGS replaces them.
    # In additive mode, keep them ON — PCGS is *added* as per-image probabilistic
    # calibration on top of BoNuS's structural supervision.
    if use_pcgs and not pcgs_warmup_active and not pcgs_additive:
        if lambda_stain_soft != 0.0 or lambda_eosin_neg != 0.0:
            # caller did not zero them — silently override to avoid double-count
            lambda_stain_soft = 0.0
            lambda_eosin_neg = 0.0

    # B9 SCR-soft: soft fg supervision in BoNuS ignore zone w/ high H stain
    loss_stain_soft = fg_logits.new_tensor(0.0)
    if lambda_stain_soft != 0.0 and "stain_h" in batch:
        loss_stain_soft = compute_stain_anchored_soft_label_loss(
            fg_logits, clu, batch["stain_h"],
            alpha=stain_soft_alpha, h_threshold=stain_soft_h_thresh,
            point_map=batch.get("point_map", None) if stain_soft_max_dist > 0 else None,
            max_dist=stain_soft_max_dist,
            target_mode=stain_soft_target_mode,
            proximity_kernel=stain_soft_proximity_kernel,
            trigger_opening=stain_soft_trigger_opening,
            adaptive_radius=adaptive_proximity,
            adapt_alpha=adapt_alpha, adapt_r_min=adapt_r_min, adapt_r_max=adapt_r_max,
            uncertainty=(outputs.get("uncertainty") if uncertainty_gate > 0 else None),
            uncertainty_pow=uncertainty_gate_pow,
            use_cpgs=use_cpgs, cpgs_k=cpgs_k, cpgs_alpha=cpgs_alpha,
            stain_e=batch.get("stain_e"),
        )
        total = total + lambda_stain_soft * loss_stain_soft

    # B12 MPS-negative: eosin-anchored hard bg
    loss_eosin_neg = fg_logits.new_tensor(0.0)
    if lambda_eosin_neg != 0.0 and "stain_e" in batch and "stain_h" in batch:
        loss_eosin_neg = compute_eosin_anchored_neg_loss(
            fg_logits, clu, batch["stain_e"], batch["stain_h"],
            tau_e=eosin_neg_tau_e, tau_h=eosin_neg_tau_h,
            point_map=batch.get("point_map", None) if eosin_neg_max_dist > 0 else None,
            max_dist=eosin_neg_max_dist, min_dist=eosin_neg_min_dist,
            fg_confidence_gate=eosin_neg_fg_gate,
            adaptive_radius=adaptive_proximity,
            adapt_alpha=adapt_alpha, adapt_r_min=adapt_r_min, adapt_r_max=adapt_r_max,
            uncertainty=(outputs.get("uncertainty") if uncertainty_gate > 0 else None),
            uncertainty_pow=uncertainty_gate_pow,
            use_cpgs=use_cpgs, cpgs_k=cpgs_k, cpgs_alpha=cpgs_alpha,
        )
        total = total + lambda_eosin_neg * loss_eosin_neg

    # L_src: eosin-anchored source-ridge on flow divergence (shape field topology)
    loss_src = fg_logits.new_tensor(0.0)
    if lambda_src != 0.0 and "stain_e" in batch and "stain_h" in batch:
        loss_src = compute_eosin_source_div_loss(
            flow_pred, clu, batch["stain_e"], batch["stain_h"],
            tau_e=src_tau_e, tau_h=src_tau_h,
            point_map=batch.get("point_map", None) if src_max_dist > 0 else None,
            max_dist=src_max_dist, margin=src_margin,
        )
        total = total + lambda_src * loss_src

    # E4 Anchor-CDF: non-parametric per-image empirical Bayesian posterior.
    loss_anchor_cdf = fg_logits.new_tensor(0.0)
    if lambda_anchor_cdf != 0.0 and "stain_h" in batch:
        loss_anchor_cdf = compute_anchor_cdf_loss(
            fg_logits, clu, batch["stain_h"],
            stain_e=batch.get("stain_e"),
            point_map=batch.get("point_map") if anchor_cdf_max_dist > 0 else None,
            max_dist=anchor_cdf_max_dist,
            n_bins=anchor_cdf_n_bins,
            smooth_eps=anchor_cdf_smooth_eps,
            use_e=anchor_cdf_use_e,
            target_clip=anchor_cdf_target_clip,
        )
        total = total + lambda_anchor_cdf * loss_anchor_cdf

    # PSPS: Point-Stain Posterior Supervision. This loss is BoNuS-free:
    # it never reads `clu` and derives anchors from point geometry only.
    loss_point_stain = fg_logits.new_tensor(0.0)
    if lambda_point_stain != 0.0 and "stain_h" in batch and "stain_e" in batch and "point_map" in batch:
        loss_point_stain = compute_point_stain_posterior_loss(
            fg_logits,
            batch["stain_h"],
            batch["stain_e"],
            batch["point_map"],
            mode=point_stain_mode,
            adaptive_radius=point_stain_adaptive_radius,
            bg_min_dist=point_stain_bg_min_dist,
        )
        total = total + lambda_point_stain * loss_point_stain

    # E5 DCSS-boundary-fg: clu==2 high-H high-E -> target=1 (85% fg per oracle).
    loss_he_boundary_fg = fg_logits.new_tensor(0.0)
    if lambda_he_boundary_fg != 0.0 and "stain_h" in batch and "stain_e" in batch:
        loss_he_boundary_fg = compute_he_quadrant_loss(
            fg_logits, clu, batch["stain_h"], batch["stain_e"],
            h_lo=he_boundary_tau_h, h_hi=1.01,
            e_lo=he_boundary_tau_e, e_hi=1.01,
            target=1.0,
            point_map=batch.get("point_map") if he_quad_max_dist > 0 else None,
            max_dist=he_quad_max_dist,
        )
        total = total + lambda_he_boundary_fg * loss_he_boundary_fg

    # E5 DCSS-gap-bg: clu==2 low-H low-E -> target=0 (99% bg per oracle).
    loss_he_gap_bg = fg_logits.new_tensor(0.0)
    if lambda_he_gap_bg != 0.0 and "stain_h" in batch and "stain_e" in batch:
        loss_he_gap_bg = compute_he_quadrant_loss(
            fg_logits, clu, batch["stain_h"], batch["stain_e"],
            h_lo=0.0, h_hi=he_gap_tau_h,
            e_lo=0.0, e_hi=he_gap_tau_e,
            target=0.0,
            point_map=batch.get("point_map") if he_quad_max_dist > 0 else None,
            max_dist=he_quad_max_dist,
        )
        total = total + lambda_he_gap_bg * loss_he_gap_bg

    # E1 PCGS — Probabilistic Chromophore-Guided Supervision.
    # Two modes:
    #   anchor (default, stateless): per-batch Gaussian fits on clu==1 (fg anchor)
    #     and clu==0 ∩ dist>bg_anchor_min_dist (bg anchor); posterior applied to clu==2.
    #     No EM, no state, no warmup — robust against the drift seen in legacy EM mode.
    #   em (legacy): sigmoid-responsibility-driven EM over voronoi-gap pixels. Kept
    #     for ablation only; observed to drift in early E1 runs.
    loss_pcgs = fg_logits.new_tensor(0.0)
    em_inputs_out = None
    if (use_pcgs and not pcgs_warmup_active
            and "stain_h" in batch and "stain_e" in batch and "point_map" in batch):
        if pcgs_mode == "anchor":
            loss_pcgs = compute_pcgs_anchor_loss(
                fg_logits, clu,
                batch["stain_h"], batch["stain_e"], batch["point_map"],
                bg_anchor_min_dist=pcgs_bg_anchor_min_dist,
                max_dist=pcgs_max_dist,
                prior_fg=pcgs_prior_fg,
                target_temperature=pcgs_target_temperature,
            )
        elif pcgs_mode == "v2":
            # PCGS-v2: bivariate Gaussian + spatial-weighted BCE (replaces hard dist cutoff)
            loss_pcgs = compute_pcgs_v2_loss(
                fg_logits, clu,
                batch["stain_h"], batch["stain_e"], batch["point_map"],
                bg_anchor_min_dist=pcgs_bg_anchor_min_dist,
                spatial_sigma=pcgs_max_dist,   # reuse max_dist as σ_d
                prior_fg=pcgs_prior_fg,
                target_temperature=pcgs_target_temperature,
            )
        else:  # "em" legacy
            if pcgs_stats is None:
                raise ValueError("pcgs_mode='em' requires pcgs_stats dict from train loop")
            ret = compute_pcgs_loss(
                fg_logits, clu,
                batch["stain_h"], batch["stain_e"], batch["point_map"],
                pcgs_stats=pcgs_stats,
                max_dist=pcgs_max_dist,
                return_em_inputs=pcgs_return_em,
            )
            if pcgs_return_em:
                loss_pcgs, em_inputs_out = ret
            else:
                loss_pcgs = ret
        total = total + pcgs_lambda * loss_pcgs

    # B17 SCMP — unified Stain-Confidence Modality Prior (replaces stain_soft+eosin_neg)
    loss_scmp = fg_logits.new_tensor(0.0)
    if lambda_scmp != 0.0 and "stain_e" in batch and "stain_h" in batch:
        loss_scmp = compute_stain_confidence_loss(
            fg_logits, clu, batch["stain_h"], batch["stain_e"],
            sharpness=scmp_sharpness,
            point_map=batch.get("point_map", None) if scmp_max_dist > 0 else None,
            max_dist=scmp_max_dist,
            confidence_floor=scmp_confidence_floor,
        )
        total = total + lambda_scmp * loss_scmp

    loss_stain_recon = fg_logits.new_tensor(0.0)
    if lambda_stain_recon != 0.0 and "h_recon" in outputs and "h_target" in outputs:
        import torch.nn.functional as F_local
        loss_stain_recon = F_local.l1_loss(outputs["h_recon"], outputs["h_target"])
        total = total + lambda_stain_recon * loss_stain_recon

    # D8 evidential MSE + KL on fg head (only if encoder provides alpha).
    loss_evidential_mse = fg_logits.new_tensor(0.0)
    loss_evidential_kl = fg_logits.new_tensor(0.0)
    if lambda_evidential_mse != 0.0 and "alpha" in outputs:
        alpha2 = outputs["alpha"]  # (B, 2, H, W)
        # target_onehot for clu==1 (fg) and clu==0 (bg). clu==2 ignored.
        fg_mask = (clu == 1).unsqueeze(1).float()
        bg_mask = (clu == 0).unsqueeze(1).float()
        valid = (fg_mask + bg_mask).clamp(0, 1)
        target = torch.cat([fg_mask, bg_mask], dim=1)  # (B, 2, H, W)
        emse = evidential_mse(alpha2, target)  # (B, 1, H, W)
        denom = valid.sum().clamp(min=1.0)
        loss_evidential_mse = (emse * valid).sum() / denom
        total = total + lambda_evidential_mse * loss_evidential_mse
        if lambda_evidential_kl != 0.0:
            ekl = evidential_kl(alpha2, target)
            loss_evidential_kl = (ekl * valid).sum() / denom
            total = total + lambda_evidential_kl * loss_evidential_kl

    # ----- SPP: Stain Polarization Posterior -----
    loss_spp = fg_logits.new_tensor(0.0)
    if use_spp and lambda_spp > 0.0 and spp_state is not None:
        if "stain_h" in batch and "stain_e" in batch and "clu" in batch:
            from chromaflow.spp_loss import compute_spp_loss, compute_polarization
            loss_spp = compute_spp_loss(
                fg_logits, batch["clu"], batch["stain_h"], batch["stain_e"],
                k=float(spp_state.k), b=float(spp_state.b), gamma=float(spp_gamma),
            )
            total = total + lambda_spp * loss_spp
            # Accumulate (rho, y) for next-epoch (k, b) refit. Only when buffer
            # has been opened by spp_state.start_epoch().
            if getattr(spp_state, "_rho_buffer", None) is not None:
                with torch.no_grad():
                    rho_acc = compute_polarization(batch["stain_h"], batch["stain_e"])
                    spp_state.accumulate(rho_acc, batch["clu"], max_pixels_per_image=2000)

    # ----- GAP: Geometric Anchor-Proximity Posterior -----
    loss_gap = fg_logits.new_tensor(0.0)
    if use_gap and lambda_gap > 0.0:
        if "d_to_clu1" in batch and "clu" in batch:
            from chromaflow.gap_loss import compute_gap_loss
            loss_gap = compute_gap_loss(
                fg_logits, batch["clu"], batch["d_to_clu1"],
                R=float(gap_R), tau=float(gap_tau),
                mode=str(gap_mode),
                d_fg_max=float(gap_d_fg_max),
                d_bg_min=float(gap_d_bg_min),
                target_fg=float(gap_target_fg),
                target_bg=float(gap_target_bg),
            )
            total = total + lambda_gap * loss_gap

    # ----- MSST: Multi-Signal Sparse-Hard Trigger -----
    loss_msst = fg_logits.new_tensor(0.0)
    if use_msst and lambda_msst > 0.0:
        need = ("d_to_clu1" in batch and "clu" in batch and "stain_h" in batch)
        if need:
            from chromaflow.msst_loss import compute_msst_loss
            loss_msst = compute_msst_loss(
                fg_logits, batch["clu"], batch["d_to_clu1"], batch["stain_h"],
                d_fg_max=float(msst_d_fg_max),
                d_bg_min=float(msst_d_bg_min),
                h_kernel=int(msst_h_kernel),
                h_fg_min=float(msst_h_fg_min),
                h_bg_max=float(msst_h_bg_max),
                target_fg=float(msst_target_fg),
                target_bg=float(msst_target_bg),
            )
            total = total + lambda_msst * loss_msst

    # ----- NICHE: Stratified Chromophore-Neutral Geometric Supervision -----
    loss_niche = fg_logits.new_tensor(0.0)
    if use_niche and lambda_niche > 0.0:
        need = ("d_to_clu1" in batch and "clu" in batch 
                and "stain_h" in batch and "stain_e" in batch)
        if need:
            from chromaflow.niche_loss import compute_niche_loss
            loss_niche = compute_niche_loss(
                fg_logits, batch["clu"], batch["d_to_clu1"],
                batch["stain_h"], batch["stain_e"],
                rho_max=float(niche_rho_max),
                d_fg_max=float(niche_d_fg_max),
                d_bg_min=float(niche_d_bg_min),
                target_fg=float(niche_target_fg),
                target_bg=float(niche_target_bg),
            )
            total = total + lambda_niche * loss_niche

    # ----- IDG: Inter-Disk Gap touching-boundary supervision -----
    loss_idg = fg_logits.new_tensor(0.0)
    if use_idg and lambda_idg > 0.0:
        need = ("d_to_clu1" in batch and "d_to_clu1_2nd" in batch
                and "clu" in batch and "stain_h" in batch)
        if need:
            from chromaflow.idg_loss import compute_idg_loss
            loss_idg = compute_idg_loss(
                fg_logits, batch["clu"], batch["stain_h"],
                batch["d_to_clu1"], batch["d_to_clu1_2nd"],
                sigma=float(idg_sigma),
                h_alpha=float(idg_h_alpha),
                trigger_q=float(idg_trigger_q),
                target=float(idg_target),
            )
            total = total + lambda_idg * loss_idg

    # ----- Boundary head: geometric touching-boundary supervision -----
    loss_boundary = fg_logits.new_tensor(0.0)
    if use_boundary and lambda_boundary > 0.0:
        if ("boundary_logits" in outputs
                and "d_to_clu1" in batch and "d_to_clu1_2nd" in batch):
            if use_gcsbb and "stain_h" in batch:
                from chromaflow.csbb_loss import compute_gcsbb_loss
                loss_boundary = compute_gcsbb_loss(
                    outputs["boundary_logits"], fg_logits, batch["clu"],
                    batch["d_to_clu1"], batch["d_to_clu1_2nd"], batch["stain_h"],
                    sigma=float(boundary_sigma), rb=float(csbb_rb),
                    h_floor=float(csbb_h_floor),
                    fg_tau=float(gcsbb_fg_tau), crowd_r=float(gcsbb_crowd_r),
                )
            elif use_csbb and "stain_h" in batch:
                from chromaflow.csbb_loss import compute_csbb_loss
                loss_boundary = compute_csbb_loss(
                    outputs["boundary_logits"], batch["clu"],
                    batch["d_to_clu1"], batch["d_to_clu1_2nd"], batch["stain_h"],
                    sigma=float(boundary_sigma), rb=float(csbb_rb),
                    h_floor=float(csbb_h_floor),
                )
            elif use_boundary_h and "stain_h" in batch:
                from chromaflow.boundary_h_loss import compute_boundary_h_loss
                loss_boundary = compute_boundary_h_loss(
                    outputs["boundary_logits"], batch["clu"],
                    batch["d_to_clu1"], batch["d_to_clu1_2nd"],
                    batch["stain_h"],
                    sigma=float(boundary_sigma),
                    h_sigma=float(boundary_h_sigma),
                    h_floor=float(boundary_h_floor),
                )
            else:
                from chromaflow.boundary_loss import compute_boundary_loss
                loss_boundary = compute_boundary_loss(
                    outputs["boundary_logits"], batch["clu"],
                    batch["d_to_clu1"], batch["d_to_clu1_2nd"],
                    sigma=float(boundary_sigma),
                )
            total = total + lambda_boundary * loss_boundary

    # ----- CGFS: DHS (dense H̃ supervision on fg head) -----
    loss_dhs = fg_logits.new_tensor(0.0)
    if use_dhs and lambda_dhs > 0.0:
        if ("clu" in batch and "stain_h" in batch):
            from chromaflow.cgfs_loss import compute_dhs_loss
            loss_dhs = compute_dhs_loss(
                fg_logits, batch["clu"], batch["stain_h"],
                sigma=float(dhs_sigma),
                target_low=float(dhs_target_low),
                target_high=float(dhs_target_high),
            )
            total = total + lambda_dhs * loss_dhs

    # ----- CGFS: HGFA (H-gradient flow direction supervision on flow head) -----
    loss_hgfa = fg_logits.new_tensor(0.0)
    if use_hgfa and lambda_hgfa > 0.0:
        if ("clu" in batch and "d_to_clu1" in batch
                and "stain_h" in batch):
            from chromaflow.cgfs_loss import compute_hgfa_loss
            loss_hgfa = compute_hgfa_loss(
                flow_pred, batch["clu"], batch["d_to_clu1"], batch["stain_h"],
                d_max=float(hgfa_d_max),
                sigma=float(hgfa_sigma),
                grad_mag_floor=float(hgfa_grad_floor),
                weight_pow=float(hgfa_weight_pow),
            )
            total = total + lambda_hgfa * loss_hgfa

    # ----- CPR: Cross-Pixel Polarity Ranking (Module A) -----
    loss_cpr = fg_logits.new_tensor(0.0)
    cpr_pair_acc = fg_logits.new_tensor(0.0)
    if use_cpr and lambda_cpr > 0.0:
        if ("clu" in batch and "stain_h" in batch and "stain_e" in batch):
            from chromaflow.cpr_loss import compute_cpr_loss
            _ret = compute_cpr_loss(
                fg_logits, batch["clu"], batch["stain_h"], batch["stain_e"],
                epoch=int(current_epoch),
                n_pairs=int(cpr_n_pairs),
                rho_diff_min=float(cpr_rho_diff_min),
                margin_init=float(cpr_margin_init),
                margin_final=float(cpr_margin_final),
                margin_warmup_epochs=int(cpr_margin_warmup_epochs),
                return_aux=True,
            )
            loss_cpr, _cpr_aux = _ret
            cpr_pair_acc = fg_logits.new_tensor(float(_cpr_aux.get("cpr_pair_acc", 0.0)))
            total = total + lambda_cpr * loss_cpr

    # ----- PEF: Polarity-Embedded Feature regularizer (Module B) -----
    loss_pef = fg_logits.new_tensor(0.0)
    if use_pef and lambda_pef > 0.0:
        if ("polarity_pred" in outputs and "stain_h" in batch 
                and "stain_e" in batch):
            pred = outputs["polarity_pred"]
            h = batch["stain_h"]
            if h.ndim == 3: h = h.unsqueeze(1)
            e = batch["stain_e"]
            if e.ndim == 3: e = e.unsqueeze(1)
            # per-image normalize H, E independently before computing rho
            h_lo = h.flatten(2).amin(dim=2).view(-1,1,1,1)
            h_hi = h.flatten(2).amax(dim=2).view(-1,1,1,1)
            h_n = (h - h_lo) / (h_hi - h_lo + 1e-6)
            e_lo = e.flatten(2).amin(dim=2).view(-1,1,1,1)
            e_hi = e.flatten(2).amax(dim=2).view(-1,1,1,1)
            e_n = (e - e_lo) / (e_hi - e_lo + 1e-6)
            rho_target = (h_n - e_n) / (h_n + e_n + 1e-6)  # in [-1, 1]
            loss_pef = torch.nn.functional.smooth_l1_loss(pred.to(rho_target.dtype),
                                          rho_target, beta=0.1)
            total = total + lambda_pef * loss_pef

    return {
        "loss": total,
        "loss_fg": loss_fg.detach(),
        "loss_flow": loss_flow.detach(),
        "loss_stain_flow": loss_stain_flow.detach(),
        "loss_point": loss_point.detach(),
        "loss_div": loss_div.detach(),
        "loss_center": loss_center.detach(),
        "loss_aff": loss_aff.detach(),
        "loss_psic": loss_psic.detach(),
        "loss_sam_fg": loss_sam_fg.detach(),
        "loss_sam_flow": loss_sam_flow.detach(),
        "loss_bim": loss_bim.detach(),
        "loss_hucaf": loss_hucaf.detach(),
        "loss_ridge": loss_ridge.detach(),
        "loss_mask_bce": loss_mask_bce.detach(),
        "loss_mask_dice": loss_mask_dice.detach(),
        "loss_mask_iou": loss_mask_iou.detach(),
        "loss_hv": loss_hv.detach(),
        "loss_h_boundary": loss_h_boundary.detach(),
        "loss_stain_dice": loss_stain_dice.detach(),
        "loss_stain_soft": loss_stain_soft.detach(),
        "loss_eosin_neg": loss_eosin_neg.detach(),
        "loss_src": loss_src.detach(),
        "loss_anchor_cdf": loss_anchor_cdf.detach(),
        "loss_point_stain": loss_point_stain.detach(),
        "loss_he_boundary_fg": loss_he_boundary_fg.detach(),
        "loss_he_gap_bg": loss_he_gap_bg.detach(),
        "loss_scmp": loss_scmp.detach(),
        "loss_stain_recon": loss_stain_recon.detach(),
        "loss_evidential_mse": loss_evidential_mse.detach(),
        "loss_evidential_kl": loss_evidential_kl.detach(),
        "loss_pcgs": loss_pcgs.detach(),
        "loss_spp": loss_spp.detach(),
        "loss_gap": loss_gap.detach(),
        "loss_msst": loss_msst.detach(),
        "loss_niche": loss_niche.detach(),
        "loss_idg": loss_idg.detach(),
        "loss_boundary": loss_boundary.detach(),
        "loss_dhs": loss_dhs.detach(),
        "loss_hgfa": loss_hgfa.detach(),
        "loss_cpr": loss_cpr.detach(),
        "loss_pef": loss_pef.detach(),
        "cpr_pair_acc": cpr_pair_acc,
        "_pcgs_em_inputs": em_inputs_out,  # None unless pcgs_return_em was True
    }
