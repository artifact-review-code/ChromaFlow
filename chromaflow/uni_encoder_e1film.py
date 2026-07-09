"""E1-FiLM encoder: NISFlowNetUNIHybridHLoRA + PCGS-FiLM modulation.

PCGS already fits per-image stain statistics from BoNuS oracle anchor regions
(clu==1 = fg, clu==0 ∩ dist>40 = bg). These 8 stats (μ_H_fg, σ_H_fg,
μ_E_fg, σ_E_fg, μ_H_bg, σ_H_bg, μ_E_bg, σ_E_bg) are normally only used as the
BCE target via the Bayesian posterior. This module exploits them at a second
location: a Feature-wise Linear Modulation (FiLM) layer on the decoder's final
64-ch feature.

Two-location consumption of the same per-image stats =
"PCGS framework with both supervision and architectural sides."

Training: stats from oracle clu==1 / clu==0∩dist>40 anchors.
Inference: stats from a fixed default vector (training-set typical values),
avoiding the anchor-unreliability issue we observed in test-time refine.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from chromaflow.uni_encoder_b1 import NISFlowNetUNIHybridHLoRA


def _per_image_norm01(x: torch.Tensor) -> torch.Tensor:
    """Per-image min-max normalize a (B,1,H,W) or (B,H,W) tensor to [0, 1]."""
    if x.ndim == 3:
        x = x.unsqueeze(1)
    lo = x.flatten(2).amin(dim=2).view(-1, 1, 1, 1)
    hi = x.flatten(2).amax(dim=2).view(-1, 1, 1, 1)
    return (x - lo) / (hi - lo + 1e-6)


def _distance_to_points(point_map: torch.Tensor, max_dist: float = 60.0) -> torch.Tensor:
    """CPU-side scipy distance transform; (B,1,H,W) or (B,H,W) -> (B,1,H,W)."""
    from scipy.ndimage import distance_transform_edt
    if point_map.ndim == 3:
        point_map = point_map.unsqueeze(1)
    device = point_map.device
    pm_np = (point_map.detach().cpu().numpy() > 0).astype("uint8")
    B, _, H, W = pm_np.shape
    out = torch.empty((B, 1, H, W), dtype=torch.float32)
    for b in range(B):
        if pm_np[b, 0].sum() == 0:
            out[b, 0].fill_(float(max_dist))
        else:
            d = distance_transform_edt(1 - pm_np[b, 0]).astype("float32")
            out[b, 0] = torch.from_numpy(d)
    return out.clamp_(0.0, float(max_dist)).to(device)


def _fit_per_image_gauss(values: torch.Tensor, mask: torch.Tensor,
                          default_mu: float, default_sigma: float,
                          sigma_min: float = 0.05, sigma_max: float = 0.40):
    """Per-image (μ, σ) over mask. Returns (μ, σ) tensors of shape (B, 1)."""
    m = mask.to(dtype=values.dtype)
    denom = m.sum(dim=(2, 3), keepdim=True).clamp(min=1.0)
    mu = (values * m).sum(dim=(2, 3), keepdim=True) / denom
    var = ((values - mu) ** 2 * m).sum(dim=(2, 3), keepdim=True) / denom
    sigma = var.clamp(min=sigma_min ** 2).sqrt().clamp(max=sigma_max)
    n_pix = m.sum(dim=(2, 3), keepdim=True)
    too_few = (n_pix < 16.0).to(values.dtype)
    mu = torch.where(too_few > 0, values.new_full(mu.shape, default_mu), mu)
    sigma = torch.where(too_few > 0, values.new_full(sigma.shape, default_sigma), sigma)
    return mu.view(-1, 1), sigma.view(-1, 1)


class NISFlowNetUNIHybridHLoRAFiLM(NISFlowNetUNIHybridHLoRA):
    """E1-T baseline + PCGS-FiLM modulation on the 64-ch final feature.

    Forward accepts a dict-style input (image_h + image_rgb + optional
    stain_h/stain_e/clu/point_map). When the stain/anchor signals are present
    (i.e. at training time, where dataset.py supplies them), per-image stats
    are fitted on-the-fly. Otherwise (inference without stain signals), a
    fixed default stats vector is used (no anchor unreliability risk).
    """

    def __init__(
        self,
        uni_path: str,
        flow_param: str = "direct",
        img_size: int = 224,
        pretrained_r34: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        film_hidden: int = 64,
        film_levels: tuple = ("final",),
        bg_anchor_min_dist: float = 40.0,
    ) -> None:
        super().__init__(
            uni_path=uni_path,
            flow_param=flow_param,
            img_size=img_size,
            pretrained_r34=pretrained_r34,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.bg_anchor_min_dist = float(bg_anchor_min_dist)
        # FiLM applied to the 64-ch final feature (between final_up and heads).
        # 8-D stats vector → MLP → (γ, β) ∈ R^64 each.
        self.film_mlp = nn.Sequential(
            nn.Linear(8, film_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(film_hidden, 2 * 64),
        )
        # Zero-init last layer so FiLM starts as identity (no-op) and only
        # activates as the MLP learns; avoids early-training destabilisation.
        nn.init.zeros_(self.film_mlp[-1].weight)
        nn.init.zeros_(self.film_mlp[-1].bias)
        # Default stats vector for inference (no anchor signals available).
        # Values match the initial pcgs_stats in train.py.
        self.register_buffer(
            "default_stats",
            torch.tensor([0.65, 0.15, 0.60, 0.18, 0.30, 0.18, 0.55, 0.20],
                         dtype=torch.float32),
        )
        print(f"[E1-FiLM] PCGS-FiLM modulation on 64-ch final feature "
              f"(hidden={film_hidden}, levels={film_levels}, zero-init γ,β).",
              flush=True)

    def _compute_pcgs_stats(self, clu, stain_h, stain_e, point_map):
        """Return (B, 8) stats vector from oracle anchors.

        Order: μ_H_fg, σ_H_fg, μ_Ecomp_fg, σ_Ecomp_fg,
               μ_H_bg, σ_H_bg, μ_E_bg, σ_E_bg
        """
        h_norm = _per_image_norm01(stain_h.float())          # (B,1,H,W)
        e_norm = _per_image_norm01(stain_e.float())
        e_comp = 1.0 - e_norm
        d = _distance_to_points(point_map, max_dist=max(self.bg_anchor_min_dist + 10.0, 60.0))
        d = d.to(h_norm.dtype)
        if clu.ndim == 3:
            clu_ = clu.unsqueeze(1)
        else:
            clu_ = clu
        fg_anchor = (clu_ == 1)
        bg_anchor = (clu_ == 0) & (d > self.bg_anchor_min_dist)

        mu_Hf, sg_Hf = _fit_per_image_gauss(h_norm, fg_anchor, 0.65, 0.15)
        mu_Ef, sg_Ef = _fit_per_image_gauss(e_comp, fg_anchor, 0.60, 0.18)
        mu_Hb, sg_Hb = _fit_per_image_gauss(h_norm, bg_anchor, 0.30, 0.18)
        mu_Eb, sg_Eb = _fit_per_image_gauss(e_norm, bg_anchor, 0.55, 0.20)
        stats = torch.cat([mu_Hf, sg_Hf, mu_Ef, sg_Ef, mu_Hb, sg_Hb, mu_Eb, sg_Eb], dim=1)
        return stats.float()

    def forward(self, x):
        """x: dict with at minimum 'image' (H-3ch) + 'image_rgb' (RGB-3ch).
        Optional fields enabling PCGS-FiLM:
            'clu' (B,H,W) long, 'stain_h' / 'stain_e' (B,H,W), 'point_map' (B,H,W).
        """
        if isinstance(x, dict):
            x_h = x["image"]
            x_rgb = x.get("image_rgb", x["image"])
        else:
            x_h = x; x_rgb = x

        H, W = x_h.shape[-2:]
        s0 = self.r34_stem(x_h)
        s1 = self.r34_layer1(self.r34_maxpool(s0))
        s2 = self.r34_layer2(s1)
        s3 = self.uni(x_rgb)
        feat = self.proj_uni(s3)
        feat = self.up3(feat, s2)
        feat = self.up2(feat, s1)
        feat = self.up1(feat, s0)
        feat = self.final_up(feat)
        if feat.shape[-2:] != (H, W):
            feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)

        # PCGS-FiLM modulation on the 64-ch final feature.
        B = feat.size(0)
        if (isinstance(x, dict)
                and "clu" in x and "stain_h" in x
                and "stain_e" in x and "point_map" in x):
            stats = self._compute_pcgs_stats(
                x["clu"], x["stain_h"], x["stain_e"], x["point_map"],
            ).to(device=feat.device, dtype=feat.dtype)
        else:
            stats = self.default_stats.to(device=feat.device, dtype=feat.dtype).unsqueeze(0).expand(B, -1).contiguous()
        gb = self.film_mlp(stats)                  # (B, 128) when feat C=64
        gamma, beta = gb.chunk(2, dim=1)           # each (B, 64)
        feat = (1.0 + gamma.view(B, -1, 1, 1)) * feat + beta.view(B, -1, 1, 1)

        fg_logits = self.fg_head(feat)
        centerness_logits = self.centerness_head(feat)
        boundary_logits = self.boundary_head(feat)
        outputs = {
            "fg_logits": fg_logits,
            "centerness_logits": centerness_logits,
            "boundary_logits": boundary_logits,
            "feat": feat,
        }
        if self.flow_param == "direct":
            outputs["flow"] = torch.tanh(self.flow_head(feat))
        else:
            phi = self.potential_head(feat)
            outputs["phi"] = phi
            from chromaflow.uni_encoder_hybrid import potential_to_flow
            outputs["flow"] = potential_to_flow(phi)
        return outputs


__all__ = ["NISFlowNetUNIHybridHLoRAFiLM"]
