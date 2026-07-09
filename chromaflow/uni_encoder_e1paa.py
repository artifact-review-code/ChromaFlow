"""E1-PaA encoder: NISFlowNetUNIHybridHLoRA + Posterior-as-Attention.

PCGS already produces a per-pixel Bayesian posterior over fg/bg from H/E
likelihoods (at training time). This module reuses the same posterior as a
spatial attention map applied to the 64-ch final feature *before* the fg head:

    feature_attended = (1 + α · (posterior - 0.5)) · feature
    fg_logits = fg_head(feature_attended)

α is a learnable scalar (init 0.1), so PaA starts essentially as a no-op and
the model decides how much to use it.

Training: posterior computed from oracle anchors (clu==1 / clu==0∩dist>40),
identical to PCGS-Anchor BCE target.
Inference: no anchor signals available, fall back to using sigmoid(fg_logits)
itself as the attention map (effectively a self-recalibration).

See uni_encoder_e1film.py for the shared per-image-norm / Gaussian-fit helpers.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from chromaflow.uni_encoder_b1 import NISFlowNetUNIHybridHLoRA
from chromaflow.uni_encoder_e1film import (
    _per_image_norm01,
    _distance_to_points,
    _fit_per_image_gauss,
)


def _gauss_pdf(x, mu, sigma):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))


class NISFlowNetUNIHybridHLoRAPaA(NISFlowNetUNIHybridHLoRA):
    """E1-T baseline + PCGS Posterior-as-Attention modulation."""

    def __init__(
        self,
        uni_path: str,
        flow_param: str = "direct",
        img_size: int = 224,
        pretrained_r34: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        bg_anchor_min_dist: float = 40.0,
    ) -> None:
        super().__init__(
            uni_path=uni_path, flow_param=flow_param, img_size=img_size,
            pretrained_r34=pretrained_r34,
            lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        )
        self.bg_anchor_min_dist = float(bg_anchor_min_dist)
        # Learnable attention strength α; starts small (≈ no-op).
        self.paa_alpha = nn.Parameter(torch.tensor(0.1))
        print(f"[E1-PaA] PCGS Posterior-as-Attention on 64-ch final feature; "
              f"α (learnable) init=0.1.", flush=True)

    def _pcgs_posterior(self, clu, stain_h, stain_e, point_map):
        h_norm = _per_image_norm01(stain_h.float())
        e_norm = _per_image_norm01(stain_e.float())
        e_comp = 1.0 - e_norm
        d = _distance_to_points(point_map, max_dist=max(self.bg_anchor_min_dist + 10.0, 60.0))
        d = d.to(h_norm.dtype)
        clu_ = clu.unsqueeze(1) if clu.ndim == 3 else clu
        fg_anchor = (clu_ == 1)
        bg_anchor = (clu_ == 0) & (d > self.bg_anchor_min_dist)
        mu_Hf, sg_Hf = _fit_per_image_gauss(h_norm, fg_anchor, 0.65, 0.15)
        mu_Ef, sg_Ef = _fit_per_image_gauss(e_comp, fg_anchor, 0.60, 0.18)
        mu_Hb, sg_Hb = _fit_per_image_gauss(h_norm, bg_anchor, 0.30, 0.18)
        mu_Eb, sg_Eb = _fit_per_image_gauss(e_norm, bg_anchor, 0.55, 0.20)
        # Reshape per-image stats to (B,1,1,1) for broadcasting.
        mu_Hf = mu_Hf.view(-1, 1, 1, 1); sg_Hf = sg_Hf.view(-1, 1, 1, 1)
        mu_Ef = mu_Ef.view(-1, 1, 1, 1); sg_Ef = sg_Ef.view(-1, 1, 1, 1)
        mu_Hb = mu_Hb.view(-1, 1, 1, 1); sg_Hb = sg_Hb.view(-1, 1, 1, 1)
        mu_Eb = mu_Eb.view(-1, 1, 1, 1); sg_Eb = sg_Eb.view(-1, 1, 1, 1)
        lik_fg = _gauss_pdf(h_norm, mu_Hf, sg_Hf) * _gauss_pdf(e_comp, mu_Ef, sg_Ef)
        lik_bg = _gauss_pdf(h_norm, mu_Hb, sg_Hb) * _gauss_pdf(e_norm, mu_Eb, sg_Eb)
        posterior = 0.5 * lik_fg / (0.5 * lik_fg + 0.5 * lik_bg + 1e-8)
        return posterior.clamp(0.01, 0.99)

    def forward(self, x):
        if isinstance(x, dict):
            x_h = x["image"]; x_rgb = x.get("image_rgb", x["image"])
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

        # Compute posterior attention map.
        att = None
        if (isinstance(x, dict) and "clu" in x and "stain_h" in x
                and "stain_e" in x and "point_map" in x):
            att = self._pcgs_posterior(x["clu"], x["stain_h"], x["stain_e"],
                                        x["point_map"]).to(feat.dtype)
        # else: leave att=None; fg_logits computed without attention. We avoid
        # using sigmoid(fg_logits) recursively here (chicken-and-egg).

        if att is not None:
            modulator = 1.0 + self.paa_alpha * (att - 0.5)
            feat = feat * modulator

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


__all__ = ["NISFlowNetUNIHybridHLoRAPaA"]
