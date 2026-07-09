"""E1-MSG encoder: NISFlowNetUNIHybridHLoRA + multi-scale gating.

Combines decoder features from two scales:
    deep    = up3 output (128-ch, then projected to 64) — semantic, /8 resolution upsampled
    shallow = up1 output (64-ch) — fine spatial detail
via a gating signal derived from the PCGS posterior entropy:
    entropy(x) = -p·log(p) - (1-p)·log(1-p)  where p = posterior at x

When PCGS posterior is *certain* (entropy low) → use deep semantic feature.
When PCGS posterior is *uncertain* (entropy high) → fall back to shallow
morphology-rich feature. The intuition: in stain-clear regions trust the
high-level fg/bg classifier; in stain-ambiguous regions trust the local
edge / boundary cues.

Gating is per-pixel.
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


class NISFlowNetUNIHybridHLoRAMSG(NISFlowNetUNIHybridHLoRA):
    """E1-T baseline + entropy-gated multi-scale fusion.

    The 64-ch final feature is replaced with a weighted blend:
        gate = sigmoid(η · (1 - 2·entropy))   where η is learnable, init=2
        final_feature = gate · final_up_out + (1 - gate) · shallow_proj_out

    `shallow_proj` is a 1×1 conv projecting the up1 (64-ch) feature.
    `gate` collapses to 1 when entropy → 0 (certain) and 0 when entropy → 1
    (uncertain). Learnable η controls slope.
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
        bg_anchor_min_dist: float = 40.0,
    ) -> None:
        super().__init__(
            uni_path=uni_path, flow_param=flow_param, img_size=img_size,
            pretrained_r34=pretrained_r34,
            lora_rank=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        )
        self.bg_anchor_min_dist = float(bg_anchor_min_dist)
        # 1×1 projection of shallow up1 (64-ch) into the same 64-ch space.
        self.shallow_proj = nn.Conv2d(64, 64, kernel_size=1)
        nn.init.zeros_(self.shallow_proj.weight)
        nn.init.zeros_(self.shallow_proj.bias)
        # Learnable slope η — controls how quickly gate decreases as entropy grows.
        self.gate_eta = nn.Parameter(torch.tensor(2.0))
        # E1-MSG v2 — Learnable bias defaulting gate strongly toward 1 (deep-only).
        # Why: v1's gate sigmoid(η·(1-2·entropy)) gave gate≈0.5 at moderate entropy,
        # which dropped in-domain MO AJI 0.6201→0.6128 because the model was forced
        # to use shallow features even on stain-clear in-domain pixels.
        # v2 with bias=2.5 default: gate(entropy=0)≈0.99, gate(entropy=0.5)≈0.92,
        # gate(entropy=1)≈0.62 — shallow only activates on highly uncertain pixels
        # (typically far-domain), preserving in-domain behaviour.
        self.gate_bias = nn.Parameter(torch.tensor(2.5))
        # E1-MSG v4 — hard threshold via STE.
        # v2 (bias=2.5) gave gate≈0.987 at low entropy, but residual 0.013 ·
        # shallow_proj contribution still hurt in-domain. v3's high-WD on
        # shallow_proj reduced its norm 27× but still cost 1pp in-domain AJI.
        # v4 enforces exact gate=1 at in-domain via a learnable entropy threshold
        # with straight-through estimator: forward uses hard step at threshold,
        # backward flows through the sigmoid surrogate.
        self.entropy_threshold = nn.Parameter(torch.tensor(0.5))
        # Default to v3 soft-gate behaviour (best so far per ablation).
        # Set to True to switch to v4 hard-gate via STE.
        self.use_hard_gate = False
        print(f"[E1-MSG] soft sigmoid gate (v3) with shallow_proj high-WD; "
              f"η init=2.0, bias init=2.5. (set use_hard_gate=True for v4 STE)",
              flush=True)

    def _posterior_entropy(self, clu, stain_h, stain_e, point_map):
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
        for t in (mu_Hf, sg_Hf, mu_Ef, sg_Ef, mu_Hb, sg_Hb, mu_Eb, sg_Eb):
            pass  # already (B,1)
        mu_Hf = mu_Hf.view(-1, 1, 1, 1); sg_Hf = sg_Hf.view(-1, 1, 1, 1)
        mu_Ef = mu_Ef.view(-1, 1, 1, 1); sg_Ef = sg_Ef.view(-1, 1, 1, 1)
        mu_Hb = mu_Hb.view(-1, 1, 1, 1); sg_Hb = sg_Hb.view(-1, 1, 1, 1)
        mu_Eb = mu_Eb.view(-1, 1, 1, 1); sg_Eb = sg_Eb.view(-1, 1, 1, 1)
        lik_fg = _gauss_pdf(h_norm, mu_Hf, sg_Hf) * _gauss_pdf(e_comp, mu_Ef, sg_Ef)
        lik_bg = _gauss_pdf(h_norm, mu_Hb, sg_Hb) * _gauss_pdf(e_norm, mu_Eb, sg_Eb)
        p = 0.5 * lik_fg / (0.5 * lik_fg + 0.5 * lik_bg + 1e-8)
        p = p.clamp(1e-3, 1.0 - 1e-3)
        entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p)) / math.log(2.0)  # ∈ [0, 1]
        return entropy

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
        feat_deep = self.proj_uni(s3)
        feat_deep = self.up3(feat_deep, s2)
        feat_deep = self.up2(feat_deep, s1)
        feat_shallow = self.up1(feat_deep, s0)             # 64-ch at near-full res
        feat_deep_full = self.final_up(feat_shallow)        # 64-ch at full res
        if feat_deep_full.shape[-2:] != (H, W):
            feat_deep_full = F.interpolate(feat_deep_full, size=(H, W), mode="bilinear", align_corners=False)
        feat_shallow_full = F.interpolate(feat_shallow, size=(H, W), mode="bilinear", align_corners=False)
        feat_shallow_full = self.shallow_proj(feat_shallow_full)

        # Compute entropy-driven gate (v4: hard threshold with STE).
        if (isinstance(x, dict) and "clu" in x and "stain_h" in x
                and "stain_e" in x and "point_map" in x):
            entropy = self._posterior_entropy(x["clu"], x["stain_h"], x["stain_e"], x["point_map"])
            entropy = entropy.to(feat_deep_full.dtype)
            if getattr(self, "use_hard_gate", False):
                # Soft surrogate for gradient flow.
                soft_gate = torch.sigmoid(
                    self.gate_eta * (self.entropy_threshold - entropy)
                )
                # Hard indicator for forward pass.
                hard = (entropy < self.entropy_threshold).to(feat_deep_full.dtype)
                # Straight-through estimator: forward = hard, backward = soft.
                gate = hard + (soft_gate - soft_gate.detach())
            else:
                # v2/v3 legacy formula: gate = sigmoid(bias + η·(1-2·entropy))
                gate = torch.sigmoid(self.gate_bias + self.gate_eta * (1.0 - 2.0 * entropy))
        else:
            # Inference fallback: gate = 1 (use deep only). shallow_proj is
            # zero-init so this collapses to baseline behaviour at start.
            gate = torch.ones_like(feat_deep_full[:, :1])

        feat = gate * feat_deep_full + (1.0 - gate) * feat_shallow_full

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


__all__ = ["NISFlowNetUNIHybridHLoRAMSG"]
