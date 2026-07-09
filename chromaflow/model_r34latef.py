"""LateFusion-R34: residual H contribution at logit level only.

Design (vs SACAF/R34Dual that collapsed):
    - Main R34 RGB pathway: 100% UNCHANGED from NISFlowNet baseline
    - H R34_lite encodes H-channel separately
    - H signal enters ONLY at logit head via additive residual:
          fg_logits = head_main(feat_rgb) + head_h(h_feat)    [head_h zero-init]
    - At init: head_h output is 0 → behavior identical to A4_R34 baseline
    - Training: head_h gradually learns small additive H-corrective signal
    - Decoder + features remain pure RGB, no contamination
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from chromaflow.model import _resnet34, UpBlock, potential_to_flow
from chromaflow.model_sacaf import _R34Lite


class NISFlowNetLateFusionR34(nn.Module):
    """Late-fusion dual-stream: H injects only at logit residual."""

    def __init__(self, pretrained: bool = True, flow_param: str = "direct",
                 h_base_ch: int = 16, h_logit_channels: int = 32) -> None:
        super().__init__()
        if flow_param not in {"direct", "potential"}:
            raise ValueError(f"Unsupported flow_param: {flow_param}")
        self.flow_param = flow_param
        self.h_logit_channels = h_logit_channels

        # Main R34 — SAME as NISFlowNet baseline
        self.main = _resnet34(pretrained=pretrained)
        self.up5 = UpBlock(512, 256, 256)
        self.up6 = UpBlock(256, 128, 128)
        self.up7 = UpBlock(128, 64, 64)
        self.up8 = UpBlock(64, 64, 64)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Main heads (same as NISFlowNet baseline)
        self.fg_head = nn.Conv2d(64, 1, kernel_size=1)
        self.centerness_head = nn.Conv2d(64, 1, kernel_size=1)
        if self.flow_param == "direct":
            self.flow_head = nn.Conv2d(64, 2, kernel_size=1)
        else:
            self.potential_head = nn.Conv2d(64, 1, kernel_size=1)

        # H branch (auxiliary)
        self.h_branch = _R34Lite(in_ch=3, base_ch=h_base_ch)
        # Project h_deep (8*h_base_ch ch, H/32) → h_logit_channels, then upsample to (H, W)
        # Use GroupNorm (not BatchNorm) for h_proj to avoid train/eval running-stats mismatch.
        # Prior BN version collapsed val_AJI to 0.007 due to BN running-stats drift after the
        # H-residual heads started contributing non-zero output.
        gn_groups = min(8, h_logit_channels)
        self.h_proj = nn.Sequential(
            nn.Conv2d(8 * h_base_ch, h_logit_channels, kernel_size=1),
            nn.GroupNorm(gn_groups, h_logit_channels),
            nn.ReLU(inplace=True),
        )

        # Auxiliary H heads — zero-init so contribution starts at 0
        self.fg_head_h = nn.Conv2d(h_logit_channels, 1, kernel_size=1)
        self.centerness_head_h = nn.Conv2d(h_logit_channels, 1, kernel_size=1)
        nn.init.zeros_(self.fg_head_h.weight)
        nn.init.zeros_(self.fg_head_h.bias)
        nn.init.zeros_(self.centerness_head_h.weight)
        nn.init.zeros_(self.centerness_head_h.bias)

        # Flow head also gets H contribution(zero-init so initial output unchanged)
        if self.flow_param == "direct":
            self.flow_head_h = nn.Conv2d(h_logit_channels, 2, kernel_size=1)
            nn.init.zeros_(self.flow_head_h.weight)
            nn.init.zeros_(self.flow_head_h.bias)
        else:
            self.potential_head_h = nn.Conv2d(h_logit_channels, 1, kernel_size=1)
            nn.init.zeros_(self.potential_head_h.weight)
            nn.init.zeros_(self.potential_head_h.bias)

        n_main = sum(p.numel() for p in self.main.parameters())
        n_h = sum(p.numel() for p in self.h_branch.parameters())
        n_decoder = sum(
            sum(p.numel() for p in m.parameters())
            for m in [self.up5, self.up6, self.up7, self.up8, self.final_up])
        print(f"[LateFusion-R34] main R34: {n_main/1e6:.2f}M, H-branch lite: {n_h/1e6:.2f}M, "
              f"decoder: {n_decoder/1e6:.2f}M; H injects only at logit residual (zero-init).",
              flush=True)

    def forward(self, x):
        if isinstance(x, dict):
            rgb = x.get("image_rgb", x.get("image"))
            h = x.get("image", rgb)
        else:
            rgb = x
            h = x

        # --- Main RGB pathway: 100% pure NISFlowNet ---
        z = self.main.conv1(rgb)
        z = self.main.bn1(z)
        s1 = self.main.relu(z)
        z = self.main.maxpool(s1)
        s2 = self.main.layer1(z)
        s3 = self.main.layer2(s2)
        s4 = self.main.layer3(s3)
        x_deep = self.main.layer4(s4)
        z = self.up5(x_deep, s4)
        z = self.up6(z, s3)
        z = self.up7(z, s2)
        z = self.up8(z, s1)
        feat = self.final_up(z)                                # (B, 64, H, W) — pure RGB feat

        # --- H branch (auxiliary deep features only) ---
        _, _, _, _, h_deep = self.h_branch(h)
        h_proj = self.h_proj(h_deep)                            # (B, h_logit_ch, H/32, W/32)
        h_proj_up = F.interpolate(h_proj, size=feat.shape[-2:],
                                    mode="bilinear", align_corners=False)

        # --- Late residual fusion at logits ---
        fg_logits = self.fg_head(feat) + self.fg_head_h(h_proj_up)
        centerness_logits = (self.centerness_head(feat)
                              + self.centerness_head_h(h_proj_up))
        outputs = {
            "fg_logits": fg_logits,
            "centerness_logits": centerness_logits,
            "feat": feat,
        }
        if self.flow_param == "direct":
            flow = torch.tanh(self.flow_head(feat) + self.flow_head_h(h_proj_up))
            outputs["flow"] = flow
        else:
            phi = self.potential_head(feat) + self.potential_head_h(h_proj_up)
            outputs["phi"] = phi
            outputs["flow"] = potential_to_flow(phi)
        return outputs


__all__ = ["NISFlowNetLateFusionR34"]
