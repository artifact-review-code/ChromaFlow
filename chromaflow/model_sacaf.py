"""SACAF (Stain-Aware Cross-Attention Fusion) encoder.

Dual-stream architecture:
    - Main stream: full R34 on RGB input (~21M params)
    - Auxiliary stream: R34-lite on H-channel input (~1.3M params, 1/4 base channels)
At each of 5 skip levels (s1, s2, s3, s4, deep), we fuse via a Cross-Modal
Channel-Spatial Gate: the H-stream features provide channel-wise and
spatial-wise modulation onto the RGB-stream features.

Design rationale (vs prior dual-stream attempts):
    - MSG (entropy-gated, A4): gate falls back to 1.0 at inference (anchor signals
      absent) → H-stream contribution effectively zero at test time. SACAF's
      cross-modal gates are computed from H features themselves, so they are
      always active.
    - FiLM (8-stats scalar conditioning, ablated): used 8 hand-picked statistics
      from PCGS anchors → at inference defaults to a fixed vector, losing
      per-image adaptivity. SACAF uses a full H-feature pyramid, providing
      per-pixel multi-scale stain-aware modulation.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock

from chromaflow.model import _resnet34, UpBlock, potential_to_flow


# =========================================================================
# Lightweight R34 trunk for H-channel branch (1/4 base channels)
# =========================================================================

class _R34Lite(nn.Module):
    """ResNet-34 style trunk with reduced channel count (base_ch=16 → /4 of std)."""

    def __init__(self, in_ch: int = 3, base_ch: int = 16) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, base_ch, kernel_size=7, stride=2,
                                padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_ch)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        c1, c2, c3, c4 = base_ch, 2 * base_ch, 4 * base_ch, 8 * base_ch

        def _make_layer(in_c, out_c, num_blocks, stride=1):
            layers = []
            down = None
            if stride != 1 or in_c != out_c:
                down = nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_c),
                )
            layers.append(BasicBlock(in_c, out_c, stride=stride, downsample=down))
            for _ in range(1, num_blocks):
                layers.append(BasicBlock(out_c, out_c))
            return nn.Sequential(*layers)

        # 3-4-6-3 like R34 (reduced channels), with stride pattern matching R34.
        self.layer1 = _make_layer(c1, c1, 3, stride=1)
        self.layer2 = _make_layer(c1, c2, 4, stride=2)
        self.layer3 = _make_layer(c2, c3, 6, stride=2)
        self.layer4 = _make_layer(c3, c4, 3, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        s1 = self.relu(x)                       # (B, base_ch, H/2, W/2)
        x = self.maxpool(s1)
        s2 = self.layer1(x)                     # (B, base_ch,   H/4)
        s3 = self.layer2(s2)                    # (B, 2*base_ch, H/8)
        s4 = self.layer3(s3)                    # (B, 4*base_ch, H/16)
        s_deep = self.layer4(s4)                # (B, 8*base_ch, H/32)
        return s1, s2, s3, s4, s_deep


# =========================================================================
# Cross-Modal Fusion: H stream gates RGB stream (channel + spatial)
# =========================================================================

class CrossModalFusion(nn.Module):
    """Stain-aware feature fusion.

    Given RGB features f_rgb ∈ ℝ^{C_rgb×H×W} and H-stream features f_h ∈
    ℝ^{C_h×H×W}, output fused features for the decoder:

        f_aligned = Conv1x1(f_h, C_rgb)            # channel align
        g_c       = sigmoid(MLP(GAP(f_aligned)))   # channel gate (B,C,1,1)
        g_s       = sigmoid(Conv7x7(f_aligned))    # spatial gate (B,1,H,W)
        f_fused   = f_rgb * (1 + α · g_c · g_s)   # multiplicative residual gating
                    + β · f_aligned               # additive residual H contribution

    α, β are small constants (0.3, 0.1) and `h_align` is zero-init so the
    layer starts as identity (no fusion). Stain features are gradually injected
    as training progresses.
    """

    def __init__(self, c_rgb: int, c_h: int,
                 alpha: float = 0.3, beta: float = 0.1) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.h_align = nn.Conv2d(c_h, c_rgb, kernel_size=1)
        nn.init.zeros_(self.h_align.weight)
        nn.init.zeros_(self.h_align.bias)

        hidden = max(c_rgb // 4, 8)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_rgb, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, c_rgb, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(c_rgb, 1, kernel_size=7, padding=3, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x_rgb: torch.Tensor, x_h: torch.Tensor) -> torch.Tensor:
        x_h_aligned = self.h_align(x_h)
        g_c = self.channel_gate(x_h_aligned)
        g_s = self.spatial_gate(x_h_aligned)
        modulation = 1.0 + self.alpha * g_c * g_s
        return x_rgb * modulation + self.beta * x_h_aligned


# =========================================================================
# SACAF Net: dual-stream + fusion + decoder + heads
# =========================================================================

class NISFlowNetSACAF(nn.Module):
    """Stain-Aware Cross-Attention Fusion network for point-supervised nuclei seg.

    Forward expects a dict with both:
        x["image_rgb"]: (B, 3, H, W) RGB tensor for main stream
        x["image"]:     (B, 3, H, W) H-channel-3ch tensor for stain stream

    Falls back to (rgb=x, h=x) if scalar tensor input — useful for legacy.
    """

    def __init__(self, pretrained: bool = True, flow_param: str = "direct",
                 h_base_ch: int = 16) -> None:
        super().__init__()
        if flow_param not in {"direct", "potential"}:
            raise ValueError(f"Unsupported flow_param: {flow_param}")
        self.flow_param = flow_param

        # Main RGB R34 (full size)
        self.main = _resnet34(pretrained=pretrained)
        # H-channel R34Lite (1/4 channels)
        self.h_branch = _R34Lite(in_ch=3, base_ch=h_base_ch)

        # Cross-modal fusion modules (one per skip level + deep)
        c_h_levels = [h_base_ch, h_base_ch, 2 * h_base_ch, 4 * h_base_ch, 8 * h_base_ch]
        c_rgb_levels = [64, 64, 128, 256, 512]
        self.fuse_s1 = CrossModalFusion(c_rgb_levels[0], c_h_levels[0])
        self.fuse_s2 = CrossModalFusion(c_rgb_levels[1], c_h_levels[1])
        self.fuse_s3 = CrossModalFusion(c_rgb_levels[2], c_h_levels[2])
        self.fuse_s4 = CrossModalFusion(c_rgb_levels[3], c_h_levels[3])
        self.fuse_deep = CrossModalFusion(c_rgb_levels[4], c_h_levels[4])

        # Decoder (same structure as NISFlowNet)
        self.up5 = UpBlock(512, 256, 256)
        self.up6 = UpBlock(256, 128, 128)
        self.up7 = UpBlock(128, 64, 64)
        self.up8 = UpBlock(64, 64, 64)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.fg_head = nn.Conv2d(64, 1, kernel_size=1)
        self.centerness_head = nn.Conv2d(64, 1, kernel_size=1)
        if self.flow_param == "direct":
            self.flow_head = nn.Conv2d(64, 2, kernel_size=1)
        else:
            self.potential_head = nn.Conv2d(64, 1, kernel_size=1)

        n_main = sum(p.numel() for p in self.main.parameters())
        n_h = sum(p.numel() for p in self.h_branch.parameters())
        n_fuse = sum(
            sum(p.numel() for p in m.parameters())
            for m in [self.fuse_s1, self.fuse_s2, self.fuse_s3, self.fuse_s4, self.fuse_deep]
        )
        print(f"[SACAF] main R34: {n_main/1e6:.2f}M, H-branch R34-lite: {n_h/1e6:.2f}M, "
              f"5 cross-modal fusion modules: {n_fuse/1e6:.2f}M",
              flush=True)

    def forward(self, x):
        if isinstance(x, dict):
            rgb = x.get("image_rgb", x.get("image"))
            h = x.get("image", rgb)
        else:
            rgb = x
            h = x

        # --- Main R34 ---
        z = self.main.conv1(rgb)
        z = self.main.bn1(z)
        s1 = self.main.relu(z)                              # (B,  64, H/2)
        z = self.main.maxpool(s1)
        s2 = self.main.layer1(z)                            # (B,  64, H/4)
        s3 = self.main.layer2(s2)                           # (B, 128, H/8)
        s4 = self.main.layer3(s3)                           # (B, 256, H/16)
        x_deep = self.main.layer4(s4)                       # (B, 512, H/32)

        # --- H R34Lite ---
        h_s1, h_s2, h_s3, h_s4, h_deep = self.h_branch(h)

        # --- Cross-modal fusion at each level ---
        s1_f = self.fuse_s1(s1, h_s1)
        s2_f = self.fuse_s2(s2, h_s2)
        s3_f = self.fuse_s3(s3, h_s3)
        s4_f = self.fuse_s4(s4, h_s4)
        x_deep_f = self.fuse_deep(x_deep, h_deep)

        # --- Decoder ---
        z = self.up5(x_deep_f, s4_f)
        z = self.up6(z, s3_f)
        z = self.up7(z, s2_f)
        z = self.up8(z, s1_f)
        feat = self.final_up(z)

        fg_logits = self.fg_head(feat)
        centerness_logits = self.centerness_head(feat)
        outputs = {
            "fg_logits": fg_logits,
            "centerness_logits": centerness_logits,
            "feat": feat,
        }
        if self.flow_param == "direct":
            outputs["flow"] = torch.tanh(self.flow_head(feat))
        else:
            phi = self.potential_head(feat)
            outputs["phi"] = phi
            outputs["flow"] = potential_to_flow(phi)
        return outputs


__all__ = ["NISFlowNetSACAF"]
