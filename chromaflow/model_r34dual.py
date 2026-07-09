"""R34Dual-simple: dual R34 backbone with additive fusion (no cross-modal attention).

Ablation baseline for SACAF. Difference from SACAF:
    - Same dual-stream backbone (main R34 on RGB + R34-lite on H)
    - Same per-level skip connections to decoder
    - DIFFERENT fusion: simple additive `x_rgb + Conv1x1(x_h)` instead of
      channel-spatial gated attention.

This isolates the contribution of the cross-modal attention mechanism in SACAF:
    - If SACAF > R34Dual-simple → attention is real contribution
    - If SACAF ≈ R34Dual-simple → dual structure alone is what matters
"""
from __future__ import annotations

import torch
import torch.nn as nn

from chromaflow.model import _resnet34, UpBlock, potential_to_flow
from chromaflow.model_sacaf import _R34Lite


class SimpleAddFusion(nn.Module):
    """Channel-align H features and additively combine with RGB features.

    No gating, no attention — purely linear merge:
        f_fused = f_rgb + 0.1 · Conv1x1(f_h)

    Init: zero-init Conv1x1 weight so layer starts as identity.
    """

    def __init__(self, c_rgb: int, c_h: int, beta: float = 0.1) -> None:
        super().__init__()
        self.beta = beta
        self.h_align = nn.Conv2d(c_h, c_rgb, kernel_size=1)
        nn.init.zeros_(self.h_align.weight)
        nn.init.zeros_(self.h_align.bias)

    def forward(self, x_rgb: torch.Tensor, x_h: torch.Tensor) -> torch.Tensor:
        return x_rgb + self.beta * self.h_align(x_h)


class NISFlowNetR34Dual(nn.Module):
    """Dual R34 (RGB main + H lite) with simple additive fusion at each scale.

    Ablation variant of SACAF: same backbone structure, no cross-modal attention.
    """

    def __init__(self, pretrained: bool = True, flow_param: str = "direct",
                 h_base_ch: int = 16) -> None:
        super().__init__()
        if flow_param not in {"direct", "potential"}:
            raise ValueError(f"Unsupported flow_param: {flow_param}")
        self.flow_param = flow_param

        self.main = _resnet34(pretrained=pretrained)
        self.h_branch = _R34Lite(in_ch=3, base_ch=h_base_ch)

        c_rgb_levels = [64, 64, 128, 256, 512]
        c_h_levels = [h_base_ch, h_base_ch, 2 * h_base_ch,
                       4 * h_base_ch, 8 * h_base_ch]
        self.fuse_s1 = SimpleAddFusion(c_rgb_levels[0], c_h_levels[0])
        self.fuse_s2 = SimpleAddFusion(c_rgb_levels[1], c_h_levels[1])
        self.fuse_s3 = SimpleAddFusion(c_rgb_levels[2], c_h_levels[2])
        self.fuse_s4 = SimpleAddFusion(c_rgb_levels[3], c_h_levels[3])
        self.fuse_deep = SimpleAddFusion(c_rgb_levels[4], c_h_levels[4])

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
        print(f"[R34Dual-simple] main R34: {n_main/1e6:.2f}M, H-branch lite: {n_h/1e6:.2f}M, "
              f"5 additive fusion modules: {n_fuse/1e6:.2f}M",
              flush=True)

    def forward(self, x):
        if isinstance(x, dict):
            rgb = x.get("image_rgb", x.get("image"))
            h = x.get("image", rgb)
        else:
            rgb = x
            h = x

        z = self.main.conv1(rgb)
        z = self.main.bn1(z)
        s1 = self.main.relu(z)
        z = self.main.maxpool(s1)
        s2 = self.main.layer1(z)
        s3 = self.main.layer2(s2)
        s4 = self.main.layer3(s3)
        x_deep = self.main.layer4(s4)

        h_s1, h_s2, h_s3, h_s4, h_deep = self.h_branch(h)

        s1_f = self.fuse_s1(s1, h_s1)
        s2_f = self.fuse_s2(s2, h_s2)
        s3_f = self.fuse_s3(s3, h_s3)
        s4_f = self.fuse_s4(s4, h_s4)
        x_deep_f = self.fuse_deep(x_deep, h_deep)

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


__all__ = ["NISFlowNetR34Dual"]
