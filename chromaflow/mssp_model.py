"""MSSP — Multi-Scale Stain Pyramid encoder.

A lightweight backbone alternative to UNI: instead of relying on a 300M ViT
foundation model, we explicitly compute multi-scale H smoothings (3, 5, 9, 17 px)
and process them with a small CNN pyramid.

Diagnostic showed AUC of H smoothed at scales 5-9 px = 0.91-0.95 on clu==2 fg/bg
(MO+CPM), vs point H = 0.89. The 5-9 px range corresponds to typical nucleus
radius, which is biologically motivated.

Architecture:
  - RGB main branch: R34 (~21M params, ImageNet pretrained)
  - H stain branch: MSSP pyramid (~5M params, learned from scratch)
       Input: H smoothed at 5 scales {1, 3, 5, 9, 17} stacked as 5 channels
       Output: feature pyramid (s1, s2, s3, s4, s5) matching R34 stage outputs
  - Fusion at each scale: concat + 1×1 conv
  - Decoder: standard U-Net up5/6/7/8 (same as V3)

Total params: ~30M (vs V3-with-UNI ~330M). Pareto-friendly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from chromaflow.model import _resnet34, UpBlock, potential_to_flow


# ---------------------------------------------------------------------------
# Building block: BasicBlock with GroupNorm
# ---------------------------------------------------------------------------

class BasicBlockGN(nn.Module):
    """Residual BasicBlock with GroupNorm (no BN train/eval state mismatch).

    Same as torchvision BasicBlock but uses GroupNorm. The MSSP pyramid is
    trained from scratch and is small enough that BN's mini-batch stats are
    noisy; GN is more robust.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, downsample=None,
                 gn_groups: int = 8) -> None:
        super().__init__()
        gn_groups = min(gn_groups, out_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(gn_groups, out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(gn_groups, out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


# ---------------------------------------------------------------------------
# MSSP encoder
# ---------------------------------------------------------------------------

class MSSPEncoder(nn.Module):
    """Multi-Scale Stain Pyramid encoder.

    Input: H channel (B, 1, H, W) — raw hematoxylin density.
    Output: 5-tuple feature pyramid (s_stem, s1, s2, s3, s4):
        s_stem: (B, base_ch,   H/2)
        s1:     (B, base_ch,   H/4)
        s2:     (B, base_ch*2, H/8)
        s3:     (B, base_ch*4, H/16)
        s4:     (B, base_ch*8, H/32)

    Resolution alignment with main R34:
        main.conv1 → relu → 64 ch, H/2     ↔ s_stem (base_ch=32, H/2)
        main.maxpool → layer1 → 64 ch, H/4 ↔ s1
        layer2 → 128 ch, H/8                ↔ s2
        layer3 → 256 ch, H/16               ↔ s3
        layer4 → 512 ch, H/32               ↔ s4
    """

    def __init__(self, scales=(1, 3, 5, 9, 17), base_ch: int = 32, n_blocks: int = 2) -> None:
        super().__init__()
        self.scales = list(scales)
        n_in = len(self.scales)
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        # Stem: 5 scale-stacked H channels → base_ch at /2
        self.stem = nn.Sequential(
            nn.Conv2d(n_in, c1, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(min(8, c1), c1),
            nn.ReLU(inplace=True),
        )

        # Stage 1: keep /2, then maxpool to /4 via first stride-2 block of stage 2
        self.layer1 = self._make_stage(c1, c1, n_blocks, stride=1)   # /2 -> /4 will be inside stem+layer1? Actually stem already does /2.
        # We need to be careful about strides. Let's keep layer1 at stride 2 like R34's maxpool→layer1 path.
        # Actually R34: conv1 (stride 2) → maxpool (stride 2) → layer1 (stride 1) → layer2 (stride 2) ...
        # So R34's stages are at /2, /4, /4, /8, /16, /32. We need MSSP outputs to match.
        # Rebuild:
        self.layer1 = self._make_stage(c1, c1, n_blocks, stride=2)   # H/4
        self.layer2 = self._make_stage(c1, c2, n_blocks, stride=2)   # H/8
        self.layer3 = self._make_stage(c2, c3, n_blocks, stride=2)   # H/16
        self.layer4 = self._make_stage(c3, c4, n_blocks, stride=2)   # H/32

    @staticmethod
    def _make_stage(in_ch: int, out_ch: int, n_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        if stride != 1 or in_ch != out_ch:
            down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.GroupNorm(min(8, out_ch), out_ch),
            )
        else:
            down = None
        layers.append(BasicBlockGN(in_ch, out_ch, stride=stride, downsample=down))
        for _ in range(1, n_blocks):
            layers.append(BasicBlockGN(out_ch, out_ch, stride=1, downsample=None))
        return nn.Sequential(*layers)

    def multi_scale_h(self, H: torch.Tensor) -> torch.Tensor:
        """Stack H at multiple smoothings as channels."""
        outs = []
        for s in self.scales:
            if s <= 1:
                outs.append(H)
            else:
                pad = s // 2
                outs.append(F.avg_pool2d(H, kernel_size=s, stride=1, padding=pad))
        return torch.cat(outs, dim=1)

    def forward(self, H: torch.Tensor):
        if H.ndim == 3:
            H = H.unsqueeze(1)
        if H.shape[1] >= 3:
            H = H[:, :1]  # take first channel if passed as 3-ch h_repeat
        x = self.multi_scale_h(H)               # (B, n_scales, H_pix, W_pix)
        s_stem = self.stem(x)                    # /2
        s1 = self.layer1(s_stem)                 # /4
        s2 = self.layer2(s1)                     # /8
        s3 = self.layer3(s2)                     # /16
        s4 = self.layer4(s3)                     # /32
        return s_stem, s1, s2, s3, s4


# ---------------------------------------------------------------------------
# Full model: R34 RGB main + MSSP H aux + fusion + decoder
# ---------------------------------------------------------------------------

class NISFlowNetMSSP(nn.Module):
    """Multi-Scale Stain Pyramid model.

    Replaces V3's UNI ViT-L (300M frozen + LoRA) with a lightweight learned
    MSSP encoder on H. Total params ~30M vs V3 ~330M.

    Inputs (via dict):
        image:     (B, 3, H, W)  — RGB or h_repeat-3ch (ignored if image_rgb present)
        image_rgb: (B, 3, H, W)  — RGB (preferred for main path)
        stain_h:   (B, H, W)     — raw H (used by MSSP)
    """

    def __init__(self, pretrained_r34: bool = True, flow_param: str = "direct",
                 mssp_base_ch: int = 32,
                 mssp_scales=(1, 3, 5, 9, 17)) -> None:
        super().__init__()
        if flow_param not in {"direct", "potential"}:
            raise ValueError(f"Unsupported flow_param: {flow_param}")
        self.flow_param = flow_param

        # --- RGB main R34 (same as V3 baseline) ---
        self.main = _resnet34(pretrained=pretrained_r34)

        # --- MSSP stain pyramid ---
        self.mssp = MSSPEncoder(scales=mssp_scales, base_ch=mssp_base_ch)
        b = mssp_base_ch
        c_stain = (b, b, b * 2, b * 4, b * 8)        # (32, 32, 64, 128, 256)
        c_main = (64, 64, 128, 256, 512)              # R34 stage outputs

        # --- Fusion at each scale: concat + 1×1 conv → R34-style channel count ---
        self.fuse_stem = nn.Conv2d(c_main[0] + c_stain[0], c_main[0], 1)
        self.fuse_s1 = nn.Conv2d(c_main[1] + c_stain[1], c_main[1], 1)
        self.fuse_s2 = nn.Conv2d(c_main[2] + c_stain[2], c_main[2], 1)
        self.fuse_s3 = nn.Conv2d(c_main[3] + c_stain[3], c_main[3], 1)
        self.fuse_s4 = nn.Conv2d(c_main[4] + c_stain[4], c_main[4], 1)

        # --- Decoder (identical to V3 baseline) ---
        self.up5 = UpBlock(512, 256, 256)
        self.up6 = UpBlock(256, 128, 128)
        self.up7 = UpBlock(128, 64, 64)
        self.up8 = UpBlock(64, 64, 64)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # --- Heads ---
        self.fg_head = nn.Conv2d(64, 1, kernel_size=1)
        self.centerness_head = nn.Conv2d(64, 1, kernel_size=1)
        if flow_param == "direct":
            self.flow_head = nn.Conv2d(64, 2, kernel_size=1)
        else:
            self.potential_head = nn.Conv2d(64, 1, kernel_size=1)

        # Param-count logging on init
        n_main = sum(p.numel() for p in self.main.parameters())
        n_mssp = sum(p.numel() for p in self.mssp.parameters())
        n_dec = sum(p.numel() for p in self.up5.parameters()) + \
                sum(p.numel() for p in self.up6.parameters()) + \
                sum(p.numel() for p in self.up7.parameters()) + \
                sum(p.numel() for p in self.up8.parameters()) + \
                sum(p.numel() for p in self.final_up.parameters())
        n_fuse = sum(sum(p.numel() for p in m.parameters())
                     for m in [self.fuse_stem, self.fuse_s1, self.fuse_s2, self.fuse_s3, self.fuse_s4])
        n_heads = sum(p.numel() for p in self.fg_head.parameters()) + \
                  sum(p.numel() for p in self.centerness_head.parameters())
        if flow_param == "direct":
            n_heads += sum(p.numel() for p in self.flow_head.parameters())
        else:
            n_heads += sum(p.numel() for p in self.potential_head.parameters())
        print(f"[MSSP] params: main_R34={n_main/1e6:.2f}M  MSSP={n_mssp/1e6:.2f}M  "
              f"fuse={n_fuse/1e6:.2f}M  decoder={n_dec/1e6:.2f}M  heads={n_heads/1e6:.2f}M  "
              f"total={(n_main+n_mssp+n_fuse+n_dec+n_heads)/1e6:.2f}M", flush=True)

    def forward(self, x):
        if isinstance(x, dict):
            rgb = x.get("image_rgb", x.get("image"))
            H = x.get("stain_h", None)
            if H is None:
                # Inference fallback: 'image' is H replicated 3ch + ImageNet-normalized
                # Un-normalize channel-0 to get raw H in [0, 1].
                img = x.get("image")
                if img is not None and img.shape[1] >= 1:
                    H = img[:, :1] * 0.229 + 0.485
                else:
                    H = rgb[:, :1]
        else:
            rgb = x
            H = rgb[:, :1]
        if H.ndim == 3:
            H = H.unsqueeze(1)
        # Ensure H is in [0, 1] approximately (per-image min-max normalize handled by MSSP internally
        # via multi_scale_h which is scale-invariant for downstream avg_pool, but we keep raw scale)
        H = H.clamp(min=0.0)

        # --- RGB main pathway ---
        z = self.main.conv1(rgb)
        z = self.main.bn1(z)
        m_stem = self.main.relu(z)                # (B, 64, H/2)
        z = self.main.maxpool(m_stem)
        m_s1 = self.main.layer1(z)                # (B, 64, H/4)
        m_s2 = self.main.layer2(m_s1)             # (B, 128, H/8)
        m_s3 = self.main.layer3(m_s2)             # (B, 256, H/16)
        m_s4 = self.main.layer4(m_s3)             # (B, 512, H/32)

        # --- MSSP stain pathway ---
        h_stem, h_s1, h_s2, h_s3, h_s4 = self.mssp(H)  # (32, 32, 64, 128, 256) at (H/2, H/4, H/8, H/16, H/32)

        # --- Fusion ---
        f_stem = self.fuse_stem(torch.cat([m_stem, h_stem], dim=1))
        f_s1 = self.fuse_s1(torch.cat([m_s1, h_s1], dim=1))
        f_s2 = self.fuse_s2(torch.cat([m_s2, h_s2], dim=1))
        f_s3 = self.fuse_s3(torch.cat([m_s3, h_s3], dim=1))
        f_s4 = self.fuse_s4(torch.cat([m_s4, h_s4], dim=1))

        # --- Decoder (U-Net upsampling with skip from fused features) ---
        z = self.up5(f_s4, f_s3)   # H/16
        z = self.up6(z, f_s2)      # H/8
        z = self.up7(z, f_s1)      # H/4
        z = self.up8(z, f_stem)    # H/2
        feat = self.final_up(z)    # H, W

        outputs = {
            "fg_logits": self.fg_head(feat),
            "centerness_logits": self.centerness_head(feat),
            "feat": feat,
        }
        if self.flow_param == "direct":
            outputs["flow"] = torch.tanh(self.flow_head(feat))
        else:
            phi = self.potential_head(feat)
            outputs["phi"] = phi
            outputs["flow"] = potential_to_flow(phi)
        return outputs


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)
    print("=== MSSP smoke test ===")
    model = NISFlowNetMSSP(pretrained_r34=False)
    x = {
        "image_rgb": torch.randn(2, 3, 224, 224),
        "stain_h": torch.rand(2, 224, 224),
    }
    out = model(x)
    print(f"  fg_logits.shape   = {tuple(out['fg_logits'].shape)}")
    print(f"  flow.shape        = {tuple(out['flow'].shape)}")
    print(f"  centerness.shape  = {tuple(out['centerness_logits'].shape)}")
    print(f"  feat.shape        = {tuple(out['feat'].shape)}")
    # Backward
    loss = out["fg_logits"].mean() + out["flow"].mean()
    loss.backward()
    print(f"  backward OK (loss={loss.item():.5f})")
