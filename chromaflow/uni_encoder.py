"""UNI encoder + matched U-Net decoder for nuclei segmentation.

This file is the addition to model.py. It adds NISFlowNetUNI:
  - Frozen UNI v1 ViT-L/16 encoder (Mahmood Lab, leakage-free of TCGA)
  - Pure-upsample decoder from /16 features
  - Same heads as NISFlowNet (fg + flow direct/potential)
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


# Re-use NISFlowNet helpers if we are appending; for self-contained snippet keep here.
def potential_to_flow(phi: torch.Tensor) -> torch.Tensor:
    """v = -grad(phi) via central differences; returns (B, 2, H, W) dx,dy."""
    phi_pad = F.pad(phi, (1, 1, 1, 1), mode="replicate")
    dphi_dx = (phi_pad[:, :, 1:-1, 2:] - phi_pad[:, :, 1:-1, :-2]) / 2.0
    dphi_dy = (phi_pad[:, :, 2:, 1:-1] - phi_pad[:, :, :-2, 1:-1]) / 2.0
    return torch.cat([-dphi_dx, -dphi_dy], dim=1)


class UNIEncoder(nn.Module):
    """Frozen UNI v1 (ViT-L/16, 303M params) feature extractor.

    Input:  (B, 3, H, W),  H, W divisible by 16
    Output: (B, 1024, H/16, W/16) feature map at /16 resolution
    """

    def __init__(self, weights_path: str, img_size: int = 224, freeze: bool = True):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise RuntimeError("timm not installed. pip install timm") from e

        self.backbone = timm.create_model(
            "vit_large_patch16_224",
            img_size=img_size,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=True,
        )
        if weights_path:
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
            # UNI ckpt is a flat dict of param names matching timm ViT
            missing, unexpected = self.backbone.load_state_dict(ckpt, strict=False)
            if missing:
                print(f"[UNI] missing keys ({len(missing)}): {missing[:5]}{' ...' if len(missing)>5 else ''}")
            if unexpected:
                print(f"[UNI] unexpected keys ({len(unexpected)}): {unexpected[:5]}{' ...' if len(unexpected)>5 else ''}")

        self.embed_dim = 1024
        self.patch_size = 16
        self.frozen = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def train(self, mode: bool = True):
        # Keep BN/dropout in eval if frozen, regardless of parent .train()
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    @torch.no_grad()
    def _forward_frozen(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        Hp, Wp = H // self.patch_size, W // self.patch_size

        if self.frozen:
            tokens = self._forward_frozen(x)
        else:
            tokens = self.backbone.forward_features(x)

        # tokens shape: (B, num_tokens, 1024)
        # timm ViT typically has CLS at index 0 → strip it
        n_tokens = tokens.shape[1]
        if n_tokens == Hp * Wp + 1:
            tokens = tokens[:, 1:]
        elif n_tokens != Hp * Wp:
            # Fallback: also strip if other registers (unlikely for UNI v1)
            tokens = tokens[:, -(Hp * Wp):]

        # (B, Hp*Wp, 1024) → (B, 1024, Hp, Wp)
        B = tokens.shape[0]
        feat = tokens.transpose(1, 2).contiguous().view(B, self.embed_dim, Hp, Wp)
        return feat


class _UpDecodeBlock(nn.Module):
    """Single decoder up-step: conv-transpose 2x + 2 conv-bn-relu."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2)
        self.body = nn.Sequential(
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(self.up(x))


class NISFlowNetUNI(nn.Module):
    """v3-style nuclei student with frozen UNI ViT-L/16 encoder.

    Decoder: /16 → /8 → /4 → /2 → /1 via pure upsample (no encoder skip — UNI
    only gives single-scale /16 features). Trainable params ≈ ~1M.
    """

    def __init__(self, uni_path: str, flow_param: str = "direct", img_size: int = 224):
        super().__init__()
        if flow_param not in {"direct", "potential"}:
            raise ValueError(f"Unsupported flow_param: {flow_param}")
        self.flow_param = flow_param

        self.encoder = UNIEncoder(uni_path, img_size=img_size, freeze=True)

        # Project 1024 -> 256
        self.proj = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # Decoder up-blocks: /16 -> /8 -> /4 -> /2 -> /1
        self.up3 = _UpDecodeBlock(256, 128)
        self.up2 = _UpDecodeBlock(128, 64)
        self.up1 = _UpDecodeBlock(64, 64)
        self.up0 = _UpDecodeBlock(64, 64)

        self.fg_head = nn.Conv2d(64, 1, 1)
        self.centerness_head = nn.Conv2d(64, 1, 1)
        if flow_param == "direct":
            self.flow_head = nn.Conv2d(64, 2, 1)
        else:
            self.potential_head = nn.Conv2d(64, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        H, W = x.shape[-2:]
        feat16 = self.encoder(x)         # (B, 1024, H/16, W/16)
        feat = self.proj(feat16)          # (B, 256, H/16, W/16)
        feat = self.up3(feat)             # /8
        feat = self.up2(feat)             # /4
        feat = self.up1(feat)             # /2
        feat = self.up0(feat)             # /1
        if feat.shape[-2:] != (H, W):
            feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)

        fg_logits = self.fg_head(feat)
        centerness_logits = self.centerness_head(feat)
        outputs = {"fg_logits": fg_logits, "centerness_logits": centerness_logits, "feat": feat}
        if self.flow_param == "direct":
            outputs["flow"] = torch.tanh(self.flow_head(feat))
        else:
            phi = self.potential_head(feat)
            outputs["phi"] = phi
            outputs["flow"] = potential_to_flow(phi)
        return outputs


if __name__ == "__main__":
    # Smoke test (skips weight loading if path not provided)
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else ""
    model = NISFlowNetUNI(uni_path=path, flow_param="direct")
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print("=== UNI smoke test ===")
    print("fg_logits:", out["fg_logits"].shape)
    print("flow:", out["flow"].shape)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"trainable params: {n_trainable/1e6:.2f} M")
    print(f"frozen params:    {n_frozen/1e6:.2f} M")
