"""DINOv2 ViT-L/14 ImageNet-pretrained encoder + matched decoder for nuclei seg.

P1a (D2 pilot) — leakage-free DINOv2-ImageNet check vs UNI pathology-pretrained.
DINOv2 ViT-L is identical architecture/size to UNI ViT-L/16 except:
  - patch size 14 (vs UNI 16) → /14 grid
  - pretrained on ImageNet-LVD-142M (NO pathology data)
  - default training res 518; we apply at 224 with interpolated pos embed.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from chromaflow.uni_encoder import _UpDecodeBlock, potential_to_flow


class DINOv2Encoder(nn.Module):
    """Frozen DINOv2 ViT-L/14 from timm, ImageNet-pretrained.

    Input: (B, 3, H, W),  H, W divisible by 14
    Output: (B, 1024, H/14, W/14)
    """
    def __init__(self, img_size: int = 224, freeze: bool = True,
                 weights_path: str | None = None):
        super().__init__()
        import timm
        # timm dinov2 model uses LayerScale (init_values from ckpt). We need patch=14.
        # Create backbone at FB's native 518 to match ckpt pos_embed; dynamic_img_size
        # then handles 224 input at forward time via on-the-fly interpolation.
        self.backbone = timm.create_model(
            "vit_large_patch14_dinov2",
            pretrained=False,
            img_size=518,
            num_classes=0,
            dynamic_img_size=True,
        )
        if weights_path:
            sd = torch.load(weights_path, map_location="cpu", weights_only=True)
            missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
            if missing: print(f"[DINOv2] missing {len(missing)}: {missing[:5]}{' ...' if len(missing)>5 else ''}")
            if unexpected: print(f"[DINOv2] unexpected {len(unexpected)}: {unexpected[:5]}{' ...' if len(unexpected)>5 else ''}")
        self.embed_dim = 1024
        self.patch_size = 14
        self.frozen = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    @torch.no_grad()
    def _forward_frozen(self, x):
        return self.backbone.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        Hp, Wp = H // self.patch_size, W // self.patch_size
        tokens = self._forward_frozen(x) if self.frozen else self.backbone.forward_features(x)
        # DINOv2 has 1 CLS + (Hp*Wp) patch tokens, or 1 CLS + 4 register + patch
        n = tokens.shape[1]
        if n == Hp * Wp + 1:
            tokens = tokens[:, 1:]
        elif n > Hp * Wp:
            tokens = tokens[:, -(Hp * Wp):]
        B = tokens.shape[0]
        feat = tokens.transpose(1, 2).contiguous().view(B, self.embed_dim, Hp, Wp)
        return feat


class NISFlowNetDINOv2(nn.Module):
    """Same architecture as NISFlowNetUNI but with DINOv2 ImageNet ViT-L/14 frozen encoder.

    Decoder: /14 → /7 → /3.5 → ... we just 4× upsample then bilinear to target.
    """
    def __init__(self, flow_param: str = "direct", img_size: int = 224):
        super().__init__()
        if flow_param not in {"direct", "potential"}:
            raise ValueError(flow_param)
        self.flow_param = flow_param
        self.encoder = DINOv2Encoder(img_size=img_size, freeze=True)
        self.proj = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.up3 = _UpDecodeBlock(256, 128)
        self.up2 = _UpDecodeBlock(128, 64)
        self.up1 = _UpDecodeBlock(64, 64)
        self.up0 = _UpDecodeBlock(64, 64)
        self.fg_head = nn.Conv2d(64, 1, 1)
        self.centerness_head = nn.Conv2d(64, 1, 1)
        self.boundary_head = nn.Conv2d(64, 1, 1)
        if flow_param == "direct":
            self.flow_head = nn.Conv2d(64, 2, 1)
        else:
            self.potential_head = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        H, W = x.shape[-2:]
        feat = self.proj(self.encoder(x))
        feat = self.up0(self.up1(self.up2(self.up3(feat))))
        if feat.shape[-2:] != (H, W):
            feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)
        outs = {
            "fg_logits": self.fg_head(feat),
            "centerness_logits": self.centerness_head(feat),
            "boundary_logits": self.boundary_head(feat),
            "feat": feat,
        }
        if self.flow_param == "direct":
            outs["flow"] = torch.tanh(self.flow_head(feat))
        else:
            phi = self.potential_head(feat)
            outs["phi"] = phi
            outs["flow"] = potential_to_flow(phi)
        return outs


if __name__ == "__main__":
    m = NISFlowNetDINOv2(flow_param="direct")
    x = torch.randn(2, 3, 224, 224)
    out = m(x)
    print("fg:", out["fg_logits"].shape, "flow:", out["flow"].shape)
    n_t = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_f = sum(p.numel() for p in m.parameters() if not p.requires_grad)
    print(f"trainable: {n_t/1e6:.2f} M  frozen: {n_f/1e6:.2f} M")
