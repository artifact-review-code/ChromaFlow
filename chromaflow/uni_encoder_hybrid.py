"""Hybrid encoder: R34 shallow stages + frozen UNI deep features + U-Net decoder with skips.

R34 conv1+bn+layer1+layer2 provides high-resolution local features at /2, /4, /8.
UNI ViT-L/16 (frozen) provides deep semantic features at /16.
Decoder uses skips for fine spatial detail at boundary.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def potential_to_flow(phi: torch.Tensor) -> torch.Tensor:
    phi_pad = F.pad(phi, (1, 1, 1, 1), mode="replicate")
    dphi_dx = (phi_pad[:, :, 1:-1, 2:] - phi_pad[:, :, 1:-1, :-2]) / 2.0
    dphi_dy = (phi_pad[:, :, 2:, 1:-1] - phi_pad[:, :, :-2, 1:-1]) / 2.0
    return torch.cat([-dphi_dx, -dphi_dy], dim=1)


def _resnet34_shallow(pretrained: bool = True) -> nn.Module:
    from torchvision import models
    if pretrained:
        try:
            weights = models.ResNet34_Weights.IMAGENET1K_V1
            rn = models.resnet34(weights=weights)
        except AttributeError:
            rn = models.resnet34(pretrained=True)
    else:
        rn = models.resnet34(weights=None)
    return rn


class UNIEncoder(nn.Module):
    """Same as uni_encoder.UNIEncoder — frozen UNI ViT-L/16, returns /16 features."""
    def __init__(self, weights_path: str, img_size: int = 224, freeze: bool = True):
        super().__init__()
        import timm
        self.backbone = timm.create_model(
            "vit_large_patch16_224",
            img_size=img_size, patch_size=16, init_values=1e-5,
            num_classes=0, dynamic_img_size=True,
        )
        if weights_path:
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(ckpt, strict=False)
        self.embed_dim = 1024
        self.patch_size = 16
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        Hp, Wp = H // self.patch_size, W // self.patch_size
        if self.frozen:
            with torch.no_grad():
                tokens = self.backbone.forward_features(x)
        else:
            tokens = self.backbone.forward_features(x)
        n_tokens = tokens.shape[1]
        if n_tokens == Hp * Wp + 1:
            tokens = tokens[:, 1:]
        elif n_tokens != Hp * Wp:
            tokens = tokens[:, -(Hp * Wp):]
        B = tokens.shape[0]
        return tokens.transpose(1, 2).contiguous().view(B, self.embed_dim, Hp, Wp)


class _DecodeBlock(nn.Module):
    """Upsample + concat skip + 2x (conv-bn-relu)."""
    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2)
        self.body = nn.Sequential(
            nn.Conv2d(c_out + c_skip, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.body(torch.cat([x, skip], dim=1))


class NISFlowNetUNIHybrid(nn.Module):
    """Hybrid encoder: trainable R34 shallow + frozen UNI deep + multi-scale U-Net decoder.

    Encoder structure:
        x (B,3,H,W)
        → R34.conv1+bn+relu  → s0  (B, 64, H/2, W/2)
        → R34.maxpool+layer1 → s1  (B, 64, H/4, W/4)
        → R34.layer2         → s2  (B,128, H/8, W/8)
        → UNI ViT-L/16       → s3  (B,1024, H/16, W/16)  [frozen]

    Decoder structure (with skip concats):
        s3 → proj → /16 (256)
        up3: /16 → /8, concat s2 → 128
        up2: /8 → /4, concat s1 → 64
        up1: /4 → /2, concat s0 → 64
        up0: /2 → /1 → 64
        heads
    """

    def __init__(self, uni_path: str, flow_param: str = "direct",
                  img_size: int = 224, pretrained_r34: bool = True):
        super().__init__()
        if flow_param not in {"direct", "potential"}:
            raise ValueError(f"Unsupported flow_param: {flow_param}")
        self.flow_param = flow_param

        # Shallow R34 (trainable)
        rn = _resnet34_shallow(pretrained=pretrained_r34)
        self.r34_stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu)  # /2  c=64
        self.r34_maxpool = rn.maxpool
        self.r34_layer1 = rn.layer1   # /4  c=64
        self.r34_layer2 = rn.layer2   # /8  c=128

        # Deep UNI (frozen)
        self.uni = UNIEncoder(uni_path, img_size=img_size, freeze=True)

        # Decoder
        self.proj_uni = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.up3 = _DecodeBlock(c_in=256, c_skip=128, c_out=128)   # /16 → /8
        self.up2 = _DecodeBlock(c_in=128, c_skip=64,  c_out=64)    # /8  → /4
        self.up1 = _DecodeBlock(c_in=64,  c_skip=64,  c_out=64)    # /4  → /2
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Heads (boundary_head is used only when L_aff is enabled for Stage-2; kept always for ckpt compat)
        self.fg_head = nn.Conv2d(64, 1, 1)
        self.centerness_head = nn.Conv2d(64, 1, 1)
        self.boundary_head = nn.Conv2d(64, 1, 1)
        if flow_param == "direct":
            self.flow_head = nn.Conv2d(64, 2, 1)
        else:
            self.potential_head = nn.Conv2d(64, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        H, W = x.shape[-2:]
        s0 = self.r34_stem(x)
        s1 = self.r34_layer1(self.r34_maxpool(s0))
        s2 = self.r34_layer2(s1)
        s3 = self.uni(x)

        feat = self.proj_uni(s3)
        feat = self.up3(feat, s2)
        feat = self.up2(feat, s1)
        feat = self.up1(feat, s0)
        feat = self.final_up(feat)
        if feat.shape[-2:] != (H, W):
            feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)

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
            outputs["flow"] = potential_to_flow(phi)
        return outputs



class _ChannelLayerNorm(nn.Module):
    """Per-pixel channel-wise LayerNorm for (B, C, H, W) tensors.

    Equivalent to permuting to (B, H, W, C), applying nn.LayerNorm(C), and permuting back.
    Used in NISFlowNetUNIHybridLN to balance R34-shallow and UNI-deep feature magnitudes
    before concat fusion (fix for D3 / Part 5 bug: 1x1 fusion conv was dominated by UNI's
    1024-d info-dense features, causing R34 shallow channels to be suppressed and
    small-nucleus recall to drop to UNI-single levels).
    """
    def __init__(self, num_channels: int):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class _DecodeBlockLN(nn.Module):
    """_DecodeBlock with ChannelLayerNorm applied to both streams before concat."""
    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2)
        self.ln_x = _ChannelLayerNorm(c_out)
        self.ln_skip = _ChannelLayerNorm(c_skip)
        self.body = nn.Sequential(
            nn.Conv2d(c_out + c_skip, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.ln_x(x)
        skip = self.ln_skip(skip)
        return self.body(torch.cat([x, skip], dim=1))


class NISFlowNetUNIHybridLN(nn.Module):
    """Variant of NISFlowNetUNIHybrid with ChannelLayerNorm in all _DecodeBlock fusions.

    Same encoder, same decoder topology, same heads. Only difference: each fusion concat
    is preceded by per-pixel channel-wise LayerNorm on both streams so their relative
    magnitudes are balanced before the 1x1 conv inside _DecodeBlockLN.body.

    Plan 1a from problem-list. Target: small-nucleus recall (<12px) recover from 0.145
    (hybrid baseline) toward R34's 0.229. Expected MO AJI gain +0.008~0.015.
    """
    def __init__(self, uni_path: str, flow_param: str = "direct",
                 img_size: int = 224, pretrained_r34: bool = True):
        super().__init__()
        if flow_param not in {"direct", "potential"}:
            raise ValueError(f"Unsupported flow_param: {flow_param}")
        self.flow_param = flow_param

        rn = _resnet34_shallow(pretrained=pretrained_r34)
        self.r34_stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu)
        self.r34_maxpool = rn.maxpool
        self.r34_layer1 = rn.layer1
        self.r34_layer2 = rn.layer2

        self.uni = UNIEncoder(uni_path, img_size=img_size, freeze=True)

        # UNI proj also gets a channel LN at output so the very first feed into up3 is balanced.
        self.proj_uni = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.up3 = _DecodeBlockLN(c_in=256, c_skip=128, c_out=128)
        self.up2 = _DecodeBlockLN(c_in=128, c_skip=64,  c_out=64)
        self.up1 = _DecodeBlockLN(c_in=64,  c_skip=64,  c_out=64)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.fg_head = nn.Conv2d(64, 1, 1)
        self.centerness_head = nn.Conv2d(64, 1, 1)
        if flow_param == "direct":
            self.flow_head = nn.Conv2d(64, 2, 1)
        else:
            self.potential_head = nn.Conv2d(64, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        H, W = x.shape[-2:]
        s0 = self.r34_stem(x)
        s1 = self.r34_layer1(self.r34_maxpool(s0))
        s2 = self.r34_layer2(s1)
        s3 = self.uni(x)

        feat = self.proj_uni(s3)
        feat = self.up3(feat, s2)
        feat = self.up2(feat, s1)
        feat = self.up1(feat, s0)
        feat = self.final_up(feat)
        if feat.shape[-2:] != (H, W):
            feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)

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
            outputs["flow"] = potential_to_flow(phi)
        return outputs


# ------------------------------------------------------------------
# Flow-Center Interleaved Guidance (IGA) — bidirectional cross-attention
# between flow features and center features. Placed after the shared
# decoder; attention runs at a small grid (default /8 of input) for
# memory, then broadcasts back via bilinear upsampling.
# ------------------------------------------------------------------
class _CrossAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        assert dim % n_heads == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor) -> torch.Tensor:
        # q_in, kv_in: (B, N, C)
        B, N, C = q_in.shape
        Nk = kv_in.shape[1]
        q = self.q_proj(q_in).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(kv_in).view(B, Nk, 2, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, C)
        return self.out_norm(self.out_proj(out))


class FlowCenterIGA(nn.Module):
    """Bidirectional Q->K,V attention between flow and center streams.
    Operates at a downsampled grid (default 28x28 for 224-input) then
    upsamples deltas back to full resolution.
    """

    def __init__(self, dim: int = 64, n_heads: int = 4, attn_grid: int = 28):
        super().__init__()
        self.attn_grid = attn_grid
        self.flow_proj = nn.Conv2d(dim, dim, 1)
        self.center_proj = nn.Conv2d(dim, dim, 1)
        self.norm_flow = nn.LayerNorm(dim)
        self.norm_center = nn.LayerNorm(dim)
        self.center_to_flow = _CrossAttention(dim, n_heads)
        self.flow_to_center = _CrossAttention(dim, n_heads)

    def forward(self, feat: torch.Tensor):
        # feat: (B, C, H, W)
        B, C, H, W = feat.shape
        feat_flow = self.flow_proj(feat)
        feat_center = self.center_proj(feat)
        gs = self.attn_grid
        flow_lo = F.adaptive_avg_pool2d(feat_flow, gs)
        center_lo = F.adaptive_avg_pool2d(feat_center, gs)
        flow_tok = self.norm_flow(flow_lo.flatten(2).transpose(1, 2))   # (B, N, C)
        center_tok = self.norm_center(center_lo.flatten(2).transpose(1, 2))
        flow_tok2 = flow_tok + self.center_to_flow(flow_tok, center_tok)
        center_tok2 = center_tok + self.flow_to_center(center_tok, flow_tok)
        flow_lo2 = flow_tok2.transpose(1, 2).reshape(B, C, gs, gs)
        center_lo2 = center_tok2.transpose(1, 2).reshape(B, C, gs, gs)
        flow_up = F.interpolate(flow_lo2, size=(H, W), mode="bilinear", align_corners=False)
        center_up = F.interpolate(center_lo2, size=(H, W), mode="bilinear", align_corners=False)
        # Residual: original projected feat + cross-attention contribution
        return feat_flow + flow_up, feat_center + center_up


class NISFlowNetUNIHybridIGA(nn.Module):
    """Hybrid encoder + Flow-Center IGA module + dual-stream heads.

    Same backbone as NISFlowNetUNIHybrid; after the shared decoder, the
    64-ch feat is split via two 1x1 convs into flow / center streams that
    exchange info via bidirectional cross-attention at /8 grid, then drive
    specialized prediction heads. fg / boundary use the average of the two
    streams; flow uses feat_flow only; centerness uses feat_center only.
    """

    def __init__(self, uni_path: str, flow_param: str = "direct",
                  img_size: int = 224, pretrained_r34: bool = True,
                  iga_attn_grid: int = 28, iga_n_heads: int = 4):
        super().__init__()
        if flow_param not in {"direct", "potential"}:
            raise ValueError(f"Unsupported flow_param: {flow_param}")
        self.flow_param = flow_param

        rn = _resnet34_shallow(pretrained=pretrained_r34)
        self.r34_stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu)
        self.r34_maxpool = rn.maxpool
        self.r34_layer1 = rn.layer1
        self.r34_layer2 = rn.layer2

        self.uni = UNIEncoder(uni_path, img_size=img_size, freeze=True)

        self.proj_uni = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.up3 = _DecodeBlock(c_in=256, c_skip=128, c_out=128)
        self.up2 = _DecodeBlock(c_in=128, c_skip=64,  c_out=64)
        self.up1 = _DecodeBlock(c_in=64,  c_skip=64,  c_out=64)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # IGA module
        self.iga = FlowCenterIGA(dim=64, n_heads=iga_n_heads, attn_grid=iga_attn_grid)

        # Heads — fg / boundary off the averaged stream; flow off flow stream; center off center stream
        self.fg_head = nn.Conv2d(64, 1, 1)
        self.centerness_head = nn.Conv2d(64, 1, 1)
        self.boundary_head = nn.Conv2d(64, 1, 1)
        if flow_param == "direct":
            self.flow_head = nn.Conv2d(64, 2, 1)
        else:
            self.potential_head = nn.Conv2d(64, 1, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        H, W = x.shape[-2:]
        s0 = self.r34_stem(x)
        s1 = self.r34_layer1(self.r34_maxpool(s0))
        s2 = self.r34_layer2(s1)
        s3 = self.uni(x)

        feat = self.proj_uni(s3)
        feat = self.up3(feat, s2)
        feat = self.up2(feat, s1)
        feat = self.up1(feat, s0)
        feat = self.final_up(feat)
        if feat.shape[-2:] != (H, W):
            feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)

        feat_flow, feat_center = self.iga(feat)
        feat_shared = 0.5 * (feat_flow + feat_center)

        fg_logits = self.fg_head(feat_shared)
        centerness_logits = self.centerness_head(feat_center)
        boundary_logits = self.boundary_head(feat_shared)
        outputs = {
            "fg_logits": fg_logits,
            "centerness_logits": centerness_logits,
            "boundary_logits": boundary_logits,
            "feat": feat_shared,
        }
        if self.flow_param == "direct":
            outputs["flow"] = torch.tanh(self.flow_head(feat_flow))
        else:
            phi = self.potential_head(feat_flow)
            outputs["phi"] = phi
            outputs["flow"] = potential_to_flow(phi)
        return outputs


# ============================================================================
# H-PS-C2F-Net: Hematoxylin-local + RGB-UNI semantic + Point-Sink Center-to-Flow
# ============================================================================
# - H-3ch goes to R34 shallow (local nuclear morphology)
# - RGB-3ch goes to frozen UNI ViT-L/16 (pathology semantic prior)
# - Standard hybrid decoder fuses both
# - Optional PS-C2F module: divergence-residual gated center-to-flow refinement
#
# The model accepts a dict {"image": h_3ch, "image_rgb": rgb_3ch} OR a 6-channel
# tensor (h_3ch concat rgb_3ch on channel dim). The dict form is preferred.


def _divergence(flow: torch.Tensor) -> torch.Tensor:
    """Central-difference divergence of a (B, 2, H, W) flow field with dx,dy ordering.
    Returns (B, 1, H, W). Same convention as losses.divergence.
    """
    vx = flow[:, 0:1]
    vy = flow[:, 1:2]
    dvx_dx = torch.zeros_like(vx)
    dvy_dy = torch.zeros_like(vy)
    dvx_dx[:, :, :, 1:-1] = 0.5 * (vx[:, :, :, 2:] - vx[:, :, :, :-2])
    dvx_dx[:, :, :, 0] = vx[:, :, :, 1] - vx[:, :, :, 0]
    dvx_dx[:, :, :, -1] = vx[:, :, :, -1] - vx[:, :, :, -2]
    dvy_dy[:, :, 1:-1, :] = 0.5 * (vy[:, :, 2:, :] - vy[:, :, :-2, :])
    dvy_dy[:, :, 0, :] = vy[:, :, 1, :] - vy[:, :, 0, :]
    dvy_dy[:, :, -1, :] = vy[:, :, -1, :] - vy[:, :, -2, :]
    return dvx_dx + dvy_dy


class PSC2FModule(nn.Module):
    """Point-Sink Center-to-Flow refinement.

    Inputs:
        feat:    (B, C, H, W) shared decoder feature
        v0:      (B, 2, H, W) initial flow (post-tanh, in [-1,1])
        c:       (B, 1, H, W) centerness logits
    Returns:
        v_refined: (B, 2, H, W)
        aux: dict with 'r_div' (residual map), 'gate', 'delta_v' for diagnostics

    Mechanism:
        sg_c = stop_grad(sigmoid(c))                       (point-sink proxy)
        r_div = |div(v0) + sg_c|                            (residual: where v0
                                                              disagrees with predicted sinks)
        Z = concat(feat, sg(c), r_div)
        gate g = sigmoid(Conv3x3 -> 1ch)
        delta_v = Conv3x3 -> 2ch
        v = tanh(v0 + gamma * g * delta_v)                  (gamma init=0 → v == v0 at start)
    """

    def __init__(self, feat_channels: int = 64, hidden: int = 64):
        super().__init__()
        in_ch = feat_channels + 1 + 1  # feat + sg(c) + r_div
        self.refine = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.gate_head = nn.Conv2d(hidden, 1, 1)
        self.dv_head = nn.Conv2d(hidden, 2, 1)
        # gamma init = 0 so module starts as identity (v == v0)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, feat: torch.Tensor, v0: torch.Tensor, c_logits: torch.Tensor):
        with torch.no_grad():
            sg_c = torch.sigmoid(c_logits).detach()  # stop-gradient on center
        div_v0 = _divergence(v0)  # (B, 1, H, W)
        r_div = (div_v0 + sg_c).abs()
        z = torch.cat([feat, sg_c, r_div], dim=1)
        h = self.refine(z)
        gate = torch.sigmoid(self.gate_head(h))
        # delta_v through a tanh so it's bounded in [-1, 1]; the gate * gamma further
        # shrinks it. At gamma=0, v = v0 exactly (preserves R1.1 baseline at init).
        delta = torch.tanh(self.dv_head(h))
        v = v0 + self.gamma * gate * delta
        # Clamp to [-1, 1] to keep flow in the standard tanh-bounded range Cellpose expects.
        v = v.clamp(-1.0, 1.0)
        return v, {"r_div": r_div, "gate": gate, "delta_v": delta}


class NISFlowNetUNIHybridH(nn.Module):
    """H-UNI hybrid: H-3ch -> R34 shallow stages; RGB-3ch -> frozen UNI ViT-L/16.

    Same decoder topology as NISFlowNetUNIHybrid. Forward expects a dict batch with
    {"image": h_3ch, "image_rgb": rgb_3ch}, OR a single tensor (treated as h_3ch and
    fed to UNI as well — back-compat mode for legacy training scripts).
    """

    def __init__(self, uni_path: str, flow_param: str = "direct",
                  img_size: int = 224, pretrained_r34: bool = True):
        super().__init__()
        if flow_param not in {"direct", "potential"}:
            raise ValueError(f"Unsupported flow_param: {flow_param}")
        self.flow_param = flow_param

        rn = _resnet34_shallow(pretrained=pretrained_r34)
        self.r34_stem = nn.Sequential(rn.conv1, rn.bn1, rn.relu)
        self.r34_maxpool = rn.maxpool
        self.r34_layer1 = rn.layer1
        self.r34_layer2 = rn.layer2

        self.uni = UNIEncoder(uni_path, img_size=img_size, freeze=True)

        self.proj_uni = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.up3 = _DecodeBlock(c_in=256, c_skip=128, c_out=128)
        self.up2 = _DecodeBlock(c_in=128, c_skip=64,  c_out=64)
        self.up1 = _DecodeBlock(c_in=64,  c_skip=64,  c_out=64)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.fg_head = nn.Conv2d(64, 1, 1)
        self.centerness_head = nn.Conv2d(64, 1, 1)
        self.boundary_head = nn.Conv2d(64, 1, 1)
        if flow_param == "direct":
            self.flow_head = nn.Conv2d(64, 2, 1)
        else:
            self.potential_head = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        """x: either Tensor (legacy) or dict with 'image' (H-3ch) and 'image_rgb' (RGB-3ch)."""
        if isinstance(x, dict):
            x_h = x["image"]
            x_rgb = x.get("image_rgb", x["image"])
        else:
            x_h = x
            x_rgb = x
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
            outputs["flow"] = potential_to_flow(phi)
        return outputs


class NISFlowNetUNIHybridHPSC2F(NISFlowNetUNIHybridH):
    """H-PS-C2F-Net: H-UNI hybrid + PS-C2F flow refinement module.

    Wraps NISFlowNetUNIHybridH and adds a PS-C2F module that refines the initial flow
    using divergence-residual gated correction. centerness is stop-gradient-protected.
    Outputs:
        flow_v0:           initial flow from base hybrid (for diagnostics / L_flow on v0)
        flow:               final refined flow (= v after PS-C2F)
        r_div, gate, delta_v: aux diagnostics
    """

    def __init__(self, uni_path: str, flow_param: str = "direct",
                  img_size: int = 224, pretrained_r34: bool = True):
        super().__init__(uni_path=uni_path, flow_param=flow_param,
                          img_size=img_size, pretrained_r34=pretrained_r34)
        self.psc2f = PSC2FModule(feat_channels=64, hidden=64)

    def forward(self, x):
        out = super().forward(x)
        v0 = out["flow"]
        v_refined, aux = self.psc2f(out["feat"], v0, out["centerness_logits"])
        out["flow_v0"] = v0
        out["flow"] = v_refined
        out["psc2f_r_div"] = aux["r_div"]
        out["psc2f_gate"] = aux["gate"]
        out["psc2f_delta_v"] = aux["delta_v"]
        return out


# ============================================================================
# HUF-Net-V1: H-UNI hybrid + contrast projection head for PSIC loss
# ============================================================================
class NISFlowNetUNIHybridHPSIC(NISFlowNetUNIHybridH):
    """H-UNI hybrid + a contrast projection head producing L2-normalized 64-dim
    pixel embeddings (used by losses.psic_loss). All other heads / decoder identical
    to NISFlowNetUNIHybridH so ckpts are mostly compatible (strict=False loading).
    """

    def __init__(self, uni_path: str, flow_param: str = "direct",
                  img_size: int = 224, pretrained_r34: bool = True,
                  contrast_dim: int = 64, contrast_hidden: int = 64):
        super().__init__(uni_path=uni_path, flow_param=flow_param,
                          img_size=img_size, pretrained_r34=pretrained_r34)
        # Contrast projection: 64-ch decoder feature -> hidden -> contrast_dim, then L2 norm
        self.contrast_proj = nn.Sequential(
            nn.Conv2d(64, contrast_hidden, 1, bias=False),
            nn.BatchNorm2d(contrast_hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(contrast_hidden, contrast_dim, 1),
        )

    def forward(self, x):
        out = super().forward(x)
        z = self.contrast_proj(out["feat"])
        z = F.normalize(z, dim=1)
        out["contrast_z"] = z
        return out


# ============================================================================
# HU-BIM: H-UNI Boundary Inference Module
# ============================================================================
class HUBoundaryInferenceModule(nn.Module):
    """Joint H+UNI boundary inference module (v2: no scalar gate).

    v1 had `gamma_inject` scalar zero-init which collapsed inject path to
    identity-bypass. v2 removes the scalar; injection is direct via a
    zero-initialized 1x1 conv. Combined with no-WD on inject_proj weights
    (handled at optimizer level by matching 'bim.' name prefix), the inject
    weights can grow from zero without being dragged back.
    """

    def __init__(self, h_ch: int = 128, uni_ch: int = 256, mid_ch: int = 64,
                 inject_ch: int = 128):
        super().__init__()
        self.h_proj = nn.Conv2d(h_ch, mid_ch, kernel_size=1, bias=False)
        self.u_proj = nn.Conv2d(uni_ch, mid_ch, kernel_size=1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * mid_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(16, mid_ch), mid_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(16, mid_ch), mid_ch),
            nn.SiLU(inplace=True),
        )
        # auxiliary boundary head (1-channel logit @ 1/8)
        self.b_head = nn.Conv2d(mid_ch, 1, kernel_size=1, bias=True)
        # injection: project mid_ch → inject_ch (decoder feat depth).
        # zero-init weight so initial inject = 0 (identity), but no scalar gate.
        # Gradient flows directly through the conv weights without a scalar
        # bottleneck that can collapse to 0.
        self.inject_proj = nn.Conv2d(mid_ch, inject_ch, kernel_size=1, bias=True)
        nn.init.zeros_(self.inject_proj.weight)
        nn.init.zeros_(self.inject_proj.bias)

    def forward(self, s2_h, f_u):
        if f_u.shape[-2:] != s2_h.shape[-2:]:
            f_u = F.interpolate(f_u, size=s2_h.shape[-2:],
                                  mode="bilinear", align_corners=False)
        s = self.h_proj(s2_h)
        u = self.u_proj(f_u)
        m = self.fuse(torch.cat([s, u], dim=1))
        b_logits = self.b_head(m)
        inject = self.inject_proj(m)  # zero at init via zero-init weight
        return m, b_logits, inject


class NISFlowNetUNIHybridHUBIM(NISFlowNetUNIHybridH):
    """H-UNI + Boundary Inference Module.

    HU-BIM produces an extra boundary feature M and an auxiliary 1ch
    boundary logit b_aux supervised by point-derivable Voronoi-ridge target.
    The injection of M into decoder is gated by a learnable scalar
    `gamma_inject` initialized to 0 so the model starts identical to H-UNI
    baseline at iteration 0.

    Outputs added to dict:
        aux_boundary_logits:  1ch @ 1/8  (sigmoid -> boundary prob)
        bim_gamma:            scalar value of gamma_inject (diagnostic)
        bim_m_norm:           ||inject||/(||feat||+eps) diagnostic
    """

    def __init__(self, uni_path: str, flow_param: str = "direct",
                  img_size: int = 224, pretrained_r34: bool = True,
                  bim_mid_ch: int = 64):
        super().__init__(uni_path=uni_path, flow_param=flow_param,
                          img_size=img_size, pretrained_r34=pretrained_r34)
        # injection channel must match feat depth after up3 (which is 128)
        self.bim = HUBoundaryInferenceModule(
            h_ch=128, uni_ch=256, mid_ch=bim_mid_ch, inject_ch=128,
        )

    def forward(self, x):
        if isinstance(x, dict):
            x_h = x["image"]
            x_rgb = x.get("image_rgb", x["image"])
        else:
            x_h = x
            x_rgb = x
        H, W = x_h.shape[-2:]
        s0 = self.r34_stem(x_h)
        s1 = self.r34_layer1(self.r34_maxpool(s0))
        s2 = self.r34_layer2(s1)
        s3 = self.uni(x_rgb)

        feat = self.proj_uni(s3)                # [B,256,1/16]
        # HU-BIM: joint boundary inference
        m, b_logits, inject = self.bim(s2, feat)
        feat = self.up3(feat, s2)               # [B,128,1/8]
        # inject M into decoder at 1/8 (zero-init conv -> identity at start;
        # no scalar gate -> gradient flows directly to inject_proj weights)
        feat = feat + inject
        with torch.no_grad():
            bim_m_norm = (inject.norm() / (feat.norm() + 1e-6)).detach()
            # diagnostic: inject_proj weight magnitude (replaces gamma_inject)
            bim_inj_w = self.bim.inject_proj.weight.abs().mean().detach()
        feat = self.up2(feat, s1)
        feat = self.up1(feat, s0)
        feat = self.final_up(feat)
        if feat.shape[-2:] != (H, W):
            feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)

        fg_logits = self.fg_head(feat)
        centerness_logits = self.centerness_head(feat)
        boundary_logits = self.boundary_head(feat)
        outputs = {
            "fg_logits": fg_logits,
            "centerness_logits": centerness_logits,
            "boundary_logits": boundary_logits,
            "feat": feat,
            "aux_boundary_logits": b_logits,           # NEW: 1ch @ 1/8 for HU-BIM loss
            "bim_inj_w": bim_inj_w,
            "bim_m_norm": bim_m_norm,
        }
        if self.flow_param == "direct":
            outputs["flow"] = torch.tanh(self.flow_head(feat))
        else:
            phi = self.potential_head(feat)
            outputs["phi"] = phi
            outputs["flow"] = potential_to_flow(phi)
        return outputs


# ============================================================================
# HU-CAF: H-UNI Cross-branch Aux Foreground (self-distill UNI from fused)
# ============================================================================
class NISFlowNetUNIHybridHUCAF(NISFlowNetUNIHybridH):
    """H-UNI + auxiliary fg head on UNI-only features.

    Adds a 2nd fg head that uses ONLY the projected UNI feature F_U (no H
    skip, no decoder). Distilled toward the main fused fg head via KL/MSE
    on sigmoid space. This forces UNI projection to learn nuclei-localizing
    features instead of generic patch tokens.

    Outputs added:
        aux_fg_logits_uni: 1ch at full resolution (upsampled from 1/16)
    """

    def __init__(self, uni_path: str, flow_param: str = "direct",
                  img_size: int = 224, pretrained_r34: bool = True):
        super().__init__(uni_path=uni_path, flow_param=flow_param,
                          img_size=img_size, pretrained_r34=pretrained_r34)
        # extra head on UNI-projected feature (256ch @ 1/16) -> 1ch
        self.aux_fg_head_uni = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, bias=True),
        )

    def forward(self, x):
        if isinstance(x, dict):
            x_h = x["image"]
            x_rgb = x.get("image_rgb", x["image"])
        else:
            x_h = x
            x_rgb = x
        H, W = x_h.shape[-2:]
        s0 = self.r34_stem(x_h)
        s1 = self.r34_layer1(self.r34_maxpool(s0))
        s2 = self.r34_layer2(s1)
        s3 = self.uni(x_rgb)

        feat_uni = self.proj_uni(s3)            # 256ch @ 1/16
        # aux fg head on UNI alone
        aux_logits = self.aux_fg_head_uni(feat_uni)  # 1ch @ 1/16
        aux_logits = F.interpolate(aux_logits, size=(H, W),
                                     mode="bilinear", align_corners=False)
        feat = self.up3(feat_uni, s2)
        feat = self.up2(feat, s1)
        feat = self.up1(feat, s0)
        feat = self.final_up(feat)
        if feat.shape[-2:] != (H, W):
            feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)

        fg_logits = self.fg_head(feat)
        centerness_logits = self.centerness_head(feat)
        boundary_logits = self.boundary_head(feat)
        outputs = {
            "fg_logits": fg_logits,
            "centerness_logits": centerness_logits,
            "boundary_logits": boundary_logits,
            "feat": feat,
            "aux_fg_logits_uni": aux_logits,    # NEW: for HU-CAF distill loss
        }
        if self.flow_param == "direct":
            outputs["flow"] = torch.tanh(self.flow_head(feat))
        else:
            phi = self.potential_head(feat)
            outputs["phi"] = phi
            outputs["flow"] = potential_to_flow(phi)
        return outputs


# ============================================================================
# HU-DAC: H-UNI Disagreement-Aware Channel gating
# ============================================================================
class HUDACModule(nn.Module):
    """Channel-wise gate on s2 derived from image-level cross-branch summary.

    GAP(F_U) and GAP(s2_H) are concatenated and passed through a small MLP
    that produces a 128-d channel scale vector. The scale modulates s2 via
        s2' = s2 * (1 + alpha * tanh(scale))
    where alpha is small (default 0.1) to bound modulation per-channel in
    [1-alpha, 1+alpha]. Final linear is zero-init so init scale=0 → identity.
    """

    def __init__(self, h_ch: int = 128, uni_ch: int = 256,
                 hidden: int = 256, alpha: float = 0.1):
        super().__init__()
        self.alpha = float(alpha)
        self.h_ch = h_ch
        self.mlp = nn.Sequential(
            nn.Linear(uni_ch + h_ch, hidden, bias=True),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, h_ch, bias=True),
        )
        # v2 fix: do NOT zero-init mlp[-1]. With BN-normalized features, GAP
        # produces v ≈ 0, so silu(mlp[0](v)) ≈ 0, and zero-init on mlp[-1]
        # combined with z=0 kills ALL gradient through this module. Instead
        # use small Kaiming-style init so initial gate has small but nonzero
        # perturbation around 1.0 (range ~ 1 ± alpha*small_scale).
        nn.init.normal_(self.mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, s2_h, f_u):
        v_h = s2_h.mean(dim=[2, 3])      # (B, 128)
        v_u = f_u.mean(dim=[2, 3])        # (B, 256)
        v = torch.cat([v_u, v_h], dim=1)  # (B, 384)
        scale = self.mlp(v)              # (B, 128)
        gate = 1.0 + self.alpha * torch.tanh(scale)
        return s2_h * gate.view(-1, self.h_ch, 1, 1), gate


class NISFlowNetUNIHybridHUDAC(NISFlowNetUNIHybridH):
    """H-UNI + image-level Disagreement-Aware Channel gating on s2 skip.

    Outputs added:
        hudac_gate_mean / hudac_gate_std: scalar diagnostics on per-channel gate
        hudac_skip_delta_norm: ||s2' - s2|| / ||s2||
    """

    def __init__(self, uni_path: str, flow_param: str = "direct",
                  img_size: int = 224, pretrained_r34: bool = True,
                  hudac_alpha: float = 0.1, hudac_hidden: int = 256):
        super().__init__(uni_path=uni_path, flow_param=flow_param,
                          img_size=img_size, pretrained_r34=pretrained_r34)
        self.hudac = HUDACModule(h_ch=128, uni_ch=256,
                                    hidden=hudac_hidden, alpha=hudac_alpha)

    def forward(self, x):
        if isinstance(x, dict):
            x_h = x["image"]
            x_rgb = x.get("image_rgb", x["image"])
        else:
            x_h = x
            x_rgb = x
        H, W = x_h.shape[-2:]
        s0 = self.r34_stem(x_h)
        s1 = self.r34_layer1(self.r34_maxpool(s0))
        s2 = self.r34_layer2(s1)
        s3 = self.uni(x_rgb)

        feat = self.proj_uni(s3)
        s2_gated, gate = self.hudac(s2, feat)
        with torch.no_grad():
            d_norm = ((s2_gated - s2).norm()
                       / (s2.norm() + 1e-6)).detach()
        feat = self.up3(feat, s2_gated)
        feat = self.up2(feat, s1)
        feat = self.up1(feat, s0)
        feat = self.final_up(feat)
        if feat.shape[-2:] != (H, W):
            feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)

        fg_logits = self.fg_head(feat)
        centerness_logits = self.centerness_head(feat)
        boundary_logits = self.boundary_head(feat)
        outputs = {
            "fg_logits": fg_logits,
            "centerness_logits": centerness_logits,
            "boundary_logits": boundary_logits,
            "feat": feat,
            "hudac_gate_mean": gate.detach().mean(),
            "hudac_gate_std": gate.detach().std(),
            "hudac_skip_delta_norm": d_norm,
        }
        if self.flow_param == "direct":
            outputs["flow"] = torch.tanh(self.flow_head(feat))
        else:
            phi = self.potential_head(feat)
            outputs["phi"] = phi
            outputs["flow"] = potential_to_flow(phi)
        return outputs


# ============================================================================
# RCF-Head: Ridge-Conditioned Flow Head (truly non-bypassable)
# ============================================================================
class HURidgeModule(nn.Module):
    """Joint H+UNI ridge module: produces a 64ch ridge feature M and a 1ch
    auxiliary ridge logit. Used by RCF-Head; M is consumed by flow head.
    """

    def __init__(self, h_ch: int = 128, u_ch: int = 256, ridge_ch: int = 64):
        super().__init__()
        self.h_proj = nn.Conv2d(h_ch, ridge_ch, kernel_size=1, bias=True)
        self.u_proj = nn.Conv2d(u_ch, ridge_ch, kernel_size=1, bias=True)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * ridge_ch, ridge_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(16, ridge_ch), ridge_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(ridge_ch, ridge_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(min(16, ridge_ch), ridge_ch),
            nn.SiLU(inplace=True),
        )
        self.b_head = nn.Conv2d(ridge_ch, 1, kernel_size=1, bias=True)

    def forward(self, s2_h, f_uni):
        u = F.interpolate(f_uni, size=s2_h.shape[-2:],
                            mode="bilinear", align_corners=False)
        h = self.h_proj(s2_h)
        u = self.u_proj(u)
        m = self.fuse(torch.cat([h, u], dim=1))
        b_logits = self.b_head(m)
        return m, b_logits


class NISFlowNetUNIHybridHURCF(NISFlowNetUNIHybridH):
    """H-UNI + Ridge-Conditioned Flow Head (RCF-Head).

    Architecture difference from HU-BIM:
      - Ridge module M (64ch @ 1/8) is NOT injected into decoder
      - Instead M is passed DIRECTLY into flow_head: v = H_flow([feat, M_up])
      - center_head and fg_head still use feat alone (protects centerF1)

    Why non-bypassable: flow_head's input dimension is dec_ch + ridge_ch.
    The flow_head MUST consume the ridge_ch channels of M (cannot bypass
    because there is no alternative feat-only flow head). If M is bad,
    flow degrades. So M is forced to learn flow-useful features.

    Outputs added:
      aux_ridge_logits: 1ch @ 1/8 (ridge prediction for SmoothL1 sup.)
      rcf_m_norm:       diagnostic, |M| / (|feat_for_flow|+eps)
    """

    def __init__(self, uni_path: str, flow_param: str = "direct",
                  img_size: int = 224, pretrained_r34: bool = True,
                  ridge_ch: int = 64):
        super().__init__(uni_path=uni_path, flow_param=flow_param,
                          img_size=img_size, pretrained_r34=pretrained_r34)
        if flow_param != "direct":
            raise NotImplementedError("RCF-Head only supports flow_param='direct'")
        self.rcf = HURidgeModule(h_ch=128, u_ch=256, ridge_ch=ridge_ch)
        # Replace flow_head with a Conv block that takes [feat, M_up] (64 + 64)
        # decoder feat depth is 64 (set by NISFlowNetUNIHybridH.final_up output)
        # ridge_ch=64. So input = 64 + 64 = 128 channels.
        dec_ch = 64
        self.flow_head = nn.Sequential(
            nn.Conv2d(dec_ch + ridge_ch, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 128),
            nn.SiLU(inplace=True),
            nn.Conv2d(128, 2, kernel_size=1, bias=True),
        )

    def forward(self, x):
        if isinstance(x, dict):
            x_h = x["image"]
            x_rgb = x.get("image_rgb", x["image"])
        else:
            x_h = x
            x_rgb = x
        H, W = x_h.shape[-2:]
        s0 = self.r34_stem(x_h)
        s1 = self.r34_layer1(self.r34_maxpool(s0))
        s2 = self.r34_layer2(s1)
        s3 = self.uni(x_rgb)

        feat_deep = self.proj_uni(s3)              # 256ch @ 1/16
        # ridge module from H + UNI (NOT injected into decoder)
        m, ridge_logits = self.rcf(s2, feat_deep)  # m: 64ch @ 1/8, ridge_logits: 1ch @ 1/8

        # standard H-UNI decoder (unchanged)
        feat = self.up3(feat_deep, s2)
        feat = self.up2(feat, s1)
        feat = self.up1(feat, s0)
        feat = self.final_up(feat)
        if feat.shape[-2:] != (H, W):
            feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)

        # fg / center heads still use feat alone (protect centerF1)
        fg_logits = self.fg_head(feat)
        centerness_logits = self.centerness_head(feat)
        boundary_logits = self.boundary_head(feat)

        # flow head consumes [feat, M_upsampled_to_full_res]
        m_up = F.interpolate(m, size=feat.shape[-2:], mode="bilinear", align_corners=False)
        flow_in = torch.cat([feat, m_up], dim=1)   # 64 + 64 = 128 ch
        flow = torch.tanh(self.flow_head(flow_in))

        with torch.no_grad():
            rcf_m_norm = (m_up.norm() / (feat.norm() + 1e-6)).detach()

        return {
            "fg_logits": fg_logits,
            "centerness_logits": centerness_logits,
            "boundary_logits": boundary_logits,
            "feat": feat,
            "flow": flow,
            "aux_ridge_logits": ridge_logits,   # for SmoothL1 ridge loss
            "rcf_m_norm": rcf_m_norm,
        }


# ============================================================================
# RCF-v2: center-safe ridge residual on baseline flow
# ============================================================================
class RidgeResidualFlowHead(nn.Module):
    """Small residual correction on top of baseline flow:
        flow = v_0 + lambda_res * tanh(Delta_v)
    where Delta_v = ConvBlock(concat[sg(F), M_up]). F is detached so the
    residual branch's gradient does NOT pollute the shared decoder; this is
    the key fix for RCF-v1's centerF1 / PQ regression.

    Final conv last layer init std=1e-3 (small but nonzero) — NOT zero-init,
    to avoid the zero-attractor that killed MTA-v1 / HU-DAC v1. lambda_res is
    a fixed scalar (NOT learnable) per design decision: learnable scalars
    repeatedly collapsed to 0 in our prior experiments.
    """

    def __init__(self, dec_ch: int = 64, ridge_ch: int = 64,
                 hidden_ch: int = 128, residual_scale: float = 0.05):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.delta_head = nn.Sequential(
            nn.Conv2d(dec_ch + ridge_ch, hidden_ch, kernel_size=3,
                        padding=1, bias=False),
            nn.GroupNorm(16, hidden_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_ch, 2, kernel_size=1, bias=True),
        )
        # small non-zero init to break zero-attractor symmetry
        nn.init.normal_(self.delta_head[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(self, base_flow: torch.Tensor, dec_feat: torch.Tensor,
                ridge_feat: torch.Tensor):
        if ridge_feat.shape[-2:] != dec_feat.shape[-2:]:
            ridge_feat = F.interpolate(ridge_feat, size=dec_feat.shape[-2:],
                                          mode="bilinear", align_corners=False)
        # detach dec_feat: residual branch does NOT push gradients into
        # shared decoder (center/fg path protected)
        x = torch.cat([dec_feat.detach(), ridge_feat], dim=1)
        delta_v = self.delta_head(x)
        flow = base_flow + self.residual_scale * torch.tanh(delta_v)
        return flow, delta_v


class NISFlowNetUNIHybridHURCFV2(NISFlowNetUNIHybridH):
    """RCF-v2: H-UNI baseline + center-safe ridge residual on flow.

    Differences from RCF-v1 (which got +0.0014 AJI but lost 0.012 centerF1):
      1. baseline flow head UNCHANGED (v_0 from H_flow(feat))
      2. ridge module inputs DETACHED (sg(s2), sg(F_U)) - ridge aux loss
         does not push gradients into H/UNI encoders or shared decoder
      3. residual head receives detached dec_feat - ridge residual does
         NOT pollute the shared decoder used by center/fg heads
      4. lambda_res is a FIXED scalar (0.05), not learnable
      5. final flow: v = v_0 + 0.05 * tanh(delta_v)

    Combined with conservative ridge target (sigma_b=3, r_core=5, lambda_b=0.02),
    this isolates ridge's effect to flow direction correction only.
    """

    def __init__(self, uni_path: str, flow_param: str = "direct",
                  img_size: int = 224, pretrained_r34: bool = True,
                  ridge_ch: int = 64, residual_scale: float = 0.05):
        super().__init__(uni_path=uni_path, flow_param=flow_param,
                          img_size=img_size, pretrained_r34=pretrained_r34)
        if flow_param != "direct":
            raise NotImplementedError("RCF-v2 only supports flow_param='direct'")
        self.rcf2 = HURidgeModule(h_ch=128, u_ch=256, ridge_ch=ridge_ch)
        self.rcf2_res = RidgeResidualFlowHead(
            dec_ch=64, ridge_ch=ridge_ch, hidden_ch=128,
            residual_scale=residual_scale,
        )

    def forward(self, x):
        if isinstance(x, dict):
            x_h = x["image"]
            x_rgb = x.get("image_rgb", x["image"])
        else:
            x_h = x
            x_rgb = x
        H, W = x_h.shape[-2:]
        s0 = self.r34_stem(x_h)
        s1 = self.r34_layer1(self.r34_maxpool(s0))
        s2 = self.r34_layer2(s1)
        s3 = self.uni(x_rgb)

        feat_deep = self.proj_uni(s3)
        # ridge module on DETACHED inputs (does not push grads into H/UNI)
        m, ridge_logits = self.rcf2(s2.detach(), feat_deep.detach())

        # standard H-UNI decoder forward (unchanged)
        feat = self.up3(feat_deep, s2)
        feat = self.up2(feat, s1)
        feat = self.up1(feat, s0)
        feat = self.final_up(feat)
        if feat.shape[-2:] != (H, W):
            feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)

        # baseline heads use feat directly (unchanged)
        fg_logits = self.fg_head(feat)
        centerness_logits = self.centerness_head(feat)
        boundary_logits = self.boundary_head(feat)
        base_flow = torch.tanh(self.flow_head(feat))

        # residual correction on flow only (dec_feat detached inside)
        flow, delta_v = self.rcf2_res(base_flow=base_flow, dec_feat=feat,
                                         ridge_feat=m)

        with torch.no_grad():
            rcf2_delta_norm = (delta_v.norm()
                                / (base_flow.norm() + 1e-6)).detach()

        return {
            "fg_logits": fg_logits,
            "centerness_logits": centerness_logits,
            "boundary_logits": boundary_logits,
            "feat": feat,
            "flow": flow,
            "base_flow": base_flow,                # for monitoring
            "aux_ridge_logits": ridge_logits,      # for SmoothL1 ridge loss
            "rcf2_delta_norm": rcf2_delta_norm,
        }


# ============================================================================
# HU-DAR: Decoder Augmented Residual (non-bypassable, no scalar gate)
# ============================================================================
class NISFlowNetUNIHybridHUDAR(NISFlowNetUNIHybridH):
    """H-UNI + Decoder Augmented Residual on s2 skip.

    Computes a direct residual on the s2 skip from UNI deep feature:
        delta_s2 = Conv1x1(F_U_upsampled_to_1/8)
        s2_aug   = s2 + delta_s2
    where Conv1x1 is **zero-initialized** so delta_s2=0 at init (model
    starts identical to H-UNI baseline).

    Crucially, there is NO scalar gate. Previous failed modules (MTA-v1,
    HU-BIM v1) collapsed because a single zero-init scalar would zero
    gradient for the entire downstream path. Here Conv1x1's weights ARE
    the gating mechanism: a 128*256-dim subspace, much harder for Adam
    to keep at exactly 0. Plus all udar_* params are placed in the
    no-weight-decay optimizer group so WD doesn't pull weights back to 0.

    Diagnostics added to forward output:
        udar_delta_norm: ||delta_s2|| / (||s2|| + eps)
        udar_w_mean:     Conv1x1.weight abs.mean (proxy for module activity)
    """

    def __init__(self, uni_path: str, flow_param: str = "direct",
                  img_size: int = 224, pretrained_r34: bool = True):
        super().__init__(uni_path=uni_path, flow_param=flow_param,
                          img_size=img_size, pretrained_r34=pretrained_r34)
        # delta path: F_U (256ch @ 1/16) -> 128ch @ 1/8 via 1x1 conv after upsample
        self.udar_proj = nn.Conv2d(256, 128, kernel_size=1, bias=False)
        nn.init.zeros_(self.udar_proj.weight)

    def forward(self, x):
        if isinstance(x, dict):
            x_h = x["image"]
            x_rgb = x.get("image_rgb", x["image"])
        else:
            x_h = x
            x_rgb = x
        H, W = x_h.shape[-2:]
        s0 = self.r34_stem(x_h)
        s1 = self.r34_layer1(self.r34_maxpool(s0))
        s2 = self.r34_layer2(s1)
        s3 = self.uni(x_rgb)

        feat = self.proj_uni(s3)                  # [B,256,1/16]
        # HU-DAR: residual on s2 from UNI deep semantics
        f_u_1_8 = F.interpolate(feat, size=s2.shape[-2:],
                                  mode="bilinear", align_corners=False)
        delta_s2 = self.udar_proj(f_u_1_8)       # 128ch @ 1/8, zero at init
        s2_aug = s2 + delta_s2
        with torch.no_grad():
            udar_delta_norm = (delta_s2.norm()
                                / (s2.norm() + 1e-6)).detach()
            udar_w_mean = self.udar_proj.weight.abs().mean().detach()
        feat = self.up3(feat, s2_aug)
        feat = self.up2(feat, s1)
        feat = self.up1(feat, s0)
        feat = self.final_up(feat)
        if feat.shape[-2:] != (H, W):
            feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)

        fg_logits = self.fg_head(feat)
        centerness_logits = self.centerness_head(feat)
        boundary_logits = self.boundary_head(feat)
        outputs = {
            "fg_logits": fg_logits,
            "centerness_logits": centerness_logits,
            "boundary_logits": boundary_logits,
            "feat": feat,
            "udar_delta_norm": udar_delta_norm,
            "udar_w_mean": udar_w_mean,
        }
        if self.flow_param == "direct":
            outputs["flow"] = torch.tanh(self.flow_head(feat))
        else:
            phi = self.potential_head(feat)
            outputs["phi"] = phi
            outputs["flow"] = potential_to_flow(phi)
        return outputs


# ============================================================================
# U2H-SRG: UNI-to-H Semantic Reliability Gate
# ============================================================================
class UniToHSkipReliabilityGate(nn.Module):
    """RGB-UNI semantic token generates a 1-channel spatial reliability gate
    that modulates the H-R34 skip feature (default: s2 @ 1/8).

    Design intent: H feature captures fine nuclear morphology but can be misled
    by overstaining / artifacts / background structures that look like nuclei.
    RGB-UNI sees full pathology context and is better at deciding whether a
    given H-local structure is a real nucleus. The gate suppresses H skip in
    unreliable regions and preserves/enhances it in reliable ones.

    Output gate G in [0, 1] via sigmoid. Final conv is zero-init so that at
    init G == sigmoid(0) == 0.5 and the modulator
        S' = S * (1 + alpha * (2G - 1))
    becomes S' == S * 1 == S, an exact identity. Training starts from H-UNI
    baseline behavior.

    Range with alpha=0.5: G=0 -> 0.5*S, G=1 -> 1.5*S (bounded modulation).

    Args:
        uni_ch:    channel count of UNI feature (e.g. 256 after proj_uni)
        hidden_ch: hidden channels in the gate generator
        alpha:     modulation strength (fixed, recommended 0.5 for v1)
        detach_uni: stop-gradient through UNI cond. so the gate doesn't push
                    gradients back into UNI's projection head (recommended True)
    """

    def __init__(self, uni_ch: int, hidden_ch: int = 128, alpha: float = 0.5,
                 detach_uni: bool = True, num_groups: int = 16):
        super().__init__()
        self.detach_uni = detach_uni
        self.alpha = float(alpha)
        ng = min(num_groups, hidden_ch)
        self.in_conv = nn.Conv2d(uni_ch, hidden_ch, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=ng, num_channels=hidden_ch)
        self.act = nn.SiLU(inplace=True)
        self.out_conv = nn.Conv2d(hidden_ch, 1, kernel_size=1, bias=True)
        # zero-init final conv -> sigmoid(0) = 0.5 -> identity gate at start
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, f_uni: torch.Tensor, h_skip: torch.Tensor):
        """Returns (h_skip_gated, gate). gate has shape (B, 1, Hs, Ws)."""
        f_cond = f_uni.detach() if self.detach_uni else f_uni
        if f_cond.shape[-2:] != h_skip.shape[-2:]:
            f_cond = F.interpolate(f_cond, size=h_skip.shape[-2:],
                                     mode="bilinear", align_corners=False)
        gate = torch.sigmoid(self.out_conv(self.act(self.norm(self.in_conv(f_cond)))))
        scale = 1.0 + self.alpha * (2.0 * gate - 1.0)
        return h_skip * scale, gate


class NISFlowNetUNIHybridHU2HSRG(NISFlowNetUNIHybridH):
    """H-UNI hybrid + UNI-to-H Semantic Reliability Gate on s2 skip.

    Identical to NISFlowNetUNIHybridH except the s2 skip (H-R34 layer2, 128ch
    @ 1/8) is passed through `u2h_srg` before entering decoder's up3 block.
    Initialization is identity (zero-init gate-net out_conv -> G=0.5 -> S'=S),
    so model starts equivalent to H-UNI baseline.

    Diagnostics in forward output:
        u2h_gate_mean: scalar mean of gate over all pixels
        u2h_gate_std:  scalar std  of gate over all pixels
        u2h_skip_delta_norm: relative L2 norm ||S' - S|| / (||S|| + eps)
    """

    def __init__(self, uni_path: str, flow_param: str = "direct",
                  img_size: int = 224, pretrained_r34: bool = True,
                  u2h_alpha: float = 0.5, u2h_hidden_ch: int = 128,
                  u2h_detach_uni: bool = True):
        super().__init__(uni_path=uni_path, flow_param=flow_param,
                          img_size=img_size, pretrained_r34=pretrained_r34)
        # UNI projection produces 256-ch feature at 1/16; s2 is 128-ch at 1/8.
        # gate net is conditioned on UNI (256ch), upsampled inside to match s2.
        self.u2h_srg = UniToHSkipReliabilityGate(
            uni_ch=256, hidden_ch=u2h_hidden_ch,
            alpha=u2h_alpha, detach_uni=u2h_detach_uni,
        )

    def forward(self, x):
        if isinstance(x, dict):
            x_h = x["image"]
            x_rgb = x.get("image_rgb", x["image"])
        else:
            x_h = x
            x_rgb = x
        H, W = x_h.shape[-2:]
        s0 = self.r34_stem(x_h)
        s1 = self.r34_layer1(self.r34_maxpool(s0))
        s2 = self.r34_layer2(s1)
        s3 = self.uni(x_rgb)

        feat = self.proj_uni(s3)                  # [B,256,h/16,w/16]
        # U2H-SRG: gate s2 with UNI-derived spatial reliability
        s2_gated, gate = self.u2h_srg(feat, s2)
        with torch.no_grad():
            u2h_delta = ((s2_gated - s2).norm()
                          / (s2.norm() + 1e-6)).detach()

        feat = self.up3(feat, s2_gated)
        feat = self.up2(feat, s1)
        feat = self.up1(feat, s0)
        feat = self.final_up(feat)
        if feat.shape[-2:] != (H, W):
            feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)

        fg_logits = self.fg_head(feat)
        centerness_logits = self.centerness_head(feat)
        boundary_logits = self.boundary_head(feat)
        outputs = {
            "fg_logits": fg_logits,
            "centerness_logits": centerness_logits,
            "boundary_logits": boundary_logits,
            "feat": feat,
            "u2h_gate_mean": gate.detach().mean(),
            "u2h_gate_std": gate.detach().std(),
            "u2h_skip_delta_norm": u2h_delta,
        }
        if self.flow_param == "direct":
            outputs["flow"] = torch.tanh(self.flow_head(feat))
        else:
            phi = self.potential_head(feat)
            outputs["phi"] = phi
            outputs["flow"] = potential_to_flow(phi)
        return outputs


# ============================================================================
# MTA: Morphology-guided Token Adapter (H-to-UNI FiLM modulation)
# ============================================================================
class MorphologyGuidedTokenAdapter(nn.Module):
    """H-to-UNI Morphology-guided Token Adapter (MTA-FiLM v1).

    Lets H-derived morphology feature produce per-channel scale/shift parameters
    that modulate the RGB-UNI semantic token feature, without touching UNI's
    pretrained weights or the H branch's gradient flow (detach_h=True).

    Final conv is zero-initialized so adapter is identity at init:
        f_uni_adapted = f_uni * (1 + eta*tanh(0)) + eta*0 = f_uni
    Training thus starts from H-UNI baseline solution.

    Args:
        uni_ch:    channel count of f_uni (the feature to be adapted)
        h_ch:      channel count of f_h (the morphology feature; usually R34 s2 = 128)
        hidden_ch: hidden channels in FiLM generator
        detach_h:  if True, stop-gradient through f_h so MTA does NOT push gradients
                   back into the H branch (recommended for first version)
        init_scale: initial value for learnable scalar eta (modulation strength)
    """

    def __init__(self, uni_ch: int, h_ch: int, hidden_ch: int = 256,
                 detach_h: bool = True, init_scale: float = 0.1,
                 num_groups: int = 32):
        super().__init__()
        self.detach_h = detach_h
        self.uni_ch = uni_ch
        ng = min(num_groups, hidden_ch)
        # build a (1x1 conv) -> GN -> SiLU -> (1x1 conv, zero-init) stack
        self.in_conv = nn.Conv2d(h_ch, hidden_ch, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=ng, num_channels=hidden_ch)
        self.act = nn.SiLU(inplace=True)
        self.out_conv = nn.Conv2d(hidden_ch, 2 * uni_ch, kernel_size=1, bias=True)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        # learnable modulation strength, small init
        self.eta = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, f_uni: torch.Tensor, f_h: torch.Tensor) -> torch.Tensor:
        if f_h.shape[-2:] != f_uni.shape[-2:]:
            f_h = F.interpolate(f_h, size=f_uni.shape[-2:],
                                  mode="bilinear", align_corners=False)
        if self.detach_h:
            f_h = f_h.detach()
        h = self.act(self.norm(self.in_conv(f_h)))
        gamma_beta = self.out_conv(h)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=1)
        return f_uni * (1.0 + self.eta * torch.tanh(gamma)) + self.eta * beta


class NISFlowNetUNIHybridHMTA(NISFlowNetUNIHybridH):
    """H-UNI hybrid + Morphology-guided Token Adapter (FiLM).

    Identical to NISFlowNetUNIHybridH except: after proj_uni produces the
    256-ch UNI feature at 1/16, the H-R34 morphology feature s2 (128ch, 1/8)
    is used (resized to 1/16, detached by default) to produce FiLM parameters
    that modulate the UNI feature before entering the decoder.

    Initialization is identity (zero-init out_conv), so the model starts
    bit-equivalent to H-UNI baseline. ckpts trained from H-UNI can be loaded
    with strict=False (only MTA submodule weights are new).

    Diagnostics exposed in forward output:
        mta_eta:       scalar value of learnable strength
        mta_delta_norm: relative L2 norm  ||f_uni' - f_uni|| / (||f_uni|| + eps)
                       (set to 0 in eval mode if not requested, but always computed for monitoring)
    """

    def __init__(self, uni_path: str, flow_param: str = "direct",
                  img_size: int = 224, pretrained_r34: bool = True,
                  mta_hidden_ch: int = 256, mta_detach_h: bool = True,
                  mta_init_scale: float = 0.1):
        super().__init__(uni_path=uni_path, flow_param=flow_param,
                          img_size=img_size, pretrained_r34=pretrained_r34)
        # H-R34 layer2 (s2) is 128 ch; projected UNI feature is 256 ch
        self.mta = MorphologyGuidedTokenAdapter(
            uni_ch=256, h_ch=128, hidden_ch=mta_hidden_ch,
            detach_h=mta_detach_h, init_scale=mta_init_scale,
        )

    def forward(self, x):
        if isinstance(x, dict):
            x_h = x["image"]
            x_rgb = x.get("image_rgb", x["image"])
        else:
            x_h = x
            x_rgb = x
        H, W = x_h.shape[-2:]
        s0 = self.r34_stem(x_h)
        s1 = self.r34_layer1(self.r34_maxpool(s0))
        s2 = self.r34_layer2(s1)
        s3 = self.uni(x_rgb)

        feat_pre = self.proj_uni(s3)            # [B,256,h/16,w/16]
        feat = self.mta(feat_pre, s2)           # H-guided modulation
        # diagnostic: relative magnitude of modulation
        with torch.no_grad():
            mta_delta_norm = ((feat - feat_pre).norm()
                               / (feat_pre.norm() + 1e-6)).detach()
        feat = self.up3(feat, s2)
        feat = self.up2(feat, s1)
        feat = self.up1(feat, s0)
        feat = self.final_up(feat)
        if feat.shape[-2:] != (H, W):
            feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)

        fg_logits = self.fg_head(feat)
        centerness_logits = self.centerness_head(feat)
        boundary_logits = self.boundary_head(feat)
        outputs = {
            "fg_logits": fg_logits,
            "centerness_logits": centerness_logits,
            "boundary_logits": boundary_logits,
            "feat": feat,
            "mta_eta": self.mta.eta.detach(),
            "mta_delta_norm": mta_delta_norm,
        }
        if self.flow_param == "direct":
            outputs["flow"] = torch.tanh(self.flow_head(feat))
        else:
            phi = self.potential_head(feat)
            outputs["phi"] = phi
            outputs["flow"] = potential_to_flow(phi)
        return outputs


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else ""
    m = NISFlowNetUNIHybrid(uni_path=path, flow_param="direct")
    x = torch.randn(2, 3, 224, 224)
    out = m(x)
    print("=== Hybrid smoke ===")
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)}")
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in m.parameters() if not p.requires_grad)
    print(f"trainable: {n_train/1e6:.2f} M")
    print(f"frozen: {n_frozen/1e6:.2f} M")
