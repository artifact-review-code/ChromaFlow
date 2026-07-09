"""SCDE — Shallow Chromophore-Disentangled Encoder.

Three independent shallow stems (conv1 + bn1 + relu + maxpool + layer1) for
RGB / H-replicated-3ch / E-replicated-3ch. Each stem starts from ImageNet R34.
Fused at s1 (/2 res) and s2 (/4 res) via concat + 1x1 conv (initialized as
1/3 mean for clean identity start). Deeper layers (layer2/3/4) shared, inherit
RGB stem's ImageNet pretraining.

Input format: (B, 5, H, W) = [R, G, B, H_raw01, E_raw01]
  - R/G/B already ImageNet-normalized (dataset.image_to_tensor).
  - H/E are raw stain_h_channel / stain_e_channel output in [0,1].
  - Model replicates H,E to 3ch and applies ImageNet norm internally so the
    pretrained stems see in-distribution inputs.

Heads (fg, centerness, flow) are unchanged from NISFlowNet.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

from chromaflow.model import _resnet34, UpBlock, potential_to_flow


class NISFlowNetSCDE(nn.Module):
    def __init__(self, pretrained: bool = True, flow_param: str = "direct") -> None:
        super().__init__()
        if flow_param not in {"direct", "potential"}:
            raise ValueError(f"Unsupported flow_param: {flow_param}")
        self.flow_param = flow_param

        net_rgb = _resnet34(pretrained=pretrained)
        net_h = _resnet34(pretrained=pretrained)
        net_e = _resnet34(pretrained=pretrained)

        # Three independent stems
        self.rgb_conv1 = net_rgb.conv1
        self.rgb_bn1 = net_rgb.bn1
        self.rgb_relu = net_rgb.relu
        self.rgb_maxpool = net_rgb.maxpool
        self.rgb_layer1 = net_rgb.layer1

        self.h_conv1 = net_h.conv1
        self.h_bn1 = net_h.bn1
        self.h_relu = net_h.relu
        self.h_maxpool = net_h.maxpool
        self.h_layer1 = net_h.layer1

        self.e_conv1 = net_e.conv1
        self.e_bn1 = net_e.bn1
        self.e_relu = net_e.relu
        self.e_maxpool = net_e.maxpool
        self.e_layer1 = net_e.layer1

        # Shared deep layers — inherit RGB pretraining
        self.layer2 = net_rgb.layer2
        self.layer3 = net_rgb.layer3
        self.layer4 = net_rgb.layer4

        # Fusion at s1 (/2 res, 64ch each) and s2 (/4 res, 64ch each)
        self.fuse_s1 = nn.Conv2d(64 * 3, 64, kernel_size=1, bias=False)
        self.fuse_s2 = nn.Conv2d(64 * 3, 64, kernel_size=1, bias=False)
        with torch.no_grad():
            for fuse in (self.fuse_s1, self.fuse_s2):
                fuse.weight.zero_()
                for i in range(64):
                    for j in range(3):
                        fuse.weight[i, i + j * 64, 0, 0] = 1.0 / 3.0

        # ImageNet normalization buffers (applied to H,E inside forward)
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Decoder + heads (identical to NISFlowNet)
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

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: (B, 5, H, W) = [R_imnet, G_imnet, B_imnet, H_raw01, E_raw01]
        rgb = x[:, :3]
        h_raw = x[:, 3:4]
        e_raw = x[:, 4:5]
        h = (h_raw.repeat(1, 3, 1, 1) - self.imagenet_mean) / self.imagenet_std
        e = (e_raw.repeat(1, 3, 1, 1) - self.imagenet_mean) / self.imagenet_std

        # Three shallow stems
        s1_rgb = self.rgb_relu(self.rgb_bn1(self.rgb_conv1(rgb)))
        s2_rgb = self.rgb_layer1(self.rgb_maxpool(s1_rgb))

        s1_h = self.h_relu(self.h_bn1(self.h_conv1(h)))
        s2_h = self.h_layer1(self.h_maxpool(s1_h))

        s1_e = self.e_relu(self.e_bn1(self.e_conv1(e)))
        s2_e = self.e_layer1(self.e_maxpool(s1_e))

        # Fuse
        s1 = self.fuse_s1(torch.cat([s1_rgb, s1_h, s1_e], dim=1))
        s2 = self.fuse_s2(torch.cat([s2_rgb, s2_h, s2_e], dim=1))

        # Shared deep
        s3 = self.layer2(s2)
        s4 = self.layer3(s3)
        bot = self.layer4(s4)

        # Decoder
        x = self.up5(bot, s4)
        x = self.up6(x, s3)
        x = self.up7(x, s2)
        x = self.up8(x, s1)
        feat = self.final_up(x)

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
