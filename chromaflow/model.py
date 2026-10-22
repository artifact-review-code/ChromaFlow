from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _resnet34(pretrained: bool) -> nn.Module:
    from torchvision import models

    if not pretrained:
        return models.resnet34(weights=None)
    try:
        weights = models.ResNet34_Weights.IMAGENET1K_V1
        return models.resnet34(weights=weights)
    except AttributeError:
        return models.resnet34(pretrained=True)


_RESNET_CHANNELS = {
    "resnet18": [64, 64, 128, 256, 512],
    "resnet34": [64, 64, 128, 256, 512],
    "resnet50": [64, 256, 512, 1024, 2048],
    "resnet101": [64, 256, 512, 1024, 2048],
}


def _resnet(name, pretrained):
    from torchvision import models
    fns = {"resnet18": models.resnet18, "resnet34": models.resnet34,
           "resnet50": models.resnet50, "resnet101": models.resnet101}
    wnames = {"resnet18": "ResNet18_Weights", "resnet34": "ResNet34_Weights",
              "resnet50": "ResNet50_Weights", "resnet101": "ResNet101_Weights"}
    fn = fns[name]
    if not pretrained:
        return fn(weights=None)
    try:
        return fn(weights=getattr(models, wnames[name]).IMAGENET1K_V1)
    except AttributeError:
        return fn(pretrained=True)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


def potential_to_flow(phi: torch.Tensor) -> torch.Tensor:
    """Convert scalar potential to dx,dy flow with central differences."""
    dphi_dx = torch.zeros_like(phi)
    dphi_dy = torch.zeros_like(phi)

    dphi_dx[:, :, :, 1:-1] = 0.5 * (phi[:, :, :, 2:] - phi[:, :, :, :-2])
    dphi_dx[:, :, :, 0] = phi[:, :, :, 1] - phi[:, :, :, 0]
    dphi_dx[:, :, :, -1] = phi[:, :, :, -1] - phi[:, :, :, -2]

    dphi_dy[:, :, 1:-1, :] = 0.5 * (phi[:, :, 2:, :] - phi[:, :, :-2, :])
    dphi_dy[:, :, 0, :] = phi[:, :, 1, :] - phi[:, :, 0, :]
    dphi_dy[:, :, -1, :] = phi[:, :, -1, :] - phi[:, :, -2, :]

    return torch.cat([-dphi_dx, -dphi_dy], dim=1)


def poisson_solve_torch(rho: torch.Tensor) -> torch.Tensor:
    """Solve the Poisson equation Δφ = rho on a periodic grid (zero-mean φ) via
    FFT. Differentiable end-to-end. rho: (B, 1, H, W) -> φ: (B, 1, H, W)."""
    _, _, H, W = rho.shape
    two_pi = 2.0 * 3.141592653589793
    with torch.cuda.amp.autocast(enabled=False):   # FFT must run in fp32, even under AMP
        r = rho.float()
        ky = torch.fft.fftfreq(H, device=rho.device).view(1, 1, H, 1) * two_pi
        kx = torch.fft.fftfreq(W, device=rho.device).view(1, 1, 1, W) * two_pi
        k2 = kx * kx + ky * ky
        k2[..., 0, 0] = float("inf")          # DC -> 0 after division (zero-mean phi)
        phi_hat = torch.fft.fft2(r) / (-k2)
        phi = torch.fft.ifft2(phi_hat).real
    return phi.to(rho.dtype)


class NISFlowNet(nn.Module):
    """Clean v3 image-only nuclei student.

    The flow output channel order is dx,dy. Inference swaps it to dy,dx before
    Cellpose dynamics.
    """

    def __init__(self, pretrained: bool = True, flow_param: str = "potential",
                 in_channels: int = 3, with_boundary: bool = False,
                 with_polarity: bool = False, with_pace: bool = False,
                 backbone: str = "resnet34") -> None:
        super().__init__()
        if flow_param not in {"direct", "potential", "poisson"}:
            raise ValueError(f"Unsupported flow_param: {flow_param}")
        self.flow_param = flow_param
        self.backbone = backbone
        self.encoder = _resnet(backbone, pretrained=pretrained)
        if in_channels != 3:
            old_conv = self.encoder.conv1
            new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                if old_conv.weight.shape == (64, 3, 7, 7):
                    if in_channels >= 3:
                        new_conv.weight[:, :3] = old_conv.weight
                        if in_channels > 3:
                            mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                            for i in range(3, in_channels):
                                new_conv.weight[:, i:i+1] = mean_w
                    else:
                        mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                        for i in range(in_channels):
                            new_conv.weight[:, i:i+1] = mean_w
                else:
                    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            self.encoder.conv1 = new_conv
        self.in_channels = in_channels

        _c0, _c1, _c2, _c3, _c4 = _RESNET_CHANNELS[self.backbone]
        self.up5 = UpBlock(_c4, _c3, 256)
        self.up6 = UpBlock(256, _c2, 128)
        self.up7 = UpBlock(128, _c1, 64)
        self.up8 = UpBlock(64, _c0, 64)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.fg_head = nn.Conv2d(64, 1, kernel_size=1)
        self.centerness_head = nn.Conv2d(64, 1, kernel_size=1)
        if self.flow_param == "direct":
            self.flow_head = nn.Conv2d(64, 2, kernel_size=1)
        elif self.flow_param == "poisson":
            self.source_head = nn.Conv2d(64, 1, kernel_size=1)
        else:
            self.potential_head = nn.Conv2d(64, 1, kernel_size=1)
        if with_boundary:
            self.boundary_head = nn.Conv2d(64, 1, kernel_size=1)
        self.decode_alpha = 0.0  # set externally at inference
        if with_pace:
            self.pace_proj = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=1), nn.GroupNorm(8, 128),
                nn.ReLU(inplace=True), nn.Conv2d(128, 64, kernel_size=1))
        if with_polarity:
            self.polarity_head = nn.Conv2d(64, 1, kernel_size=1)

    def forward_pace(self, x: torch.Tensor) -> torch.Tensor:
        e = self.encoder
        x = e.conv1(x); x = e.bn1(x); x = e.relu(x); x = e.maxpool(x)
        x = e.layer1(x); s3 = e.layer2(x)
        return F.normalize(self.pace_proj(s3), dim=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        s1 = self.encoder.relu(x)
        x = self.encoder.maxpool(s1)
        s2 = self.encoder.layer1(x)
        s3 = self.encoder.layer2(s2)
        s4 = self.encoder.layer3(s3)
        x = self.encoder.layer4(s4)

        x = self.up5(x, s4)
        x = self.up6(x, s3)
        x = self.up7(x, s2)
        x = self.up8(x, s1)
        feat = self.final_up(x)

        fg_logits_raw = self.fg_head(feat)
        centerness_logits = self.centerness_head(feat)
        outputs = {
            "centerness_logits": centerness_logits,
            "feat": feat,
        }
        if hasattr(self, "boundary_head"):
            boundary_logits = self.boundary_head(feat)
            outputs["boundary_logits"] = boundary_logits
            if (not self.training) and self.decode_alpha > 0.0:
                outputs["fg_logits"] = fg_logits_raw - self.decode_alpha * torch.sigmoid(boundary_logits)
            else:
                outputs["fg_logits"] = fg_logits_raw
        else:
            outputs["fg_logits"] = fg_logits_raw
        if hasattr(self, "polarity_head"):
            outputs["polarity_pred"] = torch.tanh(self.polarity_head(feat))
        if self.flow_param == "direct":
            outputs["flow"] = torch.tanh(self.flow_head(feat))
        elif self.flow_param == "poisson":
            rho = self.source_head(feat)
            phi = poisson_solve_torch(rho)
            outputs["phi"] = phi
            outputs["source"] = rho
            outputs["flow"] = potential_to_flow(phi)
        else:
            phi = self.potential_head(feat)
            outputs["phi"] = phi
            outputs["flow"] = potential_to_flow(phi)
        return outputs
