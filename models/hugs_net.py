"""HUGS-Net: HUe-Guided Synergistic Network.

It keeps the full-model settings identified from the ablation code:

* 32 base feature channels
* 96 groups in the group convolutions
* 6 residual convolution blocks
* a full-resolution hue-guidance branch and a half-resolution detail branch
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def rgb_to_hue(rgb: Tensor, eps: float = 1e-8) -> Tensor:
    """Convert an RGB tensor in [0, 1] to a normalized one-channel hue tensor."""
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError(f"Expected RGB input shaped [B, 3, H, W], got {tuple(rgb.shape)}")

    red, green, blue = rgb.unbind(dim=1)
    maximum, max_index = rgb.max(dim=1)
    minimum = rgb.min(dim=1).values
    delta = maximum - minimum
    safe_delta = delta.clamp_min(eps)

    hue_r = ((green - blue) / safe_delta) % 6.0
    hue_g = (blue - red) / safe_delta + 2.0
    hue_b = (red - green) / safe_delta + 4.0

    hue = torch.where(max_index == 0, hue_r, hue_g)
    hue = torch.where(max_index == 2, hue_b, hue)
    hue = torch.where(delta > eps, hue / 6.0, torch.zeros_like(hue))
    return hue.unsqueeze(1)


class ConvAct(nn.Sequential):
    """Convolution followed by an optional activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        activation: bool = True,
        transpose: bool = False,
    ) -> None:
        padding = kernel_size // 2
        convolution: nn.Module
        if transpose:
            convolution = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding - 1,
            )
        else:
            convolution = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
            )

        layers = [convolution]
        if activation:
            layers.append(nn.GELU())
        super().__init__(*layers)


class ResConvBlock(nn.Module):
    """Two 3x3 convolutions with the shortcut described for the Res-conv block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ConvAct(channels, channels, 3),
            ConvAct(channels, channels, 3, activation=False),
        )
        self.activation = nn.GELU()

    def forward(self, features: Tensor) -> Tensor:
        return self.activation(features + self.body(features))


class ResidualStack(nn.Sequential):
    def __init__(self, channels: int, depth: int) -> None:
        super().__init__(*(ResConvBlock(channels) for _ in range(depth)))


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels while the public model stays in NCHW format."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, features: Tensor) -> Tensor:
        features = features.permute(0, 2, 3, 1)
        features = self.norm(features)
        return features.permute(0, 3, 1, 2).contiguous()


class LocalTextureBlock(nn.Module):
    """Local RGB feature extractor used before group convolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(channels, channels * 2, 1, bias=False),
            nn.Conv2d(channels * 2, channels * 2, 1),
            nn.GELU(),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.body(features)


class DilatedEmbedding(nn.Module):
    """Three 3x3 convolutions with dilation rates 1, 2, and 3."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, dilation=1),
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2),
            nn.Conv2d(channels, channels, 3, padding=3, dilation=3),
            nn.GELU(),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.body(features)


class FourierEmbeddedBlock(nn.Module):
    """FEB: FFT, amplitude/phase embedding, inverse FFT, and 1x1 projection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pre_pool = nn.MaxPool2d(3, stride=1, padding=1)
        # The reference full model shares this stack between amplitude and phase.
        self.embedding = DilatedEmbedding(channels)
        self.channel_projection = nn.Conv2d(channels, channels, 1)

    def forward(self, features: Tensor) -> Tensor:
        features = self.pre_pool(features)
        spectrum = torch.fft.fft2(features, dim=(-2, -1))
        amplitude = self.embedding(torch.abs(spectrum))
        phase = self.embedding(torch.angle(spectrum))
        filtered_spectrum = amplitude * torch.exp(1j * phase)
        spatial = torch.fft.ifft2(filtered_spectrum, dim=(-2, -1)).real
        return self.channel_projection(spatial)


class ResidueGroupConvolutionBlock(nn.Module):
    """RGCB: concatenate local/Fourier features, group conv, 1x1 conv, 6 Res blocks."""

    def __init__(self, channels: int, num_res_blocks: int, groups: int) -> None:
        super().__init__()
        fused_channels = channels * 3
        if fused_channels % groups != 0:
            raise ValueError(
                f"groups={groups} must divide the fused channel count {fused_channels}"
            )

        self.rgb_norm = ChannelLayerNorm(channels)
        self.context_norm = ChannelLayerNorm(channels)
        self.local_texture = LocalTextureBlock(channels)
        self.fourier_embedding = FourierEmbeddedBlock(channels)
        self.group_projection = nn.Sequential(
            nn.Conv2d(
                fused_channels,
                fused_channels,
                3,
                padding=1,
                groups=groups,
                bias=False,
            ),
            nn.Conv2d(fused_channels, channels, 1),
        )
        self.residual_blocks = ResidualStack(channels, num_res_blocks)

    def forward(self, rgb_features: Tensor, context_features: Tensor) -> Tensor:
        local = self.local_texture(self.rgb_norm(rgb_features))
        global_context = self.fourier_embedding(self.context_norm(context_features))
        fused = torch.cat((local, global_context), dim=1)
        fused = self.group_projection(fused)
        return self.residual_blocks(fused)


class DownscaleFeatureExtractor(nn.Module):
    """Four alternating 3x3/1x1 convolutions for the DDR branch."""

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        quarter = out_channels // 4
        half = out_channels // 2
        self.body = nn.Sequential(
            ConvAct(3, quarter, 3),
            ConvAct(quarter, half, 1),
            ConvAct(half, half, 3),
            ConvAct(half, out_channels - 3, 1),
        )
        self.fuse = ConvAct(out_channels, out_channels, 1, activation=False)

    def forward(self, downscaled_rgb: Tensor) -> Tensor:
        features = torch.cat((downscaled_rgb, self.body(downscaled_rgb)), dim=1)
        return self.fuse(features)


class ElementwiseFusion(nn.Module):
    """Element-wise multiplication, 3x3 convolution, and residual addition."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.refine = ConvAct(channels, channels, 3, activation=False)

    def forward(self, coarse: Tensor, detail: Tensor) -> Tensor:
        return coarse + self.refine(coarse * detail)


class CrossScaleFusion(nn.Module):
    """Fuse two aligned feature tensors with alternating 1x1 and 3x3 convolutions."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ConvAct(in_channels, out_channels, 1),
            ConvAct(out_channels, out_channels, 3, activation=False),
        )

    def forward(self, first: Tensor, second: Tensor) -> Tensor:
        return self.body(torch.cat((first, second), dim=1))


class MultiScaleFusionBlock(nn.Module):
    """MFB used to refine each scale before the final cross-scale reconstruction."""

    def __init__(self, channels: int, num_res_blocks: int) -> None:
        super().__init__()
        self.refinement = ResidualStack(channels, num_res_blocks)

    def forward(self, features: Tensor) -> Tensor:
        return self.refinement(features)


class HGBranch(nn.Module):
    """Hue Guidance Branch (HG-Branch) in Fig. 4."""

    def __init__(
        self,
        base_channels: int,
        detail_channels: int,
        num_res_blocks: int,
        groups: int,
    ) -> None:
        super().__init__()
        self.rgb_stem = nn.Conv2d(3, base_channels, 3, padding=1)
        self.hue_stem = nn.Conv2d(1, base_channels, 3, padding=1)
        self.rgcb = ResidueGroupConvolutionBlock(
            base_channels, num_res_blocks, groups
        )
        self.scale_transfer = ConvAct(
            base_channels, detail_channels, 3, stride=2
        )

    def forward(self, rgb: Tensor, hue: Tensor) -> tuple[Tensor, Tensor]:
        rgb_features = self.rgb_stem(rgb)
        hue_features = self.hue_stem(hue)
        coarse_features = self.rgcb(rgb_features, hue_features)
        downscaled_features = self.scale_transfer(coarse_features)
        return coarse_features, downscaled_features


class DDRBranch(nn.Module):
    """Downscale Detail Refinement Branch (DDR-Branch) in Fig. 4."""

    def __init__(
        self,
        detail_channels: int,
        num_res_blocks: int,
        groups: int,
    ) -> None:
        super().__init__()
        self.feature_extractor = DownscaleFeatureExtractor(detail_channels)
        self.rgcb = ResidueGroupConvolutionBlock(
            detail_channels, num_res_blocks, groups
        )
        self.elementwise_fusion = ElementwiseFusion(detail_channels)

    def forward(
        self,
        downscaled_rgb: Tensor,
        hg_downscaled_features: Tensor,
    ) -> Tensor:
        detail_context = self.feature_extractor(downscaled_rgb)
        refined_features = self.rgcb(
            hg_downscaled_features, detail_context
        )
        return self.elementwise_fusion(refined_features, detail_context)


class MultiScaleFusionStage(nn.Module):
    """Cross-branch scale transfer and the two MFB reconstruction stages."""

    def __init__(
        self,
        base_channels: int,
        detail_channels: int,
        num_res_blocks: int,
    ) -> None:
        super().__init__()
        fusion_channels = base_channels + detail_channels

        self.low_to_high_fusion = CrossScaleFusion(
            fusion_channels, detail_channels
        )
        self.high_to_low_fusion = CrossScaleFusion(
            fusion_channels, base_channels
        )

        self.low_projection = ConvAct(base_channels, detail_channels, 1)
        self.low_mfb = MultiScaleFusionBlock(
            detail_channels, num_res_blocks
        )
        self.low_output = ConvAct(
            detail_channels, 3, 3, activation=False
        )
        self.low_to_full = ConvAct(
            detail_channels,
            base_channels,
            4,
            stride=2,
            transpose=True,
        )

        self.full_projection = ConvAct(
            fusion_channels, base_channels, 1
        )
        self.full_mfb = MultiScaleFusionBlock(
            base_channels, num_res_blocks
        )
        self.full_output = ConvAct(
            base_channels, 3, 3, activation=False
        )

    def forward(
        self,
        rgb: Tensor,
        downscaled_rgb: Tensor,
        hg_features: Tensor,
        ddr_features: Tensor,
    ) -> list[Tensor]:
        ddr_to_hg = F.interpolate(
            ddr_features,
            size=hg_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        hg_to_ddr = F.interpolate(
            hg_features,
            size=ddr_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        high_features = self.low_to_high_fusion(
            ddr_to_hg, hg_features
        )
        low_features = self.high_to_low_fusion(
            hg_to_ddr, ddr_features
        )

        low_features = self.low_projection(low_features)
        low_features = self.low_mfb(low_features)
        half_output = downscaled_rgb + self.low_output(low_features)

        upscaled_low = self.low_to_full(low_features)
        full_features = self.full_projection(
            torch.cat((upscaled_low, high_features), dim=1)
        )
        full_features = self.full_mfb(full_features)
        full_output = rgb + self.full_output(full_features)
        return [half_output, full_output]


class ChannelMLP(nn.Module):
    """Channel MLP retained by the original full-model parameter definition."""

    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        hidden_channels = channels * expansion
        self.fc1 = nn.Linear(channels, hidden_channels)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_channels, channels)

    def forward(self, features: Tensor) -> Tensor:
        return self.fc2(self.activation(self.fc1(features)))


class PaperParameterCompatibility(nn.Module):
    """Layers included in the published 1.72M parameter count.

    These layers were registered by the original complete implementation but
    were not called by its forward method. Keeping them here reproduces the
    paper's exact parameter total without changing the cleaned forward path.
    """

    def __init__(self, base_channels: int) -> None:
        super().__init__()
        detail_channels = base_channels * 2
        bottleneck_channels = base_channels * 4

        self.unused_rgb_projection = ConvAct(3, base_channels, 3)
        self.unused_bottleneck_encoder = ConvAct(
            detail_channels, bottleneck_channels, 3, stride=2
        )
        self.unused_bottleneck_decoder = ConvAct(
            bottleneck_channels,
            detail_channels,
            4,
            stride=2,
            transpose=True,
        )
        self.unused_auxiliary_head = ConvAct(
            bottleneck_channels, 3, 3, activation=False
        )
        self.unused_hg_channel_mlp = ChannelMLP(base_channels)
        self.unused_ddr_channel_mlp = ChannelMLP(detail_channels)


class HUGSNet(nn.Module):
    """Full two-branch HUGS-Net.

    Args:
        base_channels: Feature width of the full-resolution HG branch.
        num_res_blocks: Number of Res-conv blocks. The paper uses 6.
        groups: Group count used by the reference full model. The selected
            ablation implementation uses 96.

    Forward:
        ``model(rgb)`` computes hue internally. ``model(rgb, hue)`` accepts a
        precomputed normalized hue tensor for compatibility. The return value
        is ``[half_resolution_output, full_resolution_output]``.
    """

    def __init__(
        self,
        base_channels: int = 32,
        num_res_blocks: int = 6,
        groups: int = 96,
    ) -> None:
        super().__init__()
        detail_channels = base_channels * 2

        self.hg_branch = HGBranch(
            base_channels,
            detail_channels,
            num_res_blocks,
            groups,
        )
        self.ddr_branch = DDRBranch(
            detail_channels,
            num_res_blocks,
            groups,
        )
        self.multi_scale_fusion = MultiScaleFusionStage(
            base_channels,
            detail_channels,
            num_res_blocks,
        )

        # Restores the exact 1,722,982 parameters counted by the paper's
        # original complete implementation. See PaperParameterCompatibility.
        self.paper_parameter_compatibility = PaperParameterCompatibility(
            base_channels
        )

    def forward(self, rgb: Tensor, hue: Tensor | None = None) -> list[Tensor]:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError(f"Expected [B, 3, H, W], got {tuple(rgb.shape)}")
        if rgb.shape[-2] % 2 or rgb.shape[-1] % 2:
            raise ValueError("HUGS-Net requires even input height and width")

        if hue is None:
            hue = rgb_to_hue(rgb)
        if hue.ndim != 4 or hue.shape[1] != 1:
            raise ValueError(f"Expected hue shaped [B, 1, H, W], got {tuple(hue.shape)}")
        if hue.shape[-2:] != rgb.shape[-2:]:
            raise ValueError("RGB and hue spatial dimensions must match")

        # Framework overview: I_s -> I_sd and I_sh.
        downscaled_rgb = F.interpolate(
            rgb,
            scale_factor=0.5,
            mode="bilinear",
            align_corners=False,
        )

        hg_features, hg_downscaled_features = self.hg_branch(rgb, hue)
        ddr_features = self.ddr_branch(
            downscaled_rgb,
            hg_downscaled_features,
        )
        return self.multi_scale_fusion(
            rgb,
            downscaled_rgb,
            hg_features,
            ddr_features,
        )


if __name__ == "__main__":
    network = HUGSNet()
    parameter_count = sum(parameter.numel() for parameter in network.parameters())
    sample = torch.rand(1, 3, 128, 128)
    half, full = network(sample)
    print(f"Parameters: {parameter_count / 1e6:.3f}M")
    print(f"Half output: {tuple(half.shape)}")
    print(f"Full output: {tuple(full.shape)}")
