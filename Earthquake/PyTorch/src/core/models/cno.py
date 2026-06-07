"""
CNO (Convolutional Neural Operator).

Implements a U-Net style encoder-decoder architecture with the unique CNO_LReLu
activation function that performs bandlimited nonlinearity through bicubic interpolation.

This model operates on full sequences and is designed to work with spectral losses (BSP).

Adapted from the official implementation:
    https://github.com/camlab-ethz/ConvolutionalNeuralOperator

    Paper: Raonić et al., "Convolutional Neural Operators for robust and accurate learning of PDEs",
    arXiv:2302.01178.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNO_LReLu(nn.Module):
    """
    CNO activation function with bandlimited nonlinearity.
    """

    def __init__(self, in_size: int, out_size: int):
        """
        Initialize CNO_LReLu activation.

        Args:
            in_size: Input spatial size
            out_size: Output spatial size
        """
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.act = nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: upsample -> activate -> downsample.

        Args:
            x: Input tensor [B, C, T]

        Returns:
            Output tensor [B, C, out_size]
        """
        x = F.interpolate(
            x.unsqueeze(2), size=(1, 2 * self.in_size), mode="bicubic", antialias=True
        )
        x = self.act(x)
        x = F.interpolate(x, size=(1, self.out_size), mode="bicubic", antialias=True)
        return x[:, :, 0]  # Remove the extra dimension


class CNOBlock(nn.Module):
    """
    Basic CNO block: Conv1d -> BatchNorm (optional) -> Dropout (optional) -> CNO_LReLu.

    Handles spatial resolution changes through the CNO_LReLu activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_size: int,
        out_size: int,
        use_bn: bool = True,
        dropout: float = 0.0,
    ):
        """
        Initialize CNO block.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            in_size: Input spatial size
            out_size: Output spatial size
            use_bn: Whether to use BatchNorm (default True)
            dropout: Dropout rate after BatchNorm (default 0.0 = no dropout)
        """
        super().__init__()

        self.convolution = nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )

        if use_bn:
            self.batch_norm = nn.BatchNorm1d(out_channels)
        else:
            self.batch_norm = nn.Identity()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.act = CNO_LReLu(in_size=in_size, out_size=out_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNO block."""
        x = self.convolution(x)
        x = self.batch_norm(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.act(x)


class LiftProjectBlock(nn.Module):
    """
    Lift/Project block for input embedding and output projection.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        size: int,
        latent_dim: int = 64,
        dropout: float = 0.0,
        output_scale: float = 1.0,
    ):
        """
        Initialize lift/project block.

        Args:
            in_channels: Input channels
            out_channels: Output channels
            size: Spatial size (preserved)
            latent_dim: Intermediate dimension (default 64)
            dropout: Dropout rate (default 0.0 = no dropout)
            output_scale: Scale factor for output (default 1.0, use 0.1-0.3 for project)
        """
        super().__init__()

        self.output_scale = output_scale

        self.inter_CNOBlock = CNOBlock(
            in_channels=in_channels,
            out_channels=latent_dim,
            in_size=size,
            out_size=size,
            use_bn=False,
            dropout=dropout,
        )

        self.convolution = nn.Conv1d(
            in_channels=latent_dim, out_channels=out_channels, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through lift/project block."""
        x = self.inter_CNOBlock(x)
        x = self.convolution(x)
        # Apply output scaling to prevent magnitude explosion
        return x * self.output_scale


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.

    Architecture: Conv -> BN -> Dropout -> Act -> Conv -> BN -> Dropout + Skip
    Maintains same spatial size and channel count.
    """

    def __init__(
        self, channels: int, size: int, use_bn: bool = True, dropout: float = 0.0
    ):
        """
        Initialize residual block.

        Args:
            channels: Number of channels (preserved)
            size: Spatial size (preserved)
            use_bn: Whether to use BatchNorm (default True)
            dropout: Dropout rate after BatchNorm (default 0.0 = no dropout)
        """
        super().__init__()

        self.convolution1 = nn.Conv1d(
            in_channels=channels, out_channels=channels, kernel_size=3, padding=1
        )
        self.convolution2 = nn.Conv1d(
            in_channels=channels, out_channels=channels, kernel_size=3, padding=1
        )

        if use_bn:
            self.batch_norm1 = nn.BatchNorm1d(channels)
            self.batch_norm2 = nn.BatchNorm1d(channels)
        else:
            self.batch_norm1 = nn.Identity()
            self.batch_norm2 = nn.Identity()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.act = CNO_LReLu(in_size=size, out_size=size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection."""
        out = self.convolution1(x)
        out = self.batch_norm1(out)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.act(out)
        out = self.convolution2(out)
        out = self.batch_norm2(out)
        if self.dropout is not None:
            out = self.dropout(out)
        return x + out


class ResNet(nn.Module):
    """
    Stack of residual blocks.

    Applies multiple ResidualBlocks sequentially at the same resolution.
    """

    def __init__(
        self,
        channels: int,
        size: int,
        num_blocks: int,
        use_bn: bool = True,
        dropout: float = 0.0,
    ):
        """
        Initialize ResNet stack.

        Args:
            channels: Number of channels
            size: Spatial size
            num_blocks: Number of residual blocks to stack
            use_bn: Whether to use BatchNorm
            dropout: Dropout rate (default 0.0 = no dropout)
        """
        super().__init__()

        self.num_blocks = num_blocks
        self.res_nets = nn.Sequential(
            *[
                ResidualBlock(
                    channels=channels, size=size, use_bn=use_bn, dropout=dropout
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all residual blocks."""
        return self.res_nets(x)


class CNO(nn.Module):
    """
    Convolutional Neural Operator.

    Input/output shape: [batch, channels, internal_size]. CDON CNO runs at
    internal_size=4096 and is interpolated back to 4000 during evaluation.
    """

    supports_per_timestep = False

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        internal_size: int = 4096,
        down_factor: int = 4,
        n_layers: int = 4,
        n_res: int = 1,
        n_res_neck: int = 1,
        channel_multiplier: int = 54,
        use_bn: bool = True,
        latent_dim_lift: int = 64,
        dropout: float = 0.05,
    ):
        """
        Initialize CNO.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            internal_size: Processing resolution . Input must match this.
            down_factor: Downsampling factor per layer
            n_layers: Number of encoder/decoder levels
            n_res: ResBlocks per level except bottleneck
            n_res_neck: ResBlocks in bottleneck
            channel_multiplier: Channel growth factor
            use_bn: Whether to use BatchNorm
            latent_dim_lift: Intermediate dim for lift/project
            dropout: Dropout rate for regularization
        """
        super().__init__()

        self.n_layers = int(n_layers)
        self.lift_dim = channel_multiplier // 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channel_multiplier = channel_multiplier
        self.latent_dim_lift = latent_dim_lift
        self.dropout = dropout
        self.internal_size = internal_size
        self.down_factor = down_factor

        self.min_divisibility = down_factor**n_layers
        if internal_size % self.min_divisibility != 0:
            raise ValueError(
                f"Internal size {internal_size} must be divisible by {self.min_divisibility} "
                f"(down_factor^n_layers = {down_factor}^{n_layers}). "
                f"Try internal_size {((internal_size // self.min_divisibility) + 1) * self.min_divisibility}."
            )

        self.encoder_features: List[int] = [self.lift_dim]
        for i in range(self.n_layers):
            self.encoder_features.append(2**i * self.channel_multiplier)

        self.decoder_features_in = self.encoder_features[1:].copy()
        self.decoder_features_in.reverse()
        self.decoder_features_out = self.encoder_features[:-1].copy()
        self.decoder_features_out.reverse()

        for i in range(1, self.n_layers):
            self.decoder_features_in[i] = 2 * self.decoder_features_in[i]

        self.encoder_sizes: List[int] = []
        self.decoder_sizes: List[int] = []
        for i in range(self.n_layers + 1):
            self.encoder_sizes.append(internal_size // (down_factor**i))
            self.decoder_sizes.append(
                internal_size // (down_factor ** (self.n_layers - i))
            )

        self.lift = LiftProjectBlock(
            in_channels=in_channels,
            out_channels=self.encoder_features[0],
            size=internal_size,
            latent_dim=latent_dim_lift,
            dropout=dropout,
        )

        self.project = LiftProjectBlock(
            in_channels=self.encoder_features[0] + self.decoder_features_out[-1],
            out_channels=out_channels,
            size=internal_size,
            latent_dim=latent_dim_lift,
            dropout=dropout,
        )

        self.encoder = nn.ModuleList(
            [
                CNOBlock(
                    in_channels=self.encoder_features[i],
                    out_channels=self.encoder_features[i + 1],
                    in_size=self.encoder_sizes[i],
                    out_size=self.encoder_sizes[i + 1],
                    use_bn=use_bn,
                    dropout=dropout,
                )
                for i in range(self.n_layers)
            ]
        )

        self.ED_expansion = nn.ModuleList(
            [
                CNOBlock(
                    in_channels=self.encoder_features[i],
                    out_channels=self.encoder_features[i],
                    in_size=self.encoder_sizes[i],
                    out_size=self.decoder_sizes[self.n_layers - i],
                    use_bn=use_bn,
                    dropout=dropout,
                )
                for i in range(self.n_layers + 1)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                CNOBlock(
                    in_channels=self.decoder_features_in[i],
                    out_channels=self.decoder_features_out[i],
                    in_size=self.decoder_sizes[i],
                    out_size=self.decoder_sizes[i + 1],
                    use_bn=use_bn,
                    dropout=dropout,
                )
                for i in range(self.n_layers)
            ]
        )

        self.n_res = int(n_res)
        self.n_res_neck = int(n_res_neck)

        self.res_nets = nn.ModuleList(
            [
                ResNet(
                    channels=self.encoder_features[l],
                    size=self.encoder_sizes[l],
                    num_blocks=self.n_res,
                    use_bn=use_bn,
                    dropout=dropout,
                )
                for l in range(self.n_layers)
            ]
        )

        self.res_net_neck = ResNet(
            channels=self.encoder_features[self.n_layers],
            size=self.encoder_sizes[self.n_layers],
            num_blocks=self.n_res_neck,
            use_bn=use_bn,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNO.

        Args:
            x: Input tensor [batch, in_channels, internal_size]
               Must be pre-interpolated to internal_size (4096) via dataset.
               Use CDONDataset with target_signal_length=4096 for CNO.

        Returns:
            Output tensor [batch, out_channels, internal_size]

        Raises:
            ValueError: If input size doesn't match internal_size
        """
        input_size = x.shape[-1]

        if input_size != self.internal_size:
            raise ValueError(
                f"CNO requires input at internal_size={self.internal_size}, "
                f"got {input_size}. Use CDONDataset with target_signal_length={self.internal_size}."
            )

        x = self.lift(x)
        skip = []

        for i in range(self.n_layers):
            y = self.res_nets[i](x)
            skip.append(y)
            x = self.encoder[i](x)

        x = self.res_net_neck(x)

        for i in range(self.n_layers):
            if i == 0:
                x = self.ED_expansion[self.n_layers - i](x)
            else:
                x = torch.cat(
                    (x, self.ED_expansion[self.n_layers - i](skip[-i])), dim=1
                )

            x = self.decoder[i](x)

        x = torch.cat((x, self.ED_expansion[0](skip[0])), dim=1)
        x = self.project(x)

        return x
