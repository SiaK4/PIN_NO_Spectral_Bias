import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------
# Activation Function (rectangular):
# ---------------------
class CNO_LReLu(nn.Module):
    def __init__(self, in_size_hw, out_size_hw):
        super().__init__()
        self.in_h, self.in_w = in_size_hw
        self.out_h, self.out_w = out_size_hw
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = F.interpolate(x, size=(2 * self.in_h, 2 * self.in_w), mode="bicubic", antialias=True)
        x = self.act(x)
        x = F.interpolate(x, size=(self.out_h, self.out_w), mode="bicubic", antialias=True)
        return x

# --------------------
# CNO Block (rectangular):
# --------------------
class CNOBlock(nn.Module):
    def __init__(self, in_channels, out_channels, in_size_hw, out_size_hw, use_bn=True):
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = CNO_LReLu(in_size_hw=in_size_hw, out_size_hw=out_size_hw)

    def forward(self, x):
        x = self.convolution(x)
        x = self.batch_norm(x)
        return self.act(x)

# --------------------
# Lift/Project Block (rectangular):
# --------------------
class LiftProjectBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size_hw, latent_dim=64):
        super().__init__()
        self.inter_CNOBlock = CNOBlock(
            in_channels=in_channels,
            out_channels=latent_dim,
            in_size_hw=size_hw,
            out_size_hw=size_hw,
            use_bn=False,
        )
        self.convolution = nn.Conv2d(latent_dim, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.inter_CNOBlock(x)
        x = self.convolution(x)
        return x

# --------------------
# Residual Block (rectangular):
# --------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels, size_hw, use_bn=True):
        super().__init__()
        self.convolution1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.convolution2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        self.batch_norm2 = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        self.act = CNO_LReLu(in_size_hw=size_hw, out_size_hw=size_hw)

    def forward(self, x):
        out = self.convolution1(x)
        out = self.batch_norm1(out)
        out = self.act(out)
        out = self.convolution2(out)
        out = self.batch_norm2(out)
        return x + out

# --------------------
# ResNet (rectangular):
# --------------------
class ResNet(nn.Module):
    def __init__(self, channels, size_hw, num_blocks, use_bn=True):
        super().__init__()
        self.blocks = nn.Sequential(*[
            ResidualBlock(channels=channels, size_hw=size_hw, use_bn=use_bn)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return x

# --------------------
# CNO (rectangular):
# --------------------
class CNO(nn.Module):
    def __init__(
        self,
        Par,
        in_dim,                  # input channels
        out_dim,                 # output channels
        size_hw,                 # (H, W)
        N_layers,                # # of (D)/(U) stages
        N_res=2,                 # # ResBlocks per level
        N_res_neck=2,            # # ResBlocks in neck
        channel_multiplier=26,
        use_bn=True,
    ):
        super().__init__()
        self.par = Par
        self.N_layers = int(N_layers)
        self.N_res = int(N_res)
        self.N_res_neck = int(N_res_neck)
        self.lift_dim = channel_multiplier // 2
        self.in_dim = in_dim
        self.out_dim = out_dim

        H, W = size_hw
        # --- Ensure divisibility by 2**N_layers on both dims
        assert (H % (2 ** N_layers) == 0) and (W % (2 ** N_layers) == 0), \
            f"Both H={H} and W={W} must be divisible by 2**N_layers (={2**N_layers})."

        # --- Channels evolution
        enc_feats = [self.lift_dim] + [ (2 ** i) * channel_multiplier for i in range(self.N_layers) ]
        self.encoder_features = enc_feats
        self.decoder_features_in = list(reversed(enc_feats[1:]))
        self.decoder_features_out = list(reversed(enc_feats[:-1]))
        for i in range(1, self.N_layers):
            self.decoder_features_in[i] *= 2  # concatenation with skip

        # --- Spatial evolution (rectangular)
        self.encoder_sizes_hw = [ (H // (2 ** i), W // (2 ** i)) for i in range(self.N_layers + 1) ]
        self.decoder_sizes_hw = [ (H // (2 ** (self.N_layers - i)), W // (2 ** (self.N_layers - i)))
                                  for i in range(self.N_layers + 1) ]

        # --- Lift/Project
        self.lift = LiftProjectBlock(in_channels=in_dim, out_channels=self.encoder_features[0], size_hw=(H, W))
        self.project = LiftProjectBlock(
            in_channels=self.encoder_features[0] + self.decoder_features_out[-1],
            out_channels=out_dim,
            size_hw=(H, W),
        )

        # --- Encoder
        self.encoder = nn.ModuleList([
            CNOBlock(
                in_channels=self.encoder_features[i],
                out_channels=self.encoder_features[i + 1],
                in_size_hw=self.encoder_sizes_hw[i],
                out_size_hw=self.encoder_sizes_hw[i + 1],
                use_bn=use_bn,
            )
            for i in range(self.N_layers)
        ])

        # --- ED expansion to align sizes for skips
        self.ED_expansion = nn.ModuleList([
            CNOBlock(
                in_channels=self.encoder_features[i],
                out_channels=self.encoder_features[i],
                in_size_hw=self.encoder_sizes_hw[i],
                out_size_hw=self.decoder_sizes_hw[self.N_layers - i],
                use_bn=use_bn,
            )
            for i in range(self.N_layers + 1)
        ])

        # --- Decoder
        self.decoder = nn.ModuleList([
            CNOBlock(
                in_channels=self.decoder_features_in[i],
                out_channels=self.decoder_features_out[i],
                in_size_hw=self.decoder_sizes_hw[i],
                out_size_hw=self.decoder_sizes_hw[i + 1],
                use_bn=use_bn,
            )
            for i in range(self.N_layers)
        ])

        # --- ResNets at each encoder level + neck
        self.res_nets = nn.Sequential(*[
            ResNet(
                channels=self.encoder_features[l],
                size_hw=self.encoder_sizes_hw[l],
                num_blocks=self.N_res,
                use_bn=use_bn,
            )
            for l in range(self.N_layers)
        ])
        self.res_net_neck = ResNet(
            channels=self.encoder_features[self.N_layers],
            size_hw=self.encoder_sizes_hw[self.N_layers],
            num_blocks=self.N_res_neck,
            use_bn=use_bn,
        )

    def forward(self, x):
        x = (x - self.par['inp_shift']) / self.par['inp_scale']

        x = self.lift(x)
        skip = []

        # Encoder with per-level ResNet
        for i in range(self.N_layers):
            y = self.res_nets[i](x)
            skip.append(y)
            x = self.encoder[i](x)

        # Neck
        x = self.res_net_neck(x)

        # Decoder
        for i in range(self.N_layers):
            if i == 0:
                x = self.ED_expansion[self.N_layers - i](x)
            else:
                x = torch.cat((x, self.ED_expansion[self.N_layers - i](skip[-i])), dim=1)
            x = self.decoder[i](x)

        # Project
        x = torch.cat((x, self.ED_expansion[0](skip[0])), dim=1)
        x = self.project(x)

        out = x * self.par["out_scale"] + self.par["out_shift"]
        return out
