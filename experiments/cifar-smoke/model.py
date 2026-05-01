import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels, *, groups=8):
    return nn.GroupNorm(min(groups, channels), channels)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.norm2 = _group_norm(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels=3,
        base_channels=64,
        channel_multipliers=(1, 2, 4, 8),
        latent_channels=4,
        blocks_per_stage=2,
    ):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        layers, prev_channels = [], base_channels
        for i, mult in enumerate(channel_multipliers):
            channels = base_channels * mult
            for _ in range(blocks_per_stage):
                layers.append(ResBlock(prev_channels, channels))
                prev_channels = channels
            if i < len(channel_multipliers) - 1:
                layers.append(Downsample(prev_channels))
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(
            _group_norm(prev_channels),
            nn.SiLU(),
            nn.Conv2d(prev_channels, 2 * latent_channels, 3, padding=1),
        )

    def forward(self, x):
        mu, logvar = self.head(self.body(self.stem(x))).chunk(2, dim=1)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        out_channels=3,
        base_channels=64,
        channel_multipliers=(1, 2, 4, 8),
        latent_channels=4,
        blocks_per_stage=2,
    ):
        super().__init__()
        stage_channels = [base_channels * m for m in channel_multipliers]
        prev_channels = stage_channels[-1]
        self.stem = nn.Conv2d(latent_channels, prev_channels, 3, padding=1)
        layers = []
        for i, channels in enumerate(reversed(stage_channels)):
            for _ in range(blocks_per_stage):
                layers.append(ResBlock(prev_channels, channels))
                prev_channels = channels
            if i < len(stage_channels) - 1:
                layers.append(Upsample(prev_channels))
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(
            _group_norm(prev_channels),
            nn.SiLU(),
            nn.Conv2d(prev_channels, out_channels, 3, padding=1),
        )

    def forward(self, z):
        return self.head(self.body(self.stem(z)))


class CifarVaeModel(nn.Module):
    def __init__(
        self,
        *,
        base_channels=64,
        channel_multipliers=(1, 2, 4, 8),
        latent_channels=4,
        blocks_per_stage=2,
    ):
        super().__init__()
        self.encoder = Encoder(
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            latent_channels=latent_channels,
            blocks_per_stage=blocks_per_stage,
        )
        self.decoder = Decoder(
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            latent_channels=latent_channels,
            blocks_per_stage=blocks_per_stage,
        )

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
