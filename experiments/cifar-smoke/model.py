import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c, *, groups=8):
    return nn.GroupNorm(min(groups, c), c)


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.norm1, self.norm2 = _gn(c_in), _gn(c_out)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class Encoder(nn.Module):
    def __init__(self, *, in_c=3, base_c=64, ch_mults=(1, 2, 4), z_c=4, n_res=2):
        super().__init__()
        self.stem = nn.Conv2d(in_c, base_c, 3, padding=1)
        layers, c_prev = [], base_c
        for i, m in enumerate(ch_mults):
            c = base_c * m
            for _ in range(n_res):
                layers.append(ResBlock(c_prev, c))
                c_prev = c
            if i < len(ch_mults) - 1:
                layers.append(Downsample(c_prev))
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(
            _gn(c_prev), nn.SiLU(), nn.Conv2d(c_prev, 2 * z_c, 3, padding=1)
        )

    def forward(self, x):
        mu, logvar = self.head(self.body(self.stem(x))).chunk(2, dim=1)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, *, out_c=3, base_c=64, ch_mults=(1, 2, 4), z_c=4, n_res=2):
        super().__init__()
        chs = [base_c * m for m in ch_mults]
        c_prev = chs[-1]
        self.stem = nn.Conv2d(z_c, c_prev, 3, padding=1)
        layers = []
        for i, c in enumerate(reversed(chs)):
            for _ in range(n_res):
                layers.append(ResBlock(c_prev, c))
                c_prev = c
            if i < len(chs) - 1:
                layers.append(Upsample(c_prev))
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(
            _gn(c_prev), nn.SiLU(), nn.Conv2d(c_prev, out_c, 3, padding=1)
        )

    def forward(self, z):
        return self.head(self.body(self.stem(z)))


class CifarVaeModel(nn.Module):
    def __init__(self, *, base_c=64, ch_mults=(1, 2, 4), z_c=4, n_res=2):
        super().__init__()
        self.encoder = Encoder(base_c=base_c, ch_mults=ch_mults, z_c=z_c, n_res=n_res)
        self.decoder = Decoder(base_c=base_c, ch_mults=ch_mults, z_c=z_c, n_res=n_res)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * (0.5 * logvar).exp()

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar
