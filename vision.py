from torch import Tensor, nn
import torch.nn.functional as F
from modules import Attention


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(32, in_channels)
        self.nonlinear = nn.SiLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)
        x = self.nonlinear(x)
        x = self.conv(x)
        return x


class Residual(nn.Module):
    def __init__(self, n_channels: int):
        super().__init__()
        self.conv1 = ConvLayer(n_channels, n_channels, 3, 1, 1)
        self.conv2 = ConvLayer(n_channels, n_channels, 3, 1, 1)
        self.shortcut = ConvLayer(n_channels, n_channels, 1, 1, 0)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = residual + x
        return x


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.residual = Residual(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.down(x)
        x = self.residual(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.residual = Residual(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.up(x)
        x = self.residual(x)
        return x


class Attention2d(nn.Module):
    def __init__(self, hidden_size: int, head_size: int):
        super().__init__()
        self.attention = Attention(hidden_size, head_size)


class Encoder2d(nn.Module):
    def __init__(
        self,
        base_channels: int,
        num_layers: int,
        latent_size: int = 256,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(3, base_channels, 3, 1, 1)
        self.layers = nn.ModuleList(
            [
                DownBlock(base_channels * (2**i), base_channels * (2 ** (i + 1)))
                for i in range(num_layers)
            ]
        )

        last_channels = base_channels * 2**num_layers
        self.attention = Attention(last_channels, 128)
        self.norm_out = nn.GroupNorm(32, last_channels)
        self.nonlinear_out = nn.SiLU(inplace=True)
        self.conv_latent_out = nn.Conv2d(
            last_channels, latent_size, kernel_size=3, stride=1, padding=1
        )
        self.scale_norm = nn.InstanceNorm2d(latent_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_in(x)
        for layer in self.layers:
            x = layer(x)
        x = x + self.attention(x.flatten(-2).transpose(-1, -2))[0].transpose(
            -1, -2
        ).view_as(x)
        x = self.norm_out(x)
        x = self.nonlinear_out(x)
        x = self.conv_latent_out(x)
        x = self.scale_norm(x)
        return x


class Decoder2d(nn.Module):
    def __init__(
        self,
        base_channels: int,
        num_layers: int,
        latent_size: int,
    ):
        super().__init__()
        self.conv_latent_in = nn.Conv2d(
            latent_size,
            base_channels * 2**num_layers,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.attention = Attention(base_channels * 2**num_layers, 128)
        self.layers = nn.ModuleList(
            [
                UpBlock(base_channels * (2**i), base_channels * (2 ** (i - 1)))
                for i in range(num_layers, 0, -1)
            ]
        )
        self.norm_out = nn.GroupNorm(32, base_channels)
        self.nonlinear_out = nn.SiLU(inplace=True)
        self.conv_out = nn.Conv2d(base_channels, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_latent_in(x)
        x = x + self.attention(x.flatten(-2).transpose(-1, -2))[0].transpose(
            -1, -2
        ).view_as(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_out(x)
        x = self.nonlinear_out(x)
        x = self.conv_out(x)
        return x


class Autoencoder2d(nn.Module):
    def __init__(
        self,
        base_channels: int,
        num_layers: int,
        latent_size: int = 4,
    ):
        super().__init__()
        self.encoder = Encoder2d(base_channels, num_layers, latent_size)
        self.decoder = Decoder2d(base_channels, num_layers, latent_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)
