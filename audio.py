from torch import Tensor, nn
import torch
import torch.nn.functional as F
from modules import DecoderLayer


class RMSNorm(torch.nn.Module):
    def __init__(self, scale: float = 30.0, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.scale * torch.rsqrt((x**2).mean(dim=-1, keepdim=True) + self.eps)
        return x


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
        self.norm = nn.InstanceNorm1d(in_channels)
        self.nonlinear = nn.SiLU(inplace=True)
        self.conv = nn.Conv1d(
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
        self.down = nn.Conv1d(in_channels, out_channels, kernel_size=4, stride=4)
        self.residual = Residual(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.down(x)
        x = self.residual(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=4)
        self.residual = Residual(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.up(x)
        x = self.residual(x)
        return x


class Encoder1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        num_layers: int,
        latent_size: int = 256,
    ):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels, base_channels, 3, 1, 1)
        self.layers = nn.ModuleList(
            [
                DownBlock(base_channels * (2**i), base_channels * (2 ** (i + 1)))
                for i in range(num_layers)
            ]
        )
        last_channels = base_channels * 2**num_layers
        self.norm_out = nn.InstanceNorm1d(last_channels)
        self.nonlinear_out = nn.SiLU(inplace=True)
        self.conv_latent_out = nn.Conv1d(
            last_channels, latent_size, kernel_size=3, stride=1, padding=1
        )

        self.attention = DecoderLayer(last_channels, 128, last_channels * 8 // 3)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_in(x)
        for layer in self.layers:
            x = layer(x)
        x = x + self.attention(x.transpose(-1, -2))[0].transpose(-1, -2)
        x = self.norm_out(x)
        x = self.nonlinear_out(x)
        x = self.conv_latent_out(x)
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        num_layers: int,
        latent_size: int = 256,
        latent_scale: float = 30.0,
    ):
        super().__init__()
        self.latent_scale = latent_scale
        self.conv_in = nn.Conv1d(in_channels, base_channels, 3, 1, 1)
        self.layers = nn.ModuleList(
            [
                DownBlock(base_channels * (2**i), base_channels * (2 ** (i + 1)))
                for i in range(num_layers)
            ]
        )
        last_channels = base_channels * 2**num_layers
        self.norm_out = nn.InstanceNorm1d(last_channels)
        self.nonlinear_out = nn.SiLU(inplace=True)
        self.conv_latent_out = nn.Conv1d(
            last_channels, latent_size, kernel_size=3, stride=1, padding=1
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_in(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_out(x)
        x = self.nonlinear_out(x)
        x = self.conv_latent_out(x)
        x = self.sigmoid(x)
        return x


class Decoder1d(nn.Module):
    def __init__(
        self,
        out_channels: int,
        base_channels: int,
        num_layers: int,
        latent_size: int,
    ):
        super().__init__()

        in_channels = base_channels * (2**num_layers)
        self.attention = DecoderLayer(in_channels, 128, in_channels * 8 // 3)
        self.conv_latent_in = nn.Conv1d(
            latent_size,
            base_channels * 2**num_layers,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.layers = nn.ModuleList(
            [
                UpBlock(base_channels * (2**i), base_channels * (2 ** (i - 1)))
                for i in range(num_layers, 0, -1)
            ]
        )
        self.norm_out = nn.InstanceNorm1d(base_channels)
        self.nonlinear_out = nn.SiLU(inplace=True)
        self.conv_out = nn.Conv1d(
            base_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.nonlinear_out = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_latent_in(x)
        x = x + self.attention(x.transpose(-1, -2))[0].transpose(-1, -2)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_out(x)
        x = self.nonlinear_out(x)
        x = self.conv_out(x)
        return x


class Autoencoder1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        num_layers: int,
        latent_size: int = 4,
    ):
        super().__init__()
        self.encoder = Encoder1d(
            in_channels,
            base_channels,
            num_layers,
            latent_size,
        )
        self.decoder = Decoder1d(in_channels, base_channels, num_layers, latent_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:
        return self.decoder(x)
