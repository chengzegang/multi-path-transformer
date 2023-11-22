import math
from modules import Decoder, MHRMSNorm, MHLinear, Mixin
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class MHConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        super(MHConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.mixin = Mixin(out_channels)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight.data.normal_(std=0.02)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, inputs: Tensor):
        B = inputs.shape[0]
        inputs = inputs.reshape(-1, self.in_channels, *inputs.shape[-2:])
        outputs = F.conv2d(inputs, self.weight, self.bias, self.stride, self.padding)
        H, W = outputs.shape[-2:]
        outputs = (
            outputs.reshape(B, -1, self.out_channels, *outputs.shape[-2:])
            .permute(0, -2, -1, 1, 2)
            .flatten(1, 2)
        )
        scores = self.mixin(outputs)
        scores = (
            scores.reshape(B, H, W, -1, self.out_channels)
            .permute(0, 3, 4, 1, 2)
            .flatten(1, 2)
        ).contiguous()
        return scores


class MHConvTranspose2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        super(MHConvTranspose2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.weight.data.normal_(std=0.02)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, inputs: Tensor):
        B, C, H, W = inputs.shape
        inputs = inputs.reshape(-1, self.in_channels, *inputs.shape[-2:])
        outputs = F.conv_transpose2d(
            inputs, self.weight, self.bias, self.stride, self.padding
        )
        outputs = outputs.reshape(B, -1, *outputs.shape[-2:])
        return outputs


class MHVisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        hidden_size: int,
        patch_size: int,
        num_layers: int,
        head_size: int,
    ):
        super(MHVisionTransformer, self).__init__()
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.head_size = head_size
        self.token_embeds = MHConv2d(3, hidden_size, patch_size, patch_size)
        self.decoder = Decoder(hidden_size, num_layers, head_size)
        self.head = MHConvTranspose2d(hidden_size, 3, patch_size, patch_size)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = F.pixel_unshuffle(inputs, inputs.shape[-1] // self.image_size)
        token_embeds = self.token_embeds(inputs)
        shape = token_embeds.shape
        token_embeds = token_embeds.flatten(-2).transpose(-1, -2)
        logits = self.decoder(token_embeds)[0]
        logits = logits.transpose(-1, -2).view(*shape)
        outputs = self.head(logits)
        outputs = F.pixel_shuffle(outputs, int(math.sqrt(outputs.shape[1] // 3)))
        return outputs


class MPVideoTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        hidden_size: int,
        patch_size: int,
        num_layers: int,
        head_size: int,
    ):
        super(MHVisionTransformer, self).__init__()
        self.image_size = image_size
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.head_size = head_size
        self.token_embeds = MHConv2d(3, hidden_size, patch_size, patch_size)
        self.decoder = Decoder(hidden_size, num_layers, head_size)
        self.head = MHConvTranspose2d(hidden_size, 3, patch_size, patch_size)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = F.pixel_unshuffle(inputs, inputs.shape[-1] // self.image_size)
        token_embeds = self.token_embeds(inputs)
        shape = token_embeds.shape
        token_embeds = token_embeds.flatten(-2).transpose(-1, -2)
        logits = self.decoder(token_embeds)[0]
        logits = logits.transpose(-1, -2).view(*shape)
        outputs = self.head(logits)
        outputs = F.pixel_shuffle(outputs, int(math.sqrt(outputs.shape[1] // 3)))
        return outputs
