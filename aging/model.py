"""Joint registration and patch-wise age-interval regression from the paper."""

import itertools

import torch
from torch import nn

from .spatial import ScalingAndSquaring, SpatialTransformer, jacobian_determinant


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.main = nn.Conv3d(in_channels, out_channels, 3, stride, 1)
        self.norm = nn.GroupNorm(out_channels // 4, out_channels, eps=1e-5)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.activation(self.norm(self.main(x)))


class RegistrationUNet(nn.Module):
    """Concatenated-image U-Net retained from RegV4's DiffeoGenerator_V4."""

    def __init__(self):
        super().__init__()
        encoder = (16, 32, 32, 32)
        decoder = (32, 32, 32, 32, 32, 16, 16)
        self.encoder = nn.ModuleList()
        channels = 2
        for width in encoder:
            self.encoder.append(ConvBlock(channels, width, stride=2))
            channels = width
        self.decoder = nn.ModuleList()
        history = list(reversed(encoder))
        for index, width in enumerate(decoder[:4]):
            inputs = channels + history[index] if index else channels
            self.decoder.append(ConvBlock(inputs, width))
            channels = width
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.extra = nn.ModuleList()
        channels += 2
        for width in decoder[4:]:
            self.extra.append(ConvBlock(channels, width))
            channels = width

    def forward(self, x):
        history = [x]
        for layer in self.encoder:
            history.append(layer(history[-1]))
        x = history.pop()
        for layer in self.decoder:
            x = torch.cat((self.upsample(layer(x)), history.pop()), dim=1)
        for layer in self.extra:
            x = layer(x)
        return x


class Registration(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.unet = RegistrationUNet()
        self.flow = nn.Conv3d(16, 3, 3, padding=1)
        nn.init.normal_(self.flow.weight, mean=0, std=1e-5)
        nn.init.zeros_(self.flow.bias)
        self.integrate = ScalingAndSquaring(shape, steps=7)
        self.transformer = SpatialTransformer(shape)

    def forward(self, starting_image, followup_image):
        velocity = self.flow(self.unet(torch.cat((starting_image, followup_image), dim=1)))
        displacement = self.integrate(velocity)
        warped = self.transformer(starting_image, displacement)
        return warped, velocity, displacement


def local_regressor():
    # Preserve the widths actually instantiated by MOENet.encoder in RegV4.
    channels = (16, 32, 32, 32, 16)
    layers = []
    previous = 1
    for index, width in enumerate(channels):
        last = index == len(channels) - 1
        layers.extend([
            nn.Conv3d(previous, width, 1 if last else 3, padding=0 if last else 1),
            nn.BatchNorm3d(width),
        ])
        if not last:
            layers.append(nn.MaxPool3d(2))
        layers.append(nn.ReLU())
        previous = width
    layers.extend([nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(channels[-1], 1)])
    return nn.Sequential(*layers)


class LocalToGlobal(nn.Module):
    """Independent spatial regressors and population-level softmax weights."""

    def __init__(self, shape, patches_per_axis=4):
        super().__init__()
        self.experts = nn.ModuleList(local_regressor() for _ in range(patches_per_axis ** 3))
        self.relevance_logits = nn.Parameter(torch.randn(patches_per_axis ** 3))
        self.patches = []
        widths = [size // patches_per_axis for size in shape]
        for indexes in itertools.product(range(patches_per_axis), repeat=3):
            self.patches.append(tuple(
                slice(index * width, (index + 1) * width
                      if index < patches_per_axis - 1 else size)
                for index, width, size in zip(indexes, widths, shape)
            ))

    def forward(self, jacobian):
        local = torch.cat([
            expert(jacobian[(slice(None), slice(None), *patch)])
            for expert, patch in zip(self.experts, self.patches)
        ], dim=1)
        weights = self.relevance_logits.softmax(dim=0)
        global_interval = (local * weights).sum(dim=1, keepdim=True)
        return local, global_interval


class BrainAgingModel(nn.Module):
    """Predict local and global age intervals (years) from longitudinal MRI.

    Inputs: two (B, 1, D, H, W) images, individually scaled to [0, 1].
    Returned tensors support the four training losses; no gradients are
    detached between the registration network and the local regressors.
    """

    def __init__(self, shape=(176, 192, 176), patches_per_axis=4):
        super().__init__()
        self.shape = tuple(shape)
        if len(shape) != 3 or any(size <= 0 or size % 16 for size in shape):
            raise ValueError("Each of the three image dimensions must be a positive multiple of 16")
        if patches_per_axis < 1 or any(size // patches_per_axis < 32 for size in shape):
            raise ValueError("Each patch must contain at least 32 voxels along each axis")
        self.registration = Registration(shape)
        self.predictor = LocalToGlobal(shape, patches_per_axis)

    def forward(self, starting_image, followup_image):
        if (starting_image.ndim != 5 or starting_image.shape[1] != 1
                or tuple(starting_image.shape[2:]) != self.shape
                or starting_image.shape != followup_image.shape):
            raise ValueError(f"Both inputs must have shape (B, 1, {self.shape})")
        warped, velocity, displacement = self.registration(starting_image, followup_image)
        jacobian = jacobian_determinant(displacement) * (followup_image > 0)
        local, global_interval = self.predictor(jacobian)
        return {"warped": warped, "velocity": velocity,
                "local_interval": local, "global_interval": global_interval}
