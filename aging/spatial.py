"""Differentiable warping, velocity integration, and volume-change maps.

Registration layers are adapted from RegV4's VoxelMorph-derived implementation.
All flow channels follow array-axis order (D, H, W), in voxel units.
"""

import torch
from torch import nn
from torch.nn import functional as F


class SpatialTransformer(nn.Module):
    def __init__(self, shape):
        super().__init__()
        grid = torch.stack(torch.meshgrid(
            *(torch.arange(n, dtype=torch.float32) for n in shape), indexing="ij"
        )).unsqueeze(0)
        self.register_buffer("grid", grid, persistent=False)

    def forward(self, image, displacement):
        coordinates = self.grid + displacement
        coordinates = torch.stack([
            2 * coordinates[:, axis] / (size - 1) - 1
            for axis, size in enumerate(displacement.shape[2:])
        ], dim=-1)
        # grid_sample expects (x, y, z), i.e. the reverse of array-axis order.
        return F.grid_sample(
            image, coordinates[..., [2, 1, 0]], mode="bilinear",
            padding_mode="zeros", align_corners=True,
        )


class ScalingAndSquaring(nn.Module):
    def __init__(self, shape, steps=7):
        super().__init__()
        self.steps = steps
        self.transformer = SpatialTransformer(shape)

    def forward(self, velocity):
        displacement = velocity / (2 ** self.steps)
        for _ in range(self.steps):
            displacement = displacement + self.transformer(displacement, displacement)
        return displacement


def jacobian_determinant(displacement):
    """Return det(I + grad(displacement)) as (B, 1, D, H, W).

The component and derivative axes both use (D, H, W). This avoids mixing
the grid_sample coordinate convention with the network's flow convention.
Finite differences use voxel coordinates and remain differentiable.
"""
    if displacement.ndim != 5 or displacement.shape[1] != 3:
        raise ValueError("Expected displacement with shape (B, 3, D, H, W)")
    d, h, w = torch.gradient(displacement, dim=(2, 3, 4))
    a, b, c = d[:, 0] + 1, h[:, 0], w[:, 0]
    e, f, g = d[:, 1], h[:, 1] + 1, w[:, 1]
    i, j, k = d[:, 2], h[:, 2], w[:, 2] + 1
    return (a * (f * k - g * j) - b * (e * k - g * i)
            + c * (e * j - f * i)).unsqueeze(1)
