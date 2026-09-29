"""Registration, smoothness, local, and global training objectives."""

import torch
from torch.nn import functional as F


def local_ncc(fixed, warped, window=9):
    """Negative mean squared local NCC, as in RegV4 (zero-padded windows)."""
    values = torch.cat((fixed, warped, fixed.square(), warped.square(), fixed * warped), dim=1)
    kernel = fixed.new_ones((5, 1, window, window, window))
    sums = F.conv3d(values, kernel, padding=window // 2, groups=5)
    i, j, ii, jj, ij = sums.chunk(5, dim=1)
    count = window ** 3
    cross = ij - i * j / count
    var_i = ii - i.square() / count
    var_j = jj - j.square() / count
    return -(cross.square() / (var_i * var_j + 1e-5)).mean()


def velocity_smoothness(velocity):
    """RegV4's Grad('l2', loss_mult=2): twice the mean squared difference."""
    return 2 * sum(velocity.diff(dim=axis).square().mean() for axis in (2, 3, 4)) / 3


def training_loss(outputs, followup_image, interval):
    """The paper's 1 : 0.01 : 1 : 1 objective, shared by train and validation."""
    target = interval.reshape(-1, 1)
    terms = {
        "registration": local_ncc(followup_image, outputs["warped"]),
        "smoothness": velocity_smoothness(outputs["velocity"]),
        "local": F.l1_loss(outputs["local_interval"], target.expand_as(outputs["local_interval"])),
        "global": F.l1_loss(outputs["global_interval"], target),
    }
    terms["total"] = terms["registration"] + 0.01 * terms["smoothness"] + terms["local"] + terms["global"]
    return terms
