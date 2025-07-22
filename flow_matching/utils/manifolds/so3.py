# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import Tensor

from flow_matching.utils.manifolds import Manifold

########################################################
# Source: https://github.com/lenroe/manitorch
########################################################


@torch.jit.script
def l2norm(vector: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Gradient-save euclidian distance
    """
    return torch.sqrt(torch.sum(vector**2, dim=-1, keepdim=True) + eps)


@torch.jit.script
def get_rot_mat_from_6d(in_representation: Tensor, eps: float = 1e-8) -> Tensor:
    """
    in_representation: (b, *, 6)
    Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H. (2019). On the continuity of rotation representations in neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 5745-5753).
    Returns: Rotation Matrix (b, *, 3, 3)
    """
    assert in_representation.shape[-1] == 6
    a1 = in_representation[..., :3]
    a2 = in_representation[..., 3:]
    a1_norm = a1 / (l2norm(a1, eps=eps))
    b1 = a1_norm
    b2_unn = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = b2_unn / (l2norm(b2_unn, eps=eps))
    b3 = torch.cross(b1, b2, dim=-1)

    rotMat = torch.stack((b1, b2, b3), dim=-1).transpose(-1, -2)

    return rotMat


@torch.jit.script
def delta2rotMat(delta: Tensor, eps: float = 1e-6):
    xd = delta[..., 0]
    yd = delta[..., 1]
    zd = delta[..., 2]

    theta = torch.sqrt(torch.sum(delta**2, dim=-1) + eps)
    s = torch.sinc(theta / np.pi)  # need the non-normalized version of sinc here
    c = (1.0 - torch.cos(theta)) / (eps + theta**2)
    row1 = torch.stack(
        (torch.cos(theta) + c * xd**2, -s * zd + c * xd * yd, s * yd + c * xd * zd),
        -1,
    )
    row2 = torch.stack(
        (s * zd + c * xd * yd, torch.cos(theta) + c * yd**2, -s * xd + c * yd * zd),
        -1,
    )
    row3 = torch.stack(
        (-s * yd + c * xd * zd, s * xd + c * yd * zd, torch.cos(theta) + c * zd**2),
        -1,
    )
    exponential = torch.cat((row1, row2, row3), -1)
    out = exponential.unflatten(-1, (3, 3))

    return out


class SO3(Manifold):

    EPS = {torch.float32: 1e-4, torch.float64: 1e-7}

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        rot = delta2rotMat(u, eps=self.EPS[x.dtype])
        result = x @ rot
        return result

    def logmap(self, x: Tensor, y: Tensor) -> Tensor:
        X = torch.einsum(
            "...ij, ...jk -> ...ik",
            x.transpose(-1, -2),
            y,
        )
        trace = torch.einsum("...ii -> ...", X)
        argacos = (trace - 1.0) / 2.0
        # clip for numerical stability
        argacos = torch.clamp(
            argacos, -1.0 + self.EPS[x.dtype], 1.0 - self.EPS[x.dtype]
        )
        theta = torch.acos(argacos)
        vec = torch.stack(
            (
                X[..., 2, 1] - X[..., 1, 2],
                X[..., 0, 2] - X[..., 2, 0],
                X[..., 1, 0] - X[..., 0, 1],
            ),
            dim=-1,
        )
        result = (1.0 / (2.0 * torch.sinc(theta / np.pi)))[..., None] * vec
        vec_norm = torch.linalg.norm(vec, dim=-1, keepdims=True)
        vec_scaled = (vec * np.pi) / (vec_norm + self.EPS[x.dtype])
        result = torch.where(
            torch.abs(theta.unsqueeze(-1) - np.pi) < 0.01, vec_scaled, result
        )
        return result

    def projx(self, x: Tensor) -> Tensor:
        return get_rot_mat_from_6d(x.flatten(-2, -1)[..., :6], eps=self.EPS[x.dtype])

    def proju(self, x: Tensor, u: Tensor) -> Tensor:
        return u
