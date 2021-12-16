from typing import Tuple

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.optim import Adam

from geometry.intrinsic import pixels_to_rays, undistort, perspective_projection
from geometry.extrinsic import world_to_camera


def _orthographic_reconstruction(x: Tensor, R: Tensor, t: Tensor) -> Tensor:
    P = torch.cat([R, t], dim=-1)

    A = x * P[..., -1:, :-1] - P[..., :-1, :-1]
    b = -(x * P[..., -1:, -1:] - P[..., :-1, -1:])

    A, b = A.flatten(-3, -2), b.flatten(-3, -2)

    X = (A.transpose(-1, -2) @ A).inverse() @ A.transpose(-1, -2) @ b
    return X


def orthographic_reconstruction(
    p: Tensor,
    c: Tensor,
    R: Tensor,
    t: Tensor,
    K: Tensor,
    dist_coef: Tensor,
    threshold: float,
    min_views: int,
    verbose: bool = False,
) -> Tuple[Tensor, ...]:
    """

    :param p: [V, N, 2, 1]
    :param c: [V, N, 1, 1]
    :param R: [V, 3, 3]
    :param t: [V, 3, 1]
    :param K: [V, 3, 3]
    :param dist_coef: [V, 5]
    :return: [N, 3, 1]
    """
    n = p.shape[-3]
    if verbose:
        from tqdm import trange

        iterator = trange
    else:
        iterator = range

    m = c > threshold

    p = pixels_to_rays(xy=p, K=K[:, None, :, :])
    p = undistort(xy=p, dist_coef=dist_coef[:, None, :])

    P = torch.zeros(n, 3, 1)
    C = torch.zeros(n, 1, 1)
    for i in iterator(n):
        p_ = p[:, i, :, :]
        c_ = c[:, i, :, :]
        m_ = m[:, i, :, :].squeeze()
        if m_.sum() < min_views:
            m_[:] = 0.0
        else:
            P[i, :, :] = _orthographic_reconstruction(
                x=p_[m_, :, :],
                R=R[m_, :, :],
                t=t[m_, :, :],
            )
            C[i, :, :] = c_[m_, :, :].mean(dim=-3)
    return P, C


def perspective_reconstruction(
    p: Tensor,
    c: Tensor,
    R: Tensor,
    t: Tensor,
    K: Tensor,
    dist_coef: Tensor,
    threshold: float,
    min_views: int,
    max_iterations: int,
    verbose: bool = False,
):
    n = p.shape[-3]
    if verbose:
        from tqdm import trange

        iterator = trange
    else:
        iterator = range

    m = c > threshold

    p = pixels_to_rays(xy=p, K=K[:, None, :, :])
    p = undistort(xy=p, dist_coef=dist_coef[:, None, :])

    P = torch.zeros(n, 3, 1)
    C = torch.zeros(n, 1, 1)
    for i in iterator(n):
        p_ = p[:, i, :, :]
        c_ = c[:, i, :, :]
        m_ = m[:, i, :, :].squeeze()
        if m_.sum() < min_views:
            m_[:] = 0.0
        else:
            P[i, :, :] = _orthographic_reconstruction(
                x=p_[m_, :, :],
                R=R[m_, :, :],
                t=t[m_, :, :],
            )
            C[i, :, :] = c_[m_, :, :].mean(dim=-3)

    P.requires_grad_()
    optimizer = Adam([P])
    for i in iterator(max_iterations):
        optimizer.zero_grad()

        p_ = perspective_projection(
            xyz=world_to_camera(
                xyz=P[None, ...],
                R=R[..., None, :, :],
                t=t[..., None, :, :],
            ),
        )

        loss = F.l1_loss(input=m * p_, target=m * p)
        loss.backward()

        optimizer.step()

    return P.detach(), C
