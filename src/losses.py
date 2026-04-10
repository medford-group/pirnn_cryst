"""Loss functions for physics-informed training."""

import torch
import torch.nn as nn


def lossX(pred, target, noise_scale=None):
    """Data-fitting loss: weighted combination of MSE and Huber loss."""
    mse = nn.MSELoss()(pred, target)
    huber = nn.HuberLoss(delta=0.1)(pred, target)

    if noise_scale is not None:
        weight = torch.sigmoid(-noise_scale) # Higher noise -> lower weight on MSE
        return weight * mse + (1 - weight) * huber
    else:
        return 0.7 * mse + 0.3 * huber


def lossG(pred, T_seq, rhs, dt):
    """Physics loss: MSE between finite-difference derivatives and PBM."""
    dx = (pred[:, 2:] - pred[:, :-2]) / (2 * dt)
    x_mid = pred[:, 1:-1]
    T_mid = T_seq[:, 1:-1, 0]
    rhs_vals = rhs(None, x_mid, T_mid)
    return nn.MSELoss()(dx, rhs_vals)


def lossSmooth(pred):
    """Smoothness regularisation"""
    d2x = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
    return torch.mean(d2x ** 2)
