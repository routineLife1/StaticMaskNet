import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")


def gradient(data):
    D_dy = data[:, :, 1:] - data[:, :, :-1]
    D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
    return D_dx, D_dy


def grad_loss(img):
    img_dx, img_dy = gradient(img)
    dx, dy = gradient(img)
    loss_x = dx.abs()
    loss_y = dy.abs()

    return loss_x.mean() + loss_y.mean()


def grad_loss_2(img):
    img_dx, img_dy = gradient(img)
    dx, dy = gradient(img)
    loss_x = dx ** 2
    loss_y = dy ** 2

    return loss_x.mean() + loss_y.mean()


def smooth_grad_2nd(flo, image, alpha=10):
    img_dx, img_dy = gradient(image)
    weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
    weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

    dx, dy = gradient(flo)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)

    loss_x = weights_x[:, :, :, 1:] * dx2.abs()
    loss_y = weights_y[:, :, 1:, :] * dy2.abs()

    return loss_x.mean() / 2. + loss_y.mean() / 2.


class Charbonnier_L1(nn.Module):
    def __init__(self):
        super(Charbonnier_L1, self).__init__()

    def forward(self, diff, mask=None):
        if mask is None:
            loss = ((diff ** 2 + 1e-6) ** 0.5).mean()
        else:
            loss = (((diff ** 2 + 1e-6) ** 0.5) * mask).mean() / (mask.mean() + 1e-9)
        return loss
