from torch import nn
import torch


class CrossCorrLoss(nn.Module):
    """
    Implementation of pearson correlation coefficient (cross correlation) as a loss function.
    The best correlation means maximizing the coefficient --> minimizing the loss means maximizing
    the coefficient ot minimizing the negative coefficient.
    """

    def __init__(self):
        super(CrossCorrLoss, self).__init__()

    def forward(self, output, target):
        x = output
        y = target

        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        return -(torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))))


class L1NormalizedLoss(nn.Module):
    """
    Implementation of the L1 loss function, with a normalization factor (real value + prediction + 1).
    """

    def __init__(self):
        super(L1NormalizedLoss, self).__init__()

    def forward(self, output, target):
        x = output
        y = target

        return torch.mean(torch.abs(x - y) / (x + y + 1))
