import torch
from torch import nn
from torch.nn.utils.parametrizations import orthogonal


class Orthogonal(nn.Module):
    def __init__(self, hidden_dim, orthogonal_map):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.orthogonal_map = orthogonal_map
        self.orth_linear = orthogonal(nn.Linear(self.hidden_dim, self.hidden_dim))

    def forward(self, x):
        U = self.orth_linear.weight.to(x.device)
        x = x @ U
        return x

    def inverse(self, x):
        U_t = self.orth_linear.weight.to(x.device).T
        x = x @ U_t
        return x
