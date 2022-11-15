import torch
from torch import nn
from torch.nn import functional as F
from torch_sparse import SparseTensor


class GRAFFConv(nn.Module):
    def __init__(self, hidden_dim, W_type, omega_type, Q_type="zero", tau=1.):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.W_type = W_type
        self.omega_type = omega_type
        self.Q_type = Q_type
        self.tau = tau
        self._parametrize_W()
        self._parametrize_omega()
        self._parametrize_Q()

    def _parametrize_W(self):
        if self.W_type == "diag_dom":
            self.W_base = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim - 1))
            nn.init.xavier_uniform_(self.W_base)
            self.t_a = nn.Parameter(torch.Tensor(self.hidden_dim).uniform_(-1, 1))
            self.r_a = nn.Parameter(torch.Tensor(self.hidden_dim).uniform_(-1, 1))
        elif self.W_type == "diag":
            self.W_base = nn.Parameter(torch.Tensor(self.hidden_dim).uniform_(-1, 1))
        elif self.W_type == "sum":
            self.W_base = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
            nn.init.xavier_uniform_(self.W_base)
        else:
            raise ValueError(f"W type {self.W_type} not implemented")

    def _get_W(self, device):
        if self.W_type == "diag_dom":
            W_temp = torch.cat([self.W_base, torch.zeros((self.hidden_dim, 1), device=device)], dim=1)
            W = torch.stack([torch.roll(W_temp[i], shifts=i + 1, dims=-1) for i in range(self.hidden_dim)])
            W = (W + W.T) / 2
            W_sum = self.t_a * torch.abs(W).sum(dim=1) + self.r_a
            W = W + torch.diag(W_sum)
        elif self.W_type == "diag":
            W = torch.diag(self.W_base)
        elif self.W_type == "sum":
            W = (self.W_base + self.W_base.T) / 2
        else:
            raise ValueError(f"W type {self.W_type} not implemented")

        return W

    def _parametrize_omega(self):
        # Have omega be a multiple of the identity so that omega and W commute
        if self.omega_type == "identity":
            self.omega_base = 1
        elif self.omega_type == "scalar":
            self.omega_base = nn.Parameter(torch.ones(1))
        else:
            raise ValueError(f"Omega type {self.omega_type} not implemented")

    def _get_omega(self, device):
        if self.omega_type in ["identity", "scalar"]:
            omega = self.omega_base * torch.eye(self.hidden_dim, device=device)
        else:
            raise ValueError(f"Omega type {self.omega_type} not implemented")

        return omega

    def _parametrize_Q(self):
        if self.Q_type == "free":
            self.Q = nn.Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim))
        elif self.Q_type == "zero":
            self.Q = torch.zeros(self.hidden_dim, self.hidden_dim)
        else:
            raise NotImplementedError

    def _get_Q(self, device):
        if self.Q_type in ["free", "zero"]:
            Q = self.Q.to(device)
        else:
            raise ValueError(f"Q type {self.omega_type} not implemented")

        return Q

    def forward(self, x, x_0, norm_adj):
        W = self._get_W(x.device)
        omega = self._get_omega(x.device)
        Q = self._get_Q(x.device)
        x = x + self.tau * (x @ omega + norm_adj @ x @ W + x_0 @ Q)
        return x

