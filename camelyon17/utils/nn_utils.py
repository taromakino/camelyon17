import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


COV_OFFSET = 1e-6


class ResidualLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.shortcut = nn.Identity() if input_size == output_size else nn.Linear(input_size, output_size)

    def forward(self, x):
        return F.leaky_relu(self.linear(x)) + self.shortcut(x)


class ResidualMLP(nn.Module):
    def __init__(self, input_size, h_sizes, output_size):
        super().__init__()
        module_list = []
        last_size = input_size
        for h_size in h_sizes:
            module_list.append(ResidualLayer(last_size, h_size))
            last_size = h_size
        module_list.append(nn.Linear(last_size, output_size))

        self.module_list = nn.Sequential(*module_list)

    def forward(self, *args):
        return self.module_list(torch.hstack(args))


def make_dataloader(data_tuple, batch_size, is_train):
    return DataLoader(TensorDataset(*data_tuple), shuffle=is_train, batch_size=batch_size)


def one_hot(categorical, n_categories):
    batch_size = len(categorical)
    out = torch.zeros((batch_size, n_categories), device=categorical.device)
    out[torch.arange(batch_size), categorical] = 1
    return out


def arr_to_cov(low_rank, diag):
    return torch.bmm(low_rank, low_rank.transpose(1, 2)) + torch.diag_embed(F.softplus(diag) + torch.full_like(diag,
        COV_OFFSET))