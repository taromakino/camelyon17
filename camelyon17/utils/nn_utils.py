import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


COV_OFFSET = 1e-6


class DenseLinear(nn.Module):
    def __init__(self, input_size, growth_rate, bn_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, bn_size * growth_rate)
        self.fc2 = nn.Linear(bn_size * growth_rate, growth_rate)

    def bn_function(self, inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = F.leaky_relu(self.fc1(concated_features))
        return bottleneck_output

    def forward(self, input):
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input
        bottleneck_output = self.bn_function(prev_features)
        new_features = F.leaky_relu(self.fc2(bottleneck_output))
        return new_features


class DenseMLP(nn.Module):
    def __init__(self, input_size, output_size, n_layers, growth_rate, bn_size):
        super().__init__()
        self.dense_layers = nn.ModuleList()
        for i in range(n_layers):
            self.dense_layers.append(DenseLinear(input_size + i * growth_rate, growth_rate, bn_size))
        self.out = nn.Linear(input_size + n_layers * growth_rate, output_size)

    def forward(self, init_features):
        features = [init_features]
        for dense_layer in self.dense_layers:
            new_features = dense_layer(features)
            features.append(new_features)
        return self.out(torch.cat(features, 1))


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