import numpy as np
import torch.nn as nn
from torch import Tensor
from typing import Tuple
from torchvision.models.densenet import _DenseBlock


IMG_DECODE_SHAPE = (24, 6, 6)
IMG_DECODE_SIZE = np.prod(IMG_DECODE_SHAPE)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.ConvTranspose2d(num_output_features, num_output_features, 2, stride=2)


class DecoderCNN(nn.Module):
    def __init__(
        self,
        growth_rate: int = 8,
        block_config: Tuple[int, int, int, int] = (3, 3, 3, 3, 3),
        num_init_features: int = 24,
        bn_size: int = 4,
        drop_rate: float = 0,
        memory_efficient: bool = False,
    ) -> None:

        super().__init__()

        self.features = nn.Sequential()

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module("out", nn.Conv2d(num_features, 3, kernel_size=3, padding=1))

    def forward(self, x: Tensor) -> Tensor:
        out = self.features(x)
        return out