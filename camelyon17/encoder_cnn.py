import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch import Tensor
from typing import Tuple
from torchvision.models.densenet import _DenseBlock, _Transition


IMG_ENCODE_SHAPE = (48, 6, 6)
IMG_ENCODE_SIZE = np.prod(IMG_ENCODE_SHAPE)


class EncoderCNN(nn.Module):
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

        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv2d(3, num_init_features, kernel_size=3, padding=1, bias=False)),
                    ("norm0", nn.BatchNorm2d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True))
                ]
            )
        )

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

    def forward(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = F.relu(out)
        return out