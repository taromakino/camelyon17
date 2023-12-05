import torch.nn as nn
from torchvision.models import densenet121


class EncoderCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.densenet = densenet121()
        del self.densenet.classifier
        self.densenet.classifier = nn.Identity()

    def forward(self, x):
        return self.densenet(x)