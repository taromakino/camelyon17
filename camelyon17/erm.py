import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from utils.nn_utils import SkipMLP
from vae import IMG_EMBED_SIZE, CNN
from torch.optim import Adam
from torchmetrics import Accuracy


class ERMBase(pl.LightningModule):
    def __init__(self, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_metric = Accuracy('binary')
        self.val_metric = Accuracy('binary')
        self.eval_metric = Accuracy('binary')

    def training_step(self, batch, batch_idx):
        y_pred, y = self(*batch)
        loss = F.binary_cross_entropy_with_logits(y_pred, y.float())
        self.train_metric.update(y_pred, y)
        return loss

    def on_train_epoch_end(self):
        self.log('train_metric', self.train_metric.compute())

    def validation_step(self, batch, batch_idx):
        y_pred, y = self(*batch)
        loss = F.binary_cross_entropy_with_logits(y_pred, y.float())
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.val_metric.update(y_pred, y)

    def on_validation_epoch_end(self):
        self.log('val_metric', self.val_metric.compute())

    def test_step(self, batch, batch_idx):
        y_pred, y = self(*batch)
        self.eval_metric.update(y_pred, y)

    def on_test_epoch_end(self):
        self.log('eval_metric', self.eval_metric.compute())

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class ERM_X(ERMBase):
    def __init__(self, h_sizes, lr, weight_decay):
        super().__init__(lr, weight_decay)
        self.save_hyperparameters()
        self.cnn = CNN()
        self.mlp = SkipMLP(IMG_EMBED_SIZE, h_sizes, 1)

    def forward(self, x, y, e):
        batch_size = len(x)
        x = self.cnn(x).view(batch_size, -1)
        y_pred = self.mlp(x).view(-1)
        return y_pred, y


class ERM_ZC(ERMBase):
    def __init__(self, z_size, h_sizes, lr, weight_decay):
        super().__init__(lr, weight_decay)
        self.save_hyperparameters()
        self.mlp = SkipMLP(z_size, h_sizes, 1)

    def forward(self, z, y, e):
        z_c, z_s = torch.chunk(z, 2, dim=1)
        y_pred = self.mlp(z_c).view(-1)
        return y_pred, y