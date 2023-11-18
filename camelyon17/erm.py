import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from utils.nn_utils import MLP
from vae import IMG_EMBED_SIZE, CNN
from torch.optim import Adam
from torchmetrics import Accuracy


class ERMBase(pl.LightningModule):
    def __init__(self, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_acc = Accuracy('binary')
        self.val_acc = Accuracy('binary')
        self.eval_acc = Accuracy('binary')

    def training_step(self, batch, batch_idx):
        y_pred, y = self(*batch)
        loss = F.binary_cross_entropy_with_logits(y_pred, y.float())
        self.train_acc.update(y_pred, y)
        return loss

    def on_train_epoch_end(self):
        self.log('train_acc', self.train_acc.compute())

    def validation_step(self, batch, batch_idx):
        y_pred, y = self(*batch)
        self.val_acc.update(y_pred, y)

    def on_validation_epoch_end(self):
        self.log('val_acc', self.val_acc.compute())

    def test_step(self, batch, batch_idx):
        y_pred, y = self(*batch)
        self.eval_acc.update(y_pred, y)

    def on_test_epoch_end(self):
        self.log('eval_acc', self.eval_acc.compute())

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class ERM_X(ERMBase):
    def __init__(self, h_sizes, lr, weight_decay):
        super().__init__(lr, weight_decay)
        self.save_hyperparameters()
        self.cnn = CNN()
        self.mlp = MLP(IMG_EMBED_SIZE, h_sizes, 1)

    def forward(self, x, y, e):
        batch_size = len(x)
        x = self.cnn(x).view(batch_size, -1)
        y_pred = self.mlp(x).view(-1)
        return y_pred, y


class ERM_ZC(ERMBase):
    def __init__(self, z_size, h_sizes, lr, weight_decay):
        super().__init__(lr, weight_decay)
        self.save_hyperparameters()
        self.mlp = MLP(z_size, h_sizes, 1)

    def forward(self, z, y, e):
        z_c, z_s = torch.chunk(z, 2, dim=1)
        y_pred = self.mlp(z_c).view(-1)
        return y_pred, y