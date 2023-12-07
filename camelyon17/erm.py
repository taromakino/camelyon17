import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from vae import IMG_ENCODE_SIZE, EncoderCNN
from torch.optim import Adam
from torchmetrics import Accuracy


class ERM(pl.LightningModule):
    def __init__(self, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.cnn = EncoderCNN()
        self.fc = nn.Linear(IMG_ENCODE_SIZE, 1)
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_metric = Accuracy('binary')
        self.val_metric = Accuracy('binary')
        self.eval_metric = Accuracy('binary')

    def forward(self, x, y, e):
        batch_size = len(x)
        x = self.cnn(x).view(batch_size, -1)
        y_pred = self.fc(x).view(-1)
        return y_pred, y

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