import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from data import N_CLASSES, N_ENVS
from encoder_cnn import IMG_ENCODE_SIZE, EncoderCNN
from decoder_cnn import IMG_DECODE_SHAPE, IMG_DECODE_SIZE, DecoderCNN
from torch.optim import AdamW
from torchmetrics import Accuracy
from utils.nn_utils import SkipMLP, one_hot, repeat_batch, arr_to_cov


class Encoder(nn.Module):
    def __init__(self, z_size, h_sizes):
        super().__init__()
        self.z_size = z_size
        self.ecnn = EncoderCNN()
        # Parent
        self.mu_parent = SkipMLP(IMG_ENCODE_SIZE, h_sizes, z_size)
        self.offdiag_parent = SkipMLP(IMG_ENCODE_SIZE, h_sizes, z_size ** 2)
        self.diag_parent = SkipMLP(IMG_ENCODE_SIZE, h_sizes, z_size)
        # Child
        self.mu_child = SkipMLP(IMG_ENCODE_SIZE + N_CLASSES + N_ENVS, h_sizes, z_size)
        self.offdiag_child = SkipMLP(IMG_ENCODE_SIZE + N_CLASSES + N_ENVS, h_sizes, z_size ** 2)
        self.diag_child = SkipMLP(IMG_ENCODE_SIZE + N_CLASSES + N_ENVS, h_sizes, z_size)

    def parent_dist(self, x):
        batch_size = len(x)
        mu = self.mu_parent(x)
        offdiag = self.offdiag_parent(x)
        offdiag = offdiag.reshape(batch_size, self.z_size, self.z_size)
        diag = self.diag_parent(x)
        cov = arr_to_cov(offdiag, diag)
        return D.MultivariateNormal(mu, cov)

    def child_dist(self, x, y, e):
        batch_size = len(x)
        y_one_hot = one_hot(y, N_CLASSES)
        e_one_hot = one_hot(e, N_ENVS)
        mu = self.mu_child(x, y_one_hot, e_one_hot)
        offdiag = self.offdiag_child(x, y_one_hot, e_one_hot)
        offdiag = offdiag.reshape(batch_size, self.z_size, self.z_size)
        diag = self.diag_child(x, y_one_hot, e_one_hot)
        cov = arr_to_cov(offdiag, diag)
        return D.MultivariateNormal(mu, cov)

    def forward(self, x, y, e):
        x = self.ecnn(x).flatten(start_dim=1)
        parent_dist = self.parent_dist(x)
        child_dist = self.child_dist(x, y, e)
        return parent_dist, child_dist


class Decoder(nn.Module):
    def __init__(self, z_size, h_sizes):
        super().__init__()
        self.mlp = SkipMLP(z_size, h_sizes, IMG_DECODE_SIZE)
        self.decoder_cnn = DecoderCNN()

    def forward(self, z):
        batch_size = len(z)
        x_pred = self.mlp(z).view(batch_size, *IMG_DECODE_SHAPE)
        x_pred = self.decoder_cnn(x_pred).view(batch_size, -1)
        return x_pred


class Prior(nn.Module):
    def __init__(self, z_size, init_sd):
        super().__init__()
        # Parent
        self.mu_parent = nn.Parameter(torch.zeros(z_size))
        self.offdiag_parent = nn.Parameter(torch.zeros(z_size, z_size))
        self.diag_parent = nn.Parameter(torch.zeros(z_size))
        nn.init.normal_(self.mu_parent, 0, init_sd)
        nn.init.normal_(self.offdiag_parent, 0, init_sd)
        nn.init.normal_(self.diag_parent, 0, init_sd)
        # Child
        self.mu_child = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size))
        self.offdiag_child = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size, z_size))
        self.diag_child = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size))
        nn.init.normal_(self.mu_child, 0, init_sd)
        nn.init.normal_(self.offdiag_child, 0, init_sd)
        nn.init.normal_(self.diag_child, 0, init_sd)

    def parent_dist(self, batch_size):
        mu = repeat_batch(self.mu_parent, batch_size)
        offdiag = repeat_batch(self.offdiag_parent, batch_size)
        diag = repeat_batch(self.diag_parent, batch_size)
        cov = arr_to_cov(offdiag, diag)
        return D.MultivariateNormal(mu, cov)

    def child_dist(self, y, e):
        mu = self.mu_child[y, e]
        cov = arr_to_cov(self.offdiag_child[y, e], self.diag_child[y, e])
        return D.MultivariateNormal(mu, cov)

    def forward(self, y, e):
        parent_dist = self.parent_dist(len(y))
        child_dist = self.child_dist(y, e)
        return parent_dist, child_dist


class VAE(pl.LightningModule):
    def __init__(self, task, z_size, h_sizes, y_mult, prior_reg_mult, init_sd, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.task = task
        self.y_mult = y_mult
        self.prior_reg_mult = prior_reg_mult
        self.lr = lr
        self.weight_decay = weight_decay
        # q(z_c,z_s|x)
        self.encoder = Encoder(z_size, h_sizes)
        # p(x|z_c, z_s)
        self.decoder_parent = Decoder(z_size, h_sizes)
        self.decoder_child = Decoder(z_size, h_sizes)
        # p(z_c,z_s|y,e)
        self.prior = Prior(z_size, init_sd)
        # p(y|z)
        self.classifier = nn.Linear(z_size, 1)
        self.val_id_acc = Accuracy('binary')
        self.test_acc = Accuracy('binary')

    def sample_z(self, dist):
        mu, scale_tril = dist.loc, dist.scale_tril
        batch_size, z_size = mu.shape
        epsilon = torch.randn(batch_size, z_size, 1).to(self.device)
        return mu + torch.bmm(scale_tril, epsilon).squeeze(-1)

    def loss(self, x, y, e):
        batch_size = len(x)
        # z_c,z_s ~ q(z_c,z_s|x)
        posterior_parent, posterior_child = self.encoder(x, y, e)
        z_parent = self.sample_z(posterior_parent)
        z_child = self.sample_z(posterior_child)
        # E_q(z_c,z_s|x)[log p(x|z_c,z_s)]
        x_pred_parent = self.decoder_parent(z_parent)
        x_pred_child = self.decoder_child(z_child)
        x_pred = x_pred_parent + x_pred_child
        log_prob_x_z = -F.binary_cross_entropy_with_logits(x_pred, x.view(batch_size, -1), reduction='none').sum(dim=1).mean()
        # E_q(z_c|x)[log p(y|z_c)]
        y_pred = self.classifier(z_parent).view(-1)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y.float())
        # KL(q(z_c,z_s|x) || p(z_c|e)p(z_s|y,e))
        prior_parent, prior_child = self.prior(y, e)
        kl_parent = D.kl_divergence(posterior_parent, prior_parent).mean()
        kl_child = D.kl_divergence(posterior_child, prior_child).mean()
        kl = kl_parent + kl_child
        prior_reg = (torch.hstack((prior_parent.loc, prior_child.loc)) ** 2).mean()
        loss = -log_prob_x_z - self.y_mult * log_prob_y_zc + kl + self.prior_reg_mult * prior_reg
        return loss

    def training_step(self, batch, batch_idx):
        x, y, e = batch
        loss = self.loss(x, y, e)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y, e = batch
        y_pred = self.classify(x)
        if dataloader_idx == 0:
            self.val_id_acc.update(y_pred, y)
        else:
            assert dataloader_idx == 1
            self.test_acc.update(y_pred, y)

    def on_validation_epoch_end(self):
        self.log('val_id_acc', self.val_id_acc.compute())
        self.log('test_acc', self.test_acc.compute())

    def classify(self, x):
        x = self.encoder.ecnn(x).flatten(start_dim=1)
        parent_dist = self.encoder.parent_dist(x)
        z_parent = parent_dist.loc
        y_pred = self.classifier(z_parent).view(-1)
        return y_pred

    def test_step(self, batch, batch_idx):
        x, y, e = batch
        y_pred = self.classify(x)
        self.test_acc.update(y_pred, y)

    def on_test_epoch_end(self):
        self.log('test_acc', self.test_acc.compute())

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)