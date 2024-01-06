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
        self.encoder_cnn = EncoderCNN()
        # Causal
        self.mu_causal = SkipMLP(IMG_ENCODE_SIZE, h_sizes, z_size)
        self.offdiag_causal = SkipMLP(IMG_ENCODE_SIZE, h_sizes, z_size ** 2)
        self.diag_causal = SkipMLP(IMG_ENCODE_SIZE, h_sizes, z_size)
        # Spurious
        self.mu_spurious = SkipMLP(IMG_ENCODE_SIZE + N_CLASSES + N_ENVS, h_sizes, z_size)
        self.offdiag_spurious = SkipMLP(IMG_ENCODE_SIZE + N_CLASSES + N_ENVS, h_sizes, z_size ** 2)
        self.diag_spurious = SkipMLP(IMG_ENCODE_SIZE + N_CLASSES + N_ENVS, h_sizes, z_size)

    def causal_dist(self, x):
        batch_size = len(x)
        mu = self.mu_causal(x)
        offdiag = self.offdiag_causal(x)
        offdiag = offdiag.reshape(batch_size, self.z_size, self.z_size)
        diag = self.diag_causal(x)
        cov = arr_to_cov(offdiag, diag)
        return D.MultivariateNormal(mu, cov)

    def spurious_dist(self, x, y, e):
        batch_size = len(x)
        y_one_hot = one_hot(y, N_CLASSES)
        e_one_hot = one_hot(e, N_ENVS)
        mu = self.mu_spurious(x, y_one_hot, e_one_hot)
        offdiag = self.offdiag_spurious(x, y_one_hot, e_one_hot)
        offdiag = offdiag.reshape(batch_size, self.z_size, self.z_size)
        diag = self.diag_spurious(x, y_one_hot, e_one_hot)
        cov = arr_to_cov(offdiag, diag)
        return D.MultivariateNormal(mu, cov)

    def forward(self, x, y, e):
        x = self.encoder_cnn(x).flatten(start_dim=1)
        causal_dist = self.causal_dist(x)
        spurious_dist = self.spurious_dist(x, y, e)
        return causal_dist, spurious_dist


class Decoder(nn.Module):
    def __init__(self, z_size, h_sizes):
        super().__init__()
        self.mlp = SkipMLP(2 * z_size, h_sizes, IMG_DECODE_SIZE)
        self.decoder_cnn = DecoderCNN()

    def forward(self, z):
        batch_size = len(z)
        x_pred = self.mlp(z).view(batch_size, *IMG_DECODE_SHAPE)
        x_pred = self.decoder_cnn(x_pred).view(batch_size, -1)
        return x_pred


class Prior(nn.Module):
    def __init__(self, z_size, init_sd):
        super().__init__()
        # Causal
        self.mu_causal = nn.Parameter(torch.zeros(z_size))
        self.offdiag_causal = nn.Parameter(torch.zeros(z_size, z_size))
        self.diag_causal = nn.Parameter(torch.zeros(z_size))
        nn.init.normal_(self.mu_causal, 0, init_sd)
        nn.init.normal_(self.offdiag_causal, 0, init_sd)
        nn.init.normal_(self.diag_causal, 0, init_sd)
        # Spurious
        self.mu_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size))
        self.offdiag_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size, z_size))
        self.diag_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size))
        nn.init.normal_(self.mu_spurious, 0, init_sd)
        nn.init.normal_(self.offdiag_spurious, 0, init_sd)
        nn.init.normal_(self.diag_spurious, 0, init_sd)

    def causal_dist(self, batch_size):
        mu = repeat_batch(self.mu_causal, batch_size)
        offdiag = repeat_batch(self.offdiag_causal, batch_size)
        diag = repeat_batch(self.diag_causal, batch_size)
        cov = arr_to_cov(offdiag, diag)
        return D.MultivariateNormal(mu, cov)

    def spurious_dist(self, y, e):
        mu = self.mu_spurious[y, e]
        cov = arr_to_cov(self.offdiag_spurious[y, e], self.diag_spurious[y, e])
        return D.MultivariateNormal(mu, cov)

    def forward(self, y, e):
        causal_dist = self.causal_dist(len(y))
        spurious_dist = self.spurious_dist(y, e)
        return causal_dist, spurious_dist


class VAE(pl.LightningModule):
    def __init__(self, task, z_size, h_sizes, y_mult, beta, prior_reg_mult, init_sd, kl_lb, lr, weight_decay):
        super().__init__()
        self.save_hyperparameters()
        self.task = task
        self.y_mult = y_mult
        self.beta = beta
        self.prior_reg_mult = prior_reg_mult
        self.kl_lb = kl_lb
        self.lr = lr
        self.weight_decay = weight_decay
        # q(z_c,z_s|x)
        self.encoder = Encoder(z_size, h_sizes)
        # p(x|z_c, z_s)
        self.decoder = Decoder(z_size, h_sizes)
        # p(z_c,z_s|y,e)
        self.prior = Prior(z_size, init_sd)
        # p(y|z)
        self.classifier = nn.Linear(z_size, 1)
        self.val_id_acc = Accuracy('binary')
        self.val_ood_acc = Accuracy('binary')
        self.test_acc = Accuracy('binary')

    def sample_z(self, dist):
        mu, scale_tril = dist.loc, dist.scale_tril
        batch_size, z_size = mu.shape
        epsilon = torch.randn(batch_size, z_size, 1).to(self.device)
        return mu + torch.bmm(scale_tril, epsilon).squeeze(-1)

    def loss(self, x, y, e):
        batch_size = len(x)
        # z_c,z_s ~ q(z_c,z_s|x)
        posterior_causal, posterior_spurious = self.encoder(x, y, e)
        z_c = self.sample_z(posterior_causal)
        z_s = self.sample_z(posterior_spurious)
        # E_q(z_c,z_s|x)[log p(x|z_c,z_s)]
        z = torch.hstack((z_c, z_s))
        x_pred = self.decoder(z)
        log_prob_x_z = -F.binary_cross_entropy_with_logits(x_pred, x.view(batch_size, -1), reduction='none').sum(dim=1).mean()
        # E_q(z_c|x)[log p(y|z_c)]
        y_pred = self.classifier(z_c).view(-1)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y.float())
        # KL(q(z_c,z_s|x) || p(z_c|e)p(z_s|y,e))
        prior_causal, prior_spurious = self.prior(y, e)
        kl_causal = D.kl_divergence(posterior_causal, prior_causal).mean()
        kl_spurious = D.kl_divergence(posterior_spurious, prior_spurious).mean()
        kl = kl_causal + kl_spurious
        kl = torch.max(torch.full_like(kl, self.kl_lb), kl)
        prior_reg = (torch.hstack((prior_causal.loc, prior_spurious.loc)) ** 2).mean()
        loss = -log_prob_x_z - self.y_mult * log_prob_y_zc + self.beta * kl + self.prior_reg_mult * prior_reg
        return loss

    def training_step(self, batch, batch_idx):
        x, y, e = batch
        loss = self.loss(x, y, e)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y, e = batch
        y_pred = self.classify(x)
        if dataloader_idx == 0:
            loss = self.loss(x, y, e)
            self.log('val_loss', loss, on_step=False, on_epoch=True, add_dataloader_idx=False)
            self.val_id_acc.update(y_pred, y)
        elif dataloader_idx == 1:
            self.val_ood_acc.update(y_pred, y)
        else:
            assert dataloader_idx == 2
            self.test_acc.update(y_pred, y)

    def on_validation_epoch_end(self):
        self.log('val_id_acc', self.val_id_acc.compute())
        self.log('val_ood_acc', self.val_ood_acc.compute())
        self.log('test_acc', self.test_acc.compute())

    def classify(self, x):
        x = self.encoder.encoder_cnn(x).flatten(start_dim=1)
        causal_dist = self.encoder.causal_dist(x)
        z_c = causal_dist.loc
        y_pred = self.classifier(z_c).view(-1)
        return y_pred

    def test_step(self, batch, batch_idx):
        x, y, e = batch
        y_pred = self.classify(x)
        self.test_acc.update(y_pred, y)

    def on_test_epoch_end(self):
        self.log('test_acc', self.test_acc.compute())

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)