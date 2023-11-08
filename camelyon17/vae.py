import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from data import N_CLASSES, N_ENVS
from torch.optim import Adam
from torchmetrics import Accuracy
from utils.enums import Task
from utils.nn_utils import MLP, arr_to_tril, arr_to_cov, one_hot


IMAGE_EMBED_SHAPE = (32, 6, 6)
IMAGE_EMBED_SIZE = np.prod(IMAGE_EMBED_SHAPE)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        module_list = []
        channels = [3, 32, 32, 32, 32]
        for i in range(1, len(channels)):
            module_list.append(
                nn.Conv2d(
                    channels[i - 1],
                    channels[i],
                    kernel_size=4,
                    stride=2,
                    padding=1)
            )
            module_list.append(nn.LeakyReLU())
        self.module_list = nn.Sequential(*module_list)

    def forward(self, x):
        return self.module_list(x)


class DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [32, 32, 32, 32, 3]
        module_list = []
        for i in range(1, len(channels)):
            module_list.append(
                nn.ConvTranspose2d(
                    channels[i - 1],
                    channels[i],
                    kernel_size=4,
                    stride=2,
                    padding=1)
            )
            module_list.append(nn.LeakyReLU())
        self.module_list = nn.Sequential(*module_list)

    def forward(self, x):
        return self.module_list(x)


class Encoder(nn.Module):
    def __init__(self, z_size, rank, h_sizes):
        super().__init__()
        self.z_size = z_size
        self.rank = rank
        self.cnn = CNN()
        self.mu = MLP(IMAGE_EMBED_SIZE, h_sizes, 2 * z_size)
        self.low_rank = MLP(IMAGE_EMBED_SIZE, h_sizes, 2 * z_size * 2 * rank)
        self.diag = MLP(IMAGE_EMBED_SIZE, h_sizes, 2 * z_size)

    def forward(self, x):
        batch_size = len(x)
        x = self.cnn(x).view(batch_size, -1)
        mu = self.mu(x)
        low_rank = self.low_rank(x)
        low_rank = low_rank.reshape(batch_size, 2 * self.z_size, 2 * self.rank)
        diag = self.diag(x)
        return D.MultivariateNormal(mu, scale_tril=arr_to_tril(low_rank, diag))


class Decoder(nn.Module):
    def __init__(self, z_size, h_sizes):
        super().__init__()
        self.mlp = MLP(2 * z_size, h_sizes, IMAGE_EMBED_SIZE)
        self.dcnn = DCNN()

    def forward(self, x, z):
        batch_size = len(x)
        x_pred = self.mlp(z).reshape(batch_size, *IMAGE_EMBED_SHAPE)
        x_pred = self.dcnn(x_pred).view(batch_size, -1)
        return -F.binary_cross_entropy_with_logits(x_pred, x.view(batch_size, -1), reduction='none').sum(dim=1)


class Prior(nn.Module):
    def __init__(self, z_size, rank, prior_init_sd):
        super().__init__()
        # p(z_c|e)
        self.z_size = z_size
        self.mu_causal = nn.Parameter(torch.zeros(N_ENVS, z_size))
        self.low_rank_causal = nn.Parameter(torch.zeros(N_ENVS, z_size, rank))
        self.diag_causal = nn.Parameter(torch.zeros(N_ENVS, z_size))
        nn.init.normal_(self.mu_causal, 0, prior_init_sd)
        nn.init.normal_(self.low_rank_causal, 0, prior_init_sd)
        nn.init.normal_(self.diag_causal, 0, prior_init_sd)
        # p(z_s|y,e)
        self.mu_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size))
        self.low_rank_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size, rank))
        self.diag_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size))
        nn.init.normal_(self.mu_spurious, 0, prior_init_sd)
        nn.init.normal_(self.low_rank_spurious, 0, prior_init_sd)
        nn.init.normal_(self.diag_spurious, 0, prior_init_sd)

    def forward(self, y, e):
        batch_size = len(y)
        # Causal
        mu_causal = self.mu_causal[e]
        cov_causal = arr_to_cov(self.low_rank_causal[e], self.diag_causal[e])
        # Spurious
        mu_spurious = self.mu_spurious[y, e]
        cov_spurious = arr_to_cov(self.low_rank_spurious[y, e], self.diag_spurious[y, e])
        # Block diagonal
        mu = torch.hstack((mu_causal, mu_spurious))
        cov = torch.zeros(batch_size, 2 * self.z_size, 2 * self.z_size, device=y.device)
        cov[:, :self.z_size, :self.z_size] = cov_causal
        cov[:, self.z_size:, self.z_size:] = cov_spurious
        return D.MultivariateNormal(mu, cov)


class VAE(pl.LightningModule):
    def __init__(self, task, z_size, rank, h_sizes, prior_init_sd, y_mult, beta, reg_mult, lr, weight_decay, alpha,
            lr_infer, n_infer_steps):
        super().__init__()
        self.save_hyperparameters()
        self.task = task
        self.z_size = z_size
        self.y_mult = y_mult
        self.beta = beta
        self.reg_mult = reg_mult
        self.lr = lr
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.lr_infer = lr_infer
        self.n_infer_steps = n_infer_steps
        # q(z_c,z_s|x)
        self.encoder = Encoder(z_size, rank, h_sizes)
        # p(x|z_c, z_s)
        self.decoder = Decoder(z_size, h_sizes)
        # p(z_c,z_s|y,e)
        self.prior = Prior(z_size, rank, prior_init_sd)
        # p(y|z)
        self.classifier = MLP(z_size, h_sizes, 1)
        self.eval_metric = Accuracy('binary')

    def sample_z(self, dist):
        mu, scale_tril = dist.loc, dist.scale_tril
        batch_size, z_size = mu.shape
        epsilon = torch.randn(batch_size, z_size, 1).to(self.device)
        return mu + torch.bmm(scale_tril, epsilon).squeeze()

    def loss(self, x, y, e):
        # z_c,z_s ~ q(z_c,z_s|x)
        posterior_dist = self.encoder(x)
        z = self.sample_z(posterior_dist)
        # E_q(z_c,z_s|x)[log p(x|z_c,z_s)]
        log_prob_x_z = self.decoder(x, z).mean()
        # E_q(z_c|x)[log p(y|z_c)]
        z_c, z_s = torch.chunk(z, 2, dim=1)
        y_pred = self.classifier(z_c).view(-1)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y.float())
        # KL(q(z_c,z_s|x) || p(z_c|e)p(z_s|y,e))
        prior_dist = self.prior(y, e)
        kl = D.kl_divergence(posterior_dist, prior_dist).mean()
        prior_norm = (prior_dist.loc ** 2).mean()
        return log_prob_x_z, log_prob_y_zc, kl, prior_norm

    def training_step(self, batch, batch_idx):
        assert self.task == Task.VAE
        x, y, e = batch
        log_prob_x_z, log_prob_y_zc, kl, prior_norm = self.loss(x, y, e)
        loss = -log_prob_x_z - self.y_mult * log_prob_y_zc + self.beta * kl + self.reg_mult * prior_norm
        self.log('train_log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
        self.log('train_log_prob_y_zc', log_prob_y_zc, on_step=False, on_epoch=True)
        self.log('train_kl', kl, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        assert self.task == Task.VAE
        x, y, e = batch
        log_prob_x_z, log_prob_y_zc, kl, prior_norm = self.loss(x, y, e)
        loss = -log_prob_x_z - self.y_mult * log_prob_y_zc + self.beta * kl + self.reg_mult * prior_norm
        self.log('val_log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
        self.log('val_log_prob_y_zc', log_prob_y_zc, on_step=False, on_epoch=True)
        self.log('val_kl', kl, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

    def infer_loss(self, x, y, z):
        # log p(y|z_c)
        z_c, z_s = torch.chunk(z, 2, dim=1)
        y_pred = self.classifier(z_c).view(-1)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y.float(), reduction='none')
        # log q(z_c,z_s|x)
        log_prob_z_x = self.encoder(x).log_prob(z)
        loss = self.y_mult * log_prob_y_zc - self.alpha * log_prob_z_x
        return loss

    def opt_infer_loss(self, x, y_value):
        batch_size = len(x)
        z_param = nn.Parameter(self.encoder(x).loc.detach())
        y = torch.full((batch_size,), y_value, dtype=torch.long, device=self.device)
        optim = Adam([z_param], lr=self.lr_infer)
        for _ in range(self.n_infer_steps):
            optim.zero_grad()
            loss = self.infer_loss(x, y, z_param)
            loss.mean().backward()
            optim.step()
        return loss.detach().clone()

    def infer_z(self, x):
        loss_candidates = []
        y_candidates = []
        for y_value in range(N_CLASSES):
            loss_candidates.append(self.opt_infer_loss(x, y_value)[:, None])
            y_candidates.append(y_value)
        loss_candidates = torch.hstack(loss_candidates)
        y_candidates = torch.tensor(y_candidates, device=self.device)
        opt_loss = loss_candidates.min(dim=1)
        y_pred = y_candidates[opt_loss.indices]
        return opt_loss.values.mean(), y_pred

    def test_step(self, batch, batch_idx):
        assert self.task == Task.CLASSIFY
        x, y, e = batch
        with torch.set_grad_enabled(True):
            loss, y_pred = self.infer_z(x)
            self.log('loss', loss, on_step=False, on_epoch=True)
            self.eval_metric.update(y_pred, y)

    def on_test_epoch_end(self):
        assert self.task == Task.CLASSIFY
        self.log('eval_metric', self.eval_metric.compute())

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)