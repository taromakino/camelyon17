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
from utils.nn_utils import MLP, arr_to_cov, one_hot


CNN_SIZE = 512


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        module_list = []
        channels = [3, 32, 64, 128, 256, 512]
        for i in range(1, len(channels)):
            module_list.append(
                nn.Conv2d(
                    channels[i - 1],
                    channels[i],
                    kernel_size=3,
                    stride=2,
                    padding=1)
            )
            module_list.append(nn.LeakyReLU())
            module_list.append(nn.BatchNorm2d(channels[i]))
        module_list.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.module_list = nn.Sequential(*module_list)

    def forward(self, x):
        return self.module_list(x)


class DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [512, 256, 128, 64, 32, 3]
        module_list = []
        module_list.append(nn.Upsample(size=3, mode='nearest'))
        for i in range(1, len(channels)):
            module_list.append(
                nn.ConvTranspose2d(
                    channels[i - 1],
                    channels[i],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1)
            )
            module_list.append(nn.LeakyReLU())
            module_list.append(nn.BatchNorm2d(channels[i]))
        self.module_list = nn.Sequential(*module_list)

    def forward(self, x):
        return self.module_list(x)


class Encoder(nn.Module):
    def __init__(self, z_size, rank, h_sizes):
        super().__init__()
        self.z_size = z_size
        self.rank = rank
        self.cnn = CNN()
        self.mu_causal = MLP(CNN_SIZE + N_ENVS, h_sizes, z_size)
        self.low_rank_causal = MLP(CNN_SIZE + N_ENVS, h_sizes, z_size * rank)
        self.diag_causal = MLP(CNN_SIZE + N_ENVS, h_sizes, z_size)
        self.mu_spurious = MLP(CNN_SIZE + N_CLASSES + N_ENVS, h_sizes, z_size)
        self.low_rank_spurious = MLP(CNN_SIZE + N_CLASSES + N_ENVS, h_sizes, z_size * rank)
        self.diag_spurious = MLP(CNN_SIZE + N_CLASSES + N_ENVS, h_sizes, z_size)

    def forward(self, x, y, e):
        batch_size = len(x)
        x = self.cnn(x).view(batch_size, -1)
        y_one_hot = one_hot(y, N_CLASSES)
        e_one_hot = one_hot(e, N_ENVS)
        # Causal
        mu_causal = self.mu_causal(x, e_one_hot)
        low_rank_causal = self.low_rank_causal(x, e_one_hot)
        low_rank_causal = low_rank_causal.reshape(batch_size, self.z_size, self.rank)
        diag_causal = self.diag_causal(x, e_one_hot)
        cov_causal = arr_to_cov(low_rank_causal, diag_causal)
        # Spurious
        mu_spurious = self.mu_spurious(x, y_one_hot, e_one_hot)
        low_rank_spurious = self.low_rank_spurious(x, y_one_hot, e_one_hot)
        low_rank_spurious = low_rank_spurious.reshape(batch_size, self.z_size, self.rank)
        diag_spurious = self.diag_spurious(x, y_one_hot, e_one_hot)
        cov_spurious = arr_to_cov(low_rank_spurious, diag_spurious)
        # Block diagonal
        mu = torch.hstack((mu_causal, mu_spurious))
        cov = torch.zeros(batch_size, 2 * self.z_size, 2 * self.z_size, device=y.device)
        cov[:, :self.z_size, :self.z_size] = cov_causal
        cov[:, self.z_size:, self.z_size:] = cov_spurious
        return D.MultivariateNormal(mu, scale_tril=torch.linalg.cholesky(cov))


class Decoder(nn.Module):
    def __init__(self, z_size, h_sizes):
        super().__init__()
        self.mlp = MLP(2 * z_size, h_sizes, CNN_SIZE)
        self.dcnn = DCNN()

    def forward(self, x, z):
        batch_size = len(x)
        x_pred = self.mlp(z)[:, :, None, None]
        x_pred = self.dcnn(x_pred).view(batch_size, -1)
        return -F.binary_cross_entropy_with_logits(x_pred, x.view(batch_size, -1), reduction='none').sum(dim=1)


class Prior(nn.Module):
    def __init__(self, z_size, rank, prior_init_sd):
        super().__init__()
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
        posterior_dist = self.encoder(x, y, e)
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

    def infer_loss(self, x, y, e, z):
        # log p(x|z_c,z_s)
        log_prob_x_z = self.decoder(x, z)
        # log p(y|z_c)
        z_c, z_s = torch.chunk(z, 2, dim=1)
        y_pred = self.classifier(z_c).view(-1)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y.float(), reduction='none')
        # log q(z_c,z_s|x,y,e)
        log_prob_z_xye = self.encoder(x, y, e).log_prob(z)
        loss = -log_prob_x_z - self.y_mult * log_prob_y_zc - self.alpha * log_prob_z_xye
        return loss

    def make_z_param(self, x, y_value, e_value):
        batch_size = len(x)
        y = torch.full((batch_size,), y_value, dtype=torch.long, device=self.device)
        e = torch.full((batch_size,), e_value, dtype=torch.long, device=self.device)
        return nn.Parameter(self.encoder(x, y, e).loc.detach())

    def opt_infer_loss(self, x, y_value, e_value):
        batch_size = len(x)
        z_param = self.make_z_param(x, y_value, e_value)
        y = torch.full((batch_size,), y_value, dtype=torch.long, device=self.device)
        e = torch.full((batch_size,), e_value, dtype=torch.long, device=self.device)
        optim = Adam([z_param], lr=self.lr_infer)
        for _ in range(self.n_infer_steps):
            optim.zero_grad()
            loss = self.infer_loss(x, y, e, z_param)
            loss.mean().backward()
            optim.step()
        return loss.detach().clone()

    def infer_z(self, x):
        loss_candidates = []
        y_candidates = []
        for y_value in range(N_CLASSES):
            for e_value in range(N_ENVS):
                loss_candidates.append(self.opt_infer_loss(x, y_value, e_value)[:, None])
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