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
from utils.nn_utils import MLP, arr_to_cov


IMAGE_EMBED_SHAPE = (32, 6, 6)
IMAGE_EMBED_SIZE = np.prod(IMAGE_EMBED_SHAPE)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.module_list = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.module_list(x)


class DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.module_list = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        return self.module_list(x)


class Encoder(nn.Module):
    def __init__(self, z_size, rank, h_sizes):
        super().__init__()
        self.z_size = z_size
        self.rank = rank
        self.cnn = CNN()
        self.mu_causal = MLP(IMAGE_EMBED_SIZE, h_sizes, N_ENVS * z_size)
        self.low_rank_causal = MLP(IMAGE_EMBED_SIZE, h_sizes, N_ENVS * z_size * rank)
        self.diag_causal = MLP(IMAGE_EMBED_SIZE, h_sizes, N_ENVS * z_size)
        self.mu_spurious = MLP(IMAGE_EMBED_SIZE, h_sizes, N_CLASSES * N_ENVS * z_size)
        self.low_rank_spurious = MLP(IMAGE_EMBED_SIZE, h_sizes, N_CLASSES * N_ENVS * z_size * rank)
        self.diag_spurious = MLP(IMAGE_EMBED_SIZE, h_sizes, N_CLASSES * N_ENVS * z_size)

    def causal_params(self, x, e):
        batch_size = len(x)
        mu = self.mu_causal(x)
        mu = mu.reshape(batch_size, N_ENVS, self.z_size)
        mu = mu[torch.arange(batch_size), e, :]
        low_rank = self.low_rank_causal(x)
        low_rank = low_rank.reshape(batch_size, N_ENVS, self.z_size, self.rank)
        low_rank = low_rank[torch.arange(batch_size), e, :]
        diag = self.diag_causal(x)
        diag = diag.reshape(batch_size, N_ENVS, self.z_size)
        diag = diag[torch.arange(batch_size), e, :]
        cov = arr_to_cov(low_rank, diag)
        return mu, cov

    def spurious_params(self, x, y, e):
        batch_size = len(x)
        mu = self.mu_spurious(x)
        mu = mu.reshape(batch_size, N_CLASSES, N_ENVS, self.z_size)
        mu = mu[torch.arange(batch_size), y, e, :]
        low_rank = self.low_rank_spurious(x)
        low_rank = low_rank.reshape(batch_size, N_CLASSES, N_ENVS, self.z_size, self.rank)
        low_rank = low_rank[torch.arange(batch_size), y, e, :]
        diag = self.diag_spurious(x)
        diag = diag.reshape(batch_size, N_CLASSES, N_ENVS, self.z_size)
        diag = diag[torch.arange(batch_size), y, e, :]
        cov = arr_to_cov(low_rank, diag)
        return mu, cov

    def forward(self, x, y, e):
        batch_size = len(x)
        x = self.cnn(x).view(batch_size, -1)
        mu_causal, cov_causal = self.causal_params(x, e)
        mu_spurious, cov_spurious = self.spurious_params(x, y, e)
        mu = torch.hstack((mu_causal, mu_spurious))
        cov = torch.zeros(batch_size, 2 * self.z_size, 2 * self.z_size, device=y.device)
        cov[:, :self.z_size, :self.z_size] = cov_causal
        cov[:, self.z_size:, self.z_size:] = cov_spurious
        return D.MultivariateNormal(mu, scale_tril=torch.linalg.cholesky(cov))


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

    def causal_params(self, e):
        mu = self.mu_causal[e]
        cov = arr_to_cov(self.low_rank_causal[e], self.diag_causal[e])
        return mu, cov

    def spurious_params(self, y, e):
        mu = self.mu_spurious[y, e]
        cov = arr_to_cov(self.low_rank_spurious[y, e], self.diag_spurious[y, e])
        return mu, cov

    def log_prob_causal(self, z_c):
        batch_size = len(z_c)
        values = []
        for e_value in range(N_ENVS):
            e = torch.full((batch_size,), e_value, dtype=torch.long, device=z_c.device)
            dist = D.MultivariateNormal(*self.causal_params(e))
            values.append(dist.log_prob(z_c).unsqueeze(-1))
        values = torch.hstack(values)
        return torch.logsumexp(values, dim=1)

    def log_prob_spurious(self, z_s):
        batch_size = len(z_s)
        values = []
        for y_value in range(N_CLASSES):
            y = torch.full((batch_size,), y_value, dtype=torch.long, device=z_s.device)
            for e_value in range(N_ENVS):
                e = torch.full((batch_size,), e_value, dtype=torch.long, device=z_s.device)
                dist = D.MultivariateNormal(*self.spurious_params(y, e))
                values.append(dist.log_prob(z_s).unsqueeze(-1))
        values = torch.hstack(values)
        return torch.logsumexp(values, dim=1)

    def forward(self, y, e):
        batch_size = len(y)
        mu_causal, cov_causal = self.causal_params(e)
        mu_spurious, cov_spurious = self.spurious_params(y, e)
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

    def infer_loss(self, x, z):
        # log p(x|z_c,z_s)
        log_prob_x_z = self.decoder(x, z).mean()
        # log p(y|z_c)
        z_c, z_s = torch.chunk(z, 2, dim=1)
        prob_y1_zc = torch.sigmoid(self.classifier(z_c))
        prob_y0_zc = 1 - prob_y1_zc
        log_prob_y_zc = torch.log(torch.hstack((prob_y0_zc, prob_y1_zc)).max(dim=1).values).mean()
        # log p(z)
        log_prob_zc = self.prior.log_prob_causal(z_c).mean()
        log_prob_zs = self.prior.log_prob_spurious(z_s).mean()
        log_prob_z = log_prob_zc + log_prob_zs
        return log_prob_x_z, log_prob_y_zc, log_prob_z

    def infer_z(self, x):
        z_param = nn.Parameter(torch.zeros(len(x), 2 * self.z_size, device=self.device))
        nn.init.normal_(z_param)
        optim = Adam([z_param], lr=self.lr_infer)
        for _ in range(self.n_infer_steps):
            optim.zero_grad()
            log_prob_x_z, log_prob_y_zc, log_prob_z = self.infer_loss(x, z_param)
            loss = -log_prob_x_z - self.y_mult * log_prob_y_zc - self.alpha * log_prob_z
            loss.backward()
            optim.step()
        z_c, z_s = torch.chunk(z_param, 2, dim=1)
        y_pred = self.classifier(z_c).view(-1)
        return log_prob_x_z, log_prob_y_zc, log_prob_z, loss, y_pred

    def test_step(self, batch, batch_idx):
        assert self.task == Task.CLASSIFY
        x, y, e = batch
        with torch.set_grad_enabled(True):
            log_prob_x_z, log_prob_y_zc, log_prob_z, loss, y_pred = self.infer_z(x)
            self.log('log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
            self.log('log_prob_y_zc', log_prob_y_zc, on_step=False, on_epoch=True)
            self.log('log_prob_z', log_prob_z, on_step=False, on_epoch=True)
            self.log('loss', loss, on_step=False, on_epoch=True)
            self.eval_metric.update(y_pred, y)

    def on_test_epoch_end(self):
        assert self.task == Task.CLASSIFY
        self.log('eval_metric', self.eval_metric.compute())

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)