import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from data import N_CLASSES, N_ENVS
from encoder_cnn import EncoderCNN
from decoder_cnn import DecoderCNN
from torch.optim import Adam
from torchmetrics import Accuracy
from utils.nn_utils import SkipMLP, one_hot, arr_to_cov


IMG_ENCODE_SHAPE = (48, 6, 6)
IMG_ENCODE_SIZE = np.prod(IMG_ENCODE_SHAPE)
IMG_DECODE_SHAPE = (24, 6, 6)
IMG_DECODE_SIZE = np.prod(IMG_DECODE_SHAPE)


class Encoder(nn.Module):
    def __init__(self, z_size, rank, h_sizes):
        super().__init__()
        self.z_size = z_size
        self.rank = rank
        self.encoder_cnn = EncoderCNN()
        self.mu_causal = SkipMLP(IMG_ENCODE_SIZE + N_ENVS, h_sizes, z_size)
        self.low_rank_causal = SkipMLP(IMG_ENCODE_SIZE + N_ENVS, h_sizes, z_size * rank)
        self.diag_causal = SkipMLP(IMG_ENCODE_SIZE + N_ENVS, h_sizes, z_size)
        self.mu_spurious = SkipMLP(IMG_ENCODE_SIZE + N_CLASSES + N_ENVS, h_sizes, z_size)
        self.low_rank_spurious = SkipMLP(IMG_ENCODE_SIZE + N_CLASSES + N_ENVS, h_sizes, z_size * rank)
        self.diag_spurious = SkipMLP(IMG_ENCODE_SIZE + N_CLASSES + N_ENVS, h_sizes, z_size)

    def causal_dist(self, x, e):
        batch_size = len(x)
        e_one_hot = one_hot(e, N_ENVS)
        mu = self.mu_causal(x, e_one_hot)
        low_rank = self.low_rank_causal(x, e_one_hot)
        low_rank = low_rank.reshape(batch_size, self.z_size, self.rank)
        diag = self.diag_causal(x, e_one_hot)
        cov = arr_to_cov(low_rank, diag)
        return D.MultivariateNormal(mu, cov)

    def spurious_dist(self, x, y, e):
        batch_size = len(x)
        y_one_hot = one_hot(y, N_CLASSES)
        e_one_hot = one_hot(e, N_ENVS)
        mu = self.mu_spurious(x, y_one_hot, e_one_hot)
        low_rank = self.low_rank_spurious(x, y_one_hot, e_one_hot)
        low_rank = low_rank.reshape(batch_size, self.z_size, self.rank)
        diag = self.diag_spurious(x, y_one_hot, e_one_hot)
        cov = arr_to_cov(low_rank, diag)
        return D.MultivariateNormal(mu, cov)

    def forward(self, x, y, e):
        batch_size = len(x)
        x = self.encoder_cnn(x).view(batch_size, -1)
        causal_dist = self.causal_dist(x, e)
        spurious_dist = self.spurious_dist(x, y, e)
        return causal_dist, spurious_dist


class Decoder(nn.Module):
    def __init__(self, z_size, h_sizes):
        super().__init__()
        self.mlp = SkipMLP(2 * z_size, h_sizes, IMG_DECODE_SIZE)
        self.decoder_cnn = DecoderCNN()

    def forward(self, x, z):
        batch_size = len(x)
        x_pred = self.mlp(z).view(batch_size, *IMG_DECODE_SHAPE)
        x_pred = self.decoder_cnn(x_pred).view(batch_size, -1)
        return -F.binary_cross_entropy_with_logits(x_pred, x.view(batch_size, -1), reduction='none').sum(dim=1)


class Prior(nn.Module):
    def __init__(self, z_size, rank, init_sd):
        super().__init__()
        self.z_size = z_size
        self.mu_causal = nn.Parameter(torch.zeros(N_ENVS, z_size))
        self.low_rank_causal = nn.Parameter(torch.zeros(N_ENVS, z_size, rank))
        self.diag_causal = nn.Parameter(torch.zeros(N_ENVS, z_size))
        nn.init.normal_(self.mu_causal, 0, init_sd)
        nn.init.normal_(self.low_rank_causal, 0, init_sd)
        nn.init.normal_(self.diag_causal, 0, init_sd)
        # p(z_s|y,e)
        self.mu_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size))
        self.low_rank_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size, rank))
        self.diag_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size))
        nn.init.normal_(self.mu_spurious, 0, init_sd)
        nn.init.normal_(self.low_rank_spurious, 0, init_sd)
        nn.init.normal_(self.diag_spurious, 0, init_sd)

    def causal_dist(self, e):
        mu = self.mu_causal[e]
        cov = arr_to_cov(self.low_rank_causal[e], self.diag_causal[e])
        return D.MultivariateNormal(mu, cov)

    def spurious_dist(self, y, e):
        mu = self.mu_spurious[y, e]
        cov = arr_to_cov(self.low_rank_spurious[y, e], self.diag_spurious[y, e])
        return D.MultivariateNormal(mu, cov)

    def forward(self, y, e):
        causal_dist = self.causal_dist(e)
        spurious_dist = self.spurious_dist(y, e)
        return causal_dist, spurious_dist


class VAE(pl.LightningModule):
    def __init__(self, task, z_size, rank, h_sizes, y_mult, beta, reg_mult, init_sd, lr, weight_decay, lr_infer,
            n_infer_steps):
        super().__init__()
        self.save_hyperparameters()
        self.task = task
        self.z_size = z_size
        self.y_mult = y_mult
        self.beta = beta
        self.reg_mult = reg_mult
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_infer = lr_infer
        self.n_infer_steps = n_infer_steps
        # q(z_c,z_s|x)
        self.encoder = Encoder(z_size, rank, h_sizes)
        # p(x|z_c, z_s)
        self.decoder = Decoder(z_size, h_sizes)
        # p(z_c,z_s|y,e)
        self.prior = Prior(z_size, rank, init_sd)
        # p(y|z)
        self.classifier = SkipMLP(z_size, h_sizes, 1)
        self.val_acc = Accuracy('binary')
        self.test_acc = Accuracy('binary')

    def sample_z(self, dist):
        mu, scale_tril = dist.loc, dist.scale_tril
        batch_size, z_size = mu.shape
        epsilon = torch.randn(batch_size, z_size, 1).to(self.device)
        return mu + torch.bmm(scale_tril, epsilon).squeeze()

    def loss(self, x, y, e):
        # z_c,z_s ~ q(z_c,z_s|x)
        posterior_causal, posterior_spurious = self.encoder(x, y, e)
        z_c = self.sample_z(posterior_causal)
        z_s = self.sample_z(posterior_spurious)
        # E_q(z_c,z_s|x)[log p(x|z_c,z_s)]
        z = torch.hstack((z_c, z_s))
        log_prob_x_z = self.decoder(x, z).mean()
        # E_q(z_c|x)[log p(y|z_c)]
        y_pred = self.classifier(z_c).view(-1)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y.float())
        # KL(q(z_c,z_s|x) || p(z_c|e)p(z_s|y,e))
        prior_causal, prior_spurious = self.prior(y, e)
        kl_causal = D.kl_divergence(posterior_causal, prior_causal).mean()
        kl_spurious = D.kl_divergence(posterior_spurious, prior_spurious).mean()
        prior_norm = (torch.hstack((prior_causal.loc, prior_spurious.loc)) ** 2).mean()
        return log_prob_x_z, log_prob_y_zc, kl_causal, kl_spurious, prior_norm

    def training_step(self, batch, batch_idx):
        x, y, e = batch
        log_prob_x_z, log_prob_y_zc, kl_causal, kl_spurious, prior_norm = self.loss(x, y, e)
        kl = kl_causal + kl_spurious
        loss = -log_prob_x_z - self.y_mult * log_prob_y_zc + self.beta * kl + self.reg_mult * prior_norm
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, e = batch
        log_prob_x_z, log_prob_y_zc, kl_causal, kl_spurious, prior_norm = self.loss(x, y, e)
        kl = kl_causal + kl_spurious
        loss = -log_prob_x_z - self.y_mult * log_prob_y_zc + self.beta * kl + self.reg_mult * prior_norm
        self.log('val_log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
        self.log('val_log_prob_y_zc', log_prob_y_zc, on_step=False, on_epoch=True)
        self.log('val_kl_causal', kl_causal, on_step=False, on_epoch=True)
        self.log('val_kl_spurious', kl_spurious, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def init_z(self, x, y_value, e_value):
        batch_size = len(x)
        y = torch.full((batch_size,), y_value, dtype=torch.long, device=self.device)
        e = torch.full((batch_size,), e_value, dtype=torch.long, device=self.device)
        posterior_causal, posterior_spurious = self.encoder(x, y, e)
        z_c = posterior_causal.loc
        z_s = posterior_spurious.loc
        z = torch.hstack((z_c, z_s))
        return nn.Parameter(z.detach())

    def infer_loss(self, x, y, e, z):
        # log p(x|z_c,z_s)
        log_prob_x_z = self.decoder(x, z)
        # log p(y|z_c)
        z_c, z_s = torch.chunk(z, 2, dim=1)
        y_pred = self.classifier(z_c).view(-1)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y.float(), reduction='none')
        # log q(z_c,z_s|x,y,e)
        posterior_causal, posterior_spurious = self.encoder(x, y, e)
        log_prob_zc = posterior_causal.log_prob(z_c)
        log_prob_zs = posterior_spurious.log_prob(z_s)
        log_prob_z = log_prob_zc + log_prob_zs
        loss = -log_prob_x_z - self.y_mult * log_prob_y_zc - log_prob_z
        return loss

    def opt_infer_loss(self, x, y_value, e_value):
        batch_size = len(x)
        z_param = self.init_z(x, y_value, e_value)
        y = torch.full((batch_size,), y_value, dtype=torch.long, device=self.device)
        e = torch.full((batch_size,), e_value, dtype=torch.long, device=self.device)
        optim = Adam([z_param], lr=self.lr_infer)
        for _ in range(self.n_infer_steps):
            optim.zero_grad()
            loss = self.infer_loss(x, y, e, z_param)
            loss.mean().backward()
            optim.step()
        return loss.detach().clone()

    def classify(self, x):
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
        x, y, e = batch
        with torch.set_grad_enabled(True):
            loss, y_pred = self.classify(x)
            self.test_acc.update(y_pred, y)

    def on_test_epoch_end(self):
        self.log('test_acc', self.test_acc.compute())

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)