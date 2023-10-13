import os
import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from data import N_CLASSES, N_ENVS
from torch.optim import Adam
from torchmetrics import Accuracy
from utils.enums import Task
from utils.nn_utils import MLP, arr_to_tril, arr_to_cov


CNN_SIZE = 864
PRIOR_INIT_SD = 0.01


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, mode='encode'):
        assert mode in ['encode', 'decode'], "Mode must be either 'encode' or 'decode'."
        super(DenseLayer, self).__init__()
        self.BN1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        if mode == 'encode':
            self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, 1, 1, 0)
            self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, 3, 1, 1)
        elif mode == 'decode':
            self.conv1 = nn.ConvTranspose2d(in_channels, 4 * growth_rate, 1, 1, 0)
            self.conv2 = nn.ConvTranspose2d(4 * growth_rate, growth_rate, 3, 1, 1)
        self.BN2 = nn.BatchNorm2d(4 * growth_rate)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        bn1 = self.BN1(x)
        relu1 = self.relu1(bn1)
        conv1 = self.conv1(relu1)
        bn2 = self.BN2(conv1)
        relu2 = self.relu2(bn2)
        conv2 = self.conv2(relu2)
        return torch.cat([x, conv2], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, mode='encode'):
        assert mode in ['encode', 'decode'], "Mode must be either 'encode' or 'decode'."
        super(DenseBlock, self).__init__()
        self.DL1 = DenseLayer(in_channels + (growth_rate * 0), growth_rate, mode)
        self.DL2 = DenseLayer(in_channels + (growth_rate * 1), growth_rate, mode)
        self.DL3 = DenseLayer(in_channels + (growth_rate * 2), growth_rate, mode)

    def forward(self, x):
        DL1 = self.DL1(x)
        DL2 = self.DL2(DL1)
        DL3 = self.DL3(DL2)
        return DL3


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, c_rate, mode='encode'):
        assert mode in ['encode', 'decode'], "Mode must be either 'encode' or 'decode'."
        super(TransitionBlock, self).__init__()
        out_channels = int(c_rate * in_channels)
        self.BN = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        if mode == 'encode':
            self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
            self.resize_layer = nn.AvgPool2d(2, 2)
        elif mode == 'decode':
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, 1, 1, 0)
            self.resize_layer = nn.ConvTranspose2d(out_channels, out_channels, 2, 2, 0)

    def forward(self, x):
        bn = self.BN(x)
        relu = self.relu(bn)
        conv = self.conv(relu)
        output = self.resize_layer(conv)
        return output


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.init_conv = nn.Conv2d(3, 24, 3, 2, 1)
        self.BN1 = nn.BatchNorm2d(24)
        self.relu1 = nn.ReLU()
        self.db1 = DenseBlock(24, 8, 'encode')
        self.tb1 = TransitionBlock(48, 0.5, 'encode')
        self.db2 = DenseBlock(24, 8, 'encode')
        self.tb2 = TransitionBlock(48, 0.5, 'encode')
        self.db3 = DenseBlock(24, 8, 'encode')
        self.BN2 = nn.BatchNorm2d(48)
        self.relu2 = nn.ReLU()
        self.down_conv = nn.Conv2d(48, 24, 2, 2, 0)

    def forward(self, x):
        init_conv = self.init_conv(x)
        bn1 = self.BN1(init_conv)
        relu1 = self.relu1(bn1)
        db1 = self.db1(relu1)
        tb1 = self.tb1(db1)
        db2 = self.db2(tb1)
        tb2 = self.tb2(db2)
        db3 = self.db3(tb2)
        bn2 = self.BN2(db3)
        relu2 = self.relu2(bn2)
        down_conv = self.down_conv(relu2)
        return down_conv


class DCNN(nn.Module):
    def __init__(self):
        super(DCNN, self).__init__()
        self.up_conv = nn.ConvTranspose2d(24, 24, 2, 2, 0)
        self.db1 = DenseBlock(24, 8, 'decode')
        self.tb1 = TransitionBlock(48, 0.5, 'decode')
        self.db2 = DenseBlock(24, 8, 'decode')
        self.tb2 = TransitionBlock(48, 0.5, 'decode')
        self.db3 = DenseBlock(24, 8, 'decode')
        self.BN1 = nn.BatchNorm2d(48)
        self.relu1 = nn.ReLU()
        self.de_conv = nn.ConvTranspose2d(48, 24, 2, 2, 0)
        self.BN2 = nn.BatchNorm2d(24)
        self.relu2 = nn.ReLU()
        self.out_conv = nn.ConvTranspose2d(24, 3, 3, 1, 1)

    def forward(self, z):
        up_conv = self.up_conv(z)
        db1 = self.db1(up_conv)
        tb1 = self.tb1(db1)
        db2 = self.db2(tb1)
        tb2 = self.tb2(db2)
        db3 = self.db3(tb2)
        bn1 = self.BN1(db3)
        relu1 = self.relu1(bn1)
        de_conv = self.de_conv(relu1)
        bn2 = self.BN2(de_conv)
        relu2 = self.relu2(bn2)
        output = self.out_conv(relu2)
        return output


class Encoder(nn.Module):
    def __init__(self, z_size, rank, h_sizes):
        super().__init__()
        self.z_size = z_size
        self.rank = rank
        self.cnn = CNN()
        self.mu = MLP(CNN_SIZE, h_sizes, N_CLASSES * N_ENVS * 2 * z_size)
        self.low_rank = MLP(CNN_SIZE, h_sizes, N_CLASSES * N_ENVS * 2 * z_size * rank)
        self.diag = MLP(CNN_SIZE, h_sizes, N_CLASSES * N_ENVS * 2 * z_size)

    def forward(self, x, y, e):
        batch_size = len(x)
        x = self.cnn(x).view(batch_size, -1)
        mu = self.mu(x)
        mu = mu.reshape(batch_size, N_CLASSES, N_ENVS, 2 * self.z_size)
        mu = mu[torch.arange(batch_size), y, e, :]
        low_rank = self.low_rank(x)
        low_rank = low_rank.reshape(batch_size, N_CLASSES, N_ENVS, 2 * self.z_size, self.rank)
        low_rank = low_rank[torch.arange(batch_size), y, e, :]
        diag = self.diag(x)
        diag = diag.reshape(batch_size, N_CLASSES, N_ENVS, 2 * self.z_size)
        diag = diag[torch.arange(batch_size), y, e, :]
        return D.MultivariateNormal(mu, scale_tril=arr_to_tril(low_rank, diag))


class Decoder(nn.Module):
    def __init__(self, z_size, h_sizes):
        super().__init__()
        self.mlp = MLP(2 * z_size, h_sizes, CNN_SIZE)
        self.dcnn = DCNN()

    def forward(self, x, z):
        batch_size = len(x)
        x_pred = self.mlp(z).view(batch_size, 24, 6, 6)
        x_pred = self.dcnn(x_pred).view(batch_size, -1)
        return -F.binary_cross_entropy_with_logits(x_pred, x.view(batch_size, -1), reduction='none').sum(dim=1)


class Prior(nn.Module):
    def __init__(self, z_size, rank):
        super().__init__()
        self.z_size = z_size
        self.mu_causal = nn.Parameter(torch.zeros(N_ENVS, z_size))
        self.low_rank_causal = nn.Parameter(torch.zeros(N_ENVS, z_size, rank))
        self.diag_causal = nn.Parameter(torch.zeros(N_ENVS, z_size))
        nn.init.normal_(self.mu_causal, 0, PRIOR_INIT_SD)
        nn.init.normal_(self.low_rank_causal, 0, PRIOR_INIT_SD)
        nn.init.normal_(self.diag_causal, 0, PRIOR_INIT_SD)
        # p(z_s|y,e)
        self.mu_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size))
        self.low_rank_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size, rank))
        self.diag_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size))
        nn.init.normal_(self.mu_spurious, 0, PRIOR_INIT_SD)
        nn.init.normal_(self.low_rank_spurious, 0, PRIOR_INIT_SD)
        nn.init.normal_(self.diag_spurious, 0, PRIOR_INIT_SD)

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
    def __init__(self, task, z_size, rank, h_sizes, beta, reg_mult, lr, weight_decay, lr_infer, n_infer_steps):
        super().__init__()
        self.save_hyperparameters()
        self.task = task
        self.z_size = z_size
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
        self.prior = Prior(z_size, rank)
        # p(y|z)
        self.classifier = MLP(z_size, h_sizes, 1)
        # q(z)
        self.q_z_mu = nn.Parameter(torch.full((2 * z_size,), torch.nan), requires_grad=False)
        self.q_z_var = nn.Parameter(torch.full((2 * z_size,), torch.nan), requires_grad=False)
        self.q_z_samples = []
        self.z, self.y, self.e = [], [], []
        self.eval_metric = Accuracy('binary')

    def sample_z(self, dist):
        mu, scale_tril = dist.loc, dist.scale_tril
        batch_size, z_size = mu.shape
        epsilon = torch.randn(batch_size, z_size, 1).to(self.device)
        return mu + torch.bmm(scale_tril, epsilon).squeeze()

    def q_z(self):
        return D.MultivariateNormal(self.q_z_mu, covariance_matrix=torch.diag(self.q_z_var))

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
        return log_prob_x_z, log_prob_y_zc, self.beta * kl, self.reg_mult * prior_norm

    def training_step(self, batch, batch_idx):
        assert self.task == Task.VAE
        x, y, e = batch
        log_prob_x_z, log_prob_y_zc, kl, prior_norm = self.loss(x, y, e)
        loss = -log_prob_x_z - log_prob_y_zc + kl + prior_norm
        return loss

    def validation_step(self, batch, batch_idx):
        assert self.task == Task.VAE
        x, y, e = batch
        log_prob_x_z, log_prob_y_zc, kl, prior_norm = self.loss(x, y, e)
        loss = -log_prob_x_z - log_prob_y_zc + kl + prior_norm
        self.log('val_log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
        self.log('val_log_prob_y_zc', log_prob_y_zc, on_step=False, on_epoch=True)
        self.log('val_kl', kl, on_step=False, on_epoch=True)
        self.log('val_prior_norm', prior_norm, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

    def infer_loss(self, x, z):
        batch_size = len(x)
        # log p(x|z_c,z_s)
        log_prob_x_z = self.decoder(x, z)
        # log p(y|z_c)
        z_c, z_s = torch.chunk(z, 2, dim=1)
        y_pred = self.classifier(z_c).view(-1)
        # log q(z)
        log_prob_z = self.q_z().log_prob(z)
        loss_candidates = []
        y_candidates = []
        for y_elem in range(N_CLASSES):
            y = torch.full((batch_size,), y_elem, dtype=torch.long, device=self.device)
            log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y.float(), reduction='none')
            loss_candidates.append((-log_prob_x_z - log_prob_y_zc - log_prob_z)[:, None])
            y_candidates.append(y_elem)
        loss_candidates = torch.hstack(loss_candidates).min(dim=1)
        y_candidates = torch.tensor(y_candidates, device=self.device)
        idxs = loss_candidates.indices
        y_pred = y_candidates[idxs]
        return loss_candidates.values.mean(), y_pred

    def infer_z(self, x):
        batch_size = len(x)
        z_param = nn.Parameter(torch.repeat_interleave(self.q_z().loc[None], batch_size, dim=0))
        optim = Adam([z_param], lr=self.lr_infer)
        optim_loss = torch.inf
        optim_y_pred = None
        optim_z = None
        for _ in range(self.n_infer_steps):
            optim.zero_grad()
            loss, y_pred = self.infer_loss(x, z_param)
            loss.backward()
            optim.step()
            if loss < optim_loss:
                optim_loss = loss
                optim_y_pred = y_pred
                optim_z = z_param.clone()
        return optim_loss, optim_y_pred, optim_z

    def test_step(self, batch, batch_idx):
        x, y, e = batch
        if self.task == Task.Q_Z:
            z = self.encoder(x, y, e).loc
            self.q_z_samples.append(z.detach().cpu())
        else:
            assert self.task == Task.CLASSIFY
            with torch.set_grad_enabled(True):
                loss, y_pred, z = self.infer_z(x)
                self.log('loss', loss, on_step=False, on_epoch=True)
                self.eval_metric.update(y_pred, y)
                self.z.append(z.detach().cpu())
                self.y.append(y.cpu())
                self.e.append(e.cpu())

    def on_test_epoch_end(self):
        if self.task == Task.Q_Z:
            z = torch.cat(self.q_z_samples)
            self.q_z_mu.data = torch.mean(z, 0)
            self.q_z_var.data = torch.var(z, 0)
        else:
            assert self.task == Task.CLASSIFY
            self.log('eval_metric', self.eval_metric.compute())
            z = torch.cat(self.z)
            y = torch.cat(self.y)
            e = torch.cat(self.e)
            torch.save((z, y, e), os.path.join(self.trainer.log_dir, f'version_{self.trainer.logger.version}', 'infer.pt'))

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)