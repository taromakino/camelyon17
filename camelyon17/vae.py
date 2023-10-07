import os
import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from data import N_CLASSES, N_ENVS
from torch.optim import Adam
from utils.enums import Task
from utils.nn_utils import MLP, arr_to_tril, arr_to_cov


CNN_SIZE = 864


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
        self.mu = MLP(CNN_SIZE + N_CLASSES + N_ENVS, h_sizes, 2 * z_size)
        self.low_rank = MLP(CNN_SIZE + N_CLASSES + N_ENVS, h_sizes, 2 * z_size * rank)
        self.diag = MLP(CNN_SIZE + N_CLASSES + N_ENVS, h_sizes, 2 * z_size)

    def forward(self, x, y_embed, e_embed):
        batch_size = len(x)
        x = self.cnn(x).view(batch_size, -1)
        mu = self.mu(x, y_embed, e_embed)
        low_rank = self.low_rank(x, y_embed, e_embed)
        low_rank = low_rank.reshape(batch_size, 2 * self.z_size, self.rank)
        diag = self.diag(x, y_embed, e_embed)
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
    def __init__(self, z_size, rank, h_sizes):
        super().__init__()
        self.z_size = z_size
        self.rank = rank
        # p(z_c|e)
        self.mu_causal = MLP(N_ENVS, h_sizes, z_size)
        self.low_rank_causal = MLP(N_ENVS, h_sizes, z_size * rank)
        self.diag_causal = MLP(N_ENVS, h_sizes, z_size)
        # p(z_s|y,e)
        self.mu_spurious = MLP(N_CLASSES + N_ENVS, h_sizes, z_size)
        self.low_rank_spurious = MLP(N_CLASSES + N_ENVS, h_sizes, z_size * rank)
        self.diag_spurious = MLP(N_CLASSES + N_ENVS, h_sizes, z_size)

    def forward(self, y_embed, e_embed):
        batch_size = len(y_embed)
        # Causal
        mu_causal = self.mu_causal(e_embed)
        low_rank_causal = self.low_rank_causal(e_embed)
        low_rank_causal = low_rank_causal.reshape(batch_size, self.z_size, self.rank)
        diag_causal = self.diag_causal(e_embed)
        cov_causal = arr_to_cov(low_rank_causal, diag_causal)
        # Spurious
        mu_spurious = self.mu_spurious(y_embed, e_embed)
        low_rank_spurious = self.low_rank_spurious(y_embed, e_embed)
        low_rank_spurious = low_rank_spurious.reshape(batch_size, self.z_size, self.rank)
        diag_spurious = self.diag_spurious(y_embed, e_embed)
        cov_spurious = arr_to_cov(low_rank_spurious, diag_spurious)
        # Block diagonal
        mu = torch.hstack((mu_causal, mu_spurious))
        cov = torch.zeros(batch_size, 2 * self.z_size, 2 * self.z_size, device=y_embed.device)
        cov[:, :self.z_size, :self.z_size] = cov_causal
        cov[:, self.z_size:, self.z_size:] = cov_spurious
        return D.MultivariateNormal(mu, cov)


class VAE(pl.LightningModule):
    def __init__(self, task, z_size, rank, h_sizes, beta, reg_mult, lr, weight_decay, alpha, lr_infer, n_infer_steps):
        super().__init__()
        self.save_hyperparameters()
        self.task = task
        self.z_size = z_size
        self.beta = beta
        self.reg_mult = reg_mult
        self.lr = lr
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.lr_infer = lr_infer
        self.n_infer_steps = n_infer_steps
        self.y_embed = nn.Embedding(N_CLASSES, N_CLASSES)
        self.e_embed = nn.Embedding(N_ENVS, N_ENVS)
        # q(z_c,z_s|x)
        self.encoder = Encoder(z_size, rank, h_sizes)
        # p(x|z_c, z_s)
        self.decoder = Decoder(z_size, h_sizes)
        # p(z_c,z_s|y,e)
        self.prior = Prior(z_size, rank, h_sizes)
        # p(y|z)
        self.classifier = MLP(z_size, h_sizes, 1)
        # q(z)
        self.q_z_mu = nn.Parameter(torch.full((2 * z_size,), torch.nan), requires_grad=False)
        self.q_z_var = nn.Parameter(torch.full((2 * z_size,), torch.nan), requires_grad=False)
        self.q_z_samples = []
        self.z, self.y, self.e = [], [], []

    def sample_z(self, dist):
        mu, scale_tril = dist.loc, dist.scale_tril
        batch_size, z_size = mu.shape
        epsilon = torch.randn(batch_size, z_size, 1).to(self.device)
        return mu + torch.bmm(scale_tril, epsilon).squeeze()

    def q_z(self):
        return D.MultivariateNormal(self.q_z_mu, covariance_matrix=torch.diag(self.q_z_var))

    def loss(self, x, y, e):
        y_embed = self.y_embed(y)
        e_embed = self.e_embed(e)
        # z_c,z_s ~ q(z_c,z_s|x)
        posterior_dist = self.encoder(x, y_embed, e_embed)
        z = self.sample_z(posterior_dist)
        z_c, z_s = torch.chunk(z, 2, dim=1)
        # E_q(z_c,z_s|x)[log p(x|z_c,z_s)]
        log_prob_x_z = self.decoder(x, z).mean()
        # E_q(z_c|x)[log p(y|z_c)]
        y_pred = self.classifier(z_c).view(-1)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y.float())
        # KL(q(z_c,z_s|x) || p(z_c|e)p(z_s|y,e))
        prior_dist = self.prior(y_embed, e_embed)
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

    def make_infer_params(self, batch_size):
        z_param = nn.Parameter(torch.repeat_interleave(self.q_z().loc[None], batch_size, dim=0))
        y_embed = torch.arange(N_CLASSES, device=self.device).long()
        y_embed = self.y_embed(y_embed).mean(dim=0)
        y_embed = torch.repeat_interleave(y_embed[None], batch_size, dim=0)
        y_embed_param = nn.Parameter(y_embed)
        e_embed = torch.arange(N_ENVS, device=self.device).long()
        e_embed = self.e_embed(e_embed).mean(dim=0)
        e_embed = torch.repeat_interleave(e_embed[None], batch_size, dim=0)
        e_embed_param = nn.Parameter(e_embed)
        return z_param, y_embed_param, e_embed_param

    def invert_y_embed(self, y_embed):
        y_embed = torch.repeat_interleave(y_embed[:, None, :], N_CLASSES, dim=1)
        distance = torch.norm(y_embed - self.y_embed.weight.data[None], dim=2)
        return distance.argmin(dim=1)

    def infer_loss(self, x, z, y_embed, e_embed):
        z_c, z_s = torch.chunk(z, 2, dim=1)
        # log p(x|z_c,z_s)
        log_prob_x_z = self.decoder(x, z).mean()
        # log p(y|z_c)
        y = self.invert_y_embed(y_embed)
        y_pred = self.classifier(z_c).view(-1)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y.float())
        # log p(z|y,e)
        prior_dist = self.prior(y_embed, e_embed)
        log_prob_z_ye = prior_dist.log_prob(z).mean()
        return z, log_prob_x_z, log_prob_y_zc, self.alpha * log_prob_z_ye

    def infer_z(self, x):
        z_param, y_embed_param, e_embed_param = self.make_infer_params(len(x))
        optim = Adam([z_param, y_embed_param, e_embed_param], lr=self.lr_infer)
        optim_loss = torch.inf
        optim_z = optim_log_prob_x_z = optim_log_prob_y_zc = optim_log_prob_z_ye = None
        for _ in range(self.n_infer_steps):
            optim.zero_grad()
            z, log_prob_x_z, log_prob_y_zc, log_prob_z_ye = self.infer_loss(x, z_param, y_embed_param, e_embed_param)
            loss = -log_prob_x_z - log_prob_y_zc - log_prob_z_ye
            loss.backward()
            optim.step()
            if loss < optim_loss:
                optim_z = z.clone()
                optim_log_prob_x_z = log_prob_x_z
                optim_log_prob_y_zc = log_prob_y_zc
                optim_log_prob_z_ye = log_prob_z_ye
                optim_loss = loss
        return optim_z, optim_log_prob_x_z, optim_log_prob_y_zc, optim_log_prob_z_ye, optim_loss

    def test_step(self, batch, batch_idx):
        x, y, e = batch
        if self.task == Task.Q_Z:
            y_embed = self.y_embed(y)
            e_embed = self.e_embed(e)
            z = self.encoder(x, y_embed, e_embed).loc
            self.q_z_samples.append(z.detach().cpu())
        else:
            assert self.task == Task.INFER_Z
            with torch.set_grad_enabled(True):
                z, log_prob_x_z, log_prob_y_zc, log_prob_z_ye, loss = self.infer_z(x)
                self.log('log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
                self.log('log_prob_y_zc', log_prob_y_zc, on_step=False, on_epoch=True)
                self.log('log_prob_z_ye', log_prob_z_ye, on_step=False, on_epoch=True)
                self.log('loss', loss, on_step=False, on_epoch=True)
                self.z.append(z.detach().cpu())
                self.y.append(y.cpu())
                self.e.append(e.cpu())


    def on_test_epoch_end(self):
        if self.task == Task.Q_Z:
            z = torch.cat(self.q_z_samples)
            self.q_z_mu.data = torch.mean(z, 0)
            self.q_z_var.data = torch.var(z, 0)
        else:
            assert self.task == Task.INFER_Z
            z = torch.cat(self.z)
            y = torch.cat(self.y)
            e = torch.cat(self.e)
            torch.save((z, y, e), os.path.join(self.trainer.log_dir, f'version_{self.trainer.logger.version}', 'infer.pt'))

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)