import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from data import N_CLASSES, N_ENVS
from torch.optim import Adam
from torchmetrics import Accuracy
from utils.nn_utils import MLP, arr_to_cov


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
        self.mu_causal = MLP(CNN_SIZE, h_sizes, N_ENVS * z_size)
        self.low_rank_causal = MLP(CNN_SIZE, h_sizes, N_ENVS * z_size * rank)
        self.diag_causal = MLP(CNN_SIZE, h_sizes, N_ENVS * z_size)
        self.mu_spurious = MLP(CNN_SIZE, h_sizes, N_CLASSES * N_ENVS * z_size)
        self.low_rank_spurious = MLP(CNN_SIZE, h_sizes, N_CLASSES * N_ENVS * z_size * rank)
        self.diag_spurious = MLP(CNN_SIZE, h_sizes, N_CLASSES * N_ENVS * z_size)

    def forward(self, x, y, e):
        batch_size = len(x)
        x = self.cnn(x).view(batch_size, -1)
        # Causal
        mu_causal = self.mu_causal(x)
        mu_causal = mu_causal.reshape(batch_size, N_ENVS, self.z_size)
        mu_causal = mu_causal[torch.arange(batch_size), e, :]
        low_rank_causal = self.low_rank_causal(x)
        low_rank_causal = low_rank_causal.reshape(batch_size, N_ENVS, self.z_size, self.rank)
        low_rank_causal = low_rank_causal[torch.arange(batch_size), e, :]
        diag_causal = self.diag_causal(x)
        diag_causal = diag_causal.reshape(batch_size, N_ENVS, self.z_size)
        diag_causal = diag_causal[torch.arange(batch_size), e, :]
        cov_causal = arr_to_cov(low_rank_causal, diag_causal)
        # Spurious
        mu_spurious = self.mu_spurious(x)
        mu_spurious = mu_spurious.reshape(batch_size, N_CLASSES, N_ENVS, self.z_size)
        mu_spurious = mu_spurious[torch.arange(batch_size), y, e, :]
        low_rank_spurious = self.low_rank_spurious(x)
        low_rank_spurious = low_rank_spurious.reshape(batch_size, N_CLASSES, N_ENVS, self.z_size, self.rank)
        low_rank_spurious = low_rank_spurious[torch.arange(batch_size), y, e, :]
        diag_spurious = self.diag_spurious(x)
        diag_spurious = diag_spurious.reshape(batch_size, N_CLASSES, N_ENVS, self.z_size)
        diag_spurious = diag_spurious[torch.arange(batch_size), y, e, :]
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
        x_pred = self.mlp(z).view(batch_size, 24, 6, 6)
        x_pred = self.dcnn(x_pred).view(batch_size, -1)
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
    def __init__(self, task, z_size, rank, h_sizes, y_mult, beta, reg_mult, init_sd, lr, weight_decay, alpha, lr_infer,
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
        self.alpha = alpha
        self.lr_infer = lr_infer
        self.n_infer_steps = n_infer_steps
        # q(z_c,z_s|x)
        self.encoder = Encoder(z_size, rank, h_sizes)
        # p(x|z_c, z_s)
        self.decoder = Decoder(z_size, h_sizes)
        # p(z_c,z_s|y,e)
        self.prior = Prior(z_size, rank, init_sd)
        # p(y|z)
        self.classifier = MLP(z_size, h_sizes, 1)
        self.val_acc = Accuracy('binary')
        self.test_acc = Accuracy('binary')

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
        x, y, e = batch
        log_prob_x_z, log_prob_y_zc, kl, prior_norm = self.loss(x, y, e)
        loss = -log_prob_x_z - self.y_mult * log_prob_y_zc + self.beta * kl + self.reg_mult * prior_norm
        self.log('train_log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
        self.log('train_log_prob_y_zc', log_prob_y_zc, on_step=False, on_epoch=True)
        self.log('train_kl', kl, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def init_z(self, x, y_value, e_value):
        batch_size = len(x)
        y = torch.full((batch_size,), y_value, dtype=torch.long, device=self.device)
        e = torch.full((batch_size,), e_value, dtype=torch.long, device=self.device)
        return nn.Parameter(self.encoder(x, y, e).loc.detach())

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

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y, e = batch
        with torch.set_grad_enabled(True):
            loss, y_pred = self.classify(x)
            if dataloader_idx == 0:
                self.val_acc.update(y_pred, y)
            elif dataloader_idx == 1:
                self.test_acc.update(y_pred, y)
            else:
                raise ValueError

    def on_validation_epoch_end(self):
        self.log('val_acc', self.val_acc.compute())
        self.log('test_acc', self.test_acc.compute())

    def test_step(self, batch, batch_idx):
        x, y, e = batch
        with torch.set_grad_enabled(True):
            loss, y_pred = self.classify(x)
            self.test_acc.update(y_pred, y)

    def on_test_epoch_end(self):
        self.log('test_acc', self.test_acc.compute())

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)