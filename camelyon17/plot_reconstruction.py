import matplotlib.pyplot as plt
import numpy as np
import os
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from data import make_data
from utils.enums import Task
from vae import VAE


IMAGE_SHAPE = (3, 96, 96)


def sample_prior(rng, dataloader, vae):
    idx = rng.choice(len(dataloader), 1).item()
    x, y, e = dataloader.dataset.__getitem__(idx)
    prior_dist = vae.prior(vae.y_embed(y[None]), vae.e_embed(e[None]))
    z_sample = prior_dist.sample()
    return torch.chunk(z_sample, 2, dim=1)


def plot(ax, x):
    x = x.squeeze().detach().cpu().numpy()
    x = x.transpose(1, 2, 0)
    ax.imshow(x)


def decode(vae, z):
    batch_size = len(z)
    x_pred = vae.decoder.mlp(z).view(batch_size, 24, 6, 6)
    x_pred = vae.decoder.dcnn(x_pred)
    return torch.sigmoid(x_pred)


def main(args):
    rng = np.random.RandomState(args.seed)
    task_dpath = os.path.join(args.dpath, Task.VAE.value)
    pl.seed_everything(args.seed)
    dataloader, _, _, _ = make_data(1, None)
    vae = VAE.load_from_checkpoint(os.path.join(task_dpath, f'version_{args.seed}', 'checkpoints', 'best.ckpt'))
    example_idxs = rng.choice(len(dataloader), args.n_examples, replace=False)
    for i, example_idx in enumerate(example_idxs):
        x_seed, y_seed, e_seed = dataloader.dataset.__getitem__(example_idx)
        x_seed, y_seed, e_seed = x_seed[None].to(vae.device), y_seed[None].to(vae.device), e_seed[None].to(vae.device)
        posterior_dist_seed = vae.encoder(x_seed, vae.y_embed(y_seed), vae.e_embed(e_seed))
        z_seed = posterior_dist_seed.loc
        zc_seed, zs_seed = torch.chunk(z_seed, 2, dim=1)
        fig, axes = plt.subplots(2, args.n_cols, figsize=(2 * args.n_cols, 2 * 2))
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
        plot(axes[0, 0], x_seed)
        plot(axes[1, 0], x_seed)
        x_pred = decode(vae, z_seed)
        plot(axes[0, 1], x_pred)
        plot(axes[1, 1], x_pred)
        for col_idx in range(2, args.n_cols):
            zc_sample, zs_sample = sample_prior(rng, dataloader, vae)
            x_pred_causal = decode(vae, torch.hstack((zc_sample, zs_seed)))
            x_pred_spurious = decode(vae, torch.hstack((zc_seed, zs_sample)))
            plot(axes[0, col_idx], x_pred_causal)
            plot(axes[1, col_idx], x_pred_spurious)
        fig_dpath = os.path.join(task_dpath, f'version_{args.seed}', 'fig', 'reconstruct_from_posterior')
        os.makedirs(fig_dpath, exist_ok=True)
        plt.savefig(os.path.join(fig_dpath, f'{i}.png'))
        plt.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_cols', type=int, default=10)
    parser.add_argument('--n_examples', type=int, default=10)
    main(parser.parse_args())