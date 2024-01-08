import matplotlib.pyplot as plt
import numpy as np
import os
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from data import N_CLASSES, N_ENVS, make_data
from utils.enums import Task
from vae import VAE


IMAGE_SHAPE = (3, 96, 96)


def sample_prior(rng, model):
    y = torch.tensor(rng.choice(N_CLASSES), dtype=torch.long, device=model.device)[None]
    e = torch.tensor(rng.choice(N_ENVS), dtype=torch.long, device=model.device)[None]
    prior_parent, prior_child = model.prior(y, e)
    z_parent, z_child = prior_parent.sample(), prior_child.sample()
    return z_parent, z_child


def reconstruct_x(model, z_parent, z_child):
    x_pred_parent = model.decoder_parent(z_parent)
    x_pred_child = model.decoder_child(z_child)
    x_pred = x_pred_parent + x_pred_child
    return torch.sigmoid(x_pred)


def plot(ax, x):
    x = x.squeeze().detach().cpu().numpy()
    x = x.transpose(1, 2, 0)
    ax.imshow(x)


def main(args):
    rng = np.random.RandomState(args.seed)
    task_dpath = os.path.join(args.dpath, Task.VAE.value)
    pl.seed_everything(args.seed)
    data_train, _, _, _ = make_data(1, 1, args.n_workers)
    model = VAE.load_from_checkpoint(os.path.join(task_dpath, f'version_{args.seed}', 'checkpoints', 'best.ckpt'))
    model.eval()
    example_idxs = rng.choice(len(data_train), args.n_examples, replace=False)
    for i, example_idx in enumerate(example_idxs):
        x, y, e = data_train.dataset.__getitem__(example_idx)
        x, y, e = x[None].to(model.device), y[None].to(model.device), e[None].to(model.device)
        posterior_parent, posterior_child = model.encoder(x, y, e)
        z_parent, z_child = posterior_parent.loc, posterior_child.loc
        fig, axes = plt.subplots(2, args.n_cols, figsize=(2 * args.n_cols, 2 * 2))
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
        plot(axes[0, 0], x)
        plot(axes[1, 0], x)
        x_pred = reconstruct_x(model, z_parent, z_child)
        plot(axes[0, 1], x_pred)
        plot(axes[1, 1], x_pred)
        for col_idx in range(2, args.n_cols):
            z_parent_prior, z_child_prior = sample_prior(rng, model)
            x_pred_parent = reconstruct_x(model, z_parent_prior, z_child)
            x_pred_child = reconstruct_x(model, z_parent, z_child_prior)
            plot(axes[0, col_idx], x_pred_parent)
            plot(axes[1, col_idx], x_pred_child)
        fig_dpath = os.path.join(task_dpath, f'version_{args.seed}', 'fig', 'plot_reconstruction')
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