import os
import pandas as pd
from argparse import ArgumentParser
from utils.plot import *


def acc(fpath):
    df = pd.read_csv(fpath)
    return df.test_acc.iloc[-1]


def plot(ax, args, stage, x_offset):
    values = []
    for beta in args.beta_range:
        values_row = []
        for seed in range(args.n_seeds):
            fpath = os.path.join(args.dpath, f'y_mult={args.y_mult},beta={beta}', 'classify', stage, f'version_{seed}',
                'metrics.csv')
            values_row.append(acc(fpath))
        values.append(values_row)
    values = pd.DataFrame(np.array(values).T).melt()
    values.variable += x_offset
    sns.lineplot(data=values, x='variable', y='value', errorbar='sd', err_style='bars', err_kws={'capsize': 4},
        ax=ax, label=stage.capitalize(), legend=False)
    ax.set_xticks(range(len(args.beta_range)))
    ax.set_xticklabels(args.beta_range)
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel('Accuracy')


def main(args):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plot(ax, args, 'val_id', 0)
    plot(ax, args, 'test', 0.05)
    ax.axhline(0.703, color=next_color(ax), linestyle='--', label='Baseline')
    ax.grid(alpha=0.5, linewidth=0.5)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(handles), bbox_to_anchor=[0.5, 0])
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.35)
    os.makedirs('fig', exist_ok=True)
    plt.savefig(os.path.join('fig', 'accuracy.pdf'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, default='results')
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--y_mult', type=int, default=100)
    parser.add_argument('--beta_range', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    main(parser.parse_args())