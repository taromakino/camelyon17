import numpy as np
import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


N_CLASSES = 2
N_ENVS = 3
VAL_CENTER = 1
TEST_CENTER = 2


class Camelyon17Dataset(Dataset):
    def __init__(self, dpath, df):
        self.dpath = dpath
        self.df = df
        self.transforms = transforms.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.transforms(Image.open(os.path.join(self.dpath, self.df.fname.iloc[idx])).convert('RGB'))
        y = torch.tensor(self.df.tumor.iloc[idx], dtype=torch.long)
        e = torch.tensor(self.df.center.iloc[idx])
        if not torch.isnan(e).any():
            e = e.long()
        return x, y, e


def subsample(rng, df, n_debug_examples):
    if len(df) < n_debug_examples:
        return df
    else:
        idxs = rng.choice(len(df), n_debug_examples, replace=False)
        return df.iloc[idxs]


def make_data(batch_size, n_workers, n_eval_examples):
    rng = np.random.RandomState(0)
    dpath = os.path.join(os.environ['DATA_DPATH'], 'camelyon17_v1.0')
    df = pd.read_csv(os.path.join(dpath, 'metadata.csv'), index_col=0, dtype={'patient': 'str'})
    df['fname'] = [
        f'patches/patient_{patient}_node_{node}/patch_patient_{patient}_node_{node}_x_{x}_y_{y}.png'
        for patient, node, x, y in
        df.loc[:, ['patient', 'node', 'x_coord', 'y_coord']].itertuples(index=False, name=None)
    ]
    split_dict = {
        'train': 0,
        'val_id': 1,
        'val_ood': 2,
        'test': 3
    }
    val_center_mask = (df['center'] == VAL_CENTER)
    test_center_mask = (df['center'] == TEST_CENTER)
    df.loc[val_center_mask, 'split'] = split_dict['val_ood']
    df.loc[test_center_mask, 'split'] = split_dict['test']

    df_train = df[df.split == split_dict['train']]
    df_val_id = df[df.split == split_dict['val_id']]
    df_val_ood = df[df.split == split_dict['val_ood']]
    df_test = df[df.split == split_dict['test']]

    train_envs = df_train.center.unique().tolist()
    sorted_trainval_envs = np.argsort(train_envs)
    unordered_to_ordered = dict((u, o) for u, o in zip(train_envs, sorted_trainval_envs))
    df_train.loc[:, 'center'] = [unordered_to_ordered[elem] for elem in df_train.center]
    df_val_id.loc[:, 'center'] = [unordered_to_ordered[elem] for elem in df_val_id.center]
    df_val_ood.loc[:, 'center'] = np.nan
    df_test.loc[:, 'center'] = np.nan

    if n_eval_examples is not None:
        df_train = subsample(rng, df_train, n_eval_examples)
        df_val_id = subsample(rng, df_val_id, n_eval_examples)
        df_val_ood = subsample(rng, df_val_ood, n_eval_examples)
        df_test = subsample(rng, df_test, n_eval_examples)

    data_train = DataLoader(Camelyon17Dataset(dpath, df_train), shuffle=True, pin_memory=True, batch_size=batch_size,
        num_workers=n_workers)
    data_val_id = DataLoader(Camelyon17Dataset(dpath, df_val_id), pin_memory=True, batch_size=batch_size,
        num_workers=n_workers)
    data_val_ood = DataLoader(Camelyon17Dataset(dpath, df_val_ood), pin_memory=True, batch_size=batch_size,
        num_workers=n_workers)
    data_test = DataLoader(Camelyon17Dataset(dpath, df_test), pin_memory=True, batch_size=batch_size,
        num_workers=n_workers)
    return data_train, data_val_id, data_val_ood, data_test