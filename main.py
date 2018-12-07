from __future__ import absolute_import
from __future__ import print_function

import numpy as np
np.random.seed(42)  # for reproducibility

from tqdm import tqdm

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from stft_dataset import STFTDataset
from residual import ResidualModel
from highway import HighwayModel
from masking import MaskingModel
from baseline import BaselineModel
from pytorch_utils import TrainLoop, load_checkpoint

def load_data(window_size, step_size, use_log):
    print("Loading data...")
    G_train = STFTDataset(window=window_size, step=step_size, use_log=use_log)
    G_train.load_metadata_from_desc_file('ieee_reverb_only_train.json')
    G_train.fit_stats()

    G_val = STFTDataset(window=window_size, step=step_size, use_log=use_log)
    G_val.load_metadata_from_desc_file('ieee_reverb_only_valid.json')
    G_val.feats_mean = G_train.feats_mean
    G_val.feats_std = G_train.feats_std

    return G_train, G_val


def load_noisy_data(window_size, overlap, use_log):
    print("Loading data...")
    G_train = STFTDataset(window=window_size, step=overlap, use_log=use_log)
    G_train.load_metadata_from_desc_file('ieee_noisy_train.json')
    G_train.fit_stats()

    G_val = STFTDataset(window=window_size, step=overlap, use_log=use_log)
    G_val.load_metadata_from_desc_file('ieee_noisy_valid.json')
    G_val.feats_mean = G_train.feats_mean
    G_val.feats_std = G_train.feats_std

    return G_train, G_val


def load_noisy_timit(window_size, overlap, use_log):
    print("Loading data...")
    G_train = STFTDataset(window=window_size, step=overlap, use_log=use_log)
    G_train.load_metadata_from_desc_file('timit_noisy_train.json')
    G_train.fit_stats()

    G_val = STFTDataset(window=window_size, step=overlap, use_log=use_log)
    G_val.load_metadata_from_desc_file('timit_noisy_valid.json')
    G_val.feats_mean = G_train.feats_mean
    G_val.feats_std = G_train.feats_std

    return G_train, G_val


def load_reverb_timit(window_size, overlap, use_log):
    print("Loading data...")
    G_train = STFTDataset(window=window_size, step=overlap, use_log=use_log)
    G_train.load_metadata_from_desc_file('timit_reverb_only_train.json')
    G_train.fit_stats()

    G_val = STFTDataset(window=window_size, step=overlap, use_log=use_log)
    G_val.load_metadata_from_desc_file('timit_reverb_only_valid.json')
    G_val.feats_mean = G_train.feats_mean
    G_val.feats_std = G_train.feats_std

    return G_train, G_val



def train_fn(model, optimizer, criterion, batch):
    x, y, lengths = batch

    x = Variable(x.cuda())
    y = Variable(y.cuda(), requires_grad=False)

    mask = Variable(torch.ByteTensor(x.size()).fill_(1).cuda(),
            requires_grad=False)
    for k, l in enumerate(lengths):
        mask[:l, k, :] = 0

    y_hat = model(x)

    # Apply mask
    y_hat.masked_fill_(mask, 0.0)
    y.masked_fill_(mask, 0.0)

    loss = criterion(y_hat, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.data.item()


def valid_fn(model, criterion, batch):
    x, y, lengths = batch

    x = Variable(x.cuda(), volatile=True)
    y = Variable(y.cuda(), requires_grad=False)

    mask = Variable(torch.ByteTensor(x.size()).fill_(1).cuda(),
            requires_grad=False)
    for k, l in enumerate(lengths):
        mask[:l, k, :] = 0

    y_hat = model(x)

    # Apply mask
    y_hat.masked_fill_(mask, 0.0)
    y.masked_fill_(mask, 0.0)

    val_loss = criterion(y_hat, y).data.item()
    return val_loss


if __name__ == '__main__':

    from argparse import ArgumentParser
    import os
    from glob import glob

    parser = ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--num_hidden', type=int, default=256)
    parser.add_argument('--num_blocks', type=int, default=3)
    parser.add_argument('--num_layers_per_block', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--model_type', default='residual')
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--step_size', type=int, default=16)
    parser.add_argument('--data_type', default='reverb')
    parser.add_argument('--use_log', action='store_true')
    parser.add_argument('checkpoint_path')

    args = parser.parse_args()

    try:
        train_loop = load_checkpoint(args.checkpoint_path)
    except ValueError:
        print('No checkpoints, initializing a model from scratch...')
        window_size = args.window_size # in ms
        step_size = args.step_size
        n_input = int(1e-3*window_size*16000/2 + 1)
        n_output = n_input

        if args.model_type == 'residual':
            model = ResidualModel(n_input,
                    args.num_blocks,
                    args.num_hidden,
                    args.num_layers_per_block).cuda()
        elif args.model_type == 'highway':
            model = HighwayModel(n_input,
                    args.num_blocks,
                    args.num_hidden,
                    args.num_layers_per_block).cuda()
        elif args.model_type == 'masking':
            model = MaskingModel(n_input,
                    args.num_blocks,
                    args.num_hidden,
                    args.num_layers_per_block).cuda()
        elif args.model_type == 'baseline':
            model = BaselineModel(n_input,
                    args.num_hidden,
                    args.num_layers_per_block).cuda()
        else:
            raise ValueError('model_type has to be either "residual", "highway", or "baseline"')

        print(model)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

        if args.data_type == 'reverb':
            print('Loading reverb dataset')
            G_train, G_val = load_data(window_size, step_size, args.use_log)
        elif args.data_type == 'noisy':
            print('Loading noisy dataset')
            G_train, G_val = load_noisy_data(window_size, step_size, args.use_log)
        elif args.data_type == 'noisy_timit':
            print('Loading noisy_timit dataset')
            G_train, G_val = load_noisy_timit(window_size, step_size, args.use_log)
        elif args.data_type == 'reverb_timit':
            G_train, G_val = load_reverb_timit(window_size, step_size, args.use_log)
        else:
            raise ValueError('data_type has to be either "reverb" or "noisy"')

        train_loader = DataLoader(G_train, batch_size=args.batch_size,
                collate_fn=G_train.collate_samples,
                num_workers=8, shuffle=True)
        valid_loader = DataLoader(G_val, batch_size=args.batch_size,
                collate_fn=G_train.collate_samples,
                num_workers=4)

        train_loop = TrainLoop(model,
                    optimizer, criterion,
                    train_fn, train_loader,
                    valid_fn, valid_loader,
                    checkpoint_path=args.checkpoint_path)

    train_loop.train(args.max_epochs)

