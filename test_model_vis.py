from __future__ import absolute_import
from __future__ import print_function

import sys, os

import numpy as np
np.random.seed(42)  # for reproducibility

from matplotlib import pyplot as plt
import seaborn as sns

from tqdm import tqdm
from argparse import ArgumentParser

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

#torch.multiprocessing.set_sharing_strategy('file_system')

from highway import HighwayModel
from masking import MaskingModel
from residual import ResidualModel

from stft_dataset import STFTDataset
from main import train_fn, valid_fn
from pytorch_utils import TrainLoop

from utils import spectrogram_from_file, inv_spectrogram, postprocess
from maracas.utils import wavwrite

from main import load_data, load_noisy_data, load_noisy_timit, load_reverb_timit

def test_fn(model, criterion, batch):
    x, y, lengths = batch

    x = Variable(x, volatile=True)
    y = Variable(y, requires_grad=False)

    mask = Variable(torch.ByteTensor(x.size()).fill_(1),
            requires_grad=False)
    for k, l in enumerate(lengths):
        mask[:l, k, :] = 0

    y_hat = model(x)

    # Apply mask
    y_hat.masked_fill_(mask, 0.0)
    y.masked_fill_(mask, 0.0)

    test_loss = criterion(y_hat, y).item()
    return y_hat.data.numpy(), test_loss

def test_fn_all_layers(model, criterion, batch):
    x, y, lengths = batch

    x = Variable(x)
    y = Variable(y)

    y_hats = model.forward_all(x)

    # Apply mask
    outputs = []
    losses = []
    for y_hat in y_hats[0]:
        test_loss = criterion(y_hat, y).item()
        outputs.append(y_hat.data.numpy())
        losses.append(test_loss)

    return outputs, losses, y_hats[1:]


parser = ArgumentParser()
parser.add_argument('--window_size', type=int, default=32)
parser.add_argument('--step_size', type=int, default=16)
parser.add_argument('--data_type', default='reverb')
parser.add_argument('--use_log', action='store_true')
parser.add_argument('checkpoint_path')
args = parser.parse_args()

window_size = args.window_size
step_size = args.step_size

G_test = STFTDataset(window=window_size, step=step_size, use_log=args.use_log)

if args.data_type == 'reverb':
    print('Loading reverb dataset')
    G_train, G_val = load_data(window_size, step_size, args.use_log)
    G_test.load_metadata_from_desc_file('ieee_reverb_only_test.json')
elif args.data_type == 'noisy':
    print('Loading noisy dataset')
    G_train, G_val = load_noisy_data(window_size, step_size, args.use_log)
    G_test.load_metadata_from_desc_file('ieee_noisy_test.json')
elif args.data_type == 'noisy_timit':
    print('Loading noisy_timit dataset')
    G_train, G_val = load_noisy_timit(window_size, step_size, args.use_log)
    G_test.load_metadata_from_desc_file('timit_noisy_test.json')
elif args.data_type == 'reverb_timit':
    G_train, G_val = load_reverb_timit(window_size, step_size, args.use_log)
    G_test.load_metadata_from_desc_file('timit_reverb_only_test.json')
else:
    raise ValueError('data_type has to be either "reverb" or "noisy"')

G_test.feats_mean = G_train.feats_mean
G_test.feats_std = G_train.feats_std

test_loader = DataLoader(G_test, batch_size=1,
            collate_fn=G_test.collate_samples,
            num_workers=0, shuffle=False)

print('Loading model...')
model_path = args.checkpoint_path
train_loop_last = torch.load(model_path)
best_epoch = np.argmin(train_loop_last.history['valid_loss'])
train_loop_best = torch.load(os.path.join(os.path.dirname(model_path),
                            'checkpoint_{}ep.pt'.format(best_epoch)))

savedir = os.path.join(os.path.dirname(model_path), 'samples_{}ep'.format(best_epoch))
if not os.path.isdir(savedir):
    os.mkdir(savedir)

model = train_loop_best.model.cpu()
if hasattr(model, 'blocks'):
    for block in model.blocks:
        block.rnn.flatten_parameters()

print('Generating samples...')
for k, batch in tqdm(enumerate(test_loader)):
    f = G_test.audio_paths[k]
    spec, phase = spectrogram_from_file(f, window=window_size, step=step_size, log=args.use_log)
    #ref, phase_ref = spectrogram_from_file(f, window=window, step=step)

    with torch.no_grad():
        Y_hat, test_loss, layer_outputs = test_fn_all_layers(model, train_loop_best.criterion, batch)

    # Visualize layer outputs depending on model
    if type(model) == HighwayModel:
        # need to plot both output and gate
        hiddens, masks = layer_outputs
        n_layers = len(hiddens)
        if n_layers > 1:
            fig, axes = plt.subplots(4, n_layers)
        else:
            fig, axes = plt.subplots(4, 2)
        axes[0,0].pcolormesh(batch[0].detach().cpu().numpy().squeeze())
        axes[0,1].pcolormesh(batch[1].detach().cpu().numpy().squeeze())
        for n in range(n_layers):
            axes[1,n].pcolormesh(hiddens[n].detach().cpu().numpy().squeeze())
            axes[2,n].pcolormesh(masks[n].detach().cpu().numpy().squeeze())
        for n, output in enumerate(Y_hat):
            axes[3,n].pcolormesh(output.squeeze())

    elif type(model) == ResidualModel:
        # need to plot residual only
        residuals = layer_outputs[0]
        n_layers = len(residuals)
        if n_layers > 1:
            fig, axes = plt.subplots(3, n_layers)
        else:
            fig, axes = plt.subplots(3, 2)
        axes[0,0].pcolormesh(batch[0].detach().cpu().numpy().squeeze())
        axes[0,1].pcolormesh(batch[1].detach().cpu().numpy().squeeze())
        for n in range(n_layers):
            axes[1,n].pcolormesh(residuals[n].detach().cpu().numpy().squeeze())
        for n, output in enumerate(Y_hat):
            axes[2,n].pcolormesh(output.squeeze())

    elif type(model) == MaskingModel:
        # need to plot mask only
        masks = layer_outputs[0]
        n_layers = len(masks)
        if n_layers > 1:
            fig, axes = plt.subplots(3, n_layers)
        else:
            fig, axes = plt.subplots(3, 2)
        axes[0,0].pcolormesh(batch[0].detach().cpu().numpy().squeeze())
        axes[0,1].pcolormesh(batch[1].detach().cpu().numpy().squeeze())
        for n in range(n_layers):
            axes[1,n].pcolormesh(masks[n].detach().cpu().numpy().squeeze())
        for n, output in enumerate(Y_hat):
            axes[2,n].pcolormesh(output.squeeze())
#    elif type(model) == BaselineModel:
#        # all we can plot are intermediate outputs
    else:
        raise ValueError('Cannot plot for that type of model')

    for n, Yn in enumerate(Y_hat):
        if args.use_log:
            Yn = np.exp(Yn.squeeze())
        else:
            Yn = Yn.squeeze()
        y_hat = inv_spectrogram(Yn, phase,
                fft_length=int(window_size*1e-3*16000), sample_rate=16000,
                hop_length=int(step_size*1e-3*16000))

        y_hat = postprocess(y_hat, 16000, step_size)

        conddir = os.path.join(savedir,
                os.path.split(os.path.dirname(f))[-1])
        if not os.path.isdir(conddir):
            os.mkdir(conddir)
        wavwrite(os.path.join(conddir, os.path.basename(f)[:-4] + '_{}.wav'.format(n)),
                y_hat, 16000)
        fig.savefig(os.path.join(conddir, os.path.basename(f)[:-3] + 'png'))
        plt.close()

