import torch
from torch.autograd import Variable

import numpy as np
import pickle

import os
from glob import glob
from tqdm import tqdm

def load_checkpoint(checkpoint_path):
    # Check whether checkpoint path exists and is not empty
    checkpoint_path_exists = os.path.isdir(checkpoint_path)
    past_checkpoints = glob(os.path.join(checkpoint_path, 'checkpoint*ep.pt'))

    if checkpoint_path_exists and len(past_checkpoints) > 0:
        print('Checkpoint exists, resuming...')
        epochs = list(map(lambda x: int(os.path.basename(x).split('_')[-1][:-5]),
            past_checkpoints))
        last_epoch_ckpt = past_checkpoints[np.argmax(epochs)]
        train_loop = torch.load(last_epoch_ckpt)

        fix_train_loop(train_loop)
        return train_loop
    else:
        raise ValueError('Checkpoint path does not exist or is empty')


def fix_train_loop(train_loop):
    # Hack to support changes on TrainLoop
    if not hasattr(train_loop, 'cur_epoch'):
        train_loop.cur_epoch = np.max(epochs) + 1

    # Hack to support changes on DataLoader
    if not hasattr(train_loop.train_iter, 'drop_last'):
        train_loop.train_iter.drop_last = False
        train_loop.valid_iter.drop_last = False


class TrainLoop(object):

    def __init__(self, model,
            optimizer, criterion,
            train_fn, train_iter,
            valid_fn, valid_iter,
            checkpoint_path=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_fn = train_fn
        self.train_iter = train_iter
        self.valid_fn = valid_fn
        self.valid_iter = valid_iter
        if checkpoint_path is None:
            # Save to current directory
            self.checkpoint_path = os.getcwd()
        else:
            self.checkpoint_path = checkpoint_path
        self.history = {'train_loss': [],
                'valid_loss': []}
        self.total_iters = 0
        self.cur_epoch = 0

    def train(self, n_epochs=1, n_workers=1, save_every=None):
        # Note: Logging expects the losses to be divided by the batch size

        # Set up
        if not os.path.isdir(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)

        save_every_fmt = os.path.join(self.checkpoint_path, 'checkpoint_{}it.pt')
        save_epoch_fmt = os.path.join(self.checkpoint_path, 'checkpoint_{}ep.pt')

        for epoch in range(self.cur_epoch, n_epochs):
            print('Epoch {}/{}'.format(epoch+1, n_epochs))
            train_iter = tqdm(enumerate(self.train_iter))
            self.history['train_loss'].append([])
            train_loss = self.history['train_loss'][-1]
            for t, batch in train_iter:
                train_loss.append(self.train_fn(self.model,
                    self.optimizer,
                    self.criterion,
                    batch))
                train_iter.set_postfix(loss=np.mean(train_loss))
                self.total_iters += 1
                if save_every is not None:
                    if self.total_iters % save_every == 0:
                        torch.save(self, save_every_fmt.format(self.total_iters))

            # Validation
            val_loss = 0.0
            for t, batch in enumerate(self.valid_iter):
                val_loss += self.valid_fn(self.model, self.criterion, batch)
            val_loss /= t+1
            print('Validation loss: {}'.format(val_loss))
            self.history['valid_loss'].append(val_loss)
            self.cur_epoch += 1

            # Checkpointing
            print('Checkpointing...')
            torch.save(self, save_epoch_fmt.format(epoch))



