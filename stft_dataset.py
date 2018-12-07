"""
Defines a class that is used to featurize audio clips, and provide
them to the network for training or testing.
"""

from __future__ import absolute_import, division, print_function
from functools import reduce

import torch
from torch.utils.data import Dataset
import random
import json
import warnings
import numpy as np
from utils import calc_feat_dim, spectrogram_from_file

class STFTDataset(Dataset):
    def __init__(self, step=10, window=20, max_freq=8000, desc_file=None,
            use_t60s=False, pad=0, normalize_targets=False,
            use_log=True):
        """
        Params:
            step (int): Step size in milliseconds between windows
            window (int): FFT window size in milliseconds
            max_freq (int): Only FFT bins corresponding to frequencies between
                [0, max_freq] are returned
            desc_file (str, optional): Path to a JSON-line file that contains
                labels and paths to the audio files. If this is None, then
                load metadata right away
        """
        self.feat_dim = calc_feat_dim(window, max_freq)
        self.feats_mean = np.zeros((self.feat_dim,))
        self.feats_std = np.ones((self.feat_dim,))
        if desc_file is not None:
            self.load_metadata_from_desc_file(desc_file)
        self.step = step
        self.window = window
        self.max_freq = max_freq
        self.pad = pad
        self.normalize_targets = normalize_targets

        self.use_t60s = use_t60s
        self.use_log = use_log

    def featurize(self, audio_clip, target=False):
        """ For a given audio clip, calculate the log of its Fourier Transform
        Params:
            audio_clip(str): Path to the audio clip
        """
        if target:
            pad = 0
        else:
            pad = self.pad
        return spectrogram_from_file(
            audio_clip, step=self.step, window=self.window,
            max_freq=self.max_freq, pad=pad,
            log=self.use_log)[0]

    def load_metadata_from_desc_file(self, desc_file, max_duration=10.0,):
        """ Read metadata from the description file
            (possibly takes long, depending on the filesize)
        Params:
            desc_file (str):  Path to a JSON-line file that contains labels and
                paths to the audio files
            max_duration (float): In seconds, the maximum duration of
                utterances to train or test on
        """
        audio_paths, durations, targets, t60s = [], [], [], []
        with open(desc_file) as json_line_file:
            for line_num, json_line in enumerate(json_line_file):
                try:
                    spec = json.loads(json_line)
                    if float(spec['duration']) > max_duration:
                        continue
                    audio_paths.append(spec['key'])
                    durations.append(float(spec['duration']))
                    targets.append(spec['ref'])
                    try:
                        t60s.append(spec['rt'])
                    except KeyError as e:
                        if self.use_t60s:
                            raise e
                        else:
                            pass
                except Exception as e:
                    warnings.warn('Error reading line #{}: {}'
                                .format(line_num, json_line))
                    warnings.warn(str(e))

        self.audio_paths = audio_paths
        self.durations = durations
        self.targets = targets
        self.t60s = t60s

    @staticmethod
    def sort_by_duration(durations, audio_paths, targets):
        return list(zip(*sorted(list(zip(durations, audio_paths, targets)))))

    def normalize(self, feature, eps=1e-14):
        return (feature - self.feats_mean) / (self.feats_std + eps)

    def __getitem__(self, index):
        features = self.normalize(self.featurize(self.audio_paths[index]))
        target = self.featurize(self.targets[index], target=True)
        if self.normalize_targets:
            target = self.normalize(target)
        duration = target.shape[0]

        features = torch.FloatTensor(features)
        target = torch.FloatTensor(target)

        if hasattr(self, 't60s') and self.use_t60s:
            t60 = torch.FloatTensor(self.normalize_t60(self.t60s[index]))
            return features, target, duration, t60.view(1, 16)
        else:
            return features, target, duration

    def __len__(self):
        return len(self.audio_paths)

    def fit_stats(self, k_samples=100):
        """ Estimate the mean and std of the features from the training set
        Params:
            k_samples (int): Use this number of samples for estimation
        """
        k_samples = min(k_samples, len(self.audio_paths))
        rng = random.Random(42)
        samples = rng.sample(range(len(self.audio_paths)), k_samples)
        feats = [self.featurize(self.audio_paths[s]) for s in samples]
        feats = np.vstack(feats)
        self.feats_mean = np.mean(feats, axis=0)
        self.feats_std = np.std(feats, axis=0)

        if hasattr(self, 't60s') and self.use_t60s:
            t60s = np.vstack([self.t60s[s] for s in samples])
            self.t60s_mean = np.mean(t60s, axis=0)
            self.t60s_std = np.std(t60s, axis=0)

    def normalize_t60(self, t60, eps=1e-14):
        return (t60 - self.t60s_mean) / (self.t60s_std + eps)

    @staticmethod
    def collate_samples(samples, pad=0):
        """ Used by torch.utils.data.DataLoader to join samples into
        a minibatch.
        Params:
           samples: list of tuples (features, targets, duration)
        """
        features, targets, durations = zip(*samples)
        max_length = max(durations)
        batch_size = len(features)
        n_features = features[0].size(-1)
        features_padded = torch.zeros(max_length+2*pad, batch_size, n_features)
        targets_padded = torch.zeros(max_length, batch_size, n_features)

        for k in range(batch_size):
            features_padded[:durations[k]+2*pad, k, :] = features[k]
            targets_padded[:durations[k], k, :] = targets[k]

        return features_padded, targets_padded, durations

    @staticmethod
    def collate_padded_samples_t60(samples, pad=0):
        """ Used by torch.utils.data.DataLoader to join samples into
        a minibatch.
        Params:
           samples: list of tuples (features, targets, duration)
        """
        features, targets, durations, t60 = zip(*samples)
        max_length = max(durations)
        batch_size = len(features)
        n_features = features[0].size(-1)
        features_padded = torch.zeros(max_length+2*pad, batch_size, n_features)
        targets_padded = torch.zeros(max_length, batch_size, n_features)

        for k in range(batch_size):
            features_padded[:durations[k]+2*pad, k, :] = features[k]
            targets_padded[:durations[k], k, :] = targets[k]

        t60 = torch.cat(t60, 0).contiguous()

        return features_padded, targets_padded, durations, t60

    @staticmethod
    def collate_padded_samples_t60_context(samples):
        return STFTDataset.collate_padded_samples_t60(samples, 5)

    @staticmethod
    def collate_samples_conv(samples, pad=0):
        """ Used by torch.utils.data.DataLoader to join samples into
        a minibatch.
        Params:
           samples: list of tuples (features, targets, duration)
        """
        features, targets, durations = zip(*samples)
        max_length = max(durations)
        batch_size = len(features)
        n_features = features[0].size(-1)
        features_padded = torch.zeros(batch_size, 1, n_features, max_length+2*pad)
        targets_padded = torch.zeros(max_length, batch_size, n_features)

        for k in range(batch_size):
            features_padded[k, 0, :, :durations[k]+2*pad] = features[k].t()
            targets_padded[:durations[k], k, :] = targets[k]

        return features_padded, targets_padded, durations


    @staticmethod
    def collate_samples_conv_t60(samples, pad=0):
        features, targets, durations, t60 = zip(*samples)
        max_length = max(durations)
        batch_size = len(features)
        n_features = features[0].size(-1)
        features_padded = torch.zeros(batch_size, 1, n_features, max_length+2*pad)
        targets_padded = torch.zeros(max_length, batch_size, n_features)

        for k in range(batch_size):
            features_padded[k, 0, :, :durations[k]+2*pad] = features[k].t()
            targets_padded[:durations[k], k, :] = targets[k]

        t60 = torch.cat(t60, 0).contiguous()

        return features_padded, targets_padded, durations, t60


    @staticmethod
    def collate_padded_samples(samples):
        return STFTDataset.collate_samples(samples, pad=5)

    @staticmethod
    def collate_samples_t60(samples):
        """ Used by torch.utils.data.DataLoader to join samples into
        a minibatch.
        Params:
           samples: list of tuples (features, targets, duration)
        """
        features, targets, durations, t60 = zip(*samples)
        max_length = max(durations)
        batch_size = len(features)
        n_features = features[0].size(-1)
        features_padded = torch.zeros(max_length, batch_size, n_features)
        targets_padded = torch.zeros(max_length, batch_size, n_features)

        for k in range(batch_size):
            features_padded[:durations[k], k, :] = features[k]
            targets_padded[:durations[k], k, :] = targets[k]

        t60 = torch.cat(t60, 0).contiguous()

        return features_padded, targets_padded, durations, t60


if __name__ == '__main__':
    from torch.utils.data.dataloader import DataLoaderIter
    from torch.utils.data import DataLoader
    d = STFTDataset(use_t60s=False, pad=4, normalize_targets=True)
    #d = CochleagramDataset(64, use_t60s=True)
    d.load_metadata_from_desc_file('ieee_reverb_only_valid.json')
    d.fit_stats()
    loader = DataLoader(d, 4, collate_fn=d.collate_samples)
    itr = DataLoaderIter(loader)
    x, y, z = next(itr)

