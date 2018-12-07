from maracas.dataset import Dataset
import numpy as np

import os
from glob import glob

# Make sure this is reproducible
np.random.seed(42)

noise_path = 'data/DEMAND'
noise_scenarios = map(os.path.basename, glob(os.path.join(noise_path, '*')))

d = Dataset()
d.add_speech_files('data/IEEE_dataset/train/')

for noise in noise_scenarios:
    d.add_noise_files(os.path.join(noise_path, noise, 'ch01.wav'), name=noise)

d.generate_dataset([-3, 0, 3, 6, 12, 15], 'data/noise_only_ieee')

