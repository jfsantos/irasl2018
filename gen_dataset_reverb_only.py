from maracas.dataset import Dataset
import numpy as np

# Make sure this is reproducible
np.random.seed(42)

d = Dataset()

# Speech files
d.add_speech_files('IEEE_dataset/train', recursive=False)

# Reverb files
d.add_reverb_files('rirs')
d.add_noise_files('white.wav')

d.generate_dataset([np.inf], '/home/jfsantos/data/reverb_only_ieee', files_per_condition=50, n_workers=8)

