"""
Use this script to create JSON-Line description files that can be used to
train deep-speech models through this library.
This works with data directories that are organized like LibriSpeech:
data_directory/group/speaker/[file_id1.wav, file_id2.wav, ...,
                              speaker.trans.txt]

Where speaker.trans.txt has in each line, file_id transcription
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import wave
from glob import glob
import pickle
from tqdm import tqdm
import numpy as np

def main(data_directory, ref_directory, output_file):
    refs = []
    durations = []
    keys = []
    noisetypes = []
    snrs = []
    for condition in tqdm(os.listdir(data_directory)):
        condition_path = os.path.join(data_directory, condition)
        noise_type, snr = condition.split('_')
        snr = float(snr[:-2])
        for f in glob(os.path.join(condition_path, '*.wav')):
            file_id = os.path.basename(f)
            reference = os.path.join(ref_directory, file_id)

            audio = wave.open(f)
            duration = float(audio.getnframes()) / audio.getframerate()
            audio.close()

            keys.append(f)
            durations.append(duration)
            refs.append(reference)
            snrs.append(snr)
            noisetypes.append(noise_type)

    with open(output_file, 'w') as out_file:
        for i in range(len(keys)):
            line = json.dumps({'key': keys[i],
                'duration': durations[i],
                'ref': refs[i],
                'snr': snrs[i],
                'noisetype': noisetypes[i]})
            out_file.write(line + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_directory', type=str,
                        help='Path to data directory')
    parser.add_argument('ref_directory', type=str,
                        help='Path to ref directory')
    parser.add_argument('output_file', type=str,
                        help='Path to output file')
    args = parser.parse_args()

    main(args.data_directory, args.ref_directory, args.output_file)
