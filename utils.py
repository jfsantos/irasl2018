import glob

import logging
import os
import numpy as np
import re
import soundfile
from numpy.lib.stride_tricks import as_strided
from maracas.maracas import asl_meter
from audio_tools import iterate_invert_spectrogram

logger = logging.getLogger(__name__)


def calc_feat_dim(window, max_freq):
    return int(0.001 * window * max_freq) + 1


def conv_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride


def spectrogram(samples, fft_length=256, sample_rate=2, hop_length=128, pad=0):
    """
    Compute the spectrogram for a real signal.
    The parameters follow the naming convention of
    matplotlib.mlab.specgram

    Args:
        samples (1D array): input audio signal
        fft_length (int): number of elements in fft window
        sample_rate (scalar): sample rate
        hop_length (int): hop length (relative offset between neighboring
            fft windows).

    Returns:
        x (2D array): spectrogram [frequency x time]
        freq (1D array): frequency of each row in x

    Note:
        This is a truncating computation e.g. if fft_length=10,
        hop_length=5 and the signal has 23 elements, then the
        last 3 elements will be truncated.
    """
    assert not np.iscomplexobj(samples), "Must not pass in complex numbers"

    window = np.hanning(fft_length)[:, None]
    window_norm = np.sum(window**2)

    # The scaling below follows the convention of
    # matplotlib.mlab.specgram which is the same as
    # matlabs specgram.
    scale = window_norm * sample_rate

    trunc = (len(samples) - fft_length) % hop_length
    x = samples[:len(samples) - trunc]

    # "stride trick" reshape to include overlap
    nshape = (fft_length, (len(x) - fft_length) // hop_length + 1)
    nstrides = (x.strides[0], x.strides[0] * hop_length)
    x = as_strided(x, shape=nshape, strides=nstrides)

    # window stride sanity check
    assert np.all(x[:, 1] == samples[hop_length:(hop_length + fft_length)])

    # broadcast window, compute fft over columns and square mod
    x = np.fft.rfft(x * window, axis=0)
    phase = np.angle(x)
    x = np.absolute(x)**2

    # scale, 2.0 for everything except dc and fft_length/2
    x[1:-1, :] *= (2.0 / scale)
    x[(0, -1), :] /= scale

    freqs = float(sample_rate) / fft_length * np.arange(x.shape[0])

    if pad > 0:
        x = np.pad(x, ((0, 0), (pad, pad)), 'constant')
        phase = np.pad(phase, ((0, 0), (pad, pad)), 'constant')

    return x, phase, freqs


def spectrogram_from_file(filename, step=10, window=20, max_freq=None,
                          eps=1e-14, log=True, pad=0, multichannel=False):
    """ Calculate the log of linear spectrogram from FFT energy
    Params:
        filename (str): Path to the audio file
        step (int): Step size in milliseconds between windows
        window (int): FFT window size in milliseconds
        max_freq (int): Only FFT bins corresponding to frequencies between
            [0, max_freq] are returned
        eps (float): Small value to ensure numerical stability (for ln(x))
    """
    with soundfile.SoundFile(filename) as sound_file:
        audio = sound_file.read(dtype='float32')
        sample_rate = sound_file.samplerate
        if audio.ndim >= 2 and multichannel == False:
            audio = np.mean(audio, 1)
        if max_freq is None:
            max_freq = sample_rate / 2
        if max_freq > sample_rate / 2:
            raise ValueError("max_freq must not be greater than half of "
                             " sample rate")
        if step > window:
            raise ValueError("step size must not be greater than window size")
        hop_length = int(0.001 * step * sample_rate)
        fft_length = int(0.001 * window * sample_rate)

        if multichannel:
            mags, phases = [], []
            for channel in range(audio.ndim):
                pxx, phase, freqs = spectrogram(
                        audio[:, channel], fft_length=fft_length, sample_rate=sample_rate,
                    hop_length=hop_length, pad=pad)
                mags.append(pxx)
                phases.append(phase)
            pxx = np.concatenate([m.T[np.newaxis] for m in mags], axis=0)
            phase = np.concatenate([p.T[np.newaxis] for p in phases], axis=0)
        else:
            pxx, phase, freqs = spectrogram(
                audio, fft_length=fft_length, sample_rate=sample_rate,
                hop_length=hop_length, pad=pad)

        ind = np.where(freqs <= max_freq)[0][-1] + 1
    if multichannel:
        pxx = pxx[:, :, :ind]
        phase = phase[:, :, :ind]
    else:
        pxx = pxx[:ind, :].T
        phase = phase[:ind, :].T

    if log:
        return np.log(pxx + eps), phase
    else:
        return pxx + eps, phase


def inv_spectrogram(mag, phase, fft_length=256, sample_rate=2, hop_length=128):
    """
    Compute the spectrogram inversion for a real signal
    given its squared magnitude and phase.

    Args:
        mag (2D array): input magnitude (time, freq)
        phase (2D array): input phase
        fft_length (int): number of elements in fft window
        sample_rate (scalar): sample rate
        hop_length (int): hop length (relative offset between neighboring
            fft windows).

    Returns:
        x (1D array): real signal

    """
    window = np.hanning(fft_length)
    window_squared = window**2
    window_norm = np.sum(window_squared)
    scale = window_norm * sample_rate

    # descale, 2.0 for everything except dc and fft_length/2
    mag[1:-1, :] /= (2.0 / scale)
    mag[(0, -1), :] *= scale

    xlen = hop_length*(mag.shape[0]-1) + fft_length
    x = np.zeros(xlen)
    ifft_window_sum = np.zeros(xlen)

    # compute ifft and reconstruct signal in time domain
    X = np.sqrt(mag) * np.exp(1j*phase)
    xwin = np.fft.irfft(X)
    for k in range(xwin.shape[0]):
        x[k*hop_length : k*hop_length + fft_length] += window * xwin[k, :]
        ifft_window_sum[k*hop_length : k*hop_length + fft_length] += window_squared

    approx_nonzero_indices = ifft_window_sum > 1e-20
    x[approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

    return x

def postprocess(y_hat, fs, step):
    # Postprocess output: zero first/last few samples
    y_hat[:int(step * 1e-3 * fs)] = 0
    y_hat[-int(step * 1e-3 * fs):-1] = 0
    # Check if there are still NaNs
    if np.any(np.isnan(y_hat)):
        raise ValueError('NaNs in file!')
    # Normalize energy
    y_hat = y_hat/10**(asl_meter(y_hat, fs)/20) * 10**(-26.0/20)
    return y_hat

def griffin_lim(X, window, step):
    # make X double-sided
    X = np.hstack([X, X[:, -1:1:-1]])
    X[1:] *= 0.5
    return iterate_invert_spectrogram(X, window, step)

