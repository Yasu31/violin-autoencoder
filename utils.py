import scipy
import matplotlib.pyplot as plt
import IPython.display as ipd
import numpy as np
import librosa
import librosa.display
HOP_LENGTH = 512
N_FFT = 2048


def audio2stft(raw_audio, sr):
    """convert audio file to a 2D array representing the spectrogram of short-time fourier transforms.
    This data can be fed into the NN, since it is probably easier to deal with than raw waveform data.
    Please note that the returned data does not contain all the data of the original waveform- the phase data is discarded for sake of simplicity.    Args:
        raw_audio: waveform data
        sr: sampling rate
    """
    print(
        f"Input audio:\nsamples:\t{raw_audio.shape[0]}\nsampling rate:\t{sr}\nduration:\t{raw_audio.shape[0]/sr}")
    print("-"*10)
    stft = librosa.stft(raw_audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    stft = np.abs(stft)
    print(
        f"Resulting STFT:\nsamples:\t{stft.shape[1]}\nfrequency is split into {stft.shape[0]} parts.")
    print("="*10)
    return stft


def plot_stft(stft, sr):
    """visualize FFT result into a plot
    Converts to decibels before plotting.
    """
    stft_decibel = librosa.amplitude_to_db(stft)
    plt.figure(figsize=(15, 5))
    librosa.display.specshow(
        stft_decibel, sr=sr, hop_length=HOP_LENGTH, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')


def stft2audio(stft):
    """reconstructs audio signal from the stft data.
    It can reconstruct without the phase information, but sound quality is degraded.
    """
    reconstructed_audio = librosa.istft(stft)
    return reconstructed_audio
