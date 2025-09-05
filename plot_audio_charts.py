import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import os
import numpy as np
import time

def plot_audio_waveforms():
    clean_audio, _ = sf.read('denoiser_dataset/clean/00000.wav')
    noisy_audio, _ = sf.read('test_noisy_sample.wav')
    denoised_audio, _ = sf.read('denoised_output.wav')

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle('Audio Waveform Comparison', fontsize=16)

    axes[0].plot(clean_audio, color='blue')
    axes[0].set_title('Clean Audio')
    axes[0].set_ylabel('Amplitude')

    axes[1].plot(noisy_audio, color='red')
    axes[1].set_title('Noisy Audio')
    axes[1].set_ylabel('Amplitude')

    axes[2].plot(denoised_audio, color='green')
    axes[2].set_title('Denoised Audio')
    axes[2].set_xlabel('Sample Index')
    axes[2].set_ylabel('Amplitude')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('audio_waveforms.png')
    print("Waveform chart saved to audio_waveforms.png")
    plt.close()

def plot_spectrograms():
    SR = 16000
    N_FFT = 1024
    HOP = 256

    clean_audio, _ = sf.read('denoiser_dataset/clean/00000.wav')
    noisy_audio, _ = sf.read('test_noisy_sample.wav')
    denoised_audio, _ = sf.read('denoised_output.wav')

    clean_spec = librosa.stft(clean_audio, n_fft=N_FFT, hop_length=HOP)
    noisy_spec = librosa.stft(noisy_audio, n_fft=N_FFT, hop_length=HOP)
    denoised_spec = librosa.stft(denoised_audio, n_fft=N_FFT, hop_length=HOP)

    clean_spec_db = librosa.amplitude_to_db(np.abs(clean_spec), ref=np.max)
    noisy_spec_db = librosa.amplitude_to_db(np.abs(noisy_spec), ref=np.max)
    denoised_spec_db = librosa.amplitude_to_db(np.abs(denoised_spec), ref=np.max)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Spectrogram Comparison', fontsize=16)

    librosa.display.specshow(clean_spec_db, sr=SR, hop_length=HOP, x_axis='time', y_axis='log', ax=axes[0])
    axes[0].set_title('Clean Spectrogram')

    librosa.display.specshow(noisy_spec_db, sr=SR, hop_length=HOP, x_axis='time', y_axis='log', ax=axes[1])
    axes[1].set_title('Noisy Spectrogram')

    librosa.display.specshow(denoised_spec_db, sr=SR, hop_length=HOP, x_axis='time', y_axis='log', ax=axes[2])
    axes[2].set_title('Denoised Spectrogram')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('spectrogram_comparison.png')
    print("Spectrogram chart saved to spectrogram_comparison.png")
    plt.close()

if _name_ == '_main_':
    plot_audio_waveforms()
    plot_spectrograms()
