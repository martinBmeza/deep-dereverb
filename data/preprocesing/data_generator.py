import sys, os
MAIN_PATH="/home/martin/Documents/tesis/src"
sys.path.append(MAIN_PATH)

import numpy as np
import glob
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, stft
from model.capas import Spectrogram
eps = np.finfo(float).eps #precision de punto flotante

def framing(data, winsize=256, step=256, dim=1):
    n_frames = int(data.shape[dim] / winsize)
    out = np.empty((n_frames, data.shape[0], winsize)) #+1 por el pad 
    for frame in range(n_frames):
        out[frame,:,:] = data[:,frame*winsize : (frame+1)*winsize]
    #agrego el padeado 
    #resto = data.shape[dim]%winsize
    #to_pad = winsize - resto
    #out[-1,:,:resto] = data[:, n_frames*winsize: n_frames*winsize + resto]
    #out[-1,:,resto:] = np.zeros((data.shape[0],to_pad))
    #shape --> (frames, freq, time)
    return out


def normalise(array, range_min, range_max, array_min, array_max):
    norm_array = (array - array_min) / (array_max - array_min)
    norm_array = norm_array * (range_max - range_min) + range_min
    return norm_array

def denormalise(norm_array, original_min, original_max, range_min, range_max):
    array = (norm_array - range_min) / (range_max - range_min)
    array = array * (original_max - original_min) + original_min
    return array

def irm(y, n):
    y = librosa.core.stft(y.astype('float64'), 512, 128).astype('complex128')
    n = librosa.core.stft(n.astype('float64'), 512, 128).astype('complex128')
    return (1* (np.abs(y) ** 2) / (np.abs(y) ** 2 + np.abs(n) ** 2)) ** 0.5


def generate_inputs(speech_path, rir_path):

    #Cargo los datos
    speech, speech_fs = librosa.load(speech_path, sr=None)
    rir, rir_fs = librosa.load(rir_path, sr=16000)
    if speech_fs!=rir_fs:
        raise Exception("Hay audios con distintas frecuencias de sampleo")

    # Elimino el delay del impulso
    delay_shift = np.argmax(rir)
    rir = rir[delay_shift:]

    #Convoluciono. Obtengo audio con reverb
    reverb = fftconvolve(speech, rir)

    #Padeo el audio anecoico. Obtengo el audio clean
    clean = np.pad(speech, (0,len(rir)-1), 'constant', constant_values=(eps,eps)) 

    #genero las STFTs
    stft_clean = librosa.stft(clean, n_fft=512, hop_length=128)#
    spectrogram_clean = np.abs(stft_clean)
    log_spectrogram_clean = librosa.amplitude_to_db(spectrogram_clean)

    stft_reverb = librosa.stft(reverb, n_fft=512, hop_length=128)
    spectrogram_reverb = np.abs(stft_reverb)
    log_spectrogram_reverb = librosa.amplitude_to_db(spectrogram_reverb)

    log_norm_reverb = normalise(log_spectrogram_reverb, 0, 1, -47, 39)
    log_norm_clean = normalise(log_spectrogram_clean, 0, 1, -47, 39)
    
    return [log_norm_reverb, log_norm_clean]

