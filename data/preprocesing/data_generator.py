import sys, os
MAIN_PATH="/home/martin/Documents/tesis/src"
sys.path.append(MAIN_PATH)

import numpy as np
import glob
import soundfile as sf
import librosa as lb
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, stft
from model.capas import Spectrogram

eps = np.finfo(float).eps #precision de punto flotante

#listas de speech y rirs
speech_path = os.path.join(MAIN_PATH, 'data/speech')
speech_list = glob.glob(speech_path+'/**/*.flac', recursive=True)

rir_path = os.path.join(MAIN_PATH, 'data/rir')
rir_list = glob.glob(rir_path+'/**/*.wav', recursive=True)

# procesamiento de una instancia
# Cargo los audios. Reviso igualdad de FS
speech, speech_fs = lb.load(speech_list[0], sr=None)
rir, rir_fs = lb.load(rir_list[0], sr=16000)
if speech_fs!=rir_fs:
    raise Exception("Hay audios con distintas frecuencias de sampleo")

# Elimino el delay del impulso
delay_shift = np.argmax(rir)
rir_nodelay = rir[delay_shift:]

#Convoluciono. Obtengo audio con reverb
reverb = fftconvolve(speech, rir_nodelay)

#Padeo el audio anecoico. Obtengo el audio clean
clean = np.pad(speech, (0,len(rir_nodelay)-1), 'constant', constant_values=(eps,eps)) 

#genero las STFTS
_, _, spec_clean = stft(clean, nperseg=512, noverlap=384)
_, _, spec_reverb = stft(reverb, nperseg=512, noverlap=384)

#magnitud y escala logaritmica
magspec_clean = np.log(abs(spec_clean)+eps)
magspec_reverb = np.log(abs(spec_reverb)+eps)

#Normalizacion
factor = np.max(np.abs(magspec_reverb))
magspepc_clean = magspec_clean / factor
magspec_reverb = magspec_reverb /factor

#MASCARA
mask = magspec_clean / magspec_reverb
#Compresion
k = 10
c = 0.1

mask_comp= k * ( (1 - np.exp( - c * mask)) / (1 + np.exp(- c * mask)))
#Con esto los pares serian [magspec_reverb, mask_comp]

#framing

def framing(data, winsize=256, step=256, dim=1):
    n_frames = int(data.shape[dim] / winsize)
    out = np.empty((n_frames+1, data.shape[0], winsize)) #+1 por el pad 
    for frame in range(n_frames):
        out[frame,:,:] = data[:,frame*winsize : (frame+1)*winsize]
    #agrego el padeado 
    resto = data.shape[dim]%winsize
    to_pad = winsize - resto
    out[-1,:,:resto] = data[:, n_frames*winsize: n_frames*winsize + resto]
    out[-1,:,resto:] = np.zeros((data.shape[0],to_pad))
    #shape --> (frames, freq, time)
    return out


def generate_inputs(speech_path, rir_path):

    #Cargo los datos
    speech, speech_fs = lb.load(speech_path, sr=None)
    rir, rir_fs = lb.load(rir_path, sr=16000)
    if speech_fs!=rir_fs:
        raise Exception("Hay audios con distintas frecuencias de sampleo")

    # Elimino el delay del impulso
    delay_shift = np.argmax(rir)
    rir_nodelay = rir[delay_shift:]

    #Convoluciono. Obtengo audio con reverb
    reverb = fftconvolve(speech, rir_nodelay)

    #Padeo el audio anecoico. Obtengo el audio clean
    clean = np.pad(speech, (0,len(rir_nodelay)-1), 'constant', constant_values=(eps,eps)) 

    #genero las STFTS
    _, _, spec_clean = stft(clean, nperseg=512, noverlap=384)
    _, _, spec_reverb = stft(reverb, nperseg=512, noverlap=384)

    #magnitud y escala logaritmica
    magspec_clean = np.log(abs(spec_clean)+eps)
    magspec_reverb = np.log(abs(spec_reverb)+eps)

    #Normalizacion
    factor = np.max(np.abs(magspec_reverb))
    magspepc_clean = magspec_clean / factor
    magspec_reverb = magspec_reverb /factor

    #MASCARA
    mask = magspec_clean / magspec_reverb
    #Compresion
    k = 10
    c = 0.1

    mask_comp= k * ( (1 - np.exp( - c * mask)) / (1 + np.exp(- c * mask)))
    return [magspec_reverb, mask_comp]
