import glob
import librosa
import mir_eval
import pystoi
import srmrpy as srmr
import numpy as np
import tensorflow as tf
from model.network_architecture import autoencoder
from scipy.signal import fftconvolve, stft
from utils import get_audio_list
from tqdm import tqdm
EPS = np.finfo(float).eps

def denormalise(array):
    array_min = -75
    array_max = 65
    array = (array * (array_max - array_min)) + array_min
    return array

def normalise(array):                                                                                                           
    array_min = -75
    array_max = 65
    norm_array = (array - array_min) / (array_max - array_min + EPS)
    return norm_array 


def gen_stft(audio):
    # compute STFT dismissing high freqs bin
    stft_ = librosa.stft(audio, n_fft=512, hop_length=128)[:-1,:]
    
    # only magnitude
    stft_ = np.abs(stft_)

    # logaritmic scale
    log_stft_ = librosa.amplitude_to_db(stft_)

    # normalize
    norm_stft_ = normalise(log_stft_)
    return norm_stft_


def frame_to_raw(frame):
    frame = denormalise(frame)
    frame_lin = librosa.db_to_amplitude(frame)

    # add freq bin (padding)
    frame_lin_pad = np.pad(frame_lin,((0,1),(0,0)), 'minimum') 

    # recover phase information
    frame_raw = librosa.griffinlim(frame_lin_pad, 
                                        n_iter=100,
                                        hop_length=128, 
                                        win_length=512)
    return frame_raw


def get_metricas(clean, reverb, fs):
    SRMR = srmr.srmr(reverb, fs)[0]
    SDR, _, _, _ = mir_eval.separation.bss_eval_sources(clean, reverb, compute_permutation=True)
    ESTOI = pystoi.stoi(clean, reverb, fs, extended = True)
    return SRMR, SDR[0], ESTOI


def test(CLEAN_PATH, REVERB_PATH, PESOS, CARPETA):
    clean_list = get_audio_list(CLEAN_PATH, ('.npy'))
    reverb_list = get_audio_list(REVERB_PATH, ('.npy'))
    clean_list.sort() ; reverb_list.sort()

    modelo = autoencoder()
    modelo.load_weights(PESOS)

    SRMR_reverb = []
    SDR_reverb = []
    ESTOI_reverb = []

    SRMR_dereverb = []
    SDR_dereverb = []
    ESTOI_dereverb = []

    for clean_path, reverb_path in tqdm(zip(clean_list[:1000], reverb_list), total=len(clean_list[:1000])):
        
        # read files
        fs = 16000
        clean = np.load(clean_path)
        reverb = np.load(reverb_path)

        # apply model
        espectro_in = gen_stft(reverb)
        espectro_target = gen_stft(clean)

        espectro_out = modelo.predict([espectro_in.reshape(1,256,256)])
        espectro_out = espectro_out.reshape(256,256)
	
        reverb = frame_to_raw(espectro_in)
        dereverb = frame_to_raw(espectro_out)
        clean = frame_to_raw(espectro_target)

        # get metrics for clean-reverb
        srmr, sdr, estoi = get_metricas(clean, reverb, fs)
        SRMR_reverb.append(srmr)
        SDR_reverb.append(sdr)
        ESTOI_reverb.append(estoi)
        
        # get metrics for clean-dereverb
        srmr, sdr, estoi = get_metricas(clean, dereverb, fs)
        SRMR_dereverb.append(srmr)
        SDR_dereverb.append(sdr)
        ESTOI_dereverb.append(estoi)

    # save results
    np.save(CARPETA+'SRMR_reverb.npy', SRMR_reverb)
    np.save(CARPETA+'SDR_reverb.npy', SDR_reverb)
    np.save(CARPETA+'ESTOI_reverb.npy', ESTOI_reverb)
    np.save(CARPETA+'SRMR_dereverb.npy', SRMR_dereverb)
    np.save(CARPETA+'SDR_dereverb.npy', SDR_dereverb)
    np.save(CARPETA+'ESTOI_dereverb.npy', ESTOI_dereverb)
    return
"""
if __name__ == '__main__':
    CLEAN_PATH = '/home/martin/deep-dereverb/data/test/clean/' # fijo
    REVERB_PATH = '/home/martin/deep-dereverb/data/test/real/' # puede ser | real(x) | aug | gen |
    PESOS = '/home/martin/deep-dereverb/model/ckpts/weights.03-0.0022.hdf5' # mezclado
    CARPETA = 'resultados/valores/' # puede ser | real(x) | aug | gen |
    test(CLEAN_PATH, REVERB_PATH, PESOS, CARPETA)
"""

if __name__ == '__main__': #pausado por ahora

    """ 
    CLEAN_PATH = '/home/martin/deep-dereverb/data/test/clean/' # fijo
    REVERB_PATH = '/home/martin/deep-dereverb/data/test/real/' # puede ser | real(x) | aug | gen |
    PESOS = '/home/martin/deep-dereverb/model/ckpts/reales/weights.03-0.0028.hdf5' # por ahora real
    CARPETA = 'resultados/valores/pesos_reales/real/' # puede ser | real(x) | aug | gen |
    test(CLEAN_PATH, REVERB_PATH, PESOS, CARPETA)

    REVERB_PATH = '/home/martin/deep-dereverb/data/test/aug/' # puede ser | real | aug(x) | gen |
    CARPETA = 'resultados/valores/pesos_reales/aug/' # puede ser | real | aug(x) | gen |
    test(CLEAN_PATH, REVERB_PATH, PESOS, CARPETA)

    REVERB_PATH = '/home/martin/deep-dereverb/data/test/gen/' # puede ser | real | aug | gen(x) |
    CARPETA = 'resultados/valores/pesos_reales/gen/' # puede ser | real | aug | gen(x)|
    test(CLEAN_PATH, REVERB_PATH, PESOS, CARPETA)

    REVERB_PATH = '/home/martin/deep-dereverb/data/test/real/' # puede ser | real(x) | aug | gen |
    PESOS = '/home/martin/deep-dereverb/model/ckpts/gen/weights.03-0.0024.hdf5' # por ahora gen
    CARPETA = 'resultados/valores/pesos_gen/real/' # puede ser | real(x) | aug | gen|
    test(CLEAN_PATH, REVERB_PATH, PESOS, CARPETA)

    REVERB_PATH = '/home/martin/deep-dereverb/data/test/aug/' # puede ser | real | aug(x) | gen |
    CARPETA = 'resultados/valores/pesos_gen/aug/' # puede ser | real | aug(x) | gen|
    test(CLEAN_PATH, REVERB_PATH, PESOS, CARPETA)

    REVERB_PATH = '/home/martin/deep-dereverb/data/test/gen/' # puede ser | real | aug | gen(x) |
    CARPETA = 'resultados/valores/pesos_gen/gen/' # puede ser | real | aug | gen(x)|
    test(CLEAN_PATH, REVERB_PATH, PESOS, CARPETA)

    REVERB_PATH = '/home/martin/deep-dereverb/data/test/real/' # puede ser | real(x) | aug | gen |
    PESOS = '/home/martin/deep-dereverb/model/ckpts/aug/weights.03-0.0015.hdf5' # por ahora aug
    CARPETA = 'resultados/valores/pesos_aug/real/' # puede ser | real(x) | aug | gen|
    test(CLEAN_PATH, REVERB_PATH, PESOS, CARPETA)

    REVERB_PATH = '/home/martin/deep-dereverb/data/test/aug/' # puede ser | real | aug(x) | gen |
    CARPETA = 'resultados/valores/pesos_aug/aug/' # puede ser | real | aug(x) | gen|
    test(CLEAN_PATH, REVERB_PATH, PESOS, CARPETA)

    REVERB_PATH = '/home/martin/deep-dereverb/data/test/gen/' # puede ser | real | aug | gen(x) |
    CARPETA = 'resultados/valores/pesos_aug/gen/' # puede ser | real | aug | gen(x)|
    test(CLEAN_PATH, REVERB_PATH, PESOS, CARPETA)
    """
    #mezcla! 
    CLEAN_PATH = '/home/martin/deep-dereverb/data/test/clean/' # fijo
    REVERB_PATH = '/home/martin/deep-dereverb/data/test/real/' # puede ser | real(x) | aug | gen |
    PESOS = '/home/martin/deep-dereverb/model/ckpts/weights.03-0.0022.hdf5' #mezclado
    CARPETA = 'resultados/valores/mezcla/real/' # puede ser | real(x) | aug | gen|
    test(CLEAN_PATH, REVERB_PATH, PESOS, CARPETA)

    REVERB_PATH = '/home/martin/deep-dereverb/data/test/aug/' # puede ser | real | aug(x) | gen |
    CARPETA = 'resultados/valores/mezcla/aug/' # puede ser | real | aug(x) | gen|
    test(CLEAN_PATH, REVERB_PATH, PESOS, CARPETA)

    REVERB_PATH = '/home/martin/deep-dereverb/data/test/gen/' # puede ser | real | aug | gen(x) |
    CARPETA = 'resultados/valores/mezcla/gen/' # puede ser | real | aug | gen(x)|
    test(CLEAN_PATH, REVERB_PATH, PESOS, CARPETA)

