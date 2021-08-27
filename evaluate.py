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

def normalise(array):
        norm_array = (array - array.min()) / (array.max() - array.min() + EPS)
        return norm_array, array.min(), array.max()
    
def denormalise(norm_array, original_min, original_max):
        array = norm_array * (original_max - original_min) + original_min
        return array


def gen_stft(audio):
    # compute STFT dismissing high freqs bin
    stft_ = librosa.stft(audio, n_fft=512, hop_length=128)[:-1,:]
    
    # only magnitude
    stft_ = np.abs(stft_)

    # logaritmic scale
    log_stft_ = librosa.amplitude_to_db(stft_)

    # normalize
    norm_stft_, arr_min, arr_max = normalise(log_stft_)
    return norm_stft_, arr_min, arr_max


def frame_to_raw(frame, arr_min, arr_max):
    frame = denormalise(frame, arr_min, arr_max)
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
    clean_list = get_audio_list(CLEAN_PATH)
    reverb_list = get_audio_list(REVERB_PATH)

    modelo = autoencoder()
    modelo.load_weights(PESOS)

    SRMR_reverb = []
    SDR_reverb = []
    ESTOI_reverb = []

    SRMR_dereverb = []
    SDR_dereverb = []
    ESTOI_dereverb = []

    for clean_path, reverb_path in tqdm(zip(clean_list, reverb_list), total=len(clean_list)):

        # read files
        clean, fs = librosa.load(clean_path, sr=None)
        reverb, fs = librosa.load(reverb_path, sr=None)

        # apply model
        espectro_in, arr_min, arr_max = gen_stft(reverb)
        espectro_out = modelo.predict([espectro_in.reshape(1,256,256)])
        espectro_out = espectro_out.reshape(256,256)
        dereverb = frame_to_raw(espectro_out, arr_min, arr_max)

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

if __name__ == '__main__':
    
    CLEAN_PATH = '/home/martin/deep-dereverb/data/test/clean/' # fijo
    REVERB_PATH = '/home/martin/deep-dereverb/data/test/real/' # puede ser | real(x) | aug | gen |
    PESOS = '/home/martin/deep-dereverb/model/ckpts/reales/weights.10-0.010.hdf5' # por ahora real
    CARPETA = 'resultados/pesos_reales/real/' # puede ser | real(x) | aug | gen |

    CLEAN_PATH = '/home/martin/deep-dereverb/data/test/clean/'
    REVERB_PATH = '/home/martin/deep-dereverb/data/test/aug/' # puede ser | real | aug(x) | gen |
    PESOS = '/home/martin/deep-dereverb/model/ckpts/reales/weights.10-0.010.hdf5' # por ahora real
    CARPETA = 'resultados/pesos_reales/aug/' # puede ser | real | aug(x) | gen |

    CLEAN_PATH = '/home/martin/deep-dereverb/data/test/clean/'
    REVERB_PATH = '/home/martin/deep-dereverb/data/test/gen/' # puede ser | real | aug | gen(x) |
    PESOS = '/home/martin/deep-dereverb/model/ckpts/reales/weights.10-0.010.hdf5' # por ahora real
    CARPETA = 'resultados/pesos_reales/gen/' # puede ser | real | aug | gen(x)|

    CLEAN_PATH = '/home/martin/deep-dereverb/data/test/clean/' # fijo
    REVERB_PATH = '/home/martin/deep-dereverb/data/test/real/' # puede ser | real(x) | aug | gen |
    PESOS = '/home/martin/deep-dereverb/model/ckpts/gen/weights.10-0.008.hdf5' # por ahora gen
    CARPETA = 'resultados/valores/pesos_gen/real/' # puede ser | real(x) | aug | gen|
    test(CLEAN_PATH, REVERB_PATH, PESOS, CARPETA)

    CLEAN_PATH = '/home/martin/deep-dereverb/data/test/clean/' 
    REVERB_PATH = '/home/martin/deep-dereverb/data/test/aug/' # puede ser | real | aug(x) | gen |
    PESOS = '/home/martin/deep-dereverb/model/ckpts/gen/weights.10-0.008.hdf5' # por ahora gen
    CARPETA = 'resultados/valores/pesos_gen/aug/' # puede ser | real | aug(x) | gen|
    test(CLEAN_PATH, REVERB_PATH, PESOS, CARPETA)

    CLEAN_PATH = '/home/martin/deep-dereverb/data/test/clean/'
    REVERB_PATH = '/home/martin/deep-dereverb/data/test/gen/' # puede ser | real | aug | gen(x) |
    PESOS = '/home/martin/deep-dereverb/model/ckpts/gen/weights.10-0.008.hdf5' # por ahora gen
    CARPETA = 'resultados/valores/pesos_gen/gen/' # puede ser | real | aug | gen(x)|
    test(CLEAN_PATH, REVERB_PATH, PESOS, CARPETA)

    CLEAN_PATH = '/home/martin/deep-dereverb/data/test/clean/' # fijo
    REVERB_PATH = '/home/martin/deep-dereverb/data/test/real/' # puede ser | real(x) | aug | gen |
    PESOS = '/home/martin/deep-dereverb/model/ckpts/aug/weights.10-0.006.hdf5' # por ahora aug
    CARPETA = 'resultados/pesos_aug/real/' # puede ser | real(x) | aug | gen|
    test(CLEAN_PATH, REVERB_PATH, PESOS, CARPETA)

    CLEAN_PATH = '/home/martin/deep-dereverb/data/test/clean/'
    REVERB_PATH = '/home/martin/deep-dereverb/data/test/aug/' # puede ser | real | aug(x) | gen |
    PESOS = '/home/martin/deep-dereverb/model/ckpts/aug/weights.10-0.006.hdf5' # por ahora aug
    CARPETA = 'resultados/pesos_aug/aug/' # puede ser | real | aug(x) | gen|
    test(CLEAN_PATH, REVERB_PATH, PESOS, CARPETA)

    CLEAN_PATH = '/home/martin/deep-dereverb/data/test/clean/'
    REVERB_PATH = '/home/martin/deep-dereverb/data/test/gen/' # puede ser | real | aug | gen(x) |
    PESOS = '/home/martin/deep-dereverb/model/ckpts/aug/weights.10-0.006.hdf5' # por ahora aug
    CARPETA = 'resultados/pesos_aug/gen/' # puede ser | real | aug | gen(x)|
    test(CLEAN_PATH, REVERB_PATH, PESOS, CARPETA)

