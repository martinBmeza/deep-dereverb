import numpy as np
import glob
import librosa
from scipy.signal import fftconvolve, stft
import tensorflow as tf

import sys
#sys.path.append('/home/martin/Documents/tesis/src/')
sys.path.append('/home/martin/Documents/tesis/SRMRpy/srmrpy')   
import srmrpy as srmr
import mir_eval
import pystoi
from model.network_architecture import autoencoder
EPS = np.finfo(float).eps

def get_metricas(clean, reverb, fs):
    SRMR = srmr.srmr(reverb, fs)[0]
    SDR, _, _, _ = mir_eval.separation.bss_eval_sources(clean, reverb, compute_permutation=True)
    ESTOI = pystoi.stoi(clean, reverb, fs, extended = True)
    return SRMR, SDR[0], ESTOI

def frame_to_raw(frame, arr_min, arr_max):

    frame = denormalise(frame, arr_min, arr_max)
    #Escala logaritmica
    frame_lin = librosa.db_to_amplitude(frame)

    #Necesito agregar el bin de frecuencia que le saque.
    frame_lin_pad = np.pad(frame_lin,((0,1),(0,0)), 'minimum') #Ojoooooo!

    #Para antitransformar necesito la fase. Puedo estimar a partir de griffim lim 
    frame_raw = librosa.griffinlim(frame_lin_pad, 
                                        n_iter=500,
                                        hop_length=128, 
                                        win_length=512)
    return frame_raw


def normalise(array):
        norm_array = (array - array.min()) / (array.max() - array.min() + EPS)
        return norm_array, array.min(), array.max()
    
def denormalise(norm_array, original_min, original_max):
        array = norm_array * (original_max - original_min) + original_min
        return array

def get_audio_list(path, file_types = ('.wav', '.WAV', '.flac', '.FLAC')):
    search_path = path + '/**/*'
    audio_list = []
    for file_type in file_types:
        audio_list.extend(glob.glob(search_path+file_type, recursive=True))
    return audio_list

def gen_stft(audio):
    #Genero las STFT
    stft_ = librosa.stft(audio, n_fft=512, hop_length=128)[:-1,:]# Descarto altas frecuencias
    stft_ = np.abs(stft_)
    #Escala logaritmica
    log_stft_ = librosa.amplitude_to_db(stft_)

    #Normalizacion
    norm_stft_, arr_min, arr_max = normalise(log_stft_)
    return norm_stft_, arr_min, arr_max



def predict_model(data, modelo):
  output = [layer.name for layer in modelo.layers]
  outputs = []
  output_names = []
  inputs = []
  input_names = []
  for layer in modelo.layers:
      if hasattr(layer,'is_placeholder'):
          inputs.append(layer.output)
          input_names.append(layer.name)
      elif layer.name in output:
          outputs.append(layer.output)
          output_names.append(layer.name)
      else:
          pass
  predict_fn = tf.keras.backend.function(inputs = inputs,outputs=outputs)
  activations = predict_fn(data)
  activations = {name: act for name, act in zip(output_names,activations)}
  print('orden de las entradas: /\n',input_names)
  return activations
