""" Funciones que sirven para diferentes propositos
como loggin, guardado de datos, carga de datos, etc"""
import numpy as np
import os, shutil
import glob
import librosa
from scipy.signal import fftconvolve, stft

def frame_to_raw(frame, FACTOR=70):
    #Normalizacion 
    frame_denorm = frame * FACTOR

    #Escala logaritmica
    frame_lin = librosa.db_to_amplitude(frame_denorm)

    #Necesito agregar el bin de frecuencia que le saque.
    frame_lin_pad = np.pad(frame_lin,((0,1),(0,0)), 'minimum') #Ojoooooo!

    #Para antitransformar necesito la fase. Puedo estimar a partir de griffim lim 
    frame_raw = librosa.griffinlim(frame_lin_pad, 
                                        n_iter=500,
                                        hop_length=128, 
                                        win_length=512)
    return frame_raw

def get_audio_list(path, file_types = ('.wav', '.WAV', '.flac', '.FLAC')):
    search_path = path + '/**/*'
    audio_list = []
    for file_type in file_types:
        audio_list.extend(glob.glob(search_path+file_type, recursive=True))
    return audio_list


def temporal_decompose(rir, Q_e):
    inicio = np.argmax(abs(rir)) # direct path
    early = rir[inicio:inicio+Q_e]
    late = rir[inicio+Q_e:]
    rir = rir[inicio:]

    #Igualo dimensiones para simplificar la implementacion
    early_pad = np.concatenate((early, np.zeros(len(late))))
    late_pad = np.concatenate((np.zeros(len(early)), late))
    assert len(early_pad) == len(late_pad)
    return early_pad, late_pad, rir


def get_specs_from_path(rir_path, speech_path, FACTOR=60):
    #Cargo los datos
    speech, speech_fs = librosa.load(speech_path, sr=16000)
    rir, rir_fs = librosa.load(rir_path, sr=16000)

    #Normalizo el impulso
    rir = rir / np.max(abs(rir))
    #divido parte early
    rir_early, rir_late, rir = temporal_decompose(rir, Q_e = 32)

    #Convoluciono. Obtengo audio con reverb
    reverb = fftconvolve(speech, rir)
    #Convoluciono y padeo el audio anecoico. Obtengo el audio clean
    clean = fftconvolve(speech, rir_early)

    #Genero las STFT
    stft_clean = librosa.stft(clean, n_fft=512, hop_length=128)[:-1,:]# Descarto altas frecuencias
    stft_clean = np.abs(stft_clean)
    stft_reverb = librosa.stft(reverb, n_fft=512, hop_length=128)[:-1,:]
    stft_reverb = np.abs(stft_reverb)

    #Escala logaritmica
    log_stft_clean = librosa.amplitude_to_db(stft_clean)
    log_stft_reverb = librosa.amplitude_to_db(stft_reverb)

    #Normalizacion
    norm_stft_reverb = log_stft_reverb / FACTOR
    norm_stft_clean = log_stft_clean / FACTOR
    return norm_stft_clean, norm_stft_reverb

def img_framing(data, winsize=256, step=256, dim=1):
    n_frames = int(data.shape[dim] / winsize)
    out = np.empty((n_frames, data.shape[0], winsize))
    for frame in range(n_frames):
        out[frame,:,:] = data[:,frame*winsize : (frame+1)*winsize]
    return out

def img_framing_pad(data, winsize=256, step=256, dim=1):
    time = data.shape[dim]
    n_frames = int(time/winsize)
    resto = time%winsize

    if resto == 0:
        print('Resto 0!')
        out = np.empty((n_frames, data.shape[0], winsize))
    else:
        out = np.empty((n_frames+1, data.shape[0], winsize))
        out[-1,:,:resto] = data[:,n_frames*winsize:]
        out[-1,:,resto:] = np.ones((data.shape[0],winsize-resto)) * data.min()

    for frame in range(n_frames):
        out[frame,:,:] = data[:,frame*winsize : (frame+1)*winsize]
    return out

def prepare_save_path(path):

    if os.path.isdir(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)
    return(path)


def normalizer( array, range_min, range_max, arr_min, arr_max, mode):
    if mode == 'normalise':
        array = (array - arr_min) / (arr_max - arr_min)
        array = array * (range_max - range_min) + range_min
        return array

    if mode == 'denormalise':
        array = (array - range_min) / (range_max - range_min)
        array = array * (arr_max - arr_min) + arr_min
        return array


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
