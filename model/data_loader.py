"""Alimentar con datos al bucle de entrenamiento"""
import keras
import numpy as np
import os
import glob
import random
import librosa as lb
from scipy.signal import fftconvolve, stft
MAIN_PATH="/home/martin/Documents/tesis/src"
eps = np.finfo(float).eps

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
    #out = np.expand_dims(out, axis=(3))
    #shape --> (frames, freq, time, channels)
    return out



class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dict_speech, dict_rir):
        'Initialization'
        self.dict_speech = dict_speech
        self.dict_rir = dict_rir
        self.number_speech = len(dict_speech)
        self.number_rir = len(dict_rir)
        self.on_epoch_end()

    def __len__(self):

        'Denotes the number of batches per epoch'
        return 1
       # return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        index_speech = random.randint(0, self.number_speech-1)
        index_rir = random.randint(0, self.number_rir-1)

        # Generate data
        X, y = self.__data_generation(index_speech, index_rir)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
#        self.indexes = np.arange(len(self.list_IDs))
#        if self.shuffle == True:
#            np.random.shuffle(self.indexes)

    def __data_generation(self, index_speech, index_rir):

        #Cargo los datos
        speech, speech_fs = lb.load(self.dict_speech[index_speech], sr=None)
        rir, rir_fs = lb.load(self.dict_rir[index_rir], sr=16000)
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

        magspec_reverb = framing(magspec_reverb)
        mask_comp = framing(mask_comp)
        
        return [magspec_reverb, mask_comp], magspec_reverb

def build_generators(MAIN_PATH):
    """
    Crea instancias de la clase DataGenerator (para entrenamiento y valdiacion) a partir de un diccionario donde se determinan los parametros
    del generador de datos, y el path principal.

    PARAMETROS:
        -MAIN_PATH (str) path principal de la carpeta de trabajo
        -params (dict) diccionario con los campos 'dim'(int), 'batch_size'(int), 'shuffle'(bool) para configurar el generador de datos
        -subpath (str) path de la carpeta dentro de data/ de donde tomar los datos. Por defecto esta asignada a 'data_ready' que es donde se encuentran
        los datos procesados. Puede ser util cambiarla a la carpeta data_dummy para trabajar con los datos dummy en ocasiones de debuggeo

    SALIDA:
        -training_generator (DataGenerator) instancia de clase que contiene los datos para pasarse a una instancia de entrenamiento y proveer
            los datos de entrenamiento al modelo
        -validation_generator (DataGenerator) instancia de clase que contiene los datos para pasarse a una instancia de entrenamiento y proveer
            los datos de validacion al modelo

        """
    list_speech = glob.glob(os.path.join(MAIN_PATH, 'data/speech')+'/**/*.flac', recursive=True)
    dict_speech = {i:j for i,j in enumerate(list_speech)}

    list_rir = glob.glob(os.path.join(MAIN_PATH, 'data/rir')+'/**/*.wav', recursive=True)
    dict_rir = {i:j for i,j in enumerate(list_rir)}

    # Generators
    training_generator = DataGenerator(dict_speech, dict_rir)

    return training_generator
