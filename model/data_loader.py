"""Alimentar con datos al bucle de entrenamiento"""
import sys
MAIN_PATH='/home/martin/deep-dereverb/model'
sys.path.append(MAIN_PATH) #Para poder importar archivos .py como librerias
from tensorflow.keras.utils import Sequence
import numpy as np
import os
import glob
import random
import librosa
import soundfile as sf
import pandas as pd

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, dataframe, list_IDs, batch_size=8, shuffle=True):
        'Initialization'
        self.dataframe = dataframe
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        x_clean = np.empty((self.batch_size, 256, 256))
        x_reverb = np.empty((self.batch_size, 256, 256))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):  
            reverb, clean = gen_stft(self.dataframe, ID)
            x_clean[i], x_reverb[i] = clean, reverb
        #import pdb; pdb.set_trace() 
        return x_reverb, x_clean  # [input, ground truth]


def build_generators(dataframe, batch, alpha=0.9):
    #seleccion random de sets
    audio_numbers = list(range(0, len(dataframe)))
    random.shuffle(audio_numbers)
    train_n = int(len(audio_numbers)*alpha)
    validation_n = len(audio_numbers) - train_n

    partition = {'train' : audio_numbers[:train_n],
                'val' : audio_numbers[train_n:]}
    # Generators
    train_gen=DataGenerator(dataframe,partition['train'], batch_size=batch)
    val_gen=DataGenerator(dataframe,partition['val'], batch_size=batch)
    return train_gen, val_gen

EPS = np.finfo(float).eps

def normalise(array):
    array_min = -75
    array_max = 65
    norm_array = (array - array_min) / (array_max - array_min + EPS)
    return norm_array

def gen_stft(dataframe, ID):
    clean_path = dataframe.iat[ID, 0]
    reverb_path = dataframe.iat[ID, 1]

    clean = np.load(clean_path)
    reverb = np.load(reverb_path)

    #Genero las STFT
    stft_clean = librosa.stft(clean, n_fft=512, hop_length=128)[:-1,:]# Descarto altas frecuencias
    stft_clean = np.abs(stft_clean)
    stft_reverb = librosa.stft(reverb, n_fft=512, hop_length=128)[:-1,:]
    stft_reverb = np.abs(stft_reverb)

    #Escala logaritmica
    log_stft_clean = librosa.amplitude_to_db(stft_clean)
    log_stft_reverb = librosa.amplitude_to_db(stft_reverb)

    #Normalizacion
    norm_stft_reverb = normalise(log_stft_reverb)
    norm_stft_clean = normalise(log_stft_clean)
    return norm_stft_reverb, norm_stft_clean

