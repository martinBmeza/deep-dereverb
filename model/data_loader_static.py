"""Alimentar con datos al bucle de entrenamiento"""
import keras 
import numpy as np
import os
import glob
import random


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, path='', batch_size=8, dim=(257,256), n_channels=1,  shuffle=True):
        'Initialization'
        self.path = path
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
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

        x_clean = np.empty((self.batch_size, 257, 256))
        x_reverb = np.empty((self.batch_size, 257, 256))
        Y = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            
            [x_clean[i], x_reverb[i]] = np.load(self.path + str(ID) + '.npy')
            Y[i] = x_reverb[i]
           
        return [x_clean, x_reverb], Y

def build_generators(params):
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
    
    audio_list = glob.glob(params['path']+'/**/*.npy', recursive = True)
    
    #seleccion random de sets
    audio_numbers = list(range(0, len(audio_list)))
    random.shuffle(audio_numbers)
    train_n = int(len(audio_numbers)*0.9)
    validation_n = len(audio_numbers) - train_n
    
    partition = {'train' : audio_numbers[:train_n], 'validation' : audio_numbers[train_n:]}
    
    # Generators
    training_generator = DataGenerator(partition['train'], **params)
    validation_generator = DataGenerator(partition['validation'], **params)
    
    print('Cantidad de datos para entrenamiento:', len(partition['train']))
    print('Cantidad de datos para validacion:', len(partition['validation']))
    return training_generator, validation_generator
