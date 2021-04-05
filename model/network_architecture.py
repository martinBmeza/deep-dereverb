"""Modelo de red utilizado"""

MAIN_PATH = '/home/mrtn/Documents/modelo_deep'
import sys, os
sys.path.append(MAIN_PATH)
import tensorflow as tf
import tensorflow.keras.layers as tfkl
import numpy as np
import sys
from model.capas import *

def dereverb_autoencoder():
    
    tf.keras.backend.clear_session()
    eps = np.finfo(float).eps

    audio_in = tfkl.Input((257,256), name = 'Entrada_audio')
    mask_in = tfkl.Input((257,256), name = 'Entrada_mascara')

    #Acondicionamiento
    audio = tf.expand_dims(audio_in,axis=-1, name='expand_audio')
    audio = tfkl.Cropping2D(((0,1),(0,0)), name='Crop_audio')(audio)

    mask = tf.expand_dims(mask_in,axis=-1, name= 'expand_mask')
    mask = tfkl.Cropping2D(((0,1),(0,0)), name = 'Crop_mask')(mask)


    #ENCODER
    enc = tfkl.Conv2D(64, kernel_size=(4,4), strides=2, padding='SAME', input_shape=(256,256,1), name='CONV1')(audio)
    enc_1 = tfkl.LeakyReLU(alpha = 0.2, name = 'ACT1')(enc)

    enc_2 = tfkl.Conv2D(128, kernel_size=(4,4), strides=2, padding='SAME', name='CONV2')(enc_1)
    enc_2 = tfkl.BatchNormalization(name='BATCH2')(enc_2)
    enc_2 = tfkl.LeakyReLU(alpha = 0.2, name='ACT2')(enc_2)

    enc_3 = tfkl.Conv2D(256, kernel_size=(4,4), strides=2, padding='SAME', name='CONV3')(enc_2)
    enc_3 = tfkl.BatchNormalization(name='BATCH3')(enc_3)
    enc_3 = tfkl.LeakyReLU(alpha = 0.2, name='ACT3')(enc_3)

    enc_4 = tfkl.Conv2D(512, kernel_size=(4,4), strides=2, padding='SAME', name='CONV4')(enc_3)
    enc_4 = tfkl.BatchNormalization(name='BATCH4')(enc_4)
    enc_4 = tfkl.LeakyReLU(alpha = 0.2, name='ACT4')(enc_4)

    enc_5 = tfkl.Conv2D(512, kernel_size=(4,4), strides=2, padding='SAME', name='CONV5')(enc_4)
    enc_5 = tfkl.BatchNormalization(name='BATCH5')(enc_5)
    enc_5 = tfkl.LeakyReLU(alpha = 0.2, name='ACT5')(enc_5)

    enc_6 = tfkl.Conv2D(512, kernel_size=(4,4), strides=2, padding='SAME', name='CONV6')(enc_5)
    enc_6 = tfkl.BatchNormalization(name = 'BATCH6')(enc_6)
    enc_6 = tfkl.LeakyReLU(alpha = 0.2, name='ACT6')(enc_6)

    enc_7 = tfkl.Conv2D(512, kernel_size=(4,4), strides=2, padding='SAME', name='CONV7')(enc_6)
    enc_7 = tfkl.BatchNormalization(name = 'BATCH7')(enc_7)
    enc_7 = tfkl.LeakyReLU(alpha = 0.2, name='ACT7')(enc_7)

    enc_8 = tfkl.Conv2D(512, kernel_size=(4,4), strides=2, padding='SAME', name='CONV8')(enc_7)
    enc_8 = tfkl.BatchNormalization(name = 'BATCH8')(enc_8)
    enc_8 = tfkl.ReLU()(enc_8)


    #DECODER
    dec = tfkl.UpSampling2D(size=(2,2), interpolation = 'nearest')(enc_8)
    dec = tfkl.Conv2D(512, kernel_size=(4,4), strides=1, padding='SAME', name='CONV9')(dec)
    #dec = tfkl.Conv2DTranspose(256, kernel_size=(4,4), strides=2, padding='SAME', name='CONV9')(enc_8)
    dec = tfkl.BatchNormalization(name = 'BATCH9')(dec)
    dec = tfkl.Dropout(rate = 0.5)(dec)
    dec = tfkl.ReLU()(dec)
    dec = tfkl.Concatenate(axis=-1)([dec, enc_7])

    dec = tfkl.UpSampling2D(size=(2,2), interpolation = 'nearest')(dec)
    dec = tfkl.Conv2D(512, kernel_size=(4,4), strides=1, padding='SAME', name='CONV10')(dec)
    #dec = tfkl.Conv2DTranspose(256, kernel_size=(4,4), strides=2, padding='SAME', name='CONV10')(dec)
    dec = tfkl.BatchNormalization(name = 'BATCH10')(dec)
    dec = tfkl.Dropout(rate = 0.5)(dec)
    dec = tfkl.ReLU()(dec)
    dec = tfkl.Concatenate(axis=-1)([dec, enc_6])

    dec = tfkl.UpSampling2D(size=(2,2), interpolation = 'nearest')(dec)
    dec = tfkl.Conv2D(512, kernel_size=(4,4), strides=1, padding='SAME', name='CONV11')(dec)
    #dec = tfkl.Conv2DTranspose(256, kernel_size=(4,4), strides=2, padding='SAME', name='CONV11')(dec)
    dec = tfkl.BatchNormalization(name = 'BATCH11')(dec)
    dec = tfkl.Dropout(rate = 0.5)(dec)
    dec = tfkl.ReLU()(dec)
    dec = tfkl.Concatenate(axis=-1)([dec, enc_5])

    dec = tfkl.UpSampling2D(size=(2,2), interpolation = 'nearest')(dec)
    dec = tfkl.Conv2D(512, kernel_size=(4,4), strides=1, padding='SAME', name='CONV12')(dec)
    #dec = tfkl.Conv2DTranspose(256, kernel_size=(4,4), strides=2, padding='SAME', name='CONV12')(dec)
    dec = tfkl.BatchNormalization(name = 'BATCH12')(dec)
    dec = tfkl.ReLU()(dec)
    dec = tfkl.Concatenate(axis=-1)([dec, enc_4])

    dec = tfkl.UpSampling2D(size=(2,2), interpolation = 'nearest')(dec) #PROVISORIO
    dec = tfkl.Conv2D(256, kernel_size=(4,4), strides=1, padding='SAME', name='CONV13')(dec)
    #dec = tfkl.Conv2DTranspose(128, kernel_size=(4,4), strides=2, padding='SAME', name='CONV13')(dec)
    dec = tfkl.BatchNormalization(name = 'BATCH13')(dec)
    dec = tfkl.ReLU()(dec)
    dec = tfkl.Concatenate(axis=-1)([dec, enc_3])

    dec = tfkl.UpSampling2D(size=(2,2), interpolation = 'nearest')(dec)
    dec = tfkl.Conv2D(128, kernel_size=(4,4), strides=1, padding='SAME', name='CONV14')(dec)
    #dec = tfkl.Conv2DTranspose(64, kernel_size=(4,4), strides=2, padding='SAME', name='CONV14')(dec)
    dec = tfkl.BatchNormalization(name = 'BATCH14')(dec)
    dec = tfkl.ReLU()(dec)
    dec = tfkl.Concatenate(axis=-1)([dec, enc_2])

    dec = tfkl.UpSampling2D(size=(2,2), interpolation = 'nearest')(dec)
    dec = tfkl.Conv2D(64, kernel_size=(4,4), strides=1, padding='SAME', name='CONV15')(dec)
    #dec = tfkl.Conv2DTranspose(32, kernel_size=(4,4), strides=2, padding='SAME', name='CONV15')(dec)
    dec = tfkl.BatchNormalization(name = 'BATCH15')(dec)
    dec = tfkl.ReLU()(dec)
    dec = tfkl.Concatenate(axis=-1)([dec, enc_1])

    dec = tfkl.UpSampling2D(size=(2,2), interpolation = 'nearest')(dec)
    dec = tfkl.Conv2D(1, kernel_size=(4,4), strides=1, padding='SAME', activation='tanh', name='CONV16')(dec)
    #dec = tfkl.Conv2DTranspose(1, kernel_size=(4,4), strides=2, padding='SAME', activation='tanh', name='CONV16')(dec)

    err = MSE()([dec,mask])
    #err = MSE()([dec,spec_y])
    modelo = tf.keras.Model(inputs=[audio_in, mask_in],outputs=[err])

    #Compilacion
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss=mean_loss) #defino el optimizador y le indico que use laue defini
    return modelo

import numpy as np 
from model.capas import Spectrogram, TranslateRange, MSE

def basic_autoencoder():
    eps = np.finfo(float).eps

    x = tfkl.Input((32000), name = 'Entrada_Noisy')
    y = tfkl.Input((32000), name = 'Target_Clean')

    #GENERACION DE ESPECTROS
    spec_x = Spectrogram(1024,512,name='STFT_X')(x)
    spec_x = tf.math.log(spec_x+eps)
    spec_x = TranslateRange(original_range=[-5, 5],target_range=[0,1.0])(spec_x)
    spec_x = tf.expand_dims(spec_x,axis=-1, name= 'expand_X')
    spec_x = tfkl.Cropping2D(((0,1),(0,1)), name = 'Crop_X')(spec_x)

    spec_y = Spectrogram(1024,512,name='STFT_Y')(y)
    spec_y = tf.expand_dims(spec_y,axis=-1, name='expand_Y')

    #ENCODER
    enc = tfkl.Conv2D(16, kernel_size=(3,3), padding='SAME', activation='relu', input_shape=(64,64,1), name='convNET1')(spec_x)
    enc = tfkl.Conv2D(16, kernel_size=(3,3), strides=2, padding='SAME', activation='relu', name='convNET2')(enc)
    enc = tfkl.Conv2D(32, kernel_size=(3,3), strides=2, padding='SAME', activation='relu', name='convNET3')(enc)

    #DECODER
    dec = tfkl.Conv2DTranspose(32, kernel_size=(3,3), strides=2, padding='SAME', activation='relu', name='deconvNET1')(enc)
    dec = tfkl.Conv2DTranspose(16, kernel_size=(3,3), strides=2, padding='SAME', activation='relu', name='deconvNET2')(dec)
    dec = tfkl.Conv2D(1, kernel_size=(3,3), padding='SAME', activation='relu', name='deconvNET3')(dec)
    
    estimated = tfkl.Multiply()([dec,spec_x])
    estimated = tf.pad(estimated,((0,0),(0,1),(0,1),(0,0)), name='SALIDA')
    err = MSE()([estimated,spec_y])
    model = tf.keras.Model(inputs=[x,y],outputs=[err])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),loss=mean_loss)

    return model



def mean_loss(y_true, y_pred):
    """
    Custom loss. El error cuadratico ya se calcula en la net, solo lo reduzco a un escalar
    """
    return tf.reduce_mean(y_pred)