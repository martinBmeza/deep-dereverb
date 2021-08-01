"""Modelo de red utilizado"""

MAIN_PATH='/home/martin/deep-dereverb'
import sys, os
sys.path.append(MAIN_PATH)
import tensorflow as tf
import tensorflow.keras.layers as tfkl
import numpy as np
import sys
from model.capas import *


def autoencoder():
    
    # Autoencoder params
    enc_args = {'kernel_size' : (5,5),
                    'strides' : 2,
                    'padding' : 'SAME'}

    dec_args = {'kernel_size' : (5,5),
                'strides' : 1,
                'padding' : 'SAME'}
    
    #f = [64, 128, 256, 512, 512, 512, 512, 512] # filters
    
    f = [8, 16, 32, 64, 128, 256, 512, 1024] # filters

    tf.keras.backend.clear_session()
    eps = np.finfo(float).eps

    reverb_in = tfkl.Input((256,256), name = 'Entrada_reverb')
    #clean_in = tfkl.Input((256,256), name = 'Entrada_clean')

    #Acondicionamiento
    reverb = tf.expand_dims(reverb_in,axis=-1, name='Reverb')
    #reverb = tfkl.Cropping2D(((0,1),(0,0)), name='ESPECTRO_REVERB')(reverb)

    #clean = tf.expand_dims(clean_in,axis=-1, name= 'Clean')
    #clean = tfkl.Cropping2D(((0,1),(0,0)), name = 'ESPECTRO_CLEAN')(clean)


    #ENCODER
    enc = tfkl.Conv2D(f[0], kernel_size=(5,5), strides=2, padding='SAME', input_shape=(256,256,1), name='CONV1')(reverb)
    enc_1 = tfkl.LeakyReLU(alpha = 0.2, name = 'ACT1')(enc)

    enc_2 = tfkl.Conv2D(f[1], kernel_size=(5,5), strides=2, padding='SAME', name='CONV2')(enc_1)
    enc_2 = tfkl.BatchNormalization(name='BATCH2')(enc_2)
    enc_2 = tfkl.LeakyReLU(alpha = 0.2, name='ACT2')(enc_2)

    enc_3 = tfkl.Conv2D(f[2], kernel_size=(5,5), strides=2, padding='SAME', name='CONV3')(enc_2)
    enc_3 = tfkl.BatchNormalization(name='BATCH3')(enc_3)
    enc_3 = tfkl.LeakyReLU(alpha = 0.2, name='ACT3')(enc_3)

    enc_4 = tfkl.Conv2D(f[3], kernel_size=(5,5), strides=2, padding='SAME', name='CONV4')(enc_3)
    enc_4 = tfkl.BatchNormalization(name='BATCH4')(enc_4)
    enc_4 = tfkl.LeakyReLU(alpha = 0.2, name='ACT4')(enc_4)

    enc_5 = tfkl.Conv2D(f[4], kernel_size=(5,5), strides=2, padding='SAME', name='CONV5')(enc_4)
    enc_5 = tfkl.BatchNormalization(name='BATCH5')(enc_5)
    enc_5 = tfkl.LeakyReLU(alpha = 0.2, name='ACT5')(enc_5)

    enc_6 = tfkl.Conv2D(f[5], kernel_size=(5,5), strides=2, padding='SAME', name='CONV6')(enc_5)
    enc_6 = tfkl.BatchNormalization(name = 'BATCH6')(enc_6)
    enc_6 = tfkl.LeakyReLU(alpha = 0.2, name='ACT6')(enc_6)

    enc_7 = tfkl.Conv2D(f[6], kernel_size=(5,5), strides=2, padding='SAME', name='CONV7')(enc_6)
    enc_7 = tfkl.BatchNormalization(name = 'BATCH7')(enc_7)
    enc_7 = tfkl.LeakyReLU(alpha = 0.2, name='ACT7')(enc_7)

    enc_8 = tfkl.Conv2D(f[7], kernel_size=(5,5), strides=2, padding='SAME', name='CONV8')(enc_7)
    enc_8 = tfkl.BatchNormalization(name = 'BATCH8')(enc_8)
    enc_8 = tfkl.ReLU()(enc_8)


    #DECODER
    dec = tfkl.UpSampling2D(size=(2,2), interpolation = 'nearest')(enc_8)
    dec = tfkl.Conv2D(f[6], kernel_size=(5,5), strides=1, padding='SAME', name='CONV9')(dec)
    #dec = tfkl.Conv2DTranspose(256, kernel_size=(4,4), strides=2, padding='SAME', name='CONV9')(enc_8)
    dec = tfkl.BatchNormalization(name = 'BATCH9')(dec)
    dec = tfkl.Dropout(rate = 0.5)(dec)
    dec = tfkl.ReLU()(dec)
    #dec = tfkl.Concatenate(axis=-1)([dec, enc_7])
    dec = tfkl.Add()([dec, enc_7])

    dec = tfkl.UpSampling2D(size=(2,2), interpolation = 'nearest')(dec)
    dec = tfkl.Conv2D(f[5], kernel_size=(5,5), strides=1, padding='SAME', name='CONV10')(dec)
    #dec = tfkl.Conv2DTranspose(256, kernel_size=(4,4), strides=2, padding='SAME', name='CONV10')(dec)
    dec = tfkl.BatchNormalization(name = 'BATCH10')(dec)
    dec = tfkl.Dropout(rate = 0.5)(dec)
    dec = tfkl.ReLU()(dec)
    #dec = tfkl.Concatenate(axis=-1)([dec, enc_6])
    dec = tfkl.Add()([dec, enc_6])

    dec = tfkl.UpSampling2D(size=(2,2), interpolation = 'nearest')(dec)
    dec = tfkl.Conv2D(f[4], kernel_size=(5,5), strides=1, padding='SAME', name='CONV11')(dec)
    #dec = tfkl.Conv2DTranspose(256, kernel_size=(4,4), strides=2, padding='SAME', name='CONV11')(dec)
    dec = tfkl.BatchNormalization(name = 'BATCH11')(dec)
    dec = tfkl.Dropout(rate = 0.5)(dec)
    dec = tfkl.ReLU()(dec)
    #dec = tfkl.Concatenate(axis=-1)([dec, enc_5])
    dec = tfkl.Add()([dec, enc_5])

    dec = tfkl.UpSampling2D(size=(2,2), interpolation = 'nearest')(dec)
    dec = tfkl.Conv2D(f[3], kernel_size=(5,5), strides=1, padding='SAME', name='CONV12')(dec)
    #dec = tfkl.Conv2DTranspose(256, kernel_size=(4,4), strides=2, padding='SAME', name='CONV12')(dec)
    dec = tfkl.BatchNormalization(name = 'BATCH12')(dec)
    dec = tfkl.ReLU()(dec)
    #dec = tfkl.Concatenate(axis=-1)([dec, enc_4])
    dec = tfkl.Add()([dec, enc_4])

    dec = tfkl.UpSampling2D(size=(2,2), interpolation = 'nearest')(dec) #PROVISORIO
    dec = tfkl.Conv2D(f[2], kernel_size=(5,5), strides=1, padding='SAME', name='CONV13')(dec)
    #dec = tfkl.Conv2DTranspose(128, kernel_size=(4,4), strides=2, padding='SAME', name='CONV13')(dec)
    dec = tfkl.BatchNormalization(name = 'BATCH13')(dec)
    dec = tfkl.ReLU()(dec)
    #dec = tfkl.Concatenate(axis=-1)([dec, enc_3])
    dec = tfkl.Add()([dec, enc_3])

    dec = tfkl.UpSampling2D(size=(2,2), interpolation = 'nearest')(dec)
    dec = tfkl.Conv2D(f[1], kernel_size=(5,5), strides=1, padding='SAME', name='CONV14')(dec)
    #dec = tfkl.Conv2DTranspose(64, kernel_size=(4,4), strides=2, padding='SAME', name='CONV14')(dec)
    dec = tfkl.BatchNormalization(name = 'BATCH14')(dec)
    dec = tfkl.ReLU()(dec)
    #dec = tfkl.Concatenate(axis=-1)([dec, enc_2])
    dec = tfkl.Add()([dec, enc_2])

    dec = tfkl.UpSampling2D(size=(2,2), interpolation = 'nearest')(dec)
    dec = tfkl.Conv2D(f[0], kernel_size=(5,5), strides=1, padding='SAME', name='CONV15')(dec)
    #dec = tfkl.Conv2DTranspose(32, kernel_size=(4,4), strides=2, padding='SAME', name='CONV15')(dec)
    dec = tfkl.BatchNormalization(name = 'BATCH15')(dec)
    dec = tfkl.ReLU()(dec)
    #dec = tfkl.Concatenate(axis=-1)([dec, enc_1])
    dec = tfkl.Add()([dec, enc_1])


    dec = tfkl.UpSampling2D(size=(2,2), interpolation = 'nearest')(dec)
    dec = tfkl.Conv2D(1, kernel_size=(5,5), strides=1, padding='SAME', activation='relu', name='SALIDA_DEL_DECODER')(dec)
    #dec = tfkl.Activation('tanh', name = 'SALIDA_DEL_DECODER')(dec)
    #dec = tfkl.Conv2DTranspose(1, kernel_size=(4,4), strides=2, padding='SAME', activation='tanh', name='CONV16')(dec)

    clean_predict = tfkl.multiply([dec, reverb], name = 'CLEAN_PREDICT')
    #clean_predict = tf.pad(clean_predict, ((0,0),(0,1),(0,0),(0,0)), mode='CONSTANT', constant_values=0)

    modelo = tf.keras.Model(inputs=[reverb_in], outputs=[clean_predict])

    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

    return modelo

def mean_loss(y_true, y_pred):
    """
    Custom loss. El error cuadratico ya se calcula en la net, solo lo reduzco a un escalar
    """
    return tf.reduce_mean(y_pred)


