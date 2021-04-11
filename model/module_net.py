MAIN_PATH='/home/martin/Documents/tesis/src'
import sys, os
sys.path.append(MAIN_PATH)
import tensorflow as tf
import tensorflow.keras.layers as tfkl
import numpy as np
import sys
from model.capas import *

def encoder_layer(in_flow, n_filt, kernel, ID, batch_norm=True, leaky_relu=True):
    """
    in_flow : layer : layer de entrada
    n_filt : int : cantidad de filtros de la capa convolucional
    kernel : tuple : tamaño del kernel de la capa convolucional, dos dimensiones 
    ID : string : nombre de la capa 
    batch_norm : boolean : aplicar o no batch normalization
    leaky_relu : boolean : aplicar o no activacion leaky relu 
    """
    enc = tfkl.Conv2D(n_filt, kernel_size=kernel, strides=2, padding='SAME', name=ID)(in_flow)
    if batch_norm:
        enc = tfkl.BatchNormalization()(enc)
    if leaky_relu:
        enc = tfkl.LeakyReLU(alpha = 0.2)(enc)
    else:
        enc = tfkl.ReLU()(enc)
    return enc

def decoder_layer(in_flow, n_filt, kernel, ID, batch_norm=True, relu=True, in_concat=None):
    """
    in_flow : layer
    in_concat : layer
    n_filt : int
    kernel : tuple 
    ID : string
    batch_norm : boolean
    leaky_relu : boolean

    """
    dec = tfkl.UpSampling2D(size=(2,2), interpolation='nearest')(in_flow)
    dec = tfkl.Conv2D(n_filt, kernel_size=kernel, strides=1, padding='SAME', name=ID)(dec)
    if batch_norm:
        dec = tfkl.BatchNormalization()(dec)
    if relu:
        dec = tfkl.ReLU()(dec)
    else:
        dec = tf.keras.activations.tanh(dec)
    if in_concat != None:
        dec = tfkl.Concatenate(axis=-1)([dec, in_concat])

    return dec

#armo el modelo 

tf.keras.backend.clear_session()
eps = np.finfo(float).eps

audio_in = tfkl.Input((257,256), name = 'Entrada_audio')
mask_in = tfkl.Input((257,256), name = 'Entrada_mascara')

#Acondicionamiento
audio = tf.expand_dims(audio_in,axis=-1, name='expand_audio')
audio = tfkl.Cropping2D(((0,1),(0,0)), name='Crop_audio')(audio)

mask = tf.expand_dims(mask_in,axis=-1, name= 'expand_mask')
mask = tfkl.Cropping2D(((0,1),(0,0)), name = 'Crop_mask')(mask)

#encoder
enc_1 = encoder_layer(audio, 8, (4,4), 'conv1', batch_norm=False, leaky_relu=True)
enc_2 = encoder_layer(enc_1, 16, (4,4), 'conv2', batch_norm=True, leaky_relu=True)
enc_3 = encoder_layer(enc_2, 32, (4,4), 'neck', batch_norm=True, leaky_relu=True)

#decoder
dec = decoder_layer(enc_3, 16, (4,4), 'dconv1', batch_norm=True, relu=True, in_concat=enc_2)
dec = decoder_layer(dec, 8, (4,4), 'dconv2', batch_norm=True, relu=True, in_concat=enc_1)
dec = decoder_layer(dec, 1, (4,4), 'out', batch_norm=False, relu=False, in_concat=None)


err = MSE()([dec,mask])
modelo = tf.keras.Model(inputs=[audio_in, mask_in],outputs=[err])

#Compilacion
modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss=lambda y_true, y_pred : tf.reduce_mean(ypred)) #defino el optimizador y le indico que use laue defini

modelo.summary()

