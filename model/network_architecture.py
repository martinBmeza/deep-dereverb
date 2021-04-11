"""Modelo de red utilizado"""

MAIN_PATH='/home/martin/Documents/tesis/src'
import sys, os
sys.path.append(MAIN_PATH)
import tensorflow as tf
import tensorflow.keras.layers as tfkl
import numpy as np
import sys
from model.capas import *


def enc_layer(in_flow, n_filt, kernel, ID, batch_norm=True, leaky_relu=True):
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
    return  enc

def dec_layer(in_flow, n_filt, kernel, ID, batch_norm=True, dropout=False, relu=True, in_concat=None):
    """
    in_flow : layer
    in_concat : layer
    n_filt : int
    kernel : tuple 
    ID : string
    batch_norm : boolean
    dropout : boolean
    leaky_relu : boolean

    """
    dec = tfkl.UpSampling2D(size=(2,2), interpolation='nearest')(in_flow)
    dec = tfkl.Conv2D(n_filt, kernel_size=kernel, strides=1, padding='SAME', name=ID)(dec)
    if batch_norm:
        dec = tfkl.BatchNormalization()(dec)
    if dropout:
        dec = tfkl.Dropout(rate=0.5)(dec)
    if relu:
        dec = tfkl.ReLU()(dec)
    else:
        dec = tf.keras.activations.tanh(dec)
    if in_concat != None:
        dec = tfkl.Concatenate(axis=-1)([dec, in_concat])

    return dec



def modelo_de_prueba():

    tf.keras.backend.clear_session()
    eps = np.finfo(float).eps
    kernel = (4,4)

    audio_in = tfkl.Input((257,256), name = 'Entrada_audio')
    mask_in = tfkl.Input((257,256), name = 'Entrada_mascara')

    #Acondicionamiento
    audio = tf.expand_dims(audio_in,axis=-1, name='expand_audio')
    audio = tfkl.Cropping2D(((0,1),(0,0)), name='Crop_audio')(audio)

    mask = tf.expand_dims(mask_in,axis=-1, name= 'expand_mask')
    mask = tfkl.Cropping2D(((0,1),(0,0)), name = 'Crop_mask')(mask)

    #ENCODER
    enc_1 = enc_layer(audio,8,kernel, 'conv1', batch_norm=False)
    enc_2 = enc_layer(enc_1, 16, kernel, 'conv2')
    enc_3 = enc_layer(enc_2, 32, kernel, 'neck')

    #DECODER
    dec = dec_layer(enc_3, 16, kernel, 'dec1', in_concat=enc_2)
    dec = dec_layer(dec, 8, kernel, 'dec2', in_concat=enc_1)
    dec = dec_layer(dec, 1, kernel, 'out', batch_norm=False, relu=False)

    err = MSE()([dec,mask])
    modelo = tf.keras.Model(inputs=[audio_in, mask_in],outputs=[err])

    #Compilacion
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss=mean_loss) #defino el optimizador y le indico que use laue defini
    return modelo

def dereverb_autoencoder():
    """
    """
    kernel=(4,4)
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
    enc_1 = enc_layer(audio,64,kernel, 'conv1', batch_norm=False)
    enc_2 = enc_layer(enc_1, 128, kernel, 'conv2')
    enc_3 = enc_layer(enc_2, 256, kernel, 'conv3')
    enc_4 = enc_layer(enc_3, 512, kernel, 'conv4')
    enc_5 = enc_layer(enc_4, 512, kernel, 'conv5')
    enc_6 = enc_layer(enc_5, 512, kernel, 'conv6')
    enc_7 = enc_layer(enc_6, 512, kernel, 'conv7')
    enc_8 = enc_layer(enc_7, 512, kernel, 'neck', leaky_relu=False)

    #DECODER
    dec = dec_layer(enc_8, 512, kernel, 'dec1', dropout=True, in_concat=enc_7)
    dec = dec_layer(dec, 512, kernel, 'dec3', dropout=True, in_concat=enc_6) 
    dec = dec_layer(dec, 512, kernel, 'dec4', dropout=True, in_concat=enc_5)
    dec = dec_layer(dec, 512, kernel, 'dec5', in_concat=enc_4)
    dec = dec_layer(dec, 256, kernel, 'dec6', in_concat=enc_3)
    dec = dec_layer(dec, 128, kernel, 'dec7', in_concat=enc_2)
    dec = dec_layer(dec, 64, kernel, 'dec8', in_concat=enc_1)
    dec = dec_layer(dec, 1, kernel, 'out', batch_norm=False, relu=False)

    err = MSE()([dec,mask])
    #err = MSE()([dec,spec_y])
    modelo = tf.keras.Model(inputs=[audio_in, mask_in],outputs=[err])

    #Compilacion
    modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss=mean_loss) #defino el optimizador y le indico que use laue defini
    return modelo

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
