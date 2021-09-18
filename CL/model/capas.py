# -*- coding: utf-8 -*-
"""
@author: mrtn
"""
import tensorflow as tf
import tensorflow.keras.layers as tfkl
from scipy.signal import stft, istft 
import numpy as np
#import keras


class Spectrogram(tfkl.Layer):
    def __init__(self,win_size,hop_size,fft_size=None,calculate='magnitude',window=tf.signal.hann_window,pad_end=False,name=None, trainable=False):
        super(Spectrogram, self).__init__(name=name)

        self.stft_args = {'ws': win_size,
                  'hs': hop_size,
                  'ffts': fft_size,
                  'win': window,
                  'pad': pad_end,
                  'calculate': calculate}

    def call(self,x):
        stft = tf.signal.stft(
                signals=x,
                frame_length=self.stft_args['ws'],
                frame_step=self.stft_args['hs'],
                fft_length=self.stft_args['ffts'],
                window_fn=self.stft_args['win'],
                pad_end=self.stft_args['pad'])

        calculate = self.stft_args['calculate']
        if calculate == 'magnitude':
            return tf.abs(stft)
        elif calculate == 'complex':
            return stft
        elif calculate == 'phase':
            return tf.math.angle(stft)
        else:
            raise Exception("{} not recognized as calculate parameter".format(calculate))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'win_size': self.stft_args['ws'],
            'hop_size': self.stft_args['hs'],
            'fft_size': self.stft_args['ffts'],
            'calculate': self.stft_args['calculate'],
            'window': self.stft_args['win'],
            'pad_end': self.stft_args['pad']
        })
        return config

class SoftMask(tfkl.Layer):
    def __init__(self,name=None):
        super(SoftMask,self).__init__(name=name)

    def call(self,x):
        sources = x[0]
        to_mask = x[1]
        total_sources = tf.expand_dims(tf.reduce_sum(sources,axis=1),axis=1)
        mask = sources/(total_sources+1e-9)
        to_mask = tf.expand_dims(to_mask,axis=1)
        return mask*to_mask

class TranslateRange(tfkl.Layer):
    def __init__(self,name=None,trainable=False,original_range=None,target_range=None):
        super(TranslateRange,self).__init__(name=name)
        self.original_range = original_range
        self.target_range = target_range

    def call(self,x):
        original = self.original_range[1] - self.original_range[0]
        target = self.target_range[1] - self.target_range[0]
        
        #centro_o = self.original_range[1]-self.original_range[0]
        #centro_t = self.target_range[1]-self.target_range[0]
        #offset = centro_t - centro_o
        #x_range = self.original_range[1] - self.original_range[0]
        #y_range = self.target_range[1] - self.target_range[0]

        #return y_range*(x + offset)/x_range
        return (((x - self.original_range[0]) * target) / original) + self.target_range[0]
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'original_range': self.original_range,
            'target_range': self.target_range,
        })
        return config

class Normalize(tfkl.Layer):    
    def __init__(self,name=None,trainable=False,original_range=None,target_range=None):
        super(Normalize,self).__init__(name=name)
        self.original_range = original_range
        self.target_range = target_range

    def call(self,x):
        factor = tf.math.reduce_max(tf.math.abs(x))
        return tf.math.scalar_mul(1/factor, x) 
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'original_range': self.original_range,
            'target_range': self.target_range,
        })
        return config

class Normalize_pos(tfkl.Layer):    
    def __init__(self,name=None,trainable=False,original_range=None,target_range=None):
        super(Normalize_pos,self).__init__(name=name)
        self.original_range = original_range
        self.target_range = target_range

    def call(self,x):
        factor = tf.math.reduce_max(tf.math.abs(x))
        salida = tf.math.scalar_mul(1/(2*factor), x) 
        return  salida + 1
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'original_range': self.original_range,
            'target_range': self.target_range,
        })
        return config





class MSE(tfkl.Layer):
    #x[0]: predicted
    #x[1]: original
    def __init__(self,name=None,trainable=False,lnorm=2,offset=1e-9,normalize=False):
        super(MSE,self).__init__(name=name)
        self.offset = offset
        self.normalize = normalize
        self.lnorm = lnorm
    
    def call(self,x):
        mse_error = tf.abs(x[0] - x[1])**self.lnorm
        if self.normalize:
            mse_error = mse_error/(self.offset + tf.abs(x[1])**self.lnorm)
        return mse_error
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'offset': self.offset,
            'normalize': self.normalize,
            'lnorm' : self.lnorm
        })
        return config


