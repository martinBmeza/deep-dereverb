import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np
import sys, os 
import glob
import librosa
from scipy.signal import fftconvolve, stft
from IPython.display import Audio
import tqdm

class TestDatasetGenerator():
    '''Generate the test audiofiles to use in the model evaluation process'''

    def __init__(self, name, audio_path, rir_path):
        self.name = name
        self.audio_list = None  
        self.rir_list = None 
        self._get_file_list(audio_path, rir_path)

    def _get_file_list(self, audio_path, rir_path):
        '''Generate path files list from audio path '''
        data_types = ('.wav', '.WAV', '.flac', '.FLAC', '.mp3', '.MP3') 
        search_audio_path = audio_path + '/**/*' 
        search_rir_path = rir_path + '/**/*'
        
        audio_list = []
        rir_list = []
        
        print('Scanning for audio files...')
        for types in tqdm.tqdm(data_types): 
            audio_list.extend(glob.glob(search_audio_path+types, recursive = True))
            rir_list.extend(glob.glob(search_rir_path+types, recursive = True))
        self.audio_list = audio_list 
        self.rir_list = rir_list 





class Evaluate_model():
    ''' Load a model and weights to make predictions from data entries'''

    def __init__(self, input_size, samplerate):
        self.input_size = input_size
        self.samplerate = samplerate
        self.weights_path = weights_path
        self.audio_list = None 




    def audio_framing(self, audio):                                                                                                                            
                                                                                                                                                                          
        n_frames = int(len(audio)/self.input_size)                                                                                                                                
        resto = len(audio)%self.input_size                                                                                                                                        
                                                                                                                                                                          
        if resto == 0:                                                                                                                                                    
            out = np.empty((n_frames, self.input_size))                                                                                                                           
        else:                                                                                                                                                             
            out = np.empty((n_frames+1, self.input_size))                                                                                                                         
            out[-1,:resto] = audio[n_frames*self.input_size:]                                                                                                                     
            out[-1,resto:] = np.zeros(self.input_size-resto)                                                                                                                      
                                                                                                                                                                          
        for frame in range(n_frames):                                                                                                                                     
            out[frame,:] = audio[frame*self.input_size:(frame+1)*self.input_size]                                                                                                         
        audio_pad = np.concatenate((audio, np.zeros(self.input_size-resto)))
        
        return out, audio_pad

    
















































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


