""" Funciones que sirven para diferentes propositos
como loggin, guardado de datos, carga de datos, etc"""
import numpy as np
import os, shutil

def prepare_save_path(path):

    if os.path.isdir(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)
    return(path)


def temporal_decompose(rir, fs, win = 0.0010):
    t_d = np.argmax(rir) # direct path
    t_o = int((win) * fs) #tolerance window in samples
    
    if t_d - t_o < 0:
        start = 0
    else:
        start = t_d - t_o
        
    if t_d + t_o > len(rir) - 1:
        end = len(rir) - 1
    else:
        end = t_d + t_o + 1
    
    early = rir[start:end]
    complete = rir[start:]
    return early, complete

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
  #print('orden de las entradas: /\n',input_names)                                                                                                                     
  return activations

