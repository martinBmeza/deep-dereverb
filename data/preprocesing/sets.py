"""
La idea de este archivo es ordenar la generación de los diferentes conjuntos de datos, previendo sus dimensiones y las diferentes fuentes.
"""
import sys, os, random
import numpy as np 
import matplotlib.pyplot as plt
import glob
import librosa as lb
import soundfile as sf
import tqdm
from data_generator import generate_inputs, framing
#----------------------------PATHS---------------------------------------------
path_datos = "/mnt/datasets" #path donde tengo el dataset
sys.path.append(path_datos)
save_path = "/mnt/datasets/npy_data/"

#------------------------------------------------------------------------------
#listo las bases de datos a usar 

#SPEECH
speech_list = glob.glob(os.path.join(path_datos, 'clean_voice/LibriSpeech/dev-clean', '**/*.flac'), recursive=True)
speec_duraciones=[sf.info(audio).frames for audio in speech_list]
tiempo_speech = np.sum(speec_duraciones)/sf.info(speech_list[0]).samplerate
print('En total hay {:.2f} horas de speech'.format(tiempo_speech/60/60))

#RIR
rir_list = glob.glob(os.path.join(path_datos, 'impulsos', '**/*.wav'), recursive=True)
rir_duraciones=[sf.info(rir).frames for rir in rir_list]
print('En total hay {} respuestas al impulso'.format(len(rir_list)))

#genero diccionario de rir para seleccionar aleatoriamente
dict_rir = {i:j for i,j in enumerate(rir_list)}

contador = 0
for speech_path in tqdm.tqdm(speech_list):
    rir_path = dict_rir[random.randint(0, len(rir_list)-1)] #rir aleatoria
    magspec_reverb, mask_comp = generate_inputs(speech_path, rir_path)
    magspec_reverb = framing(magspec_reverb)
    mask_comp = framing(mask_comp)
    for frame in range(magspec_reverb.shape[0]):
        np.save(save_path+str(contador)+'.npy',[magspec_reverb[frame,:,:], mask_comp[frame,:,:]])
        contador+=1
        #print('impulso: {} \n'.format(rir_path))
        #print('cantidad de frames: {} \n'.format(frame+1))

