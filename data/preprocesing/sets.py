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
from data_generator import generate_inputs, framing, audio_framing
#----------------------------PATHS---------------------------------------------
path_datos = "/mnt/datasets" #path donde tengo el dataset
sys.path.append(path_datos)
save_path = "/mnt/datasets/npy_data/"

#------------------------------------------------------------------------------
#listo las bases de datos a usar 

#SPEECH
speech_list = glob.glob(os.path.join(path_datos, 'clean_voice/TIMIT', '**/*.WAV'), recursive=True)


#RIR
rir_list = glob.glob(os.path.join(path_datos, 'impulsos/reales/processed', '**/*.wav'), recursive=True)
dict_rir = {i:j for i,j in enumerate(rir_list)} #genero diccionario de rir para seleccionar aleatoriamente


contador = 0
for speech_path in tqdm.tqdm(speech_list):
    rir_path = dict_rir[random.randint(0, len(rir_list)-1)] #rir aleatoria
    audio_reverb, audio_clean = generate_inputs(speech_path, rir_path)
    audio_reverb = audio_framing(audio_reverb)
    audio_clean = audio_framing(audio_clean)
    for frame in range(audio_reverb.shape[0]):
        np.save(save_path+str(contador)+'.npy',[audio_reverb[frame,:], audio_clean[frame,:]])
        contador+=1
    #print('impulso: {} \n'.format(rir_path))
    #print('cantidad de frames: {} \n'.format(frame+1))
 
