import sys, os, random
sys.path.append('/home/martin/Documents/tesis/src')
import numpy as np
import glob
import shutil
import soundfile as sf
import librosa
import pickle
import tqdm
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, stft
from utils import temporal_decompose, normalizer, prepare_save_path

eps = np.finfo(float).eps #precision de punto flotante

def img_framing(data, winsize=256, step=256, dim=1):
    
    n_frames = int(data.shape[dim] / winsize)
    out = np.empty((n_frames, data.shape[0], winsize)) #+1 por el pad 
    for frame in range(n_frames):
        out[frame,:,:] = data[:,frame*winsize : (frame+1)*winsize]
   
    return out

'''
def audio_framing(audio, winsize = 32640):                                                                                                                            
                                                                                                                                                                      
    n_frames = int(len(audio)/winsize)                                                                                                                                
    resto = len(audio)%winsize                                                                                                                                        
                                                                                                                                                                      
    if resto == 0:                                                                                                                                                    
        out = np.empty((n_frames, winsize))                                                                                                                           
    else:                                                                                                                                                             
        out = np.empty((n_frames+1, winsize))                                                                                                                         
        out[-1,:resto] = audio[n_frames*winsize:]                                                                                                                     
        out[-1,resto:] = np.zeros(winsize-resto)                                                                                                                      
                                                                                                                                                                      
    for frame in range(n_frames):                                                                                                                                     
        out[frame,:] = audio[frame*winsize:(frame+1)*winsize]                                                                                                         
    audio_pad = np.concatenate((audio, np.zeros(winsize-resto)))
    
    return out, audio_pad
'''

def audio_framing(audio, winsize = 32640):
    ''' Devuelvo solo frames enteros, sin padear'''

    n_frames = int(len(audio)/winsize) 
    out = np.empty((n_frames, winsize))
    
    for frame in range(n_frames):
        out[frame,:] = audio[frame*winsize:(frame+1)*winsize]
    
    return out, audio 


def generate_inputs(speech_path, rir_path):

    #Cargo los datos
    speech, speech_fs = librosa.load(speech_path, sr=16000)
    rir, rir_fs = librosa.load(rir_path, sr=16000)
    
    #Normalizo el impulso 
    rir = rir / np.max(abs(rir))

    #Divido parte early
    rir_early, rir_complete = temporal_decompose(rir, rir_fs)
    
    #Convoluciono. Obtengo audio con reverb
    reverb = fftconvolve(speech, rir_complete)

    #Convoluciono y padeo el audio anecoico. Obtengo el audio clean
    clean = fftconvolve(speech, rir_early)
    clean = np.pad(clean, (0,len(rir_complete)-len(rir_early)), 'constant', constant_values=(eps,eps)) 

    return [reverb, clean]

def building_loop(speech_list, rir_list):
    dict_rir = {i:j for i,j in enumerate(rir_list)} #genero diccionario de rir para seleccionar aleatoriamente 
    contador = 0
    for speech_path in tqdm.tqdm(speech_list):
        rir_path = dict_rir[random.randint(0, len(rir_list)-1)] #rir aleatoria
        audio_reverb, audio_clean = generate_inputs(speech_path, rir_path)
        audio_reverb, _ = audio_framing(audio_reverb)
        audio_clean, _ = audio_framing(audio_clean)
        
        for frame in range(audio_reverb.shape[0]):
            np.save(save_path+str(contador)+'.npy',[audio_reverb[frame,:], audio_clean[frame,:]])
            contador+=1
 
#----------------------------PATHS---------------------------------------------
#Experimento 1 
EXP_FILE = '/home/martin/Documents/tesis/src/experiments/datasets/exp1.pkl'
with open(EXP_FILE, 'rb') as f:
    exp_1 = pickle.load(f)

save_path = prepare_save_path(exp_1['out_path'])+'/'
speech_list = exp_1['clean_train']
rir_list = exp_1['real_train'] + exp_1['sim_train']
building_loop(speech_list, rir_list)
#------------------------------------------------------------------------------          
