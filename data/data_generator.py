import sys, os, random
sys.path.append('/home/martin/Documents/tesis/src')
import numpy as np
import tqdm
from utils import *

def building_loop(speech_list, rir_list, save_path): 
    contador = 0
    for speech_path in tqdm.tqdm(speech_list):

        rir_path = np.random.choice(rir_list)

        #Genero los espectros
        clean_stft, reverb_stft = get_specs_from_path(rir_path, speech_path, FACTOR=70)

        #Framing
        clean_frames = img_framing_pad(clean_stft, winsize=256, step=256, dim=1)
        reverb_frames = img_framing_pad(reverb_stft, winsize=256, step=256, dim=1)

        for frame in range(clean_frames.shape[0]):
            np.save(save_path+str(contador)+'.npy',[reverb_frames[frame,:,:], clean_frames[frame,:,:]])
            contador+=1

#----------------------------PATHS---------------------------------------------
#Prueba General 

save_path = '/mnt/datasets/npy_data/con_aumentados/'
speech_path = '/mnt/datasets/clean_voice/dev-clean'
speech_list = get_audio_list(speech_path)

#rir_path = '/mnt/datasets/impulsos/reales/C4DM'
rir_path = '/home/martin/Documents/tesis/src/aumentacion/aumentados'
rir_list = get_audio_list(rir_path)

building_loop(speech_list, rir_list, save_path)
#------------------------------------------------------------------------------          



"""
#----------------------------PATHS---------------------------------------------
#Experimento 1 
EXP_FILE = '/home/martin/Documents/tesis/src/experiments/datasets/exp1.pkl'
with open(EXP_FILE, 'rb') as f:
    exp_1 = pickle.load(f)

save_path = prepare_save_path(exp_1['out_path'])+'/'
speech_list = exp_1['clean_train']
#rir_list = exp_1['real_train'] + exp_1['sim_train']
rir_list = exp_1['sim_train']
building_loop(speech_list, rir_list[0:2])
#------------------------------------------------------------------------------          
"""
