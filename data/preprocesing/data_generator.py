import sys, os, random
MAIN_PATH="/home/martin/Documents/tesis/src"
sys.path.append(MAIN_PATH)
import numpy as np
import glob
import shutil
import soundfile as sf
import librosa
import pickle
import tqdm
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, stft

eps = np.finfo(float).eps #precision de punto flotante

def prepare_save_path(path):

    if os.path.isdir(path):
        shutil.rmtree(path)
        os.mkdir(path)
    else:
        os.mkdir(path)
    
    return(path)


def framing(data, winsize=256, step=256, dim=1):
    n_frames = int(data.shape[dim] / winsize)
    out = np.empty((n_frames, data.shape[0], winsize)) #+1 por el pad 
    for frame in range(n_frames):
        out[frame,:,:] = data[:,frame*winsize : (frame+1)*winsize]
    #agrego el padeado 
    #resto = data.shape[dim]%winsize
    #to_pad = winsize - resto
    #out[-1,:,:resto] = data[:, n_frames*winsize: n_frames*winsize + resto]
    #out[-1,:,resto:] = np.zeros((data.shape[0],to_pad))
    #shape --> (frames, freq, time)

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
    
    n_frames = int(len(audio)/winsize)
    #resto = len(audio)%winsize
    
    out = np.empty((n_frames, winsize))
    #if resto == 0:
    #    out = np.empty((n_frames, winsize))
    #else:
    #    out = np.empty((n_frames+1, winsize))
    #    out[-1,:resto] = audio[n_frames*winsize:]
    #    out[-1,resto:] = np.zeros(winsize-resto)
    
    for frame in range(n_frames):
        out[frame,:] = audio[frame*winsize:(frame+1)*winsize]
    
    return out, audio 


def normalise(array, range_min, range_max, array_min, array_max):
    norm_array = (array - array_min) / (array_max - array_min)
    norm_array = norm_array * (range_max - range_min) + range_min
    return norm_array

def denormalise(norm_array, original_min, original_max, range_min, range_max):
    array = (norm_array - range_min) / (range_max - range_min)
    array = array * (original_max - original_min) + original_min
    return array


def generate_inputs(speech_path, rir_path):

    #Cargo los datos
    speech, speech_fs = librosa.load(speech_path, sr=16000)
    rir, rir_fs = librosa.load(rir_path, sr=16000)
    if speech_fs!=rir_fs:
        raise Exception("Hay audios con distintas frecuencias de sampleo")

    # Elimino el delay del impulso
    delay_shift = np.argmax(rir)
    rir = rir[delay_shift:]

    #Normalizo el impulso 
    rir = rir / np.max(abs(rir))

    #Convoluciono. Obtengo audio con reverb
    reverb = fftconvolve(speech, rir)

    #Padeo el audio anecoico. Obtengo el audio clean
    clean = np.pad(speech, (0,len(rir)-1), 'constant', constant_values=(eps,eps)) 

    #genero las STFTs
    #stft_clean = librosa.stft(clean, n_fft=512, hop_length=128)#
    #spectrogram_clean = np.abs(stft_clean)
    #log_spectrogram_clean = librosa.amplitude_to_db(spectrogram_clean)

    #stft_reverb = librosa.stft(reverb, n_fft=512, hop_length=128)
    #spectrogram_reverb = np.abs(stft_reverb)
    #log_spectrogram_reverb = librosa.amplitude_to_db(spectrogram_reverb)

    #log_norm_reverb = normalise(log_spectrogram_reverb, 0, 1, -47, 39)
    #log_norm_clean = normalise(log_spectrogram_clean, 0, 1, -47, 39)
    
    #return [log_norm_reverb, log_norm_clean]

    return [reverb, clean]


#----------------------------PATHS---------------------------------------------
#Experimento 1 
EXP_FILE = '/home/martin/Documents/tesis/src/data/exp1.pkl'
with open(EXP_FILE, 'rb') as f:
    exp_1 = pickle.load(f)

save_path = prepare_save_path(exp_1['out_path'])+'/'
speech_list = exp_1['clean_train']
rir_list = exp_1['real_train'] + exp_1['sim_train']
#------------------------------------------------------------------------------

dict_rir = {i:j for i,j in enumerate(rir_list)} #genero diccionario de rir para seleccionar aleatoriamente

def building_loop(speech_list, rir_list):

    contador = 0
    for speech_path in tqdm.tqdm(speech_list):
        rir_path = dict_rir[random.randint(0, len(rir_list)-1)] #rir aleatoria
        audio_reverb, audio_clean = generate_inputs(speech_path, rir_path)
        audio_reverb, _ = audio_framing(audio_reverb)
        audio_clean, _ = audio_framing(audio_clean)
        
        #import pdb; pdb.set_trace()
        for frame in range(audio_reverb.shape[0]):
            np.save(save_path+str(contador)+'.npy',[audio_reverb[frame,:], audio_clean[frame,:]])
            contador+=1
           
building_loop(speech_list, rir_list)
