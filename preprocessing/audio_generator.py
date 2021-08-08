import sys, os, random
sys.path.append('/home/martin/deep-dereverb/')
import numpy as np
import glob
import tqdm
import librosa
import soundfile as sf
from scipy.signal import fftconvolve

def reverberacion(speech_path, rir_path, clean_save, reverb_save, c, target=False):
    # Cargo los audios
    speech, speech_fs = librosa.load(speech_path, sr=16000)
    rir, rir_fs = librosa.load(rir_path, sr=16000)
    
    speech = speech / np.max(abs(speech))
    reverb = generar_reverb(speech, rir)
    clean = speech
    ventana = int(32640)
    for i in range(len(clean)//ventana):   
        if target:
            clean_cut = clean[int(ventana*i):int(ventana*(i+1))]
            clean_fn = clean_save+'{:06d}'.format(c)+ '.wav'
            sf.write(clean_fn, clean_cut, speech_fs)
 
        reverb_cut = reverb[int(ventana*i):int(ventana*(i+1))]
        reverb_fn = reverb_save+'{:06d}'.format(c)+ '.wav'
        sf.write(reverb_fn, reverb_cut, speech_fs) 
        c +=1
    return c


def get_audio_list(path, file_types = ('.wav', '.WAV', '.flac', '.FLAC')):
    search_path = path + '/**/*'
    audio_list = []
    for file_type in file_types:
        audio_list.extend(glob.glob(search_path+file_type, recursive=True))
    return audio_list


def temporal_decompose(rir, Q_e):
    inicio = np.argmax(abs(rir)) # direct path
    early = rir[inicio:inicio+Q_e]
    late = rir[inicio+Q_e:]
    rir = rir[inicio:]
    return rir


def generar_reverb(speech, rir):
    # Normalizo el impulso y el speech
    rir = rir / np.max(abs(rir))

    # Divido parte early - elimino el delay
    rir_completa = temporal_decompose(rir, Q_e = 32)

    # Convoluciono. Obtengo audio con reverb
    reverb = fftconvolve(speech, rir_completa)[:len(speech)]
    reverb = reverb / np.max(abs(reverb))
    return reverb

save_clean_path = '/mnt/datasets/train/clean/'
save_real_path = '/mnt/datasets/train/real/'
save_gen_path = '/mnt/datasets/train/gen/'
save_aug_path = '/mnt/datasets/train/aug/'

speech_path = '/mnt/datasets/clean_voice/train-clean-100'
speech_list = glob.glob(speech_path+'/**/*.flac', recursive=True)

rir_real_path = '/mnt/datasets/impulsos/reales/C4DM'
rir_real_list =glob.glob(rir_real_path+'/**/*.wav', recursive=True)

rir_gen_path = '/home/martin/deep-dereverb/preprocessing/rir_generacion/generados'
rir_gen_list =glob.glob(rir_gen_path+'/**/*.wav', recursive=True)

rir_aug_path = '/home/martin/deep-dereverb/preprocessing/rir_aumentacion/aumentados'
rir_aug_list = glob.glob(rir_aug_path+'/**/*.wav', recursive=True)

n = 0
for speech_path in tqdm.tqdm(speech_list):
    
    rir_real_rnd = np.random.choice(rir_real_list)
    rir_gen_rnd = np.random.choice(rir_gen_list)
    rir_aug_rnd = np.random.choice(rir_aug_list)
    
    _ = reverberacion(speech_path, rir_real_rnd, save_clean_path, save_real_path, n, target=True)
    _ = reverberacion(speech_path, rir_gen_rnd, save_clean_path, save_gen_path, n, target=False)
    n = reverberacion(speech_path, rir_aug_rnd, save_clean_path, save_aug_path, n, target=False)
   


