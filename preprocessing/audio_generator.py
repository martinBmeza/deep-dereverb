import sys, os, random
sys.path.append('/home/martin/deep-dereverb/')
import numpy as np
import glob
import tqdm
import librosa
import soundfile as sf
from scipy.signal import fftconvolve

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
    # Convoluciono y padeo el audio anecoico. Obtengo el audio clean
    return reverb


# FALTA HACER: Pasar esto a una funcion, donde los parametros
#sean los paths de entrada, salida y alguna otra referencia mas
#si es necesaria para despues

#Lo mejor seria hacer en un sol bucle la generacion de todos los datos,
#asi me aseguro de darle el mismo nombre al audio clean y a TODOS sus
#correspondientes versiones reverberadas. Tampoco son tantas, 
#Raeal, sintetica, aumentada. De esta manera genero la menor cantidad de datos
#y como tengo esta numeracion homogenea entre sets, puedo hacer la seleccion de 
#conjuntos de datos sin preocuparme por overlapping entre audios ni nada de eso

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

    # Cargo los audios
    speech, speech_fs = librosa.load(speech_path, sr=16000)
    rir_real, rir_fs = librosa.load(rir_real_rnd, sr=16000)
    rir_gen, rir_fs = librosa.load(rir_gen_rnd, sr=16000)
    rir_aug, rir_fs = librosa.load(rir_aug_rnd, sr=16000)
    
    speech = speech / np.max(abs(speech))
        
    #import pdb; pdb.set_trace()
    reverb_real = generar_reverb(speech, rir_real)
    reverb_gen = generar_reverb(speech, rir_gen)
    reverb_aug = generar_reverb(speech, rir_aug)
    
    #CAMBIO:ESTIMO SPEECH Y NO PARTE EARLY
    #clean = fftconvolve(speech, rir_early)

    clean = speech
    
    # Guardo los pares en carpetas
    clean_fn = save_clean_path+'{:06d}'.format(n)+ '.wav'
    reverb_real_fn = save_real_path+'{:06d}'.format(n)+ '.wav'
    reverb_gen_fn = save_gen_path+'{:06d}'.format(n)+ '.wav'
    reverb_aug_fn =save_aug_path+'{:06d}'.format(n)+ '.wav'

    sf.write(clean_fn, clean, speech_fs)
    sf.write(reverb_real_fn, reverb_real, speech_fs )
    sf.write(reverb_gen_fn, reverb_gen, speech_fs ) 
    sf.write(reverb_aug_fn, reverb_aug, speech_fs )
    n +=1
##########################################################################

