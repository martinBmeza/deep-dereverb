"""
Manejo de paths para construir los sets de entrenamiento, validacion y testeo
"""
import pickle 
import glob 
from processing import audio_number 

def get_audio_list(path, file_types = ('.wav', '.WAV', '.flac', '.FLAC')):
    search_path = path + '/**/*'
    audio_list = []
    for file_type in file_types:
        audio_list.extend(glob.glob(search_path+file_type, recursive=True))
    return audio_list
"""
# TRAIN
save_clean_path = '/home/martin/deep-dereverb/CL/data/train/clean/'
save_reverb_path = '/home/martin/deep-dereverb/CL/data/train/reverb/'

speech_path = '/mnt/Tesis/train/speech/'
speech_list = glob.glob(speech_path+'/**/*.flac', recursive=True)
speech_list = audio_number(speech_list)
print('{} archivos de habla'.format(len(speech_list)))

rir_path = '/home/martin/rir_analysis/notebooks/rir_aug_database/'
rir_list =glob.glob(rir_path+'/**/*.wav', recursive=True)
print('{} Respuestas al impulso'.format(len(rir_list)))

save = {'clean' : save_clean_path,
        'reverb' : save_reverb_path}

lists = {'speech' : speech_list,
         'rir' : rir_list}

train = {'save' : save, 'list' : lists}

if __name__ == '__main__':
    with open('paths.pkl', 'wb') as f:
        pickle.dump(train, f)

"""
# TEST
save_clean_path = '/home/martin/deep-dereverb/CL/data/test/clean/'
save_reverb_path = '/home/martin/deep-dereverb/CL/data/test/reverb/'

speech_path = '/mnt/Tesis/test/speech/'
speech_list = glob.glob(speech_path+'/**/*.flac', recursive=True)
speech_list = audio_number(speech_list)
print('{} archivos de habla'.format(len(speech_list)))

rir_path = '/home/martin/rir_analysis/notebooks/rir_aug_database/' 
rir_list =glob.glob(rir_path+'/**/*.wav', recursive=True)
print('{} Respuestas al impulso reales'.format(len(rir_list)))


save = {'clean' : save_clean_path,
        'reverb' : save_reverb_path}

lists = {'speech' : speech_list,
         'rir' : rir_list}

test = {'save' : save, 'list' : lists}

if __name__ == '__main__':
    with open('test_paths.pkl', 'wb') as f:
        pickle.dump(test, f)
