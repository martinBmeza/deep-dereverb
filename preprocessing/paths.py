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

# TRAIN
save_clean_path = '/home/martin/deep-dereverb/data/train/clean/'
save_real_path = '/home/martin/deep-dereverb/data/train/real/'
save_gen_path = '/home/martin/deep-dereverb/data/train/gen/'
save_aug_path = '/home/martin/deep-dereverb/data/train/aug/'

speech_path = '/mnt/Tesis/train/speech/'
speech_list = glob.glob(speech_path+'/**/*.flac', recursive=True)
speech_list = audio_number(speech_list)
print('{} archivos de habla'.format(len(speech_list)))

rir_real_path = '/mnt/Tesis/train/rir_reales/'
rir_real_list =glob.glob(rir_real_path+'/**/*.wav', recursive=True)
print('{} Respuestas al impulso reales'.format(len(rir_real_list)))

rir_gen_path = '/mnt/Tesis/train/rir_sinteticas'
rir_gen_list =glob.glob(rir_gen_path+'/**/*.wav', recursive=True)
print('{} Respuestas al impulso sinteticas'.format(len(rir_gen_list)))

rir_aug_path = '/mnt/Tesis/train/rir_reales_aumentadas'
rir_aug_list = glob.glob(rir_aug_path+'/**/*.wav', recursive=True)
print('{} Respuestas al impulso reales aumentadas'.format(len(rir_aug_list)))

save = {'clean' : save_clean_path,
        'real' : save_real_path,
        'gen' : save_gen_path,
        'aug' : save_aug_path}

lists = {'speech' : speech_list,
         'real' : rir_real_list,
         'gen' : rir_gen_list,
         'aug' : rir_aug_list }

train = {'save' : save, 'list' : lists}

if __name__ == '__main__':
    with open('paths.pkl', 'wb') as f:
        pickle.dump(train, f)


# TEST
save_clean_path = '/home/martin/deep-dereverb/data/test/clean/'
save_real_path = '/home/martin/deep-dereverb/data/test/real/'
save_gen_path = '/home/martin/deep-dereverb/data/test/gen/'
save_aug_path = '/home/martin/deep-dereverb/data/test/aug/'

speech_path = '/mnt/Tesis/test/speech/'
speech_list = glob.glob(speech_path+'/**/*.flac', recursive=True)
speech_list = audio_number(speech_list)
print('{} archivos de habla'.format(len(speech_list)))

rir_real_path = '/mnt/Tesis/test/rir_reales/' 
rir_real_list =glob.glob(rir_real_path+'/**/*.wav', recursive=True)
print('{} Respuestas al impulso reales'.format(len(rir_real_list)))

rir_gen_path = '/mnt/Tesis/test/rir_gen/' 
rir_gen_list =glob.glob(rir_gen_path+'/**/*.wav', recursive=True)
print('{} Respuestas al impulso sinteticas'.format(len(rir_gen_list)))

rir_aug_path = '/mnt/Tesis/test/rir_aug'
rir_aug_list =glob.glob(rir_aug_path+'/**/*.wav', recursive=True)
print('{} Respuestas al impulso aumentadas'.format(len(rir_aug_list)))



save = {'clean' : save_clean_path,
        'real' : save_real_path,
        'aug' : save_aug_path,
        'gen' : save_gen_path}

lists = {'speech' : speech_list,
         'real' : rir_real_list,
         'aug' : rir_aug_list,
         'gen' : rir_gen_list}

test = {'save' : save, 'list' : lists}

if __name__ == '__main__':
    with open('test_paths.pkl', 'wb') as f:
        pickle.dump(test, f)

