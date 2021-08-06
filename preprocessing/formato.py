import glob
import librosa
import soundfile as sf
import tqdm 


AUDIO_PATH = '/mnt/datasets/clean_voice/train-clean-100'
SAVE_PATH = '/mnt/datasets/clean_voice/formateado'
TARGET_FS = 16000
TARGET_FORMAT = '.wav'

def get_audio_list(path, file_types = ('.wav', '.WAV', '.flac', '.FLAC')):
    search_path = path + '/**/*'
    audio_list = []
    for file_type in file_types:
        audio_list.extend(glob.glob(search_path+file_type, recursive=True))
    return audio_list


def reformat(path, target_fs, target_format, save_path):
    audio, fs = librosa.load(path, sr=target_fs)
    filename = save_path+'/speech_' +'{:06d}'.format(contador)+target_format
    #import pdb; pdb.set_trace()
    sf.write(filename, audio, fs)


audio_lista = get_audio_list(AUDIO_PATH)

contador = 0
for audio in tqdm.tqdm(audio_lista):
    reformat(audio, TARGET_FS, TARGET_FORMAT, SAVE_PATH)    
    contador +=1
