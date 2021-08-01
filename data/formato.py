import glob
import librosa
import soundfile as sf
import tqdm 


SAVE_PATH = '/home/martin/deep-dereverb/data/speech'
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
    filename = save_path + '/speech_' + str(contador) + target_format
    #import pdb; pdb.set_trace()
    sf.write(filename, audio, fs)


audio_path = '/mnt/datasets/clean_voice/dev-clean'
audio_lista = get_audio_list(audio_path)

contador = 0
for audio in tqdm.tqdm(audio_lista):
    reformat(audio, TARGET_FS, TARGET_FORMAT, SAVE_PATH)    
    contador +=1
