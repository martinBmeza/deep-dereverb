import librosa
import soundfile as sf
import numpy as np
import glob
import pandas as pd

def get_audio_list(path, file_types = ('.wav', '.WAV', '.flac', '.FLAC')):
    search_path = path + '/**/*'
    audio_list = []
    for file_type in file_types:
        audio_list.extend(glob.glob(search_path+file_type, recursive=True))
    return audio_list

def audio_chunk(paths, win_size=32640, hop_size=32640):
    """Recibe un path de audio y devuelve el path, los puntos de comienzo
	y final para recortar el audio, y la frecuencia de sampleo."""
    clean_path = paths[0]
    reverb_path = paths[1]
  
    nframes = sf.info(clean_path, verbose = True).frames
    
    start = np.array(range(0,nframes-win_size,hop_size), dtype='int32')
    end = start + win_size
    return  clean_path, reverb_path, start, end

def speech_dataframe(clean_path, reverb_path):
    clean_list = get_audio_list(clean_path)
    clean_list.sort()

    reverb_list = get_audio_list(reverb_path)
    reverb_list.sort()
    audio_list = [[clean_list[i], reverb_list[i]] for i in range(len(clean_list))]


    dicts = [{'clean_path':audio[0],'reverb_path':audio[1],\
            'start': audio[2],'end':audio[3]}\
             for audio in map(audio_chunk, audio_list)]

    df_speech = pd.concat(map(pd.DataFrame, dicts), axis=0, ignore_index = True)
    
    return df_speech

def speech_fixdataframe(clean_path, reverb_path):
    clean_list = get_audio_list(clean_path)
    clean_list.sort()

    reverb_list = get_audio_list(reverb_path)
    reverb_list.sort()
    audio_list = [[clean_list[i], reverb_list[i]] for i in range(len(clean_list))]


    dicts = [{'clean_path':audio[0],'reverb_path':audio[1]} for audio in audio_list]

    df_speech = pd.DataFrame(dicts)
    return df_speech



if __name__ == "__main__":
    
    CLEAN_PATH = '/home/martin/deep-dereverb/data/clean/'
    REVERB_PATH = '/home/martin/deep-dereverb/data/real/'
    SAVE_PATH = '/home/martin/deep-dereverb/data/dataset_reales.pkl'
    dataframe = speech_fixdataframe(CLEAN_PATH, REVERB_PATH)
    dataframe.to_pickle(SAVE_PATH)

    CLEAN_PATH = '/mnt/datasets/train/clean/'
    REVERB_PATH = '/mnt/datasets/train/aug/'
    SAVE_PATH = '/home/martin/deep-dereverb/data/dataset_aug.pkl'
    dataframe = speech_fixdataframe(CLEAN_PATH, REVERB_PATH)
    dataframe.to_pickle(SAVE_PATH)

    CLEAN_PATH = '/mnt/datasets/train/clean/'
    REVERB_PATH = '/mnt/datasets/train/gen/'
    SAVE_PATH = '/home/martin/deep-dereverb/data/dataset_gen.pkl'
    dataframe = speech_fixdataframe(CLEAN_PATH, REVERB_PATH)
    dataframe.to_pickle(SAVE_PATH)

