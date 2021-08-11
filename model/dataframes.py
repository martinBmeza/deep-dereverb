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
    
    CLEAN_PATH = '/home/martin/deep-dereverb/data/train/clean/'
    REVERB_PATH = '/home/martin/deep-dereverb/data/train/real/'
    SAVE_PATH = '/home/martin/deep-dereverb/data/train/dataset_reales.pkl'
    dataframe = speech_fixdataframe(CLEAN_PATH, REVERB_PATH)
    print(len(dataframe))
    dataframe.to_pickle(SAVE_PATH)

    REVERB_PATH = '/home/martin/deep-dereverb/data/train/aug/'
    SAVE_PATH = '/home/martin/deep-dereverb/data/train/dataset_aug.pkl'
    dataframe = speech_fixdataframe(CLEAN_PATH, REVERB_PATH)
    print(len(dataframe))
    dataframe.to_pickle(SAVE_PATH)

    REVERB_PATH = '/home/martin/deep-dereverb/data/train/gen/'
    SAVE_PATH = '/home/martin/deep-dereverb/data/train/dataset_gen.pkl'
    dataframe = speech_fixdataframe(CLEAN_PATH, REVERB_PATH)
    print(len(dataframe))
    dataframe.to_pickle(SAVE_PATH)

