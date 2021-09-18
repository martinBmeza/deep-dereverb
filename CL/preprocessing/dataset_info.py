import matplotlib.pyplot as plt 
import numpy as np 
import glob
import soundfile as sf
import tqdm


class DatasetInfo: 

    def __init__(self, dataset_folder_path, dataset_ID = 'DATASET'):
        self.folder_path = dataset_folder_path
        self.dataset_ID = dataset_ID
        self.audio_list = None
        self.amount_of_files = None
        self.durations = []
        self.samplerate = set()
        self.extensions = set()
        self._scan_audio_files()
        self.calculate_values()
 
    def _scan_audio_files(self):
        
        data_types = ('.wav', '.WAV', '.flac', '.FLAC', '.mp3', '.MP3') 
        search_path = self.folder_path + '/**/*'
        self.audio_list = []

        print('Scanning for audio files...')
        for types in tqdm.tqdm(data_types): 
            self.audio_list.extend(glob.glob(search_path+types, recursive = True))
    

    def calculate_values(self):
        
        self.amount_of_files = len(self.audio_list)

        print('Calculating parameters...')
        for audio in tqdm.tqdm(self.audio_list):
            audio_info = sf.info(audio)
            self.durations.append(audio_info.duration)
            self.samplerate.add(audio_info.samplerate)
            self.extensions.add(audio_info.format_info)

    def print_info(self):

        print('-'*15+self.dataset_ID+' Info'+'-'*15)
        print('Amount of audio files: {}'.format(self.amount_of_files))
        print('Total duration in hours: {:0.2f}'.format(np.sum(self.durations)/(3600)))
        print('Durations mean/std: {:0.2f}/{:0.2f}'.format(np.mean(self.durations), np.std(self.durations)))
        print('Samplerates found: {}'.format(self.samplerate))
        print('File extensions found: {}'.format(self.extensions))

LIBRISPEECH = DatasetInfo('/mnt/datasets/clean_voice/dev-clean', 'LibriSpeech Corpus')
TIMIT = DatasetInfo('/mnt/datasets/clean_voice/TIMIT', 'TIMIT Corpus')

TIMIT.print_info()
LIBRISPEECH.print_info()
