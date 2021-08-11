import pickle 
import numpy as np
from processing import reverberacion, reverb_multiprocessing, clean_multiprocessing

with open('paths.pkl', 'rb') as f:
    paths = pickle.load(f)

# Generation of real reverbs
audio_list = paths['list']['speech']
rir_list = paths['list']['real']
save = paths['save']['real']
reverb_multiprocessing(audio_list, rir_list, save)

# Synthetic reverbs
audio_list = paths['list']['speech']
rir_list = paths['list']['gen']
save = paths['save']['gen']
reverb_multiprocessing(audio_list, rir_list, save)

# Augmented reverbs
audio_list = paths['list']['speech']
rir_list = paths['list']['aug']
save = paths['save']['aug']
reverb_multiprocessing(audio_list, rir_list, save)

# Clean speech 
audio_list = paths['list']['speech']
save = paths['save']['clean']
clean_multiprocessing(audio_list, save)
