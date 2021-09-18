import pickle 
import numpy as np
from processing import reverberacion, reverb_multiprocessing, clean_multiprocessing
"""
# TRAIN 
with open('paths.pkl', 'rb') as f:
    paths = pickle.load(f)

# Generation of  reverbs
audio_list = paths['list']['speech']
rir_list = paths['list']['rir']
save = paths['save']['reverb']
reverb_multiprocessing(audio_list, rir_list, save)

# Clean speech 
save = paths['save']['clean']
clean_multiprocessing(audio_list, save)

"""
# TEST

with open('test_paths.pkl', 'rb') as f:
    paths = pickle.load(f)

# Generacion reales 
audio_list = paths['list']['speech']
rir_list = paths['list']['rir']
save = paths['save']['reverb']
reverb_multiprocessing(audio_list, rir_list, save)

# Clean speech 
save = paths['save']['clean']
clean_multiprocessing(audio_list, save)
