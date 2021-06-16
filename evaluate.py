import tensorflow as tf 
import matplotlib.pyplot as plt 
import numpy as np
import sys, os, random 
import glob
import librosa
import soundfile as sf
from scipy.signal import fftconvolve, stft
import tqdm
from model.network_architecture import autoencoder
from utils import temporal_decompose, normalizer
import mir_eval
import pystoi
sys.path.append('/home/martin/Documents/tesis/SRMRpy/srmrpy')
import srmr
import csv
eps = np.finfo(float).eps


#METRICAS


#def reverb_ref(clean, reverb, dereverb):

class Evaluate_model():
    ''' Load a model and weights to make predictions from data entries'''

    def __init__(self, audio_list, rir_list, weights_path, input_size=32640, samplerate=16000, spec_min=-55, spec_max=39, model=autoencoder()):
        self.audio_list = audio_list
        self.rir_list = rir_list
        self.weights_path = weights_path
        self.input_size = input_size
        self.samplerate = samplerate
        self.spec_min = spec_min
        self.spec_max = spec_max
        self.metrics = {}
        self.model = model
        self._model_loader()
        self.SDR = []
        self.SRMR = []
        self.ESTOI = []

    def _model_loader(self):
        self.model.load_weights(self.weights_path)

    def generate_audios(self, speech_path, rir_path):
        speech, _ = librosa.load(speech_path, sr=self.samplerate)
        rir, _ = librosa.load(rir_path, sr=self.samplerate)
        
        rir = rir/np.max(abs(rir))
        rir_early, rir_complete = temporal_decompose(rir, self.samplerate)

        reverb = fftconvolve(speech, rir_complete)
        clean = fftconvolve(speech, rir_early)
        clean = np.pad(clean,(0,len(rir_complete)-len(rir_early)), 'constant', constant_values=(eps,eps))
        return reverb, clean

    def audio_framing(self, audio):
        n_frames = int(len(audio)/self.input_size)                                                                                                                                
        resto = len(audio)%self.input_size                                                                                                                                        
                                                                                                                                                                          
        if resto == 0:                                                                                                                                                    
            frames = np.empty((n_frames, self.input_size))
        else:                                                                                                                                                             
            frames = np.empty((n_frames+1, self.input_size))                                                                                                
            frames[-1,:resto] = audio[n_frames*self.input_size:]                                                                                                                 
            frames[-1,resto:] = np.zeros(self.input_size-resto)                                                                                                                   
                                                                                                                                                                          
        for frame in range(n_frames):                                                                                                                                     
            frames[frame,:] = audio[frame*self.input_size:(frame+1)*self.input_size]
        return frames, self.input_size-resto
    
    def preProcessing(self, audio):
        '''Process applied inside data loader before net input'''
        stft = librosa.stft(audio, n_fft=512, hop_length=128)                                                                                         
        spectrogram = np.abs(stft)                                                                                                              
        log_spectrogram = librosa.amplitude_to_db(spectrogram)                                                                                  
        log_norm = normalizer(log_spectrogram, 0, 1, self.spec_min, self.spec_max, mode = 'normalise')
        return log_norm.reshape(1, 257, 256)

    def predict_audio(self, speech_path, rir_path):
        reverb, clean = self.generate_audios(speech_path, rir_path)
        reverb_frames, n_pad = self.audio_framing(reverb) #Mismas dimensiones con pads
        
        predict = []
        for frame_index in range(reverb_frames.shape[0]):
            frame_ready = self.preProcessing(reverb_frames[frame_index])
            frame_dereverb = self.model.predict(frame_ready)
            frame_dereverb = normalizer(frame_dereverb.reshape(257, 256), 0, 1, self.spec_min, self.spec_max, mode = 'denormalise')
            frame_dereverb = librosa.db_to_amplitude(frame_dereverb)
            frame_predict = librosa.griffinlim(frame_dereverb, n_iter=1000, hop_length=128, win_length=512)
            predict = np.concatenate((predict, frame_predict))
        predict = predict[:-n_pad]
        return reverb, clean, predict

    def srmrpy(self, clean, reverb, dereverb):
        srmr_dereverb = srmr.srmr(dereverb, self.samplerate)[0]
        srmr_reverb = srmr.srmr(reverb, self.samplerate)[0]
        return [float(srmr_reverb),float( srmr_dereverb)]

    def sdr(self, clean, reverb, dereverb):
        sdr_dereverb, _, _, _ = mir_eval.separation.bss_eval_sources(clean, dereverb, compute_permutation=True)
        sdr_reverb, _, _, _ = mir_eval.separation.bss_eval_sources(clean, reverb, compute_permutation=True)
        return [float(sdr_reverb), float(sdr_dereverb)]

    def estoi(self, clean, reverb, dereverb):
        estoi_reverb = pystoi.stoi(clean, reverb, self.samplerate, extended = True)
        estoi_dereverb =  pystoi.stoi(clean, dereverb, self.samplerate, extended = True)
        return [float(estoi_reverb), float(estoi_dereverb)]


    def calculate_metrics(self):
        #Puedo hacer un diccionario que llame a las metricas que quiero usar
        for speech in tqdm.tqdm(self.audio_list):
            for rir in self.rir_list:
                reverb, clean, dereverb = self.predict_audio(speech, rir)
                self.SRMR.append(self.srmrpy(clean, reverb, dereverb))
                self.SDR.append(self.sdr(clean, reverb, dereverb))
                self.ESTOI.append(self.estoi(clean, reverb, dereverb))
        return #reverb, clean, dereverb
    
    def save_results(self, filename):
        srmr_reverb = [i[0] for i in self.SRMR]
        srmr_dereverb = [i[1] for i in self.SRMR]
        sdr_reverb = [i[0] for i in self.SDR]
        sdr_dereverb = [i[1] for i in self.SDR]
        estoi_reverb = [i[0] for i in self.ESTOI]
        estoi_dereverb = [i[1] for i in self.ESTOI]

        with open(filename, 'w') as f:
            fieldnames = ['SRMR-reverb', 'SRMR-dereverb','SDR-reverb', 'SDR-dereverb','ESTOI-reverb', 'ESTOI-dereverb',]
            writer = csv.DictWriter(f, fieldnames = fieldnames)
            writer.writeheader()
            for i in range(len(self.audio_list)):
                writer.writerow({'SRMR-reverb' : srmr_reverb[i],
                                 'SRMR-dereverb': srmr_dereverb[i],
                                 'SDR-reverb': sdr_reverb[i],
                                 'SDR-dereverb': sdr_dereverb[i],
                                 'ESTOI-reverb': estoi_reverb[i],
                                 'ESTOI-dereverb': estoi_dereverb[i]})
            f.close()

import pickle
EXP_FILE = '/home/martin/Documents/tesis/src/experiments/datasets/exp1.pkl'
with open(EXP_FILE, 'rb') as f:
    exp_1 = pickle.load(f)

audio_list = exp_1['clean_test']
rir_list = exp_1['real_test']['octagon']
weights_path = '/home/martin/Documents/tesis/src/model/ckpts/weights.hdf5'

evaluacion = Evaluate_model(audio_list, rir_list, weights_path)
evaluacion.calculate_metrics()
evaluacion.save_results('experiments/csv/exp1/octagon.csv')
   
