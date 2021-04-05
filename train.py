

"""
Bucle de entrenamiento
"""

import sys
MAIN_PATH="/home/martin/Documents/tesis/src"
sys.path.append(MAIN_PATH) #Para poder importar archivos .py como librerias

#Data generators
from model.data_loader import build_generators
loadpath = MAIN_PATH + '/data/data_ready_img/'
params = {'dim': (256,257), 'batch_size': 2, 'shuffle': True, 'path' : loadpath}
training_generator, validation_generator = build_generators(MAIN_PATH, params)

#defino el modelo
from model.network_architecture import dereverb_autoencoder
modelo = dereverb_autoencoder()
modelo.summary()

#callbacks


#entrenando 
history = modelo.fit(training_generator, epochs= 10)

