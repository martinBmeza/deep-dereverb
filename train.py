"""
Bucle de entrenamiento
"""
import sys
import tensorflow as tf
MAIN_PATH="/home/martin/Documents/tesis/src"
sys.path.append(MAIN_PATH) #Para poder importar archivos .py como librerias

#Data generators
from model.data_loader import build_generators
params = {'path':'/mnt/datasets/npy_data/experimento_1/', 'batch_size' : 5, 'dim' : (256, 257)}
training_generator, validation_generator = build_generators(params)

#defino el modelo
from model.network_architecture import autoencoder
#modelo = dereverb_autoencoder()
modelo = autoencoder()
modelo.summary()

cbks = [tf.keras.callbacks.EarlyStopping(monitor='loss',restore_best_weights=True, patience=2),
        tf.keras.callbacks.ModelCheckpoint('/home/martin/Documents/tesis/src/model/ckpts/weights.{epoch:02d}-{loss:.3f}.hdf5'),
        tf.keras.callbacks.TensorBoard(log_dir='tb_logs',profile_batch=0, update_freq='batch', histogram_freq=1)]


#Entreno
history = modelo.fit(training_generator, validation_data =  validation_generator, use_multiprocessing = True, callbacks = cbks, workers=12, epochs=10)


