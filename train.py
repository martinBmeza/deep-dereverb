"""
Bucle de entrenamiento
"""
import sys
import tensorflow as tf
MAIN_PATH="/home/martin/Documents/tesis/src"
sys.path.append(MAIN_PATH) #Para poder importar archivos .py como librerias

#Data generators
from model.data_loader_static import build_generators
params = {'path':'/mnt/datasets/npy_data/', 'batch_size' : 8, 'dim' : (257,256)}
training_generator, validation_generator = build_generators(params)

#defino el modelo
from model.network_architecture import dereverb_autoencoder
modelo = dereverb_autoencoder()
modelo.summary()

#callbacks

#cbks = [tf.keras.callbacks.EarlyStopping(monitor='loss',restore_best_weights=True),
#        tf.keras.callbacks.ModelCheckpoint('/home/martin/Documents/tesis/src/model/ckpts/weights.{epoch:02d}-{loss:.2f}.hdf5'),
#        tf.keras.callbacks.TensorBoard(log_dir='tb_logs',profile_batch=0, update_freq='batch', histogram_freq=1)]
#agregar argumento callbacks=cbaks en modelo.fit

cbks = [tf.keras.callbacks.EarlyStopping(monitor='loss',restore_best_weights=True, patience=3),
        tf.keras.callbacks.ModelCheckpoint('/home/martin/Documents/tesis/src/model/ckpts/weights.{epoch:02d}-{loss:.2f}.hdf5'),
        tf.keras.callbacks.TensorBoard(log_dir='tb_logs',profile_batch=0, update_freq='batch', histogram_freq=1)]
#agregar argumento callbacks=cbaks en modelo.fit


#entrenando                                                                                                                                                                 
history = modelo.fit(training_generator,validation_data =  validation_generator, use_multiprocessing = True, callbacks = cbks, workers=12, epochs=10)


