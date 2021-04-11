"""
Bucle de entrenamiento
"""
import sys
import tensorflow as tf
MAIN_PATH="/home/martin/Documents/tesis/src"
sys.path.append(MAIN_PATH) #Para poder importar archivos .py como librerias

#Data generators
from model.data_loader import build_generators
training_generator = build_generators(MAIN_PATH)

#defino el modelo
from model.network_architecture import dereverb_autoencoder, modelo_de_prueba
modelo = dereverb_autoencoder()
#modelo = modelo_de_prueba()
modelo.summary()

#callbacks

#cbks = [tf.keras.callbacks.EarlyStopping(monitor='loss',restore_best_weights=True),
#        tf.keras.callbacks.ModelCheckpoint('/home/martin/Documents/tesis/src/model/ckpts/weights.{epoch:02d}-{loss:.2f}.hdf5'),
#        tf.keras.callbacks.TensorBoard(log_dir='tb_logs',profile_batch=0, update_freq='batch', histogram_freq=1)]
#agregar argumento callbacks=cbaks en modelo.fit


#entrenando                                                                                                                                                                 
history = modelo.fit(training_generator, workers=12, steps_per_epoch=None, epochs=1)


