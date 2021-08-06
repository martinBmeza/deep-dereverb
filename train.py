import sys
import tensorflow as tf
#MAIN_PATH="/home/martin/Documents/tesis/src"
#sys.path.append(MAIN_PATH) #Para poder importar archivos .py como librerias

#Data generators
from model.data_loader import build_generators

CLEAN_PATH = '/home/martin/deep-dereverb/data/prueba/clean/'
REVERB_PATH =  '/home/martin/deep-dereverb/data/prueba/reverb/'
BATCH_SIZE = 8
train_gen, val_gen = build_generators(CLEAN_PATH,REVERB_PATH,BATCH_SIZE)

#defino el modelo
from model.network_architecture import autoencoder
modelo = autoencoder()

cbks = [tf.keras.callbacks.EarlyStopping(monitor='loss',restore_best_weights=True, patience=2),
        tf.keras.callbacks.ModelCheckpoint('/home/martin/Documents/tesis/src/model/ckpts/weights.{epoch:02d}-{loss:.3f}.hdf5')]
        #tf.keras.callbacks.TensorBoard(log_dir='tb_logs',profile_batch=0, update_freq='epoch', histogram_freq=1)]

modelo.summary()
#Entreno
#modelo.load_weights('/home/martin/Documents/tesis/src/model/ckpts/weights_TIMIT.hdf5')
history = modelo.fit(train_gen,
                     validation_data =  val_gen,
                     use_multiprocessing = True,
                     workers=12, max_queue_size=16384, epochs=10)
                     #callbacks = cbks)
#modelo.save_weights('/home/martin/Documents/tesis/src/model/ckpts/weights_prueba_general.hdf5')
