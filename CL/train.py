import tensorflow as tf
import pandas as pd
import numpy as np
from model.data_loader import build_generators

BATCH_SIZE = 8

DATAFRAME = pd.read_pickle('data/train/dataset.pkl')
train_gen, val_gen = build_generators(DATAFRAME, BATCH_SIZE)

# Defino el modelo
from model.network_architecture import autoencoder
modelo = autoencoder()

cbks = [tf.keras.callbacks.EarlyStopping(monitor='loss',restore_best_weights=True, patience=2),
        tf.keras.callbacks.ModelCheckpoint('/home/martin/deep-dereverb/CL/model/ckpts/weights.{epoch:02d}-{loss:.4f}.hdf5')]

modelo.summary()

#Entreno
#modelo.load_weights('/home/martin/Documents/tesis/src/model/ckpts/weights_TIMIT.hdf5')
history = modelo.fit(train_gen,
                     validation_data =  val_gen,
                     use_multiprocessing = True,
                     workers=12, max_queue_size=256, epochs=1,
                     callbacks = cbks, shuffle=True)
#np.save('/home/martin/deep-dereverb/model/logs/history.npy', history.history)
#modelo.save_weights('/home/martin/Documents/tesis/src/model/ckpts/weights_prueba_general.hdf5')
