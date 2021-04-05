"""Contiene el bucle principal utilizado para probar y evaluar el modelo sobre set de datos de prueba"""
import tensorflow as tf

def predict_model(data, modelo):
  output = [layer.name for layer in modelo.layers]
  outputs = []
  output_names = []
  inputs = []
  input_names = []
  for layer in modelo.layers:
      if hasattr(layer,'is_placeholder'):
          inputs.append(layer.output)
          input_names.append(layer.name)
      elif layer.name in output:
          outputs.append(layer.output)
          output_names.append(layer.name)
      else:
          pass
  predict_fn = tf.keras.backend.function(inputs = inputs,outputs=outputs)
  activations = predict_fn(data)
  activations = {name: act for name, act in zip(output_names,activations)}
  print('orden de las entradas: /\n',input_names)
  return activations
