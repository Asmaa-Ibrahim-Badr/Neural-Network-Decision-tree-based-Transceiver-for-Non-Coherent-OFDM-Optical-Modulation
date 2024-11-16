import tensorflow as tf

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model('model_tree15_noise20db') # path to the SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)