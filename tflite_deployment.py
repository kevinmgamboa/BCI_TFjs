"""
Created on Mon Oct 18 22:28:59 2021

@author: kevin
"""
# Importing libraries
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# Personal Libraries
from helpers_and_functions import config, main_functions as mpf, utils
import databases as dbs
# %%
# ------------------------------------------------------------------------------------
#                            Prepares the model
# ------------------------------------------------------------------------------------
# loads the model
model_path = 'log_savings/alex_20211018_194037'
model = tf.keras.models.load_model(model_path+'/all_folds_best_model.h5')
# converts the model to tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
# Save the model.
with open(model_path + '/model.tflite', 'wb') as f:
  f.write(tflite_model)
# Loads the model as an interpreter
interpreter = tf.lite.Interpreter(model_path=model_path + '/model.tflite')
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# labels
labels = ['conscious', 'unconscious']
# %%
# ------------------------------------------------------------------------------------
#                            Loads the signal
# ------------------------------------------------------------------------------------
# Feature Applied
feature_function = mpf.raw_chunks_to_spectrograms

# initialize database database
database = dbs.eeg()
# path to dataset
ane_data_path = 'datasets/alex'
# loads [x_epochs, y_labels]
database.load_epochs_labels(ane_data_path, selected_channels=config.channels_sleep_montage, sleep_montage=True)
# Normalize the dataset between [-1,1]
database.transform(mpf.nor_dataset)
# applying dataset transformation e.g. 'spectrogram'
database.transform(feature_function, name='eeg_spectrogram')
# make dataset ready for training
database.get_ready_for_training()
# %%
# ------------------------------------------------------------------------------------
#                            Implementing realtime Spectrogram
# ------------------------------------------------------------------------------------
# initializing variables
predictions = []
# Loop where signal is transformed into spectrogram
for n in tqdm(range(425)):
    # makes the prediction
    prediction = utils.predict_tfl(interpreter, input_details, output_details,
                                   np.expand_dims(database.data['test']['epochs'][n], axis=0))

    predictions.append(prediction)

predictions = np.array(list(predictions))
#%%
plt.plot(predictions)
plt.plot(database.data['test']['labels'])
plt.show()