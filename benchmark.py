"""
Created on Wed Jun  9 21:18:16 2021
@author: kevin machado gamboa
"""
# -----------------------------------------------------------------------------
#                           Libraries Needed
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt
# Personal Libraries
from helpers_and_functions import config, main_functions as mpf, utils
import databases as dbs
# ML library
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# %%
# Feature Applied
feature_function = mpf.raw_chunks_to_spectrograms
# ------------------------------------------------------------------------------------
#                               Loading dataset
# ------------------------------------------------------------------------------------
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
# -----------------------------------------------------------------------------
#                             Importing Model
# -----------------------------------------------------------------------------
transfered_model = tf.keras.models.load_model('log_savings/9685_sleep_20210727_112442_spectrogram_sequential_4/9685_sleep_20210727_112442_spectrogram_sequential_4_model.h5')
transfered_model.summary()

predictions = transfered_model.predict(database.data['train']['epochs'])
# Results
plt.figure()
plt.plot(database.data['train']['labels'])
plt.plot(mpf.vec_nor(predictions))
plt.show()