"""
Configuration File
------------------
Created on Mon Jun 21 10:37:31 2021
@author: Kevin MAchado Gamboa
"""
EPOCH_LENGTH = 3
# Channels used
channels = ['Iz', 'O2', 'Oz', 'O1', 'PO8', 'PO4', 'POz', 'PO3', 'PO7', 'P8',
            'P6', 'P4', 'P2', 'Pz', 'P1', 'P3', 'P5', 'P7', 'TP10', 'TP8', 'CP6',
            'CP4', 'CP2', 'CPz', 'CP1', 'CP3', 'CP5', 'TP7', 'TP9', 'T8', 'C6',
            'C4', 'C2', 'Cz', 'C1', 'C3', 'C5', 'T7', 'FT8', 'FC6', 'FC4', 'FC2',
            'FCz', 'FC1', 'FC3', 'FC5', 'FT7', 'F8', 'F6', 'F4', 'F2', 'Fz', 'F1',
            'F3', 'F5', 'F7', 'AF4', 'AFz', 'AF3', 'Fp2', 'Fpz', 'Fp1']

channels_r = ['Iz', 'O2', 'Oz', 'O1', 'PO8', 'PO4', 'POz', 'PO3', 'PO7', 'P8',
              'P6', 'P4', 'P2', 'Pz', 'P1', 'P3', 'P5', 'P7', 'TP8', 'CP6',
              'CP4', 'CP2', 'CPz', 'CP1', 'CP3', 'CP5', 'TP7', 'TP9', 'T8', 'C6',
              'C4', 'C2', 'Cz', 'C1', 'C3', 'C5', 'T7', 'FT8', 'FC6', 'FC4', 'FC2',
              'FCz', 'FC1', 'FC3', 'FC5', 'FT7', 'F8', 'F6', 'F4', 'F2', 'Fz', 'F1',
              'F3', 'F5', 'F7', 'AF4', 'AFz', 'AF3', 'Fp2', 'Fpz', 'Fp1']

# Channels for sleep montage
channels_sleep_montage = ['Fp2', 'Cz', 'Pz', 'Oz']
# Define EEG bands
eeg_bands = {'Delta': (1, 4),
             'Theta': (4, 8),
             'Alpha': (8, 12),
             'Beta': (12, 30),
             'Gamma': (30, 49)}

# ------------------------------------------------------------------
NUM_FOLDS = 5
