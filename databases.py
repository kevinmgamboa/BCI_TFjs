"""
Sleep Data
----------
Load and prepare the dataset. In the future this will aso creates different dataset fold to run different experiments of generalization error.
Created on Tue Mar  2 12:10:27 2021
@author: Kevin Machado Gamboa
"""
import os
import mne
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

# Personal Libraries
from helpers_and_functions import config, main_functions as mpf

# -----------------------------------------------------------------------------
#                           Class for loading EEG data
# -----------------------------------------------------------------------------
class eeg:
    """
    Manage the loading and pre-processing for the Anaesthesia dataset

            Parameters
            ----------
            ane_data_path : string
                path to the Anaesthesia dataset

            Returns
            -------
            data : dict
                dictionary with eeg recordings
    """

    def __init__(self):
        # dataset dictionary
        self.data = {}
        # Initializing information variable
        self.info = dict()

    def load_epochs_labels(self, ane_data_path, selected_channels=config.channels_sleep_montage,
                           sleep_montage=True, n_test=0.3):
        # Initializing container for dataset
        self.data = dict()

        self.info = dict()
        # initializes subject ids counter
        s_ids = []
        # loading process
        for n, eeg_file in enumerate(tqdm(os.listdir(ane_data_path))):
            # get subject id
            s_id = eeg_file[11:15]
            # check if id in counter
            if s_id not in s_ids:
                s_ids.append(s_id)
                self.data[s_id] = {'train': {"epochs": [], "labels": []},
                                   'test': {"epochs": [], "labels": []}}
            # Loads eyes closed
            if eeg_file.endswith("sleep.edf"):
                # loads file and selected eeg channels
                raw = mne.io.read_raw(os.path.join(ane_data_path, eeg_file)).pick(selected_channels)
                if sleep_montage:
                    # applies montage
                    raw = mpf.sleep_montage(raw)
                # applies filter
                raw.load_data().filter(1, 49, fir_design='firwin')
                # creating equally-spaced Events arrays
                raw = mne.make_fixed_length_epochs(raw, duration=config.EPOCH_LENGTH,
                                                   preload=False)
                # splits dataset keeping class distribution
                train_x, test_x, train_y, test_y = train_test_split(raw.get_data(), raw.events[:, 2] - 1,
                                                                    test_size=n_test)
                # stores training epochs and labels in data container
                self.data[s_id]['train']["epochs"].append(train_x)
                self.data[s_id]['train']["labels"].append(train_y)
                # stores test epochs and labels in data container
                self.data[s_id]['test']["epochs"].append(test_x)
                self.data[s_id]['test']["labels"].append(test_y)

            # Loads eyes open
            if eeg_file.endswith("awake.edf"):
                # loads file and selected eeg channels
                raw = mne.io.read_raw(os.path.join(ane_data_path, eeg_file)).pick(selected_channels)
                if sleep_montage:
                    # applies montage
                    raw = mpf.sleep_montage(raw)
                # applies filter
                raw.load_data().filter(1, 49, fir_design='firwin')
                # creating equally-spaced Events arrays
                raw = mne.make_fixed_length_epochs(raw, duration=config.EPOCH_LENGTH,
                                                   preload=False)
                # splits dataset keeping class distribution
                train_x, test_x, train_y, test_y = train_test_split(raw.get_data(), raw.events[:, 2], # label = 1
                                                                    test_size=n_test)
                # stores training epochs and labels in data container
                self.data[s_id]['train']["epochs"].append(train_x)
                self.data[s_id]['train']["labels"].append(train_y)
                # stores test epochs and labels in data container
                self.data[s_id]['test']["epochs"].append(test_x)
                self.data[s_id]['test']["labels"].append(test_y)

        # Concatenation all examples in list
        for id in self.data.keys():
            self.data[id]['train']['labels'] = np.array(np.concatenate(self.data[id]['train']['labels']))
            self.data[id]['train']['epochs'] = np.array(np.concatenate(self.data[id]['train']['epochs']))

            self.data[id]['test']['labels'] = np.array(np.concatenate(self.data[id]['test']['labels']))
            self.data[id]['test']['epochs'] = np.array(np.concatenate(self.data[id]['test']['epochs']))

            # information samples
            self.info[id] = {'n_samples': {'train': len(self.data[id]['train']['labels']),
                                           'test': len(self.data[id]['test']['labels'])}}
            # time information
            self.info[id].update({'eeg_time':
                                     {'total': mpf.eeg_time(
                                         sum(self.info[id]['n_samples'].values()) * config.EPOCH_LENGTH),
                                      'train': mpf.eeg_time(
                                          list(self.info[id]['n_samples'].values())[0] * config.EPOCH_LENGTH),
                                      'test': mpf.eeg_time(
                                          list(self.info[id]['n_samples'].values())[1] * config.EPOCH_LENGTH)}})
            # class balance information
            self.info[id].update({'class_balance': {'train':
                {'value': dict(
                    pd.Series(self.data[id]['train']['labels']).value_counts()),
                    'percentage': dict(
                        pd.Series(self.data[id]['train']['labels']).value_counts() /
                        self.info[id]['n_samples']['train'])
                },
                'test':
                    {'value': dict(
                        pd.Series(self.data[id]['test']['labels']).value_counts()),
                        'percentage': dict(
                            pd.Series(self.data[id]['test']['labels']).value_counts() /
                            self.info[id]['n_samples']['test'])}}})

    def get_binary_labels(self):
        for id in self.data.keys():
            # converts eye open into 0 & sedation into 1
            self.data[id]['train']['labels'] = pd.Series(self.data[id]['train']['labels']).replace(1, 0).replace(2, 1).to_numpy()
            self.data[id]['test']['labels'] = pd.Series(self.data[id]['test']['labels']).replace(1, 0).replace(2, 1).to_numpy()
            # class balance information
            self.info[id]['class_balance'] = {'train':
                                              {'value': dict(pd.Series(self.data[id]['train']['labels']).value_counts()),
                                               'percentage': dict(pd.Series(self.data[id]['train']['labels']).value_counts() /
                                                                  self.info[id]['n_samples']['train'])
                                               },
                                          'test':
                                              {'value': dict(pd.Series(self.data[id]['test']['labels']).value_counts()),
                                               'percentage': dict(pd.Series(self.data[id]['test']['labels']).value_counts() /
                                                                  self.info[id]['n_samples']['test'])}}

        print('Anaesthesia dataset with binary labels: [0=>conscious, 1=>unconscious]')

    def transform(self, my_function, name='unspecify'):
        for id in self.data.keys():
            print('transforming training dataset')
            # transforming training dataset
            self.data[id]['train']['epochs'] = my_function(self.data[id]['train']['epochs'])
            print('transforming test dataset')
            # transforming validation dataset
            self.data[id]['test']['epochs'] = my_function(self.data[id]['test']['epochs'])
            # information: Data Shape
            self.info[id].update({'data_shape': np.shape(self.data[id]['test']['epochs'][0])})
            # name for transformation
            self.info.update({'transformation': name})

    def get_ready_for_training(self):
        for id in self.data.keys():
            self.data[id]['train']['epochs'] = np.expand_dims(self.data[id]['train']['epochs'], 1)
            self.data[id]['test']['epochs'] = np.expand_dims(self.data[id]['test']['epochs'], 1)