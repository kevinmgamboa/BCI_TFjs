"""

Created on Wed Jun  9 21:18:16 2021
@author: kevin machado gamboa
"""
# -----------------------------------------------------------------------------
#                           Libraries Needed
# -----------------------------------------------------------------------------
import os
import copy
import numpy as np
import pandas as pd
# Personal Libraries
from helpers_and_functions import config, main_functions as mpf, utils
import databases as dbs
# Libraries for training process
from sklearn.model_selection import KFold
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

import modelhub as mh

# Creates folder to store experiment
date = utils.create_date()
os.mkdir('log_savings/alex' + date)
# %%
# ------------------------------------------------------------------------------------
#                               Loading dataset
# ------------------------------------------------------------------------------------
# initialize eeg database
dataset = dbs.eeg()
# path to dataset
ane_data_path = 'datasets/alex'
# loads [x_epochs, y_labels]
dataset.load_epochs_labels(ane_data_path, selected_channels=config.channels_sleep_montage, sleep_montage=True)
# converts labels to [0=>conscious, 1=>unconscious]
dataset.get_binary_labels()
# Normalize the dataset between [-1,1]
dataset.transform(mpf.nor_dataset)
# applying dataset transformation e.g. 'spectrogram'
dataset.transform(mpf.raw_chunks_to_spectrograms, name=date + 'SPECTROGRAM')
# make dataset ready for training
dataset.get_ready_for_training()

# %%
# ------------------------------------------------------------------------------------
#                                Cross-Validation
# ------------------------------------------------------------------------------------
acc_patient_score, loss_patient_score = [], []
cm_per_participant = []

for i, n in enumerate(dataset.data.keys()):
    print(100 * '#')
    print(f'##################### Participant {1 + i} from {len(dataset.data)} #####################')
    print(100 * '#')
    # Initializing variables
    cm_per_fold = []
    # number of train epochs
    train_epochs = 10
    # early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
    #                                               verbose=1, restore_best_weights=True)
    kfold = KFold(n_splits=config.NUM_FOLDS, shuffle=True)
    # init fold number
    fn = 1

    # stores the models with best acc & other info
    model_best = {'model': [],
                  'score': [],
                  'tra_with': [],
                  'val_with': [],
                  'train_history': [],
                  'initial_weights': [],
                  'test_acc_per_fold': [],
                  'test_loss_per_fold': [],
                  'transformation': dataset.info['transformation']}
    # training parameters
    parameters = {'lr': 1e-5,
                  'num_filters': 10,
                  'kernel_size': 3,
                  'dense_units': 10,
                  'out_size': 1}

    p_count = 0  # early stopping counter
    patient = 5  # wait n epochs for error to keep decreasing, is not stop

    all_folds_best_test_score = 0.0
    for tra, val in kfold.split(dataset.data[n]['train']['epochs'], dataset.data[n]['train']['labels']):
        # Call the hub
        hub = mh.simple_cnn(param=parameters)
        # build model structure
        hub.build_model_structure(dataset.info[n]['data_shape'])
        # compile model
        hub.compile()
        # initializing model
        model_best_fold = tf.keras.models.clone_model(hub.model)
        # initial model weights
        ini_wei = hub.model.get_weights()
        # initializing training history
        train_history = []
        # defines an initial score
        pre_score = [1.0, 0.0]
        # Generate a print
        print(100 * '-')
        print(f'--------------------------------- Training for fold {fn} ---------------------------------')
        # -----------------------------------------------------------------------------
        #                               Train-Test Loop
        # -----------------------------------------------------------------------------
        for n_ep in range(train_epochs):
            print('------- train score -------')
            # Train the model ---------------------------------------------------------
            train_score = hub.model.fit(dataset.data[n]['train']['epochs'][tra],
                                        dataset.data[n]['train']['labels'][tra],
                                        validation_data=(dataset.data[n]['train']['epochs'][val],
                                                         dataset.data[n]['train']['labels'][val]),
                                        epochs=1)  # , callbacks=[early_stop])
            print('------- test score -------')
            # Evaluates on Test set ----------------------------------------------------
            test_scores = hub.model.evaluate(dataset.data[n]['test']['epochs'],
                                             dataset.data[n]['test']['labels'])
            # saves train history
            train_score = list(np.concatenate(list(train_score.history.values())))
            # Adding test score to train score
            train_score.extend(test_scores)
            # Train history including the score in test set
            train_history.append(train_score)

            # Stores the best model in the fold ----------------------------------------
            if test_scores[1] > pre_score[1]:
                print(f'new best score in the fold: {test_scores[1]:.4}')
                # saves best model INSIDE FOLD
                hub.model.save('log_savings/alex' + date + '/best_fold_model.h5')
                # Saves best model from ALL FOLDS
                # if test_scores[1] > all_folds_best_test_score:
                #     print(f'new best model from ALL FOLDS {test_scores[1]:.4} ')
                #     all_folds_best_test_score = test_scores[1]
                #     # saves best model
                #     hub.model.save('log_savings/alex' + date + '/all_folds_best_model.h5')
                # updating previous score
                pre_score = copy.copy(test_scores)
            else:  # Stopping criteria:
                p_count += 1
                if p_count >= patient:
                    print('Early Stopping !!! Error hasnt decreased')
                    p_count = 0
                    break
        # -----------------------------------------------------------------------------
        #                          Stores Data from Each Fold
        # -----------------------------------------------------------------------------
        # save train history
        model_best['train_history'].append(train_history)
        # save best score from fold
        train_history = pd.DataFrame(train_history)
        idx = train_history[5].idxmax()  # max idx test acc
        model_best['score'].append(train_history[5][idx])
        # saves segments of data the model was trained with
        model_best['tra_with'].append(tra)
        model_best['val_with'].append(val)
        # save model initial weights
        model_best['initial_weights'].append(ini_wei)

        print(
            f'Best score fold {fn}: {hub.model.metrics_names[0]}: {train_history[4][idx]:.4}; {hub.model.metrics_names[1]}: {train_history[5][idx] * 100:.4}%')
        # Adds test score
        model_best['test_acc_per_fold'].append(train_history[5][idx] * 100)
        model_best['test_loss_per_fold'].append(train_history[4][idx])
        # confusion matrix per fold
        # -------------------------
        # Load best model in fold
        model_best_fold = tf.keras.models.load_model('log_savings/alex' + date + '/best_fold_model.h5')
        # # Confusion matrix of best model in fold
        # cm_per_fold.append(utils.get_confusion_matrix(model_best_fold, dataset.data[n]['test'],
        #                                               dataset.info[n]['class_balance']['test']['value']))
        # Increase fold number
        fn += 1
    # Saving Scores per Patient
    # -----------------------------------------------------------------------------------------------------------------
    acc_patient_score.append(model_best['test_acc_per_fold'])
    loss_patient_score.append(model_best['test_loss_per_fold'])
    # cm_per_participant.append(cm_per_fold)
##%%
# ------------------------------------------------------------------------------------
#                                    Final Results
# ------------------------------------------------------------------------------------
acc_patient_score, loss_patient_score = np.array(acc_patient_score), np.array(loss_patient_score)
# mean acc & loss in cross validation per subject
mean_acc = [np.mean(score) for score in acc_patient_score]
mean_loss = [np.mean(score) for score in loss_patient_score]

# Boxplot the average participant test acc per fold
title = f'Participants Cross-Validation Score Distribution Across {config.NUM_FOLDS} Folds\n ' \
        f'for {len(acc_patient_score)}'
utils.boxplot_within_patient_acc_fold(acc_patient_score, loss_patient_score, title)

# Plots mean Test acc & loss in cross validation per subject
utils.plot_mean_acc_within_patient(mean_acc, mean_loss)

# confusion matrix dataframe across participants
df = utils.cm_participants_to_df(cm_per_participant, loss_patient_score)

utils.boxplot_evaluation_metrics_from_df(df)
df.to_csv('log_savings/alex' + date + '/' + date + '_data.csv')
