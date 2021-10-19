"""
Created on Tue Oct  19 22:13:16 2021
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
# ------------------------------------------------------------------------------------
#                                Cross-Validation
# ------------------------------------------------------------------------------------
# Creates folder to store experiment
date = utils.create_date()
save_path = 'log_savings/alex_' + date
os.mkdir(save_path)
# confusion matrix per fold variable
cm_per_fold = []
# number of train epochs
train_epochs = 50
# defining a K-fold object
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
              'transformation': database.info['transformation']}
# training parameters
parameters = {'lr': 1e-5,
              'num_filters': 10,
              'kernel_size': 3,
              'dense_units': 10,
              'out_size': 1}

p_count = 0  # early stopping counter
patient = 10  # wait n epochs for error to keep decreasing, is not stop

all_folds_best_test_score = 0.0
for tra, val in kfold.split(database.data['train']['epochs'], database.data['train']['labels']):
    # import model
    model = tf.keras.models.load_model('log_savings/9933_20210713_185628_sequential_27.h5')
    # compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(parameters['lr']),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=tf.keras.metrics.BinaryAccuracy(name='accuracy'))
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
        train_score = model.fit(database.data['train']['epochs'][tra],
                                database.data['train']['labels'][tra],
                                validation_data=(database.data['train']['epochs'][val],
                                                     database.data['train']['labels'][val]),
                                    epochs=1)#, callbacks=[early_stop])
        print('------- test score -------')
        # Evaluates on Test set ----------------------------------------------------
        test_scores = model.evaluate(database.data['test']['epochs'], database.data['test']['labels'])
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
            model.save(save_path + '/best_fold_model.h5')
            # Saves best model from ALL FOLDS
            if test_scores[1] > all_folds_best_test_score:
                print(f'new best model from ALL FOLDS {test_scores[1]:.4} ')
                all_folds_best_test_score = test_scores[1]
                # saves best model
                model_name = f'/{int(10000*all_folds_best_test_score)}_best_model.h5'
                model.save(save_path + '/_best_model.h5')
            # updating previous score
            pre_score = copy.copy(test_scores)
            # reset the stopping patient counter
            p_count = 0
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
    # id best model from training
    idx = train_history[5].idxmax()  # max idx test acc
    model_best['score'].append(train_history[5][idx])
    # saves segments of data the model was trained with
    model_best['tra_with'].append(tra)
    model_best['val_with'].append(val)
    # # save model initial weights
    # model_best['initial_weights'].append(ini_wei)

    print(
        f'Best score fold {fn-1}: {model.metrics_names[0]}: {train_history[4][idx]:.4}; {model.metrics_names[1]}: {train_history[5][idx] * 100:.4}%')
    # Adds test score
    model_best['test_acc_per_fold'].append(train_history[5][idx] * 100)
    model_best['test_loss_per_fold'].append(train_history[4][idx])
    # confusion matrix per fold
    # -------------------------
    # Load best model in fold
    model_best_fold = tf.keras.models.load_model(save_path + '/best_fold_model.h5')
    # Confusion matrix of best model in fold
    cm_per_fold.append(utils.get_confusion_matrix(model_best_fold, database.data['test'],
                                                  database.info['class_balance']['test']['value']))
    # Increase fold number
    fn += 1
#%%
# ------------------------------------------------------------------------------------
#                                    Final Results
# ------------------------------------------------------------------------------------
# confusion matrix dataframe across participants
df = utils.cm_fold_to_df(cm_per_fold)
utils.boxplot_evaluation_metrics_from_df(df, x_axes='fold', save_path=save_path+'/ev_metrics')

# plots train history for the best model
utils.plot_train_test_history(model_best, save_path=save_path+'/tt_history.png')

# Plots the confusion matrix of the best model the folds
cm_categories = {0: 'Conscious', 1: 'Unconscious'}
labels = [' True Pos', ' False Neg', ' False Pos', ' True Neg']
utils.make_confusion_matrix(cm_per_fold[np.argmax(model_best['score'])], group_names=labels, categories=cm_categories,
                            class_balance=database.info['class_balance']['test']['value'],
                            title='Confusion Matrix of Best Model', save_path=save_path+'/cf_matrix.png')
# Save info best model as json
utils.extract_best_convert_json(model_best, path=save_path)
# deleting unnecessary files
os.remove(save_path + '/best_fold_model.h5')
# # zip the experiment folder
# shutil.make_archive(save_path+'/experiment', 'zip', save_path)
#%%
# # ------------------------------------------------------------------------------------
# #                                    Final Results
# # ------------------------------------------------------------------------------------
import tensorflowjs as tfjs
model = tf.keras.models.load_model(save_path + model_name)
tfjs.converters.save_keras_model(model, save_path + '/model_tfjs')