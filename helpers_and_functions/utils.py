"""
Created on Wed Jul 14 11:52:11 2021
@author: Kevin Machado Gamboa
"""
import os
import random as rd
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# importing personal libraries
from helpers_and_functions import config, main_functions as mpf
import databases as dbs


def plot_train_test_history(model_best):
    """
    @param model_best: dictionary with model information
    @return: Plots for history
    """
    fig = plt.figure()
    # Converts train history into DataFrame
    train_history = pd.DataFrame(model_best['train_history'][np.argmax(np.array(model_best['score']))],
                                 columns=['loss', 'acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc'])
    # Line stiles
    styles_acc = ['b', 'y.', 'r--']
    ax1 = fig.add_subplot(211)
    # plots train history
    train_history[['acc', 'val_acc', 'test_acc']].plot(title='Accuracy Train-Val-Test',
                                                       style=styles_acc, linewidth=1.0,
                                                       grid=True, ax=ax1)
    ax2 = fig.add_subplot(212)
    # plots loss history
    train_history[['loss', 'val_loss', 'test_loss']].plot(title='Loss Train-Val-Test',
                                                          style=styles_acc, linewidth=1.0,
                                                          grid=True, ax=ax2)
    plt.xlabel('Epochs')
    plt.show()


def print_cross_validation_scores(test_acc_per_fold, test_loss_per_fold):
    """
    Provides a summary of scores after k-fold cross validation
    @param test_acc_per_fold: list of accuracy scores
    @param test_loss_per_fold: list of loss scores
    @return: None
    """
    # == Provide average scores ==
    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i, test_acc_fold in enumerate(test_acc_per_fold):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i + 1} - Test Loss: {test_loss_per_fold[i]:.4} - Test Accuracy: {test_acc_fold:.4}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Mean Test Accuracy: {np.mean(test_acc_per_fold):.4} (+- {np.std(test_acc_per_fold):.4})')
    print(f'> Mean Loss: {np.mean(test_loss_per_fold):.4}')
    print('------------------------------------------------------------------------')


def cm_from_class_scores(score_0: float,
                         score_1: float,
                         class_balance) -> np.ndarray:
    """
Calculates confusion matrix from positive and negative scores
    @param score_0: score of predicting the 0 class
    @param score_1: score of predicting the 1 class
    @param class_balance: dictionary with class labels and balance
    @return: confusion matrix ndarray
    """
    # Calculating Confusion Matrix
    true_p = round(score_1 * class_balance[1])
    false_n = round((1 - score_1) * class_balance[1])
    true_n = round(score_0 * class_balance[0])
    false_p = round((1 - score_0) * class_balance[0])
    conf_matrix = np.array([[true_p, false_n], [false_p, true_n]], dtype=int)
    return conf_matrix


def evaluation_metrics(conf_matrix: np.ndarray, out: str) -> str:
    """
Calculates ML score measurements such as precision, recall, f1-score
    @param out: desired output
    @param conf_matrix: confusion matrix
    @return: string with the ML measurements
    """
    # Accuracy is sum of diagonal divided by total observations
    accuracy = np.trace(conf_matrix) / float(np.sum(conf_matrix))

    # if it is a binary confusion matrix, show some more stats
    if len(conf_matrix) == 2:
        # Correction for tp+fp=0 and tp+fn=0
        tpfp, tpfn = np.sum(conf_matrix[:, 0]), np.sum(conf_matrix[0, :])
        if tpfn == 0.0 or tpfp == 0.0:
            precision = 0.0
            recall = 0.0
        else:
            precision = conf_matrix[0, 0] / tpfp
            recall = conf_matrix[0, 0] / tpfn
        f1_score = [(2 * precision * recall / (precision + recall)) if (precision + recall != 0.0) else 0.0][0]
        if out == 'text':
            stats = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy, precision, recall, f1_score)
        else:
            stats = np.array([accuracy, precision, recall, f1_score])
    else:
        if out == 'text':
            stats = "\n\nAccuracy={:0.3f}".format(accuracy)
        else:
            stats = accuracy
    return stats


def get_confusion_matrix(model, data, class_balance):
    """
    Method for obtaining confusion matrix
    @param model:
    @param data:
    @param class_balance:
    @return:
    """

    # 1. get indices for positives and negatives labels
    idx_0, idx_1 = data['labels'] == 0, data['labels'] == 1
    # 2. Evaluates predictions in both positive and negative labels
    print('------- negative class prediction -------')
    test_scores_0 = model.evaluate(data['epochs'][idx_0],
                                   data['labels'][idx_0])
    print('------- positive class prediction -------')
    test_scores_1 = model.evaluate(data['epochs'][idx_1],
                                   data['labels'][idx_1])
    # 3. Calculating Confusion Matrix from positive and negative scores
    con_matrix = cm_from_class_scores(test_scores_0[1], test_scores_1[1], class_balance)

    return con_matrix


def cm_fold_to_df(cm_per_fold):
    """
    Conerts Confusion Matrix per fold to data frame
    @param cm_per_fold: cm per participant
    @return: dataframe
    """
    # initialize dataframe
    data_f = pd.DataFrame(
        columns=['true_p', 'false_n', 'false_p', 'true_n', 'acc', 'precision', 'recall', 'f1', 'fold'])
    for num, cmpp in enumerate(cm_per_fold):
        # adds the fold no and participant no to each cm
        cmpp = np.array([np.append(cmpp.flatten(), np.append(evaluation_metrics(cmpp, out=None),
                                                             np.array([num + 1])))])
        # method to append np.array([[true_p, false_n], [false_p, true_n]]
        data_f = data_f.append(pd.DataFrame(cmpp, columns=['true_p', 'false_n', 'false_p', 'true_n',
                                                           'acc', 'precision', 'recall', 'f1', 'fold']),
                               ignore_index=True)

    # converts columns into integers
    col = ["true_p", "false_n", "false_p", "true_n"]
    data_f[col] = data_f[col].astype(int)

    return data_f


def cm_participants_to_df(cm_per_par, melt=False):
    """
    Conerts Confusion Matrix per participants to data frame
    @param cm_per_par: cm per participant
    @return: dataframe
    """
    # initialize dataframe
    data_f = pd.DataFrame(
        columns=['true_p', 'false_n', 'false_p', 'true_n', 'acc', 'precision', 'recall', 'f1', 'fold', 'participant'])
    for num, cmpp in enumerate(cm_per_par):
        # adds the fold no and participant no to each cm
        cmpp = np.array([np.append(cmpf.flatten(), np.append(evaluation_metrics(cmpf, out=None),
                                                             np.array([f + 1, num + 1]))) for f, cmpf in
                         enumerate(cmpp)])
        # method to append np.array([[true_p, false_n], [false_p, true_n]]
        data_f = data_f.append(pd.DataFrame(cmpp, columns=['true_p', 'false_n', 'false_p', 'true_n',
                                                           'acc', 'precision', 'recall', 'f1', 'fold', 'participant']),
                               ignore_index=True)
    if melt:
        # melts cm values into two columns
        data_f = pd.melt(data_f, id_vars=['acc', 'precision', 'recall', 'f1', 'fold', 'participant'], var_name='cm',
                         value_name='value_cm')
    # converts columns into integers
    col = ["true_p", "false_n", "false_p", "true_n", "participant"]
    data_f[col] = data_f[col].astype(int)

    return data_f


def make_confusion_matrix(conf_matrix,
                          group_names=None,
                          categories='auto',
                          class_balance=None,
                          title=None
                          ):
    """
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    conf_matrix:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    class_balance: The value of the distribution of classes
    title:         Title for the heatmap. Default is None.
    """
    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(conf_matrix.size)]

    if group_names and len(group_names) == conf_matrix.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks
    # show the raw number in the confusion matrix.
    group_counts = ["{0:0.0f}\n".format(value) for value in conf_matrix.flatten()]
    # # shows the percentage of cm with respect entire dataset
    # group_percentages = ["{0:.2%}".format(value) for value in conf_matrix.flatten() / np.sum(conf_matrix)]
    # shows the percentage of cm with respect to each class
    class_p = np.array([val / list(class_balance.values())[1 - i] for i, val in enumerate(conf_matrix)])
    class_p = [f"{value:.2%}" for value in class_p.flatten()]

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in
                  zip(group_labels, group_counts, class_p)]
    box_labels = np.asarray(box_labels).reshape(conf_matrix.shape[0], conf_matrix.shape[1])

    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    stats_text = evaluation_metrics(conf_matrix, out='text')
    # makes xlabel
    x_cat = list(np.flip([f'from {class_balance[cl]} \n' + str(categories[cl]) for cl in categories.keys()]))

    # MAKE THE HEATMAP VISUALIZATION
    plt.figure()
    sns.heatmap(conf_matrix, annot=box_labels, fmt="", cmap='gray_r', cbar=True,
                xticklabels=np.flip(list(categories.values())),
                yticklabels=x_cat)
    plt.ylabel('True label')
    plt.xlabel('Predicted label' + stats_text)
    if title:
        plt.title(title)

    plt.show()


def imshow_samples(dataset, classes):
    """
    show image samples from the dataset
    @param dataset: dataset transformed
    @param classes: the vector of classes
    @return: shows the figure
    """
    # extract number of classes (num_c)
    num_c = len(classes)
    # defines the figure subplots
    fig, axes = plt.subplots(nrows=5, ncols=num_c, figsize=(15, 10))
    # class selector
    class_s = 0
    # Specification for barcode plot
    barprops = dict(aspect='auto', cmap='binary', interpolation='nearest')

    for row in axes:
        for col in row:
            # selects a random category (the index)
            sample_idx = rd.sample(list(np.where(dataset['labels'] == (class_s % num_c))[0]), 1)
            # selects the epoch corresponded to category above
            spec_chunk = dataset['epochs'][sample_idx][0][0]
            # shows epoch
            col.imshow(spec_chunk, **barprops)
            # Increase class selector
            class_s += 1
    # Creating labels per columns and rows
    cols = [classes[n] for n in classes]
    rows = ['Sample {}'.format(n) for n in range(1, 6)]

    # Adds titles per columns
    _ = [axes.set_title(col) for axes, col in zip(axes[0], cols)]

    for axes, row in zip(axes[:, 0], rows):
        axes.annotate(row, xy=(0, 0.5), xytext=(-axes.yaxis.labelpad - 5, 0),
                      xycoords=axes.yaxis.label, textcoords='offset points',
                      size='large', ha='right', va='center')

    fig.tight_layout()
    plt.show()


def create_date():
    """
    Creates a date including year-day-time(h:m:s)
    @return: string with date
    """
    return str(datetime.now()).replace(' ', '_').replace('-', '').replace(':', '')[:-7]


def check_benchmark(model_best, database='sleep'):
    """

    @param model_best: dictionary with model information
    @param database: (str) either 'sleep' or 'database'
    @return: None
    """
    # reads benchmark list
    ben_l = pd.read_csv('log_savings/benchmarks_' + database + '.csv', header=0)
    # compares model max outcome with benchmark
    max_o = ben_l.value_best_achieved.max()  # , ben_l.value_best_achieved.idxmax()
    # and ben_l.in_data_form[idx]
    # get the index for model with best score
    index = np.argmax(np.array(model_best['score']))
    # get the best score and multiplied to compare it
    best_score = int(np.array(model_best['score'])[index] * 10000)
    # stores the model in the benchmark
    if best_score > max_o:
        print(f'New Benchmark Found .. \nold: {max_o}\nnew: {best_score}')
        # date for logs
        date = str(datetime.now()).replace(' ', '_').replace('-', '').replace(':', '')[:-7]
        # creates file name
        export_dir = 'log_savings/' + str(int(best_score)) + '_' + database + '_' + \
                     date + '_' + model_best['transformation'] + '_' + model_best['name']
        # makes a folder
        os.mkdir(export_dir)
        # saving the model
        model_best['model'][index].save(export_dir + '/' + export_dir[8:] + '_model.h5')
        # filling columns for logs
        col_fill = [[export_dir[8:], model_best['name'], 'accuracy', best_score, config.NUM_FOLDS,
                     "{:.4f}".format(np.mean(model_best['test_acc_per_fold'])),
                     "{:.4f}".format(np.std(model_best['test_acc_per_fold'])),
                     "{:.4f}".format(np.mean(model_best['test_loss_per_fold'])),
                     model_best['transformation']]]
        # convert column into Dataframe
        ben_l = pd.DataFrame(col_fill)
        # updating benchmark.csv
        ben_l.to_csv('log_savings/benchmarks_' + database + '.csv', index=False, mode='a', header=False)
        # saving segments parameters for reproducibility
        with open(export_dir + '/' + export_dir[8:] + '_data_segments.npy', 'wb') as file:
            np.save(file, model_best['tra_with'][index])
            np.save(file, model_best['val_with'][index])
        # saving initial weights for reproducibility
        with open(export_dir + '/' + export_dir[8:] + '_initial_weights_.npy', 'wb') as file:
            for _, wei in enumerate(model_best['initial_weights'][index]):
                np.save(file, wei)
        # Stores train history
        train_history = pd.DataFrame(model_best['train_history'][index],
                                     columns=['loss', 'acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc'])
        train_history.to_pickle(export_dir + '/' + export_dir[8:] + '_train_history.pkl')


def super_test(model, transform_func, dataset='sleep', n_files=5):
    """
    @param n_files:
    @param model:
    @param dataset:
    @param transform_func:
    @return:
    """

    if dataset == 'sleep':
        # initialize sleep database
        sleep = dbs.sleep()
        # loads [x_epochs, y_labels]
        sleep.load_epochs_labels(t_files=n_files)
        # converts labels to [0=>conscious,5* 1=>unconscious]
        sleep.get_binary_labels()
        # Normalize the dataset between [-1,1]
        sleep.transform(mpf.nor_dataset)
        # applying dataset transformation e.g. 'spectrogram'
        sleep.transform(transform_func)
        # make dataset ready for training
        sleep.get_ready_for_training()
        # make train and test into 1
        sleep.make_them_one()
        ##############################
        print(model.evaluate(sleep.data['epochs'], sleep.data['labels']))
        # Get confusion matrix
        conf_mat = get_confusion_matrix(model, sleep.data,
                                        sleep.info['class_balance']['value'])
        # Plot confusion matrix
        cm_categories = {0: 'Conscious', 1: 'Unconscious'}
        labels = [' True Pos', ' False Neg', ' False Pos', ' True Neg']
        make_confusion_matrix(conf_mat, group_names=labels,
                              categories=cm_categories,
                              class_balance=sleep.info['class_balance']['value'])

    elif dataset == 'database':
        # initialize database database
        anaesthesia = dbs.anaesthesia()
        # path to dataset
        ane_data_path = r'datasets\Kongsberg_anesthesia_data\EEG_resampled_100Hz'
        # loads [x_epochs, y_labels]
        anaesthesia.load_epochs_labels(ane_data_path, selected_channels=config.channels_sleep_montage,
                                       sleep_montage=True)
        # converts labels to [0=>conscious, 1=>unconscious]
        anaesthesia.get_binary_labels()
        # Normalize the dataset between [-1,1]
        anaesthesia.transform(mpf.nor_dataset)
        # applying dataset transformation e.g. 'spectrogram'
        anaesthesia.transform(transform_func)
        # make dataset ready for training
        anaesthesia.get_ready_for_training()
        # make train and test into 1
        anaesthesia.make_them_one()
        ##############################
        print(model.evaluate(anaesthesia.data['epochs'], anaesthesia.data['labels']))
        # Get confusion matrix
        conf_mat = get_confusion_matrix(model, anaesthesia.data,
                                        anaesthesia.info['class_balance']['value'])
        # Plot confusion matrix
        cm_categories = {0: 'Conscious', 1: 'Unconscious'}
        labels = [' True Pos', ' False Neg', ' False Pos', ' True Neg']
        make_confusion_matrix(conf_mat, group_names=labels,
                              categories=cm_categories,
                              class_balance=anaesthesia.info['class_balance']['value'])


# -----------------------------------------------------------------------------
#                               Within training plots
# -----------------------------------------------------------------------------
def boxplot_evaluation_metrics_from_df(data_frame, x_axes):
    """
    Plots Confusion Matrix and Evaluation Metrics from DataFrame
    @param data_frame: Pandas DataFrame with Metrics
    @return:
    """
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    sns.barplot(data=data_frame, x=x_axes, y='true_n')
    plt.subplot(2, 2, 2)
    sns.barplot(data=data_frame, x=x_axes, y='true_p')
    plt.subplot(2, 2, 3)
    sns.barplot(data=data_frame, x=x_axes, y='false_p')
    plt.subplot(2, 2, 4)
    sns.barplot(data=data_frame, x=x_axes, y='false_n')
    plt.suptitle(f'Confusion Matrix Scores\n in {config.NUM_FOLDS} Fold Cross-Validation \nAcross Participants')
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    sns.barplot(data=data_frame, x=x_axes, y='acc')
    plt.subplot(2, 2, 2)
    sns.barplot(data=data_frame, x=x_axes, y='precision')
    plt.subplot(2, 2, 3)
    sns.barplot(data=data_frame, x=x_axes, y='recall')
    plt.subplot(2, 2, 4)
    sns.barplot(data=data_frame, x=x_axes, y='f1')
    plt.suptitle(f'Evaluation Metrics Scores\n in {config.NUM_FOLDS} Fold Cross-Validation \nAcross Participants')
    plt.show()


def boxplot_within_patient_acc_fold(acc_score_data, loss_score_data, title):
    """
    Boxplots the average participant test acc per fold
    @param loss_score_data: loss matrix num_participants X folds_cross-validation
    @param acc_score_data: acc matrix num_participants X folds_cross-validation
    @param title: figure title
    @return: image plotted
    """
    plt.subplot(1, 2, 1)
    sns.boxplot(data=acc_score_data, linewidth=1.5)
    plt.ylabel('Accuracy')
    plt.xlabel('Fold number')
    plt.xticks(np.arange(0, 5), np.arange(1, 6))

    plt.subplot(1, 2, 2)
    plt.ylabel('Loss')
    plt.xlabel('Fold number')
    sns.boxplot(data=acc_score_data, linewidth=1.5)
    plt.xticks(np.arange(0, 5), np.arange(1, 6))
    plt.suptitle(title)
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=acc_score_data.transpose(), linewidth=1.5)
    plt.ylabel('Accuracy')
    plt.xlabel('Participants')
    plt.xticks(np.arange(0, len(acc_score_data)), np.arange(1, len(acc_score_data) + 1))
    plt.subplot(1, 2, 2)
    plt.ylabel('Loss')
    plt.xlabel('Participants')
    sns.boxplot(data=loss_score_data.transpose(), linewidth=1.5)
    plt.xticks(np.arange(0, len(acc_score_data)), np.arange(1, len(acc_score_data) + 1))
    plt.suptitle(f'Cross-Validation Score Distribution Across {len(loss_score_data)} Participants')
    plt.show()


def plot_mean_acc_within_patient(mean_test_acc, mean_test_loss):
    '''
    Plots mean acc & loss in cross validation per subject
    @param mean_test_acc: average test acc vector of length n participants
    @param mean_test_loss: average test loss vector of length n participants
    @return: figure
    '''
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.ylabel('Accuracy')
    sns.boxplot(data=mean_test_acc, linewidth=1.5)
    plt.xticks([0], [' '])
    plt.subplot(1, 2, 2)
    plt.ylabel('Loss')
    sns.boxplot(data=mean_test_loss, linewidth=1.5)
    plt.xticks([0], [' '])
    plt.suptitle(f'Distribution of Average Test Score Within Participants\n{len(mean_test_acc)} Subjects')
    plt.show()

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.ylabel('Accuracy')
    plt.plot(mean_test_acc, 'r*', linewidth=1.5)
    plt.xticks(np.arange(0, len(mean_test_acc)), np.arange(1, len(mean_test_acc) + 1))
    plt.subplot(2, 1, 2)
    plt.ylabel('Loss')
    plt.plot(mean_test_loss, 'b*', linewidth=1.5)
    plt.xticks(np.arange(0, len(mean_test_acc)), np.arange(1, len(mean_test_acc) + 1))
    plt.xlabel('Participant')
    plt.suptitle('Best Test Score Per Participant')
    plt.show()

# plt.stem(np.arange(1, len(mean_acc) + 1), mean_acc, 'r*', use_line_collection=True)
# plt.show()
