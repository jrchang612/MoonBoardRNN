import numpy as np
import os
import copy
import pickle
import heapq
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

def save_pickle(data, file_name):
    """
    Saves data as pickle format
    """
    with open(file_name, 'wb') as f:
        pickle.dump(data, f)
    return None


def plot_confusion_matrix(Y_true, Y_predict, title = None):
    """
    Plot the confusion matrix.
    """
    labels = ['V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13']
    conf_matrix = metrics.confusion_matrix(Y_true, Y_predict)
    df_cm = pd.DataFrame((conf_matrix/np.sum(conf_matrix, axis = 1, keepdims = True)), 
                         index = [i for i in labels],
                         columns = [i for i in labels])
    plt.figure(dpi = 150, figsize = (10,7))
    sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues)
    plt.xlabel('predicted grade')
    plt.ylabel('actual grade')
    if title:
        plt.title(title)
    plt.show()
    return

def plot_history(history_all, model_name):
    """
    Plot the training history of the model
    """
    # Plot training & validation accuracy values
    acc = []
    val_acc = []
    loss = []
    val_loss = []
    for history in history_all:
        acc.append(history.history['sparse_categorical_accuracy'])
        val_acc.append(history.history['val_sparse_categorical_accuracy'])
        loss.append(history.history['loss'])
        val_loss.append(history.history['val_loss'])

    acc = sum(acc, [])
    val_acc = sum(val_acc, [])
    loss = sum(loss, [])
    val_loss = sum(val_loss, [])

    fig, axes = plt.subplots(nrows = 1, ncols = 2, dpi = 150)
    axes[0].plot(acc)
    axes[0].plot(val_acc)
    axes[0].set_title('Accuracy of '+ model_name)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(['Train', 'Dev'], loc='upper left')

    axes[1].plot(loss)
    axes[1].plot(val_loss)
    axes[1].set_title('Loss of ' + model_name)
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(['Train', 'Dev'], loc='upper left')
    plt.tight_layout()
    
    history_package = {'acc': acc, 
                       'val_acc': val_acc, 
                       'loss': loss, 
                       'val_loss': val_loss}
    return history_package

def plot_history_package(history_package, model_name):
    """
    Plot the training history of the model (from saved history package)
    """
    # Plot training & validation accuracy values
    acc = history_package['acc']
    val_acc = history_package['val_acc']
    loss = history_package['loss']
    val_loss = history_package['val_loss']

    fig, axes = plt.subplots(nrows = 1, ncols = 2, dpi = 150)
    axes[0].plot(acc)
    axes[0].plot(val_acc)
    axes[0].set_title('Accuracy of '+ model_name)
    axes[0].set_ylabel('Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(['Train', 'Dev'], loc='upper left')

    axes[1].plot(loss)
    axes[1].plot(val_loss)
    axes[1].set_title('Loss of ' + model_name)
    axes[1].set_ylabel('Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(['Train', 'Dev'], loc='upper left')
    plt.tight_layout()

    return

def compute_accuracy(y_true, y_predict):
    """
    Compute the accuracy of the model
    - complete_accurate: the model output match the expected output exactly
    - roughly_accurate: the model output is only different from the expected output by less than or equal to 1 grade.
    """
    complete_accurate = np.sum(y_true == y_predict)/len(y_true)
    roughly_accurate = np.sum(np.abs(y_true - y_predict) <= 1)/len(y_true)
    return (complete_accurate, roughly_accurate)