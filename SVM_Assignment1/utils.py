import numpy as np
import pandas as pd
from typing import Tuple
from matplotlib import pyplot as plt


def get_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # load the data
    train_df = pd.read_csv('data/mnist_train.csv')
    test_df = pd.read_csv('data/mnist_test.csv')

    X_train = train_df.drop('label', axis=1).values
    y_train = train_df['label'].values

    X_test = test_df.drop('label', axis=1).values
    y_test = test_df['label'].values

    return X_train, X_test, y_train, y_test


def normalize(X_train, X_test) -> Tuple[np.ndarray, np.ndarray]:
    # normalize the data
    # Normalize data pixels in between -1 to 1
    X_train_norm, X_test_norm = ((X_train/255)-0.5)*2, ((X_test/255)-0.5)*2
    return X_train_norm, X_test_norm
    # raise NotImplementedError


def plot_metrics(metrics) -> None:
    k_values = [m[0] for m in metrics]
    accuracy_values = [m[1] for m in metrics]
    precision_values = [m[2] for m in metrics]
    recall_values = [m[3] for m in metrics]
    f1_score_values = [m[4] for m in metrics]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # plot accuracy
    axs[0, 0].plot(k_values, accuracy_values, marker='o')
    axs[0, 0].set_title('Accuracy vs. Number of Principal Components')
    axs[0, 0].set_xlabel('Number of Principal Components')
    axs[0, 0].set_ylabel('Accuracy')

    # plot precision
    axs[0, 1].plot(k_values, precision_values, marker='o')
    axs[0, 1].set_title('Precision vs. Number of Principal Components')
    axs[0, 1].set_xlabel('Number of Principal Components')
    axs[0, 1].set_ylabel('Precision')

    # plot recall
    axs[1, 0].plot(k_values, recall_values, marker='o')
    axs[1, 0].set_title('Recall vs. Number of Principal Components')
    axs[1, 0].set_xlabel('Number of Principal Components')
    axs[1, 0].set_ylabel('Recall')

    # plot f1 score
    axs[1, 1].plot(k_values, f1_score_values, marker='o')
    axs[1, 1].set_title('F1 Score vs. Number of Principal Components')
    axs[1, 1].set_xlabel('Number of Principal Components')
    axs[1, 1].set_ylabel('F1 Score')

    plt.tight_layout()

    # save the plots
    plt.savefig('performance_metrics.png')
