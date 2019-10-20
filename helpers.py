import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import jaccard_similarity_score, adjusted_rand_score


def import_txt(filename):
    data = pd.read_csv(filename, sep="\t", header=None)

    data = data.sample(frac=1, random_state=42)

    X = data.iloc[:, 2:].copy()
    y = data.iloc[:, 1].copy()

    unique_labels = list(set(y))

    return X, y, unique_labels

# TODO: implement from scratch


def get_validation(ground_truth, predicted, type):
    if type == "rand":
        return rand(ground_truth, predicted)
    elif type == "jaccard":
        return jaccard(ground_truth, predicted)

    # if type == "rand":
    #     return adjusted_rand_score(ground_truth, predicted)
    # elif type == "jaccard":
    #     return jaccard_similarity_score(ground_truth, predicted)


def rand(y, predicted):

    n = y.shape[0]

    predicted_bools = np.zeros((n, n))
    y_bools = np.zeros((n, n))

    for i in range(n):
        predicted_bools[i][i] = y_bools[i][i] = 1
        for j in range(i+1, n):
            if predicted[i] == predicted[j]:
                predicted_bools[i][j] = predicted_bools[j][i] = 1
            if y[i] == y[j]:
                y_bools[i][j] = y_bools[j][i] = 1

    numerator = np.sum(np.logical_and(
        predicted_bools, y_bools))
    denominator = np.sum(np.logical_or(
        predicted_bools, y_bools))
    return numerator / denominator


def jaccard(y, predicted):

    n = y.shape[0]

    predicted_bools = np.zeros((n, n))
    y_bools = np.zeros((n, n))

    for i in range(n):
        predicted_bools[i][i] = y_bools[i][i] = 1
        for j in range(i+1, n):
            if predicted[i] == predicted[j]:
                predicted_bools[i][j] = predicted_bools[j][i] = 1
            if y[i] != -1 and y[j] != -1 and y[i] == y[j]:
                y_bools[i][j] = y_bools[j][i] = 1

    numerator = np.sum(np.logical_and(
        predicted_bools, y_bools))
    denominator = np.sum(np.logical_or(
        predicted_bools, y_bools))
    return numerator / denominator


def pca(X):
    pca = PCA(n_components=2)
    pca.fit(X)
    new_X = pca.transform(X)

    return new_X


def scatter(X1, X2, labels, unique_labels):
    unique_encoded = [i for i in range(len(unique_labels))]

    colors = [plt.cm.jet(float(i)/max(unique_encoded))
              for i, u in enumerate(unique_labels)]

    for i, u in enumerate(unique_labels):
        xi = [X1[j] for j in range(len(X1)) if labels[j] == u]
        yi = [X2[j] for j in range(len(X2)) if labels[j] == u]
        plt.scatter(xi, yi, c=colors[i], label=str(u))
    plt.legend()

    plt.show()
