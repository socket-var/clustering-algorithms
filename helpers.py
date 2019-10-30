import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def import_txt(filename):
    data = pd.read_csv(filename, sep="\t", header=None)

    data = data.sample(frac=1, random_state=42)

    X = data.iloc[:, 2:]
    y = data.iloc[:, 1]

    unique_labels = list(set(y))

    return np.array(X), np.array(y), unique_labels

# TODO: implement from scratch


def _make_incidence_matrix(labels):
    incidence_matrix = np.zeros((len(labels), len(labels)))

    for i in range(len(labels)):
        for j in range(len(labels)):
            incidence_matrix[i][j] = 1 if labels[i] == labels[j] else 0
    return incidence_matrix


def _get_confusion_matrix(gt_matrix, p_matrix):
    TT = TF = FT = FF = 0

    for i in range(len(p_matrix)):
        for j in range(len(gt_matrix)):

            if p_matrix[i][j] != gt_matrix[i][j]:
                if p_matrix[i][j] == 1:
                    TF += 1
                else:
                    FT += 1
            else:
                if p_matrix[i][j] == 1:
                    TT += 1
                else:
                    FF += 1

    return TT, TF, FT, FF


def jaccard(ground_truth, predicted):
    gt_matrix = _make_incidence_matrix(ground_truth)
    p_matrix = _make_incidence_matrix(predicted)

    TT, TF, FT, _ = _get_confusion_matrix(gt_matrix, p_matrix)
    return float(TT) / (TT + TF + FT)


def rand(ground_truth, predicted):
    gt_matrix = _make_incidence_matrix(ground_truth)
    p_matrix = _make_incidence_matrix(predicted)

    TT, TF, FT, FF = _get_confusion_matrix(gt_matrix, p_matrix)
    return float(TT + FF) / (TT + TF + FT + FF)


def pca(X):
    pca = PCA(n_components=2)
    pca.fit(X)
    new_X = pca.transform(X)

    return new_X


def scatter(X1, X2, labels, unique_labels, algo="", file_name=""):
    plt.figure(figsize=[17, 12])
    if algo and file_name:
        plt.title("Algorithm: {} Dataset: {}".format(algo, file_name))
    unique_encoded = [i for i in range(len(unique_labels))]

    colors = [plt.cm.jet(float(i)/max(unique_encoded))
              for i, u in enumerate(unique_labels)]

    for i, u in enumerate(unique_labels):
        xi = [X1[j] for j in range(len(X1)) if labels[j] == u]
        yi = [X2[j] for j in range(len(X2)) if labels[j] == u]
        plt.scatter(xi, yi, c=colors[i], label=str(u), s=35)

    plt.legend()

    plt.show()
