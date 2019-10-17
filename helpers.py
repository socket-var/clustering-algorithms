import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import jaccard_similarity_score, adjusted_rand_score


def import_txt(filename):
    data = pd.read_csv(filename, sep="\t", header=None)

    data = data.sample(frac=1, random_state=42)

    X = data.iloc[:, 2:]
    y = data.iloc[:, 1]

    unique_labels = list(set(y))

    return X, y, unique_labels

# TODO: implement from scratch


def get_validation(ground_truth, predicted, type):
    if type == "rand":
        return jaccard_similarity_score(ground_truth, predicted)
    elif type == "jaccard":
        return adjusted_rand_score(ground_truth, predicted)


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
