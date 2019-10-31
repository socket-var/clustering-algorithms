# 1. import data
import helpers
import numpy as np
import __gmm


def _eval_input(a):
    return eval(a) if a else ""


file_name = input("Enter the dataset file name: ")

X, y, unique_labels = helpers.import_txt(file_name)

print(X[:5])
print(y[:5])
print(unique_labels)


# 2. Declare any specific inputs to the program and call the algorithm
num_clusters = int(input("Enter the number of Guassians: "))
mu = _eval_input(input("Enter mean: "))
sigma = _eval_input(input("Enter covariance: "))
pi = _eval_input(input("Enter pi: "))

convergence_threshold = input("Enter convergence threshold: ")
max_iter = input("Enter maximum number of iterations: ")
smoothing_value = input("Enter a smoothing value: ")
convergence_threshold = _eval_input(
    convergence_threshold)
max_iter = int(max_iter) if max_iter else None
smoothing_value = _eval_input(smoothing_value)

# 3. Perform DBSCAN
if mu and sigma and pi and convergence_threshold and smoothing_value and max_iter:
    model = __gmm.GMM(X, num_clusters, mu, sigma, pi,
                      convergence_threshold, smoothing_value, max_iter)
else:
    model = __gmm.GMM(X, num_clusters, smoothing_value=smoothing_value)
predicted = model.fit()

# get the predicted labels
predicted = np.argmax(predicted, axis=1)+1

rand_score = helpers.rand(y, predicted)
jaccard_score = helpers.jaccard(y, predicted)
unique_predicted = list(set(predicted))

print(rand_score)
print(jaccard_score)

# # 5. Visualize using PCA
new_X = X
if X.shape[1] > 2:
    new_X = helpers.pca(X)
helpers.scatter(new_X[:, 0], new_X[:, 1],
                predicted, unique_predicted)
