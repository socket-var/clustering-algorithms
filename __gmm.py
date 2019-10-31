from sklearn.cluster import KMeans
import numpy as np
import math
from scipy.stats import multivariate_normal


class GMM:
    def __init__(self, X, num_clusters, mu=None, sigma=None, pi=None, convergence_threshold=10**-1, smoothing_value=10**-15, max_iter=50):
        self.X = X
        self.num_clusters = num_clusters
        self.mu = np.array(mu) if mu else None
        self.sigma = np.array(sigma) if sigma else None
        self.pi = np.array(pi) if pi else None
        self.convergence_threshold = convergence_threshold
        # smoothing value for iyer = 5 * 10**-8 cho = 10**-15
        self.smoothing_value = smoothing_value
        self.max_iter = max_iter
        self.predicted = np.empty((self.X.shape[0], self.num_clusters))

    def fit(self):
        N = self.X.shape[0]
        M = self.X.shape[1]

        if (isinstance(self.mu, np.ndarray) and isinstance(self.sigma, np.ndarray) and isinstance(self.pi, np.ndarray)):
            start = True
        else:
            start = False
            KM = KMeans(n_clusters=self.num_clusters, random_state=42)

            kmeans = KM.fit(self.X)
            labels = kmeans.labels_
            # one-hot encoding
            self.predicted = np.zeros((N, self.num_clusters))
            self.predicted[np.arange(N), labels] = 1

        i = 1
        curr_likelihood = last_likelihood = None

        while (curr_likelihood == None or last_likelihood == None or math.fabs(curr_likelihood - last_likelihood) > self.convergence_threshold) and i < self.max_iter:

            if not start:
                pred_sum = np.sum(self.predicted, axis=0)
                self.pi = pred_sum / N
                self.mu = np.dot(self.X.T, self.predicted) / pred_sum
                self.sigma = np.zeros((self.num_clusters, M, M))

                for k in range(self.num_clusters):
                    for n in range(N):
                        reduced_X = np.reshape(self.X[n]-self.mu[:, k], (M, 1))
                        self.sigma[k] += self.predicted[n, k] * \
                            np.dot(reduced_X, reduced_X.T)
                    self.sigma[k] /= pred_sum[k]

            else:
                start = False

            last_likelihood = curr_likelihood
            curr_likelihood = self.log_likelihood()

            print("Iteration: {} , log likelihood: {}".format(i, curr_likelihood))

            for j in range(N):
                for k in range(self.num_clusters):
                    pdf = multivariate_normal.pdf(
                        self.X[j], mean=self.mu[:, k], cov=self.sigma[k])
                    self.predicted[j, k] = self.pi[k] * pdf
                self.predicted[j] /= np.sum(self.predicted[j])

            i += 1
        print(self.mu, self.sigma, self.pi)
        return self.predicted

    # Calculate negative log likelihood
    def log_likelihood(self):
        N = self.predicted.shape[0]
        self.num_clusters = self.predicted.shape[1]
        loss = 0
        for n in range(N):
            for k in range(self.num_clusters):
                np.fill_diagonal(
                    self.sigma[k], self.sigma[k].diagonal()+self.smoothing_value)
                loss += self.predicted[n, k]*math.log(self.pi[k])
                loss += self.predicted[n, k] * \
                    multivariate_normal.logpdf(
                        self.X[n], mean=self.mu[:, k], cov=self.sigma[k])
        return loss
