from sklearn.cluster import KMeans
import numpy as np
import math
from scipy.stats import multivariate_normal


class GMM:
    def __init__(self, X, num_clusters, smoothing_value):
        self.X = X
        self.num_clusters = num_clusters
        # smoothing value for iyer = 5 * 10**-8 cho = 10**-15
        self.smoothing_value = smoothing_value if smoothing_value else 1

    def fit(self):
        N = self.X.shape[0]
        M = self.X.shape[1]

        KM = KMeans(n_clusters=self.num_clusters, random_state=42)

        kmeans = KM.fit(self.X)
        labels = kmeans.labels_
        # one-hot encoding
        self.predicted = np.zeros((N, self.num_clusters))
        self.predicted[np.arange(N), labels] = 1

        i = 1
        curr_likelihood = last_likelihood = None

        while (curr_likelihood == None or last_likelihood == None or math.fabs(curr_likelihood - last_likelihood) > 1e-1) and i < 50:

            pred_sum = np.sum(self.predicted, axis=0)
            pi = pred_sum / N
            mu = np.dot(self.X.T, self.predicted) / pred_sum
            sigma = np.zeros((self.num_clusters, M, M))

            for k in range(self.num_clusters):
                for n in range(N):
                    reduced_X = np.reshape(self.X[n]-mu[:, k], (M, 1))
                    sigma[k] += self.predicted[n, k] * \
                        np.dot(reduced_X, reduced_X.T)
                sigma[k] /= pred_sum[k]

            last_likelihood = curr_likelihood
            curr_likelihood = self.log_likelihood(mu, sigma, pi)

            print("Iteration: {} , log likelihood: {}".format(i, curr_likelihood))

            for j in range(N):
                for k in range(self.num_clusters):
                    pdf = multivariate_normal.pdf(
                        self.X[j], mean=mu[:, k], cov=sigma[k])
                    self.predicted[j, k] = pi[k] * pdf
                self.predicted[j] /= np.sum(self.predicted[j])

            i += 1

        return self.predicted

    # Calculate negative log likelihood
    def log_likelihood(self, mu, sigma, pi):
        N = self.predicted.shape[0]
        self.num_clusters = self.predicted.shape[1]
        loss = 0
        for n in range(N):
            for k in range(self.num_clusters):
                np.fill_diagonal(
                    sigma[k], sigma[k].diagonal()+self.smoothing_value)
                loss += self.predicted[n, k]*math.log(pi[k])
                loss += self.predicted[n, k] * \
                    multivariate_normal.logpdf(
                        self.X[n], mean=mu[:, k], cov=sigma[k])
        return loss
