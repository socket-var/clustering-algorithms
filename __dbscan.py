import numpy as np


class DBSCAN:
    def __init__(self, X, eps, min_pts):
        self.X = X
        self.eps = eps
        self.min_pts = min_pts
        self.predicted = np.zeros(self.X.shape[0], dtype="int")

    def fit(self):

        cluster_id = 0

        for point_idx, point in enumerate(self.X):
            if self.predicted[point_idx] == 0:  # already visited
                neighbor_points = self._regionQuery(point)
                if len(neighbor_points) < self.min_pts:
                    self.predicted[point_idx] = -1  # assign point to noise
                else:
                    cluster_id += 1
                    self.predicted[point_idx] = cluster_id
                    self._expand_cluster(neighbor_points,
                                         cluster_id)

        return self.predicted

    def _expand_cluster(self, neighbor_points, cluster_id):
        for point_idx in neighbor_points:

            if self.predicted[point_idx] == -1:
                self.predicted[point_idx] = cluster_id  # border points

            if self.predicted[point_idx] == 0:
                self.predicted[point_idx] = cluster_id
                neighbors = self._regionQuery(self.X[point_idx])
                if len(neighbors) >= self.min_pts:
                    neighbor_points += neighbors

    def _regionQuery(self, point):

        neighbor_points = []

        for idx, neighbor in enumerate(self.X):
            if np.linalg.norm(point - neighbor) <= self.eps:
                neighbor_points.append(idx)

        return neighbor_points
