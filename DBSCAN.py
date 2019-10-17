# 1. import data
import helpers
import numpy as np

file_name = input("Enter the dataset file name: ")

X, y, unique_labels = helpers.import_txt(file_name)

print(X.head())
print(y.head())
print(unique_labels)

# 2. Write functions for the algorithm
# perform DBSCAN


def dbscan(X, epsilon, min_pts):
    predicted = np.zeros(X.shape[0])
    cluster = 0

    for idx, point in enumerate(X.T):
        if not predicted[idx]:
            neighbor_points = region_query(X, point, epsilon)
            if len(neighbor_points) < min_pts:
                predicted[idx] = -1
            else:
                cluster += 1
                predicted[idx] = cluster
                neighbor_points, predicted = expand_cluster(X, idx, point, neighbor_points,
                                                            cluster, epsilon, min_pts, predicted)
    return predicted


def expand_cluster(X, point_idx, point, neighbor_points, cluster, epsilon, min_pts, predicted):

    for neighbor in neighbor_points:
        if not predicted[point_idx]:
            neighbors = region_query(X, point, epsilon)
            if len(neighbors) >= min_pts:
                neighbor_points.append(neighbor)

        predicted[point_idx] = cluster

    return neighbor_points, predicted


def region_query(X, point, epsilon):
    neighbor_points = []

    for candidate_point in X:
        distance = np.linalg.norm(point-candidate_point)
        if distance <= epsilon and point != neighbor_points:
            neighbor_points.append(candidate_point)

    return neighbor_points


# 3. Declare any specific inputs to the program and call the algorithm
epsilon = float(input("Enter epsilon value: "))

min_pts = float(input("Enter min_pts value: "))

predicted = dbscan(X, epsilon, min_pts)


# 4. Find Rand index and Jaccard

rand_score = helpers.get_validation(y, predicted, type="rand")
jaccard_score = helpers.get_validation(y, predicted, type="jaccard")

print(predicted)
print(rand_score)
print(jaccard_score)

# 5. Visualize using PCA

new_X = helpers.pca(X)
helpers.scatter(new_X[:, 0], new_X[:, 1],
                predicted, unique_labels)
