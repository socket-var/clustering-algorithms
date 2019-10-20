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

    for idx in range(X.shape[0]):
        if predicted[idx] == 0:
            neighbor_points = region_query(X, idx, epsilon)
            if len(neighbor_points) >= min_pts:
                cluster += 1
                predicted = expand_cluster(X, idx, neighbor_points,
                                           cluster, epsilon, min_pts, predicted)
            else:
                predicted[idx] = -1

    unique, counts = np.unique(predicted, return_counts=True)

    unique = [int(i) for i in unique]
    print("Cluster: count = " + str(dict(zip(unique, counts))))

    centroids = {}
    for x in range(int(np.min(predicted)), int(np.max(predicted)) + 1):
        if x == 0:
            continue
        centroids[x] = np.asarray(np.where(predicted == x))+1

    # Printing Centroids
    print("Cluster: points in cluster = ")
    for key, value in centroids.items():
        print(str(key) + ":" + str(value))

    return predicted


def expand_cluster(X, pt_idx, neighbor_points, cluster, epsilon, min_pts, predicted):
    predicted[pt_idx] = cluster
    for neighbor in neighbor_points:
        if predicted[neighbor] == -1 or predicted[neighbor] == 0:
            predicted[neighbor] = cluster

        if predicted[neighbor] == 0:
            neighbor_pts = region_query(X, neighbor, epsilon)

            if len(neighbor_pts) >= min_pts:
                neighbor_points += neighbor_pts

    return predicted


def region_query(X, point_idx, epsilon):
    neighbor_points = []

    for idx in range(X.shape[0]):

        distance = np.linalg.norm(X[point_idx] - X[idx])
        if distance <= epsilon:
            neighbor_points.append(idx)

    return neighbor_points


# 3. Declare any specific inputs to the program and call the algorithm
epsilon = float(input("Enter epsilon value: "))

min_pts = float(input("Enter min_pts value: "))

predicted = dbscan(np.array(X), epsilon, min_pts)


# 4. Find Rand index and Jaccard

rand_score = helpers.get_validation(y, predicted, type="rand")
jaccard_score = helpers.get_validation(y, predicted, type="jaccard")

# print(predicted)
print(rand_score)
print(jaccard_score)

# 5. Visualize using PCA

new_X = helpers.pca(X)
helpers.scatter(new_X[:, 0], new_X[:, 1],
                predicted, unique_labels)
