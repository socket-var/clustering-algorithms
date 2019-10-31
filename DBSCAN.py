# 1. import data
import helpers
import numpy as np
import __dbscan

file_name = input("Enter the dataset file name: ")

X, y, unique_labels = helpers.import_txt(file_name)

print(X[:5])
print(y[:5])
print(unique_labels)


# 2. Declare any specific inputs to the program and call the algorithm
epsilon = float(input("Enter epsilon value: "))

min_pts = float(input("Enter min_pts value: "))

# 3. Perform DBSCAN
model = __dbscan.DBSCAN(X, epsilon, min_pts)
predicted = model.fit()
unique, counts = np.unique(predicted, return_counts=True)
print("Counts by cluster:")
for key, value in zip(unique, counts):
    print("{}: {}".format(key, value))

# 4. Find Rand index and Jaccard

rand_score = helpers.rand(y, predicted)
jaccard_score = helpers.jaccard(y, predicted)
unique_predicted = list(set(predicted))
print(predicted)
print(rand_score)
print(jaccard_score)

# print(adjusted_rand_score(y, predicted))
# print(jaccard_similarity_score(y, predicted))

# 5. Visualize using PCA
new_X = X
if X.shape[1] > 2:
    new_X = helpers.pca(X)

helpers.scatter(new_X[:, 0], new_X[:, 1],
                predicted, unique_predicted, "DBSCAN", file_name)
