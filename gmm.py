# 1. import data
import helpers
import numpy as np
import __gmm

file_name = input("Enter the dataset file name: ")

X, y, unique_labels = helpers.import_txt(file_name)

print(X[:5])
print(y[:5])
print(unique_labels)


# 2. Declare any specific inputs to the program and call the algorithm
num_clusters = int(input("Enter the number of Guassians: "))

# 3. Perform DBSCAN
model = __gmm.GMM(X, num_clusters)
predicted = model.fit()

# get the predicted labels
predicted = np.argmax(predicted, axis=1)+1

rand_score = helpers.rand(y, predicted)
jaccard_score = helpers.jaccard(y, predicted)
unique_predicted = list(set(predicted))

print(rand_score)
print(jaccard_score)

# # 5. Visualize using PCA

new_X = helpers.pca(X)
helpers.scatter(new_X[:, 0], new_X[:, 1],
                predicted, unique_predicted)
