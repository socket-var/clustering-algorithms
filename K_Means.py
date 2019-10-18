import helpers
import sys
import numpy as np 

file_name = sys.argv[1]
no_cluster = int(sys.argv[2])
iterations = int(sys.argv[3])

X,y,unique_labels = helpers.import_txt(file_name)

def get_data(filename, num_of_clusters, centroidsArr, num_of_iterations):
    centroid_array = np.array(get_initial_clusters(y,num_of_clusters))

def get_initial_clusters(original_data,number_of_clusters):
    return (np.random.choice(y.shape[0], number_of_clusters, replace=False))

get_data(file_name, no_cluster, centroid_id, iterations)






# print(X)
# print(y)
# print(unique_labels)


