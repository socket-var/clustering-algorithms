import helpers
import sys
import numpy as np 

file_name = sys.argv[1]
no_cluster = int(sys.argv[2])
iterations = int(sys.argv[3])
centroid_val = []

def get_initial_clusters(original_data,number_of_clusters):
    return (np.random.choice(original_data.shape[0], number_of_clusters, replace=False))

def get_data(file_name,number_of_clusters,centroid_arr,number_of_iterations):
    with open(file_name) as textFile:
        lines = [line.replace("\n","").split("\t") for line in textFile]
        data = np.array(lines, dtype='float')
        # print(data)
        if(len(centroid_arr)==0):
            centroid_no=np.asarray(get_initial_clusters(data,no_cluster))
        else:
            centroid_no=np.asarray(centroid_arr)
        print(centroid_no)
        for i in range(len(centroid_no)):
            centroid_val.append(data[centroid_no[i]-1])
        
        print(centroid_val)
        centroids = np.asarray(centroid_val)
        centroids = centroids[:,2:]
        print("Actual Centroid vals")
        print(centroids)
        # print(centroids)

get_data(file_name,no_cluster,[],iterations)


