import helpers
import sys
import numpy as np 
from scipy.spatial import distance

# Returns random clusters as initial clusters
def get_initial_clusters(original_data,num_of_clusters):
    return (np.random.choice(original_data.shape[0], num_of_clusters, replace=False))

# Processes the gene data and gives the initial cluster centers along with gene expressions
def get_data(file_name,num_of_clusters,centroid_arr,num_of_iterations):
    with open(file_name) as textFile:
        lines = [line.replace("\n","").split("\t") for line in textFile]
        data = np.array(lines, dtype='float')
        # print(data)
        centroid_no = np.asarray(get_initial_clusters(data,no_cluster))
        # print(centroid_no)
        for i in range(len(centroid_no)):
            centroid_val.append(data[centroid_no[i]-1])
        
        # print(centroid_val)
        centroids = np.asarray(centroid_val)
        centroids = centroids[:,2:]
        # print("Actual Centroid vals")
        # print(centroids)
        kmeans(data,centroids,iterations,no_cluster,iteration_count)

# Compute Euclidean distance for all the genes wrt Clusters.
# Get the IDs of genes assigned to the different clusters.
def kmeans(data,centroids,num_of_iterations,num_of_clusters,iteration_count):
    clusters = [[] for _ in range(num_of_clusters)]
    clusters_id = [[] for _ in range(num_of_clusters)]
    # print(centroids[0])
    # print(centroids)
    for i in range(data.shape[0]):
        gene_data = data[i,2:data.shape[1]]
        dist = []
        for j in range(centroids.shape[0]):
            dist.append(distance.euclidean(gene_data,centroids[j]))
        clusters[dist.index(min(dist))].append(gene_data)
        clusters_id[dist.index(min(dist))].append(i+1)
        # print(clusters)
    # print(clusters_id)
    iteration_count+=1


file_name = sys.argv[1]
no_cluster = int(sys.argv[2])
iterations = int(sys.argv[3])
centroid_val = []
iteration_count = 0

get_data(file_name,no_cluster,[],iterations)


