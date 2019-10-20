import helpers
import sys
import numpy as np 
from scipy.spatial import distance
from operator import add

# Returns random clusters as initial clusters
def get_initial_clusters(original_data,num_of_clusters):
    return (np.random.choice(original_data.shape[0], num_of_clusters, replace=False))

# Processes the gene data and gives the initial cluster centers along with gene expressions
def get_data(file_name,num_of_clusters,centroid_arr,num_of_iterations):
    with open(file_name) as textFile:
        lines = [line.replace("\n","").split("\t") for line in textFile]
        data = np.array(lines, dtype='float')
        centroid_no = np.asarray(get_initial_clusters(data,no_cluster))
        
        for i in range(len(centroid_no)):
            centroid_val.append(data[centroid_no[i]-1])
        
        centroids = np.asarray(centroid_val)
        centroids = centroids[:,2:]
        kmeans(data,centroids,iterations,no_cluster,iteration_count)

# Compute Euclidean distance for all the genes wrt Clusters.
# Get the IDs of genes assigned to the different clusters.
def kmeans(data,centroids,num_of_iterations,num_of_clusters,iteration_count):
    clusters = [[] for _ in range(num_of_clusters)]
    clusters_id = [[] for _ in range(num_of_clusters)]
    
    for i in range(data.shape[0]):
        gene_data = data[i,2:data.shape[1]]
        dist = []
        for j in range(centroids.shape[0]):
            dist.append(distance.euclidean(gene_data,centroids[j]))
        clusters[dist.index(min(dist))].append(gene_data)
        clusters_id[dist.index(min(dist))].append(i+1)
        
    iteration_count+=1
    new_centroids(data,centroids,clusters,clusters_id,num_of_iterations,num_of_clusters,iteration_count)

# get new centroids 
def new_centroids(data,centroids,clusters,clusters_id,num_of_iterations,num_of_clusters,iteration_count):
    
    new_cluster = [[float(0) for _ in range(centroids.shape[1])] for _ in range(num_of_clusters)]
    new_cluster = np.asarray(new_cluster)

    for i in range(len(clusters)):
        for new in clusters[i]:
            new_cluster[i] = list(map(add,new_cluster[i],new))
        for j in range(len(new_cluster[i])):
            new_cluster[i][j] = float(new_cluster[i][j])/len(clusters[i])
    
    print(centroids)
    print("New")
    print(new_cluster)         
        
        
file_name = sys.argv[1]
no_cluster = int(sys.argv[2])
iterations = int(sys.argv[3])
centroid_val = []
iteration_count = 0

get_data(file_name,no_cluster,[],iterations)


