import helpers
import sys
import numpy as np 
from scipy.spatial import distance
from numpy import linalg
from operator import add
import sklearn.metrics

# Returns random clusters as initial clusters
def get_initial_clusters(original_data,num_of_clusters):
    return (np.random.choice(original_data.shape[0], num_of_clusters, replace=False))

# Processes the gene data and gives the initial cluster centers along with gene expressions
def get_data(file_name,num_of_clusters,centroid_arr,num_of_iterations):
    with open(file_name) as textFile:
        lines = [line.replace("\n","").split("\t") for line in textFile]
        data = np.array(lines, dtype='float')
        
        if(centroid_arr==[]):
            centroid_no = np.array(get_initial_clusters(data,no_cluster))
            print(centroid_no)
        else:
            centroid_no = np.array(centroid_arr)
            print(centroid_no)
        
        centroid_val = []
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

    for i in range(0,len(data)):
        gene_data = data[i,2:]
        dist = []
        minval = float('inf') 
        minIndex = -1
        for j in range(0,5):
            if(distance.euclidean(gene_data,centroids[j])<minval):
                minval = distance.euclidean(gene_data,centroids[j])
                minIndex = j
        
        clusters[minIndex].append(gene_data)
        clusters_id[minIndex].append(i+1)
    
    iteration_count+=1
    print("Iteration Count ",iteration_count)
    
    new_centroids(data,centroids,clusters,clusters_id,iterations,no_cluster,iteration_count)

# get new centroids and call k-means until the previous centroids and current centroids converge
def new_centroids(data,centroids,clusters,clusters_id,num_of_iterations,num_of_clusters,iteration_count):
    #new_centroid = [[float(0) for _ in range(centroids.shape[1])] for _ in range(num_of_clusters)]
    new_centroid = []

    for i in range(len(clusters)):
        val = np.array(clusters[i])
        new_centroid.append((np.sum(val,0))/len(clusters[i]))
        
    new_centroid = np.array(new_centroid)
    sums = np.sum(np.array(centroids)-np.array(new_centroid))
    d = dict()
    if(sums==0 or iteration_count==iterations):
        print("Converged")
        for x in range(len(clusters_id)):
            for y in range(len(clusters_id[x])):
                d[clusters_id[x][y]] = x+1
        
        vals = [d[x] for x in sorted(d.keys())]
        ground_truth = list(map(int,data[:,1]))

        print("Jaccard")
        ja = helpers.jaccard(ground_truth,vals)
        print(ja)
        
        print("Rand index")
        rd = helpers.rand(ground_truth,vals)
        print(rd)
        
        unique_predicted = list(set(vals))
        new_x = helpers.pca(data[:,2:])
        helpers.scatter(new_x[:,0],new_x[:,1],vals,unique_predicted)
    else:
        kmeans(data,new_centroid,iterations,no_cluster,iteration_count)

        
file_name = sys.argv[1]
no_cluster = int(sys.argv[2])
iterations = int(sys.argv[3])
centroid_val = []
iteration_count = 0
kmean_count = 0
overall_iterations = 0
max_jaccard = float('-inf')
max_clusters = []

get_data(file_name,no_cluster,[],iterations)



