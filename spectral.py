import numpy as np
import csv
import sys
from scipy.spatial import distance
import math
import helpers
import sklearn.cluster
from operator import add

# Computes the Laplacian matrix
def getLaplacianMatrix(simMatrix):
	for i in range(0,len(simMatrix)):
		deg = np.sum(simMatrix[i])
		simMatrix[i] = -1*simMatrix[i]
		simMatrix[i][i] = deg
	return simMatrix

# Computes the similarity matrix
def getSimilarityMatrix(GeneExpressions,sigma):
	simMatrix = []
	for i in range(0,len(GeneExpressions)):
		temp = []
		for j in range(0,len(GeneExpressions)):
			temp.append(0)
		simMatrix.append(temp)
	for i in range(0,len(GeneExpressions)):
		for j in range(i+1,len(GeneExpressions)):
			#print("calculating distance for :",i,j)
			num =  -1*getEucledianDist(GeneExpressions,i,j)
			deno = sigma*sigma
			frac = float(num)/deno
			gaussianval = math.exp(frac)
			simMatrix[i][j] = gaussianval
			simMatrix[j][i] = gaussianval
	simMatrix = np.array(simMatrix)
	return simMatrix

# Computes eucledian distances for two points
def getEucledianDist(GeneExpressions,index1,index2):
	p1 = np.array(GeneExpressions[index1])
	p2 = np.array(GeneExpressions[index2])
	p3 = np.square(p1-p2)
	return np.sqrt(np.sum(p3))

filename = sys.argv[1]
sigma = float(sys.argv[2]) 
k = int(sys.argv[3])
num_clusters = int(sys.argv[4])

GeneExpressions = []
Groundtruth = []
rowsnumbers = []
rownumber = 0
uniquelabels = set()

# Read data from file
with open(filename) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    for row in csv_reader:
    	line = row[2:]
    	line = list(map(float, line))
    	GeneExpressions.append(line)
    	Groundtruth.append(int(row[1]))
    	uniquelabels.add(int(row[1]))
    	rowsnumbers.append([rownumber])
    	rownumber+=1

simMatrix = getSimilarityMatrix(GeneExpressions,sigma)
lapMatrix = getLaplacianMatrix(simMatrix)

# Computes eigen values and eigen vectors for laplacian matrix
eigval,eigvec = np.linalg.eig(lapMatrix)

#print(eigval)
min1 = sys.maxsize
min2 = sys.maxsize
min1Index = -1
min2Index = -1
for i in range(0,len(eigval)):
	if(eigval[i]<min1):
		min2 = min1
		min2Index = min1Index
		min1 = eigval[i]
		min1Index = i
	elif(eigval[i]<min2):
		min2 = eigval[i]
		min2Index = i

print("Minimum values")
print(min1,min2)
print("Minimum indices")
print(min1Index,min2Index)

eigvec1 = eigvec[min1Index]
eigvec2 = eigvec[min2Index]

reducedSpace = []
reducedSpace.append(eigvec1)
reducedSpace.append(eigvec2)
reducedSpace = np.array(reducedSpace)
reducedSpace = np.ndarray.transpose(reducedSpace)
print(reducedSpace)
print(np.shape(reducedSpace))

# kmeans = sklearn.cluster.KMeans(n_clusters=5,random_state=0).fit(reducedSpace)
# print("Library")
# print(kmeans.labels_)
# print(set(kmeans.labels_))
# Run K-means on reduced space n*k to produce k clusters
centroid_val = []
iterations = 100
iteration_count = 0
def get_initial_clusters(original_data,num_of_clusters):
    return (np.random.choice(original_data.shape[0], num_of_clusters, replace=False))

centroid_no = np.asarray(get_initial_clusters(reducedSpace,5))
# print(centroid_no)
for i in range(len(centroid_no)):
    centroid_val.append(reducedSpace[centroid_no[i]-1])

centroids = np.asarray(centroid_val)
# print(centroids)

def new_centroids(reducedSpace,centroids,clusters,clusters_id,num_of_iterations,num_clusters,iteration_count):
    new_centroid = []

    for i in range(len(clusters)):
        val = np.array(clusters[i])
        new_centroid.append((np.sum(val,0))/len(clusters[i]))
        
    new_centroid = np.array(new_centroid)
    sums = np.sum(np.array(centroids)-np.array(new_centroid))
    # sums = 0
    d = dict()
    if(sums==0 or iteration_count==iterations):
        print("Converged")
        for x in range(len(clusters_id)):
            for y in range(len(clusters_id[x])):
                d[clusters_id[x][y]] = x+1
        
        vals = [d[x] for x in sorted(d.keys())]
        vals = np.array(vals)
        print(vals)
        print(set(vals))
        # ground_truth = list(map(int,GeneExpressions[:,1]))

        print("Jaccard")
        ja = helpers.jaccard(Groundtruth,vals)
        print(ja)
        
        print("Rand index")
        rd = helpers.rand(Groundtruth,vals)
        print(rd)

    else:
        kmeans(reducedSpace,new_centroid,iterations,num_clusters,iteration_count)

def kmeans(reducedSpace,centroids,num_of_iterations,num_of_clusters,iteration_count):
    clusters = [[] for _ in range(num_clusters)]
    clusters_id = [[] for _ in range(num_clusters)]
    
    for i in range(reducedSpace.shape[0]):
        gene_data = reducedSpace[i,:]
        dist = []
        minval = float('inf') 
        minIndex = -1
        for j in range(0,num_clusters):
            if(distance.euclidean(gene_data,centroids[j])<minval):
                minval = distance.euclidean(gene_data,centroids[j])
                minIndex = j
        
        clusters[minIndex].append(gene_data)
        clusters_id[minIndex].append(i+1)
            
    iteration_count+=1
    print("Iteration Count ",iteration_count)
    new_centroids(reducedSpace,centroids,clusters,clusters_id,iterations,num_clusters,iteration_count)

kmeans(reducedSpace,centroids,iterations,num_clusters,iteration_count)







