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
			num =  getEucledianDist(GeneExpressions,i,j)
			deno = sigma*sigma
			#print(num,deno)
			frac = num/deno
			gaussianval = math.exp(-1*frac)
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
given_clusters = sys.argv[5]
given_clusters = given_clusters[1:-1].split(',')
if(given_clusters[0] == ''):
	given_clusters = []
if(len(given_clusters) != 0):
	given_clusters = list(map(int,given_clusters))

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
#print(simMatrix)
#print(lapMatrix)
# Computes eigen values and eigen vectors for laplacian matrix
eigval,eigvec = np.linalg.eig(lapMatrix)

#print(eigval)
eigDict = {}

for i in range(len(eigval)):
	eigDict[eigval[i]] = eigvec[i]

# print(eigDict)
sorted_eigval = [x for x in sorted(eigDict.keys())]
corresponding_eigvec = [eigDict[x] for x in sorted(eigDict.keys())]

minimum_eigvals = sorted_eigval[:k]
minimum_eigvec = []
eigList = np.array(corresponding_eigvec[:k])

for i in range(len(eigList)):
	minimum_eigvec.append(np.real(eigList[i]))
	#print(minimum_eigvals[i])
	#print(eigList[i])
#minimum_eigvec = np.array(corresponding_eigvec[:k])
minimum_eigvec = np.array(minimum_eigvec)
reducedSpace = np.ndarray.transpose(minimum_eigvec)
#print(np.shape(reducedSpace))

centroid_val = []
iterations = 100
iteration_count = 0

def get_initial_clusters(original_data,num_clusters):
	return (np.random.choice(original_data.shape[0], num_clusters, replace=False))

if(len(given_clusters)==0):
	centroid_no = np.asarray(get_initial_clusters(reducedSpace,num_clusters))
else:
	centroid_no = np.asarray(given_clusters)

for i in range(len(centroid_no)):
	centroid_val.append(reducedSpace[centroid_no[i]-1])

centroids = np.asarray(centroid_val)

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

		unique_predicted = list(set(vals))
		new_x = helpers.pca(GeneExpressions)
		helpers.scatter(new_x[:,0],new_x[:,1],vals,unique_predicted)

	else:
		kmeans(reducedSpace,new_centroid,iterations,num_clusters,iteration_count)

def kmeans(reducedSpace,centroids,num_of_iterations,num_clusters,iteration_count):
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







