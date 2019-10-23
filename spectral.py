import numpy as np
import csv
import sys
import math

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
    	Groundtruth.append([rownumber,int(row[1])])
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







