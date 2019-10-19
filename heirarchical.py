import numpy as np
import csv

def getEucledianDist(GeneExpressions,index1,index2):
	p1 = np.array(GeneExpressions[index1])
	p2 = np.array(GeneExpressions[index2])
	p3 = np.square(p1-p2)
	return np.sqrt(np.sum(p3))


def getDistMatrix(GeneExpressions):
	distMatrix = []
	for i in range(0,len(GeneExpressions)):
		temp = []
		for j in range(0,len(GeneExpressions)):
			temp.append(0)
		distMatrix.append(temp)
	for i in range(0,len(GeneExpressions)):
		for j in range(i+1,len(GeneExpressions)):
			print("calculating distance for :",i,j)
			dist =  getEucledianDist(GeneExpressions,i,j)
			distMatrix[i][j] = dist
			distMatrix[j][i] = dist
	distMatrix = np.array(distMatrix)
	return distMatrix



filename1 = "cho.txt"
filename2 = "iyer.txt"


GeneExpressions = []
Groundtruth = []
rowsnumbers = []
rownumber = 0
uniquelabels = set()

# Read data from file

with open(filename1) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    for row in csv_reader:
    	line = row[2:]
    	line = list(map(float, line))
    	GeneExpressions.append(line)
    	Groundtruth.append([rownumber,int(row[1])])
    	uniquelabels.add(int(row[1]))
    	rowsnumbers.append(rownumber)
    	rownumber+=1

# Compute Initial distance matrix

distMatrix = getDistMatrix(GeneExpressions)
print(distMatrix)

# print(GeneExpressions[385])
# print(Groundtruth[385])
# print(len(GeneExpressions))
# print(len(Groundtruth))
# print(uniquelabels)
# print(rowsnumbers)



