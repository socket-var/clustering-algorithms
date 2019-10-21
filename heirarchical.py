import numpy as np
import csv
import sys

# Update distance matrix at each iteration
def updateDistMatrix(distMatrix,rowsnumbers):
	row,column = getMin(distMatrix)
	rdim = column
	c1dist = distMatrix[row]
	c2dist = distMatrix[column]
	newdist = []
	for i in range(0,len(c1dist)):
		if(i!=column):
			newdist.append(min(c1dist[i],c2dist[i]))
	#c1 = rowsnumbers[row]
	#c2 = rowsnumbers[column]
	rowsnumbers[row].extend(rowsnumbers[column])
	distMatrix = np.delete(distMatrix,rdim,0)
	distMatrix = np.delete(distMatrix,rdim,1)
	#c1 = np.concatenate((c1,c2),axis=0)
	for i in range(0,len(distMatrix)):
		distMatrix[i][row] = newdist[i]
		distMatrix[row][i] = newdist[i]
	#rowsnumbers[row] = c1
	rowsnumbers.pop(column)
	#rowsnumbers = np.delete(rowsnumbers,column)
	#print(distMatrix)
	#print(rowsnumbers)
	return distMatrix,rowsnumbers

# Gets the minimum from the distance matrix
def getMin(distMatrix):
	gmin = sys.maxsize
	row = 0
	col = 0
	for i in range(0,len(distMatrix)-1):
		for j in range(i+1,len(distMatrix[i])):
			if(distMatrix[i][j] < gmin):
				gmin = distMatrix[i][j]
				row = i
				col = j
	return row,col


# Computes eucledian distances for two points
def getEucledianDist(GeneExpressions,index1,index2):
	p1 = np.array(GeneExpressions[index1])
	p2 = np.array(GeneExpressions[index2])
	p3 = np.square(p1-p2)
	return np.sqrt(np.sum(p3))

# Computes the initial Distance matrix
def getDistMatrix(GeneExpressions):
	distMatrix = []
	for i in range(0,len(GeneExpressions)):
		temp = []
		for j in range(0,len(GeneExpressions)):
			temp.append(0)
		distMatrix.append(temp)
	for i in range(0,len(GeneExpressions)):
		for j in range(i+1,len(GeneExpressions)):
			#print("calculating distance for :",i,j)
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
with open(filename2) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    for row in csv_reader:
    	line = row[2:]
    	line = list(map(float, line))
    	GeneExpressions.append(line)
    	Groundtruth.append([rownumber,int(row[1])])
    	uniquelabels.add(int(row[1]))
    	rowsnumbers.append([rownumber])
    	rownumber+=1

# Get initial distance matrix
distMatrix = getDistMatrix(GeneExpressions)

# Testing data
#distMatrix = [[0.00,0.71,5.66,3.61,4.24,3.20],[0.71,0.00,4.95,2.92,3.54,2.50],[5.66,4.95,0.00,2.24,1.41,2.50],[3.61,2.92,2.24,0.00,1.00,0.50],[4.24,3.54,1.41,1.00,0.00,1.12],[3.20,2.50,2.50,0.50,1.12,0.00]]
#rowsnumbers = [[0],[1],[2],[3],[4],[5]]
distMatrix = np.array(distMatrix)
#print(distMatrix)
while(len(distMatrix)>=2):
	#row,column = getMin(distMatrix)
	print(distMatrix)
	print(rowsnumbers)
	distMatrix,rowsnumbers = updateDistMatrix(distMatrix,rowsnumbers)
	#raise NotImplementedError
#print(row,column)




