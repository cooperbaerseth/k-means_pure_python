#Seth Baer K-Means Python Implementation

from sys import argv
import random
import math
import operator

class ClustDP:
    dimensionality = 0
    data = []
    cluster = 0
    distToClust = 0.0

    def __init__(self, rawData):
        self.data = rawData

    def setDistToClust(self, dist):
        self.distToClust = dist

    def setCluster(self, c):
        self.cluster = c

def readFile(fileName):
    #open file for reading
    f = open(fileName, "r")

    # get numExamples and dimensionality
    firstLine = map(int, f.readline().split())
    numExamples = firstLine[0]
    dimensionality = firstLine[1]

    # allocate 2d list for data
    data = [[0.0 for i in range(dimensionality)] for j in range(numExamples)]

    for n in range(numExamples):
        row = map(float, f.readline().split())
        data[n] = [row[i] for i in range(len(row))]

    f.close()

    return data, numExamples, dimensionality

#returns a numEx by dimension list of ClustDP objects
def populateClustDP(rawData, numEx, dimension):
    ClustDP.dimensionality = dimension

    clustDPList = []
    for i in range(numEx):
        clustDPList.append(ClustDP(rawData[i]))

    return clustDPList

def chooseInitialClusts(clustData, numEx, numClusts):
    centroidIndexes = random.sample(range(numEx), numClusts)
    centroids = []
    for i in range(numClusts):
        centroids.append(clustData[centroidIndexes[i]].data)
        #print ("Original Data: \t" + str(clustData[centroidIndexes[i]].data) + "\nCentroid i: \t" + str(centroids[i]) + '\n')
    return centroids

    '''for i in range(numClusts):
        for j in range(ClustDP.dimensionality):
            centroids[i][j] = clustData[centroidIndexes[i]].data[j]'''

def calc2Dist(x1, x2):
    return math.sqrt(sum(map(lambda a,b: (a - b)**2, x1, x2)))

def cluster(dataPoints, centroids):
    for i in range(len(dataPoints)):
        distList = [calc2Dist(dataPoints[i].data, centroids[j]) for j in range(len(centroids))]
        dataPoints[i].setCluster(distList.index(min(distList)))
        dataPoints[i].distToClust = min(distList)

        '''for j in range(len(centroids)):              #test distList
            print "DataPoint:\t", dataPoints[i].data, "\n", "Centroid:\t", centroids[j], '\n', "Distance:\t", calc2Dist(dataPoints[i].data, centroids[j]), "\n\n"'''

    #print all clusters numbers
    '''for i in range(len(dataPoints)):
        print i, ":", dataPoints[i].cluster'''

def computeTotalSSE(dataPoints, centroids):
    perClustSSE = [0.0 for i in range(len(centroids))]
    for i in range(len(dataPoints)):
        perClustSSE[dataPoints[i].cluster] += calc2Dist(centroids[dataPoints[i].cluster], dataPoints[i].data)**2
    return sum(perClustSSE)

def recalcCentroids(dataPoints, centroids):
    newCentroids = [ [0.0]*ClustDP.dimensionality for _ in range(len(centroids)) ]
    numInClusts = [0]*len(centroids)

    #add up dimensions of all points in cluster
    for i in range(len(dataPoints)):
        newCentroids[dataPoints[i].cluster] = map(operator.add, newCentroids[dataPoints[i].cluster], dataPoints[i].data)
        numInClusts[dataPoints[i].cluster] += 1

    #debug
    #print "Number in Clusters:", numInClusts

    #divide by number of points in respective cluster
    for i in range(len(numInClusts)):
        for j in range(ClustDP.dimensionality):
            newCentroids[i][j] = newCentroids[i][j] / numInClusts[i]

    #debug
    #print "Old Centroids:", centroids, '\n', "New Centroids:", newCentroids
    #print "Centroid movement:", map(calc2Dist, centroids, newCentroids)
    return newCentroids



''' ************* '''
''' BEGIN PROGRAM '''
''' ************* '''

#get arguments
fileName = argv[1]
numClusts = int(argv[2])
maxIters = int(argv[3])
convThresh = float(argv[4])
numRuns = int(argv[5])

#create out file
index = fileName.find('.')
outFile = open(fileName[0:index] + '_test.txt', 'w')

#read file from argument
rawData, numExamples, dim = readFile(fileName)

#populate clustDP data structure
clustDPList = populateClustDP(rawData, numExamples, dim)

finalSSEs = [0.0]*numRuns

#repeat numRuns num of times
for i in range(numRuns):
    print "Run", i+1, "\n", "------"
    print >> outFile, "Run", i+1, "\n", "------"

    # generate initial clusters
    centroids = chooseInitialClusts(clustDPList, numExamples, numClusts)

    iter = 1
    prevSSE = 0.0
    totalSSE = float("inf")
    while iter < maxIters:
        prevSSE = totalSSE

        # compute clustering
        cluster(clustDPList, centroids)

        # recompute centroids
        centroids = recalcCentroids(clustDPList, centroids)

        # compute clustering SSE
        totalSSE = computeTotalSSE(clustDPList, centroids)
        print "Iteration", iter, ":", "SSE =", totalSSE
        print >> outFile, "Iteration", iter, ":", "SSE =", totalSSE

        if(abs(prevSSE - totalSSE) / prevSSE < convThresh):
            break

        iter += 1
    finalSSEs[i] = totalSSE
    print "\n"
    print >> outFile, "\n"

print "Best Run:", finalSSEs.index(min(finalSSEs)) + 1, "SSE =", min(finalSSEs)
print >> outFile, "Best Run:", finalSSEs.index(min(finalSSEs)) + 1, "SSE =", min(finalSSEs)

outFile.close()

''' ************* '''
'''  END PROGRAM  '''
''' ************* '''