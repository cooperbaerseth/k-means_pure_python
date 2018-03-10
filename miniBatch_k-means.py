'''
Seth Baer K-Means Python Implementation

This implementation runs the basic k-means algorithm along with the minibatch k-means
algorithm and compares the two.

'''

from sys import argv
import random
import math
import operator
from datetime import datetime

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
            if numInClusts[i] != 0:
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
miniB_size = float(argv[6])

#create out file
index = fileName.find('.')
outFile_miniB = open(fileName[0:index] + '_test_MiniB.txt', 'w')


#read file from argument
rawData, numExamples, dim = readFile(fileName)

#populate clustDP data structure
clustDPList = populateClustDP(rawData, numExamples, dim)

finalSSEs = [float("inf")]*numRuns
bestCentroids = [[0.0]*ClustDP.dimensionality for _ in range(numClusts)]
bestSSE = float("inf")
saveInitCentroids = [[0.0]*ClustDP.dimensionality for _ in range(numRuns)]

##########################################
########### MINI BATCH K-MEANS ###########
##########################################
print "*********** BEGIN MINI BATCH ***********"
print >> outFile_miniB, "*********** BEGIN MINI BATCH ***********"

# start minibatch k-means time
startMiniB_time = datetime.now()

#repeat numRuns num of times
for i in range(numRuns):

    print "Run", i+1, "\n", "------"
    print >> outFile_miniB, "Run", i+1, "\n", "------"

    miniBatchInd = random.sample(range(len(clustDPList)), int(round(len(clustDPList) * miniB_size)))
    miniBatch = []
    for j in miniBatchInd:
        miniBatch.append(clustDPList[j])

    # generate initial clusters
    centroids = chooseInitialClusts(miniBatch, len(miniBatch), numClusts)
    saveInitCentroids[i] = centroids   #save minibatch initial centroids to run regular k-means with

    iter = 1
    prevSSE = 0.0
    totalSSE = float("inf")
    while iter <= maxIters:
        prevSSE = totalSSE

        # compute clustering
        cluster(miniBatch, centroids)

        # recompute centroids
        centroids = recalcCentroids(miniBatch, centroids)

        # compute clustering SSE
        totalSSE = computeTotalSSE(miniBatch, centroids)
        print "Iteration", iter, ":", "SSE =", totalSSE
        print >> outFile_miniB, "Iteration", iter, ":", "SSE =", totalSSE

        if(abs(prevSSE - totalSSE) / prevSSE < convThresh):
            break

        #get new miniBatch
        miniBatchInd = random.sample(range(len(clustDPList)), int(round(len(clustDPList) * miniB_size)))
        miniBatch = []
        for j in miniBatchInd:
            miniBatch.append(clustDPList[j])

        # keep track of centroids with the best total SSE
        if totalSSE < bestSSE or (iter == 1 and i == 1):
            bestSSE = totalSSE
            bestCentroids = centroids
            bestRun = i+1

        iter += 1

    finalSSEs[i] = totalSSE
    print "\n"
    print >> outFile_miniB, "\n"


#end minibatch k-means time
endMiniB_time = datetime.now()
miniB_deltaTime = endMiniB_time - startMiniB_time

print "Best Run:", bestRun, "SSE =", bestSSE
print >> outFile_miniB, "Best Run:", bestRun, "SSE =", bestSSE

#calculate SSE for all data using best centroids from minibatch runs
cluster(clustDPList, bestCentroids)
totalDPSSE = computeTotalSSE(clustDPList, bestCentroids)

print "SSE on Full Dataset Using Best Mini Batch ( batch size", miniB_size*100, "% ) Clustering =", totalDPSSE
print >> outFile_miniB, "SSE on Full Dataset Using Best Mini Batch (batch size", miniB_size*100, "% ) Clustering =", totalDPSSE

print "Time elapsed during algorithm:", miniB_deltaTime.total_seconds()
print >> outFile_miniB, "Time elapsed during algorithm:", miniB_deltaTime.total_seconds()

print "*********** END MINI BATCH ***********"
print >> outFile_miniB, "*********** END MINI BATCH ***********"

#######################################
########### REGULAR K-MEANS ###########
#######################################
print "*********** BEGIN REGULAR K-MEANS ***********"
print >> outFile_miniB, "*********** BEGIN REGULAR K-MEANS ***********"

#start k-means time
startKmeans_time = datetime.now()

#repeat numRuns num of times
for i in range(numRuns):
    #start regular kmeans time

    print "Run", i+1, "\n", "------"
    print >> outFile_miniB, "Run", i+1, "\n", "------"

    # use initial centroids from mini batch
    centroids = saveInitCentroids[i]

    iter = 1
    prevSSE = 0.0
    totalSSE = float("inf")
    while iter <= maxIters:
        prevSSE = totalSSE

        # compute clustering
        # print "before:", clustDPList[34].cluster, clustDPList[84].cluster, clustDPList[134].cluster, clustDPList[200].cluster, "\n"
        cluster(clustDPList, centroids)
        # print "after:", clustDPList[34].cluster, clustDPList[84].cluster, clustDPList[134].cluster, clustDPList[200].cluster

        # recompute centroids
        centroids = recalcCentroids(clustDPList, centroids)

        # compute clustering SSE
        totalSSE = computeTotalSSE(clustDPList, centroids)
        print "Iteration", iter, ":", "SSE =", totalSSE
        print >> outFile_miniB, "Iteration", iter, ":", "SSE =", totalSSE

        if(abs(prevSSE - totalSSE) / prevSSE < convThresh):
            break

        iter += 1
    finalSSEs[i] = totalSSE
    print "\n"
    print >> outFile_miniB, "\n"

#end regular kmeans time
endKmeans_time = datetime.now()
kmeans_deltaTime = endKmeans_time - startKmeans_time

print "Best Run:", finalSSEs.index(min(finalSSEs)) + 1, "SSE =", min(finalSSEs)
print >> outFile_miniB, "Best Run:", finalSSEs.index(min(finalSSEs)) + 1, "SSE =", min(finalSSEs)

print "Time elapsed during algorithm:", kmeans_deltaTime.total_seconds()
print >> outFile_miniB, "Time elapsed during algorithm:", kmeans_deltaTime.total_seconds()

print "*********** END REGULAR K-MEANS ***********"
print >> outFile_miniB, "*********** END REGULAR K-MEANS ***********"

#print summary
print '\n',\
    "*******************************", '\n',\
    "*********** SUMMARY ***********", '\n',\
    "*******************************", '\n'
print "Regular K-Means:\n", "Best SSE:", min(finalSSEs), "\nCPU Time:", kmeans_deltaTime.total_seconds(), "\n\n"
print "Mini Batch K-Means:\n", "Best SSE:", totalDPSSE, "\nCPU Time:", miniB_deltaTime.total_seconds(), "\n\n"

#print summary to file
print >> outFile_miniB, '\n',\
    "*******************************", '\n',\
    "*********** SUMMARY ***********", '\n',\
    "*******************************", '\n'
print >> outFile_miniB, "Regular K-Means:\n", "Best SSE:", min(finalSSEs), "\nCPU Time:", kmeans_deltaTime.total_seconds(), "\n\n"
print >> outFile_miniB, "Mini Batch K-Means:\n", "Best SSE:", totalDPSSE, "\nCPU Time:", miniB_deltaTime.total_seconds(), "\n\n"

outFile_miniB.close()

''' ************* '''
'''  END PROGRAM  '''
''' ************* '''