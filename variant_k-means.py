
'''
Seth Baer K-Means Python Implementation
This k-means implementation allows for the option to use two different cluster
initialization methods, and two different data regularization methods.

Initialization Methods: 
Random Partition
Minimax

Regularization Methods:
Minimax
Z-Score
'''

from sys import argv
import random
import math
import operator
from datetime import datetime
from math import sqrt

import numpy    #only for ease of testing some specific things

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

def minMaxEQ(x, min, max):
    if x == 0.0 and min == 0.0 and max == 0.0:
        return 0.0
    else:
        return (x - min)/(max - min)

def minMaxNorm(data):
    #get mins and maxes of all attributes
    #create 2d array of x=data, y=attributes
    attribArray = [[data[i][j] for i in range(len(data))] for j in range(len(data[0]))]

    #get mins and maxes of each attribute
    mins = [min(attribArray[i]) for i in range(len(attribArray))]
    maxs = [max(attribArray[i]) for i in range(len(attribArray))]

    #normalize data
    normData = [[0.0 for i in range(len(data[0]))] for j in range(len(data))]

    #see how many unique numbers are in each attribute FOR TESTING
    '''for i in range(len(attribArray)):
        print len(numpy.unique(attribArray[i]))'''

    for i in range(len(data)):
        for j in range(len(data[0])):
            normData[i][j] = minMaxEQ(data[i][j], mins[j], maxs[j])

    return normData

def standardDev(list):
    mean = sum(list) / float(len(list))
    differences = [x - mean for x in list]
    squareDifferences = [d ** 2 for d in differences]
    ssd = sum(squareDifferences)
    variance = ssd / float(len(list))
    return sqrt(variance)

def zScoreEQ(x, mean, std):
    if x == 0.0 and mean == 0.0 and std == 0.0:
        return 0.0
    else:
        return (x - mean)/std

def zScoreNorm(data):
    #get mins and maxes of all attributes
    #create 2d array of x=data, y=attributes
    attribArray = [[data[i][j] for i in range(len(data))] for j in range(len(data[0]))]

    #get means and stds of each attribute
    means = [sum(attribArray[i])/float(len(attribArray[i])) for i in range(len(attribArray))]
    stds = [standardDev(attribArray[i]) for i in range(len(attribArray))]

    #test mean/std with numpy
    '''x = numpy.array(attribArray[2])
    xstd = x.std()
    xmean = x.mean()'''

    #normalize data
    normData = [[0.0 for i in range(len(data[0]))] for j in range(len(data))]

    #see how many unique numbers are in each attribute FOR TESTING
    '''for i in range(len(attribArray)):
        print len(numpy.unique(attribArray[i]))'''

    for i in range(len(data)):
        for j in range(len(data[0])):
            normData[i][j] = zScoreEQ(data[i][j], means[j], stds[j])

    return normData

#returns a numEx by dimension list of ClustDP objects
def populateClustDP(rawData, numEx, dimension):
    ClustDP.dimensionality = dimension

    clustDPList = []
    for i in range(numEx):
        clustDPList.append(ClustDP(rawData[i]))

    return clustDPList

def chooseInitialClusts_randSel(clustData, numEx, numClusts):
    centroidIndexes = random.sample(range(numEx), numClusts)
    centroids = []
    for i in range(numClusts):
        centroids.append(clustData[centroidIndexes[i]].data)
        #print ("Original Data: \t" + str(clustData[centroidIndexes[i]].data) + "\nCentroid i: \t" + str(centroids[i]) + '\n')
    return centroids

def chooseInitialClusts_minmax(clustData, numEx, numClusts):
    centroids = [[float("inf") for i in range(clustData[0].dimensionality)] for j in range(numClusts)]

    #get first centroid as random point in dataset
    temp = random.randint(0, numEx - 1)
    centroids[0] = clustData[temp].data
    #print temp

    #get second centroid as the point furthest away from the first centroid
    dists = [calc2Dist(clustData[i].data, centroids[0]) for i in range(numEx)]
    centroids[1] = clustData[dists.index(max(dists))].data
    #print dists.index(max(dists))

    for i in range(2, numClusts):
        dists = [[calc2Dist(clustData[k].data, centroids[j]) for j in range(i)] for k in range(numEx)]
        minDists = map(min, dists)
        #print minDists.index(max(minDists))
        centroids[i] = clustData[minDists.index(max(minDists))].data

    return centroids

def chooseInitialClusts_randPart(clustData, numEx, numClusts):
    centroids = [[0.0 for i in range(clustData[0].dimensionality)] for j in range(numClusts)]
    centroidCounts = [0 for i in range(numClusts)]

    #give all data points a random cluster id, sum up centroids, and keep track of number of dps in clusters
    for dp in clustData:
        dp.setCluster(random.randint(0, numClusts-1))
        centroids[dp.cluster] = map(operator.add, centroids[dp.cluster], dp.data)
        centroidCounts[dp.cluster] += 1

    #divide sums by number in cluster to get average
    for i in range(len(centroids)):
        for j in range(len(centroids[0])):
            centroids[i][j] = centroids[i][j]/centroidCounts[i]

    return centroids

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
normDecision = argv[7]
initMethod = argv[8]

#create out files
index = fileName.find('.')
outFile_phase2 = open(fileName[0:index] + '_test_phase2.txt', 'w')
pass_phase2 = open('phase2_pass.txt', 'a+')


#read file from argument
rawData, numExamples, dim = readFile(fileName)

#create lists to hold statistics for averaging
initSSE = [0.0 for i in range(numRuns)]
finalSSEs = [float("inf")]*numRuns
numIters = [0 for i in range(numRuns)]

#normalize data
if(normDecision == 'minmax'):
    #min-max normalization
    rawData = minMaxNorm(rawData)
    print "Using: Minimax Normalization"
    print >> outFile_phase2, "Using: Minmax Normalization"
elif(normDecision == 'zscore'):
    #z-score normalization
    rawData = zScoreNorm(rawData)
    print "Using: Z-Score Normalization"
    print >> outFile_phase2, "Using: Z-Score Normalization"

#populate clustDP data structure
clustDPList = populateClustDP(rawData, numExamples, dim)

bestCentroids = [[0.0]*ClustDP.dimensionality for _ in range(numClusts)]
bestSSE = float("inf")
saveInitCentroids = [[0.0]*ClustDP.dimensionality for _ in range(numRuns)]


#TODO: option to run configurations on minibatch also
'''
##########################################
########### MINI BATCH K-MEANS ###########
##########################################
print "*********** BEGIN MINI BATCH ***********"
print >> outFile_phase2, "*********** BEGIN MINI BATCH ***********"

# start minibatch k-means time
startMiniB_time = datetime.now()

#repeat numRuns num of times
for i in range(numRuns):

    print "Run", i+1, "\n", "------"
    print >> outFile_phase2, "Run", i+1, "\n", "------"

    miniBatchInd = random.sample(range(len(clustDPList)), int(round(len(clustDPList) * miniB_size)))
    miniBatch = []
    for j in miniBatchInd:
        miniBatch.append(clustDPList[j])

    # generate initial clusters
    if initMethod == "partition":
        centroids = chooseInitialClusts_randPart(miniBatch, len(miniBatch), numClusts)
    elif initMethod == "selection":
        centroids = chooseInitialClusts_randSel(miniBatch, len(miniBatch), numClusts)

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
        print >> outFile_phase2, "Iteration", iter, ":", "SSE =", totalSSE

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
    print >> outFile_phase2, "\n"


#end minibatch k-means time
endMiniB_time = datetime.now()
miniB_deltaTime = endMiniB_time - startMiniB_time

print "Best Run:", bestRun, "SSE =", bestSSE
print >> outFile_phase2, "Best Run:", bestRun, "SSE =", bestSSE

#calculate SSE for all data using best centroids from minibatch runs
cluster(clustDPList, bestCentroids)
totalDPSSE = computeTotalSSE(clustDPList, bestCentroids)

print "SSE on Full Dataset Using Best Mini Batch ( batch size", miniB_size*100, "% ) Clustering =", totalDPSSE
print >> outFile_phase2, "SSE on Full Dataset Using Best Mini Batch (batch size", miniB_size*100, "% ) Clustering =", totalDPSSE

print "Time elapsed during algorithm:", miniB_deltaTime.total_seconds()
print >> outFile_phase2, "Time elapsed during algorithm:", miniB_deltaTime.total_seconds()

print "*********** END MINI BATCH ***********"
print >> outFile_phase2, "*********** END MINI BATCH ***********"
'''



#######################################
########### REGULAR K-MEANS ###########
#######################################
print "*********** BEGIN REGULAR K-MEANS ***********"
print >> outFile_phase2, "*********** BEGIN REGULAR K-MEANS ***********"

#start k-means time
startKmeans_time = datetime.now()

#repeat numRuns num of times
for i in range(numRuns):
    #start regular kmeans time

    print "Run", i+1, "\n", "------"
    print >> outFile_phase2, "Run", i+1, "\n", "------"

    # generate initial clusters
    if initMethod == "partition":
        centroids = chooseInitialClusts_randPart(clustDPList, len(clustDPList), numClusts)
    elif initMethod == "selection":
        centroids = chooseInitialClusts_randSel(clustDPList, len(clustDPList), numClusts)
    elif initMethod == "minmax":
        centroids = chooseInitialClusts_minmax(clustDPList, len(clustDPList), numClusts)

    saveInitCentroids[i] = centroids


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
        if iter == 1:
            initSSE[i] = totalSSE   #save initial SSE of run
        print "Iteration", iter, ":", "SSE =", totalSSE
        print >> outFile_phase2, "Iteration", iter, ":", "SSE =", totalSSE
        numIters[i] += 1

        if(abs(prevSSE - totalSSE) / prevSSE < convThresh):
            break

        iter += 1
    finalSSEs[i] = totalSSE
    print "\n"
    print >> outFile_phase2, "\n"

#end regular kmeans time
endKmeans_time = datetime.now()
kmeans_deltaTime = endKmeans_time - startKmeans_time

print "Best Run:", finalSSEs.index(min(finalSSEs)) + 1, "SSE =", min(finalSSEs)
print >> outFile_phase2, "Best Run:", finalSSEs.index(min(finalSSEs)) + 1, "SSE =", min(finalSSEs)

print "Time elapsed during algorithm:", kmeans_deltaTime.total_seconds()
print >> outFile_phase2, "Time elapsed during algorithm:", kmeans_deltaTime.total_seconds()

print "*********** END REGULAR K-MEANS ***********"
print >> outFile_phase2, "*********** END REGULAR K-MEANS ***********"

#print summary
print '\n',\
    "*******************************", '\n',\
    "*********** SUMMARY ***********", '\n',\
    "*******************************", '\n'
print "Regular K-Means:\n", "Best SSE:", min(finalSSEs), "\nCPU Time:", kmeans_deltaTime.total_seconds(), "\n\n"

#for minibatch
#print "Mini Batch K-Means:\n", "Best SSE:", totalDPSSE, "\nCPU Time:", miniB_deltaTime.total_seconds(), "\n\n"

#print summary to file
print >> outFile_phase2, '\n',\
    "*******************************", '\n',\
    "*********** SUMMARY ***********", '\n',\
    "*******************************", '\n'
print >> outFile_phase2, "Regular K-Means:\n", "Best SSE:", min(finalSSEs), "\nCPU Time:", kmeans_deltaTime.total_seconds(), "\n\n"

#for minibatch
#print >> outFile_phase2, "Mini Batch K-Means:\n", "Best SSE:", totalDPSSE, "\nCPU Time:", miniB_deltaTime.total_seconds(), "\n\n"

#pass results to runner through file
print >> pass_phase2, min(initSSE), '\n', min(finalSSEs), '\n', min(numIters)
pass_phase2.close()

outFile_phase2.close()

''' ************* '''
'''  END PROGRAM  '''
''' ************* '''