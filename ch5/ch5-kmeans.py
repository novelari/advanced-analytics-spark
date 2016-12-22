import os
import math
os.environ["SPARK_HOME"] = "/Users/Karim/src/spark-2.0.0-bin-hadoop2.6"
os.environ["PYSPARK_PYTHON"] = "/usr/bin/python"

from itertools import groupby
from operator import itemgetter
from pyspark import SparkContext
from pyspark.mllib.linalg import Vectors
from pyspark.ml.clustering import KMeans
from pyspark.mllib.clustering import KMeans

def zipWithIndex(ls):
    indices = range(0, len(ls))
    return zip(ls, indices)

def clusteringTake0(rawData):
    countsByLabel = rawData.map(lambda x: x.split(',').pop()).countByValue().items()
    countSorted = sorted(countsByLabel, key=itemgetter(1), reverse=True)
    for val in countSorted:
        print(val)

    def preprocessing(line):
        values = line.split(",")
        del values[1:4]
        label = values.pop()
        vector = Vectors.dense(map(lambda x: float(x),values))
        return (label, vector)

    labelsAndData = rawData.map(preprocessing)

    data = labelsAndData.values().cache()

    model = KMeans.train(data, 2)

    for centerpoint in model.clusterCenters:
        print(centerpoint)

    clusterLabelCount = labelsAndData.map(lambda x: (model.predict(x[1]), x[0] )).countByValue()

    for labelCount in clusterLabelCount.items():
        print(str(labelCount[0][0]) + " " + str(labelCount[0][1]) + " " + str(labelCount[1]))

    data.unpersist()

def distance(a, b):
    zipped = zip(a,b)
    diff = map(lambda x: x[0] - x[1],zipped)
    squared = map(lambda x: x * x,diff)
    summed = sum(squared)
    return math.sqrt(summed)

def distToCentroid(datum, model):
    cluster = model.predict(datum)
    centroid = model.clusterCenters[cluster]
    return distance(centroid, datum)

def clusteringScore(data, k):
    model = KMeans.train(data, k)
    return data.map(lambda datum: distToCentroid(datum, model)).mean()

def clusteringScore2(data, k):
    model = KMeans.train(data, k, maxIterations=10, epsilon=1.0e-6)
    return data.map(lambda datum: distToCentroid(datum, model)).mean()

def clusteringTake1(rawData):
    def preprocessing(line):
        values = line.split(",")
        del values[1:4]
        values.pop()
        return Vectors.dense(map(lambda x: float(x), values))

    data = rawData.map(preprocessing).cache()

    for k in range(5, 30, 5):
        print((k, clusteringScore(data, k)))

    for k in range(30, 100, 10):
        print((k, clusteringScore2(data, k)))

    data.unpersist()

def visualizationInR(rawData):
    def preprocessing(line):
        values = line.split(",")
        del values[1:4]
        values.pop()
        return Vectors.dense(map(lambda x: float(x), values))

    data = rawData.map(preprocessing).cache()
    model = KMeans.train(data, 100, maxIterations=10, epsilon=1.0e-6)

    sample = data.map(lambda datum: model.predict(datum) + "," + ",".join(datum)).sample(False, fraction=0.05, seed=None)
    sample.saveAsTextFile("file:///user/ds/sample")

def buildNormalizationFunction(data):
    numCols = data.first().size
    n = data.count()
    sums = data.reduce(lambda a,b: map(lambda t: t[0] + t[1], zip(a,b) ))
    sumSquares = data.aggregate([0] * numCols,
                                lambda a,b: map(lambda t: t[0]+ t[1] * t[1], zip(a,b) ),
                                lambda a,b: map(lambda t: t[0]+ t[1], zip(a,b) )
                                )
    stdevs = map(lambda x:  math.sqrt(n * x[0] - x[1] * x[1]) / n , zip(sumSquares, sums) )
    means = map(lambda x: x/n, sums)
    def f(datum):
        zipped = zip(datum, means, stdevs)
        normalizedArray = map(lambda x: (x[0]-x[1]) if x[2]<=0 else (x[0]-x[1])/x[2], zipped)
        return Vectors.dense(normalizedArray)
    return f

def clusteringTake2(rawData):
    def preprocessing(line):
        values = line.split(",")
        del values[1:4]
        values.pop()
        return Vectors.dense(map(lambda x: float(x), values))

    data = rawData.map(preprocessing)

    normalizedData = data.map(buildNormalizationFunction(data)).cache()

    for k in range(60, 120, 10):
        print((k, clusteringScore2(normalizedData, k)))

    normalizedData.unpersist()

def buildCategoricalAndLabelFunction(rawData):
    splitData = rawData.map(lambda line: line.split(','))

    protocolsList = splitData.map(lambda x: x[1]).distinct().collect()
    protocols = dict(zipWithIndex(protocolsList))

    servicesList = splitData.map(lambda x: x[2]).distinct().collect()
    services = dict(zipWithIndex(servicesList))

    tcpStatesList = splitData.map(lambda x: x[3]).distinct().collect()
    tcpStates = dict(zipWithIndex(tcpStatesList))


    def f(line):
        values = line.split(',')
        protocol = values.pop(1)
        service = values.pop(1)
        tcpState = values.pop(1)
        label = values.pop()
        vector = map(lambda x: float(x), values)

        newProtocolFeatures = [0.0] * len(protocols)
        newProtocolFeatures[protocols[protocol]] = 1.0

        newServiceFeatures = [0.0] * len(services)
        newServiceFeatures[services[service]] = 1.0

        newTcpStateFeatures = [0.0] * len(tcpStates)
        newTcpStateFeatures[tcpStates[tcpState]] = 1.0

        vector[1:1] = newTcpStateFeatures
        vector[1:1] = newServiceFeatures
        vector[1:1] = newProtocolFeatures

        return (label, Vectors.dense(vector))

    return f

def clusteringTake3(rawData):
    parseFunction = buildCategoricalAndLabelFunction(rawData)
    data = rawData.map(parseFunction).values()
    normalizedData = data.map(buildNormalizationFunction(data)).cache()

    for k in range(80, 160, 10):
        (k, clusteringScore2(normalizedData, k))

    normalizedData.unpersist()

def entropy(counts):
    values = filter(lambda x: x > 0, counts)
    n = float(sum(values))
    e = map(lambda v: -(v/n) * math.log((v/n)), values)
    return sum(e)

def clusteringScore3(normalizedLabelsAndData, k):
    model = KMeans.train(normalizedLabelsAndData.values(), k, maxIterations=10, epsilon=1.0e-6)
    labelsAndClusters = normalizedLabelsAndData.mapValues(model.predict)
    clustersAndLabels = labelsAndClusters.map(lambda t: (t[1], t[0]))
    labelsInCluster = clustersAndLabels.groupByKey().values()
    labelCounts = labelsInCluster.map(lambda a: map(lambda (k, g): len(list(g)), groupby(a)))
    n = normalizedLabelsAndData.count()
    return labelCounts.map(lambda m: sum(m) * entropy(m)).sum()/n

def clusteringTake4(rawData):
    parseFunction = buildCategoricalAndLabelFunction(rawData)
    labelsAndData = rawData.map(parseFunction)
    normalizedLabelsAndData = labelsAndData.mapValues(buildNormalizationFunction(labelsAndData.values())).cache()

    for k in range(80, 160, 10):
        (k, clusteringScore3(normalizedLabelsAndData, k))

    normalizedLabelsAndData.unpersist()

def buildAnomalyDetector(data, normalizeFunction):
    normalizedData = data.map(normalizeFunction)
    normalizedData.cache()
    model = KMeans.train(normalizedData, 150, maxIterations=10, epsilon=1.0e-6)
    normalizedData.unpersist()
    distances = normalizedData.map(lambda datum: distToCentroid(datum, model))
    threshold = distances.top(100).pop()

    def f(datum):
        return distToCentroid(normalizeFunction(datum), model) > threshold
    return f

def anomalies(rawData):
    parseFunction = buildCategoricalAndLabelFunction(rawData)
    originalAndData = rawData.map(lambda line: (line, parseFunction(line)[1]))
    data = originalAndData.values()
    normalizeFunction = buildNormalizationFunction(data)
    anomalyDetector = buildAnomalyDetector(data, normalizeFunction)
    anomalies = originalAndData.filter(lambda x: anomalyDetector(x[1])).keys()

    for val in anomalies.take(10):
        print(val)


if __name__ == "__main__":
    sc = SparkContext(appName="kmeans")

    rawData = sc.textFile("file:///Users/Karim/Downloads/kddcup.data_10_percent")
    clusteringTake0(rawData)
    clusteringTake1(rawData)
    clusteringTake2(rawData)
    clusteringTake3(rawData)
    clusteringTake4(rawData)
    anomalies(rawData)
