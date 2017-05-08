import os

os.environ["SPARK_HOME"] = "/Users/Karim/src/spark-2.0.0-bin-hadoop2.6"
os.environ["PYSPARK_PYTHON"]="/usr/bin/python"

from operator import itemgetter
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.mllib.tree import RandomForest, DecisionTree

def simpleDecisionTree(trainData, cvData):
    model = DecisionTree.trainClassifier(trainData,numClasses=7, categoricalFeaturesInfo={}, impurity="gini", maxDepth=4, maxBins=100)
    metrics = getMetrics(model, cvData)

    print(metrics.confusionMatrix())
    print(metrics.precision())

    for category in range(0,7):
        print((metrics.precision(category), metrics.recall(category)))


def getMetrics(model, data):
    labels = data.map(lambda d: d.label)
    features = data.map(lambda d: d.features)
    predictions = model.predict(features)
    predictionsAndLabels = predictions.zip(labels)
    return MulticlassMetrics(predictionsAndLabels)

def randomClassifier(trainData, cvData):
    trainPriorProbabilities = classProbabilities(trainData)
    cvPriorProbabilities = classProbabilities(cvData)
    zipProbabilities = zip(trainPriorProbabilities, cvPriorProbabilities)
    productProbabilities = map(lambda x: x[0]*x[1],zipProbabilities)
    accuracy = sum(productProbabilities)
    print(accuracy)

def classProbabilities(data):
    countsByCategory = data.map(lambda x: x.label).countByValue().items()
    countSorted = sorted(countsByCategory, key=itemgetter(0), reverse=False)
    count = map(lambda x: x[1], countSorted)
    return map(lambda x: float(x)/sum(count), count)

def evaluate(trainData, cvData, testData):
    evaluations = []
    for impurity in ["gini", "entropy"]:
        for depth in [1, 20]:
            for bins in [10, 300]:
                model = DecisionTree.trainClassifier(trainData,numClasses=7, categoricalFeaturesInfo={}, impurity=impurity, maxDepth=depth, maxBins=bins)
                accuracy = getMetrics(model, cvData).precision()
                evaluations.append(((impurity, depth, bins), accuracy))

    sorted(evaluations, key=itemgetter(1), reverse=True)
    for val in evaluations:
        print(val)

    model = DecisionTree.trainClassifier(trainData.union(cvData),numClasses=7, categoricalFeaturesInfo={}, impurity="entropy", maxDepth=20, maxBins=300)
    print(getMetrics(model, testData).precision())
    print(getMetrics(model, trainData.union(cvData)).precision())

def unencodeOneHot(line):
    values = map(lambda x: float(x), line.split(","))
    wilderness = float(values[10:14].index(1.0))
    soil = float(values[14:54].index(1.0))
    featureVector = Vectors.dense(values[0:10] + [wilderness] + [soil])
    label = values.pop() - 1
    return LabeledPoint(label, featureVector)

def evaluateCategorical(rawData):
    data = rawData.map(unencodeOneHot)
    (trainData, cvData, testData) = data.randomSplit(weights=[0.8, 0.1, 0.1])

    trainData.cache()
    cvData.cache()
    testData.cache()

    evaluations = []
    for impurity in ["gini", "entropy"]:
        for depth in [10, 20, 30]:
            for bins in [40, 300]:
                model = DecisionTree.trainClassifier(trainData, numClasses=7, categoricalFeaturesInfo={10:4, 11: 40}, impurity=impurity, maxDepth=depth, maxBins=bins)
                trainAccuracy = getMetrics(model, trainData).precision()
                cvAccuracy = getMetrics(model, cvData).precision()
                evaluations.append(((impurity, depth, bins), (trainAccuracy, cvAccuracy)))

    sorted(evaluations, key=itemgetter(1,1), reverse=True)
    for val in evaluations:
        print(val)

    model = DecisionTree.trainClassifier(trainData, numClasses=7, categoricalFeaturesInfo={10: 4, 11: 40}, impurity="entropy", maxDepth=30, maxBins=300)
    print(getMetrics(model, testData).precision())

    trainData.unpersist()
    cvData.unpersist()
    testData.unpersist()

def evaluateForest(rawData):
    data = rawData.map(unencodeOneHot)
    (trainData, cvData) = data.randomSplit(weights=[0.9, 0.1])

    trainData.cache()
    cvData.cache()

    forest = RandomForest.trainClassifier(trainData, numClasses=7, categoricalFeaturesInfo={10: 4, 11: 40}, numTrees=20, featureSubsetStrategy="auto", impurity="entropy", maxDepth=30, maxBins=300)

    metrics = getMetrics(forest, cvData)

    print(metrics.precision())

    input = "2709,125,28,67,23,3224,253,207,61,6094,0,29"
    vector = Vectors.dense(map(lambda x: float(x), input.split(",")))
    print(forest.predict(vector))

    trainData.unpersist()
    cvData.unpersist()

if __name__ == "__main__":
    sc = SparkContext(appName="rdf")

    rawData = sc.textFile("file:///Users/Karim/Downloads/covtype.data")

    def preprocessing(line):
        values = map(lambda x: float(x), line.split(","))
        last_el = values.pop()
        featureVector = Vectors.dense(values)
        label = last_el - 1
        return LabeledPoint(label, featureVector)

    data = rawData.map(preprocessing)

    (trainData, cvData, testData) = data.randomSplit(weights=[0.8, 0.1, 0.1])

    trainData.cache()
    cvData.cache()
    testData.cache()

    simpleDecisionTree(trainData, cvData)
    randomClassifier(trainData, cvData)
    evaluate(trainData, cvData, testData)
    evaluateCategorical(rawData)
    evaluateForest(rawData)

    trainData.unpersist()
    cvData.unpersist()
    testData.unpersist()