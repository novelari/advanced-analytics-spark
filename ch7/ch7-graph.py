import os, itertools, hashlib, math
import pyspark.sql.functions as func

os.environ["SPARK_HOME"] = "/Users/Karim/src/spark-2.0.0-bin-hadoop2.6"
os.environ["PYSPARK_PYTHON"] = "/usr/bin/python"
os.environ["PYSPARK_SUBMIT_ARGS"] = (
    "--packages graphframes:graphframes:0.3.0-spark2.0-s_2.11 pyspark-shell"
)

from itertools import groupby
from operator import itemgetter
from pyspark import SparkContext
from pyspark.sql import SQLContext

def loadMedline(sc, path):
    df = sc.read.format('com.databricks.spark.xml').options(rowTag='MedlineCitation') \
        .load(path)
    return df

def majorTopics(elem):
    filtered_elem = filter(lambda y: y["_MajorTopicYN"] == "Y", elem["DescriptorName"]) if elem["DescriptorName"] != None else []
    return map(lambda z: z["_VALUE"], filtered_elem)

def hashId(string):
    m = hashlib.md5()
    m.update(string.encode('utf-8'))
    return m.hexdigest()

def sortedConnectedComponents(connectedComponents):
    componentCounts = connectedComponents.rdd.map(lambda x: x[0]).countByValue().items()
    return sorted(componentCounts, key=itemgetter(1), reverse=True)

def topNamesAndDegrees(degrees, topicGraph):
    return degrees.join(topicGraph.vertices).select("topic", "degree").orderBy(func.desc("degree")).limit(10).collect()

def chiSq(YY, YB, YA, T):
    NB = T - YB
    NA = T - YA
    YN = YA - YY
    NY = YB - YY
    NN = T - NY - YN - YY
    inner = abs(YY * NN - YN * NY) - T / 2.0
    return T * math.pow(inner, 2) / (YA * NA * YB * NB)

def avgClusteringCoef(graph):
    triCountGraph = graph.triangleCount()
    maxTrisGraph  = graph.degrees.rdd.mapValues(lambda d: d * (d - 1) / 2.0)
    clusterCoefGraph = triCountGraph.select("id", "count").rdd.join(maxTrisGraph).mapValues(lambda x: 0 if x[1] ==0 else x[0]/x[1])
    return clusterCoefGraph.map(lambda x: x[1]).sum() / graph.vertices.count()

def samplePathLengths(graph, fraction= 0.02):
    replacement = False
    sample = graph.vertices.rdd.map(lambda v: v[0]).sample(replacement, fraction, 1729L)
    ids = sample.collect()
    mapVerticesRDD = graph.vertices.rdd.map(lambda v: (v[0],{v[0]: 0}) if v[0] in ids else (v[0],{}))
    mapVerticesDF = sqlContext.createDataFrame(mapVerticesRDD, ["id", "cnt"])
    mapGraph = GraphFrame(mapVerticesDF, graph.edges)
    start = {}
    #missing pregel in python


if __name__ == "__main__":
    sc = SparkContext(appName="Graph")
    sc.setCheckpointDir("./checkpoint")
    sqlContext = SQLContext(sc)

    medlineRaw = loadMedline(sqlContext, "/Users/Karim/Downloads/medline_data/medsamp2016a.xml")
    medline = medlineRaw.select("MeshHeadingList.MeshHeading.DescriptorName").rdd.map(majorTopics).cache()

    topics= medline.flatMap(lambda mesh: mesh)
    topicCounts = topics.countByValue().items()

    sortedTopicCount = sorted(topicCounts, key=itemgetter(1), reverse=True)
    for val in sortedTopicCount[0:10]:
        print(val)

    valueDist = map(lambda (k, g): (k, len(list(g))), groupby(sortedTopicCount, lambda x: x[1]))
    sortedValueDist = sorted(valueDist, key=itemgetter(1), reverse=True)
    for val in sortedValueDist[0:10]:
        print(val)

    topicPairs = medline.flatMap(lambda t: itertools.combinations(sorted(t), 2) )

    cooccurs = topicPairs.map(lambda p: (p, 1)).reduceByKey(lambda a,b: a + b)
    cooccurs.cache()
    cooccurs.count()
    top10 = cooccurs.sortBy(lambda x: x[1], False).take(10)

    for val in top10:
        print(val)

    vertices = topics.map(lambda topic: (hashId(topic), topic))
    verticesDF = sqlContext.createDataFrame(vertices.distinct(), ["id", "topic"])

    def edgeCreator(p):
        (topics, cnt) = p
        ids = sorted(map(hashId, topics))
        return (ids[0], ids[1], cnt)
    edges = cooccurs.map(edgeCreator)
    edgesDF = sqlContext.createDataFrame(edges, ["src", "dst", "count"])
    from graphframes import *
    topicGraph = GraphFrame(verticesDF, edgesDF)
    topicGraph.cache()

    connectedComponentGraph = topicGraph.connectedComponents()
    componentCounts = sortedConnectedComponents(connectedComponentGraph)
    print(len(componentCounts))
    for val in componentCounts[0:10]:
        print(val)

    hiv = topics.filter(lambda x: "HIV" in x).countByValue().items()
    for val in hiv:
        print(val)

    degrees = topicGraph.degrees.cache()
    degrees.rdd.map(lambda x: x[1]).stats()

    for val in topNamesAndDegrees(degrees, topicGraph):
        print(val)

    T = medline.count()
    topicCountsRdd = topics.map(lambda x: (hashId(x), 1)).reduceByKey(lambda a,b: a+b)
    topicCountsDF = sqlContext.createDataFrame(topicCountsRdd, ["id", "cnt"])
    topicCountGraph = GraphFrame(topicCountsDF, topicGraph.edges)

    chiEdgesRDD = topicCountGraph.triplets.select("edge", "src.cnt", "dst.cnt").rdd.map(lambda x: (x[0]["src"], x[0]["dst"], chiSq(x[0]["count"], x[1], x[2], T)))
    chiEdgesDF = sqlContext.createDataFrame(chiEdgesRDD, ["src", "dst", "chi"])
    chiSquaredGraph = GraphFrame(topicCountsDF, chiEdgesDF)
    chiSquaredGraph.edges.rdd.map(lambda x: x[2]).stats()

    interesting = GraphFrame(topicCountsDF, chiEdgesDF.filter("chi >19.5"))
    interesting.edges.count()

    interestingComponentCounts = sortedConnectedComponents(interesting.connectedComponents())
    print(len(interestingComponentCounts))
    for val in interestingComponentCounts[0:10]:
        print(val)

    interestingDegrees = interesting.degrees.cache()
    interestingDegrees.rdd.map(lambda x: x[1]).stats()

    for val in topNamesAndDegrees(interestingDegrees, topicGraph):
        print(val)

    avgCC = avgClusteringCoef(interesting)

    # paths = samplePathLengths(interesting)
    #
    # paths.map(lambda x: x[2]).filter(lambda x: x > 0).stats()
    #
    # hist = paths.map(lambda x: x[2]).countByValue()
    # for val in sorted(hist):
    #     print(val)
