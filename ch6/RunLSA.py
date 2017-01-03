import os

os.environ["SPARK_HOME"] = "/Users/Karim/src/spark-2.0.0-bin-hadoop2.6"
os.environ["PYSPARK_PYTHON"] = "/usr/bin/python"

from ParseWikipedia import *
from svd import *
from operator import itemgetter
from stanford_corenlp_pywrapper import CoreNLP
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.mllib.linalg.distributed import RowMatrix


def preprocessing(sampleSize, numTerms, sc):
    pages = readFile('/Users/Karim/Downloads/wiki-nlp/enwiki-latest-pages-articles1.xml', sc).sample(False, sampleSize, 11L)
    plainText = wikiXmlToPlainText(pages)

    stopWords = sc.broadcast(loadStopWords("/Users/Karim/PycharmProjects/AAspark/ch6/stopwords.txt")).value

    def lemmaMapper(itr):
        pipeline = CoreNLP(configdict={'annotators': "tokenize,ssplit,pos,lemma"},
                           corenlp_jars=["./stanford-corenlp-full-2015-04-20/*"])
        return map(lambda tc: (tc[0], plainTextToLemmas(tc[1], stopWords, pipeline)), itr)

    lemmatized = plainText.mapPartitions(lemmaMapper)

    filtered = lemmatized.filter(lambda l: len(l[1]) > 1)

    return documentTermMatrix(filtered, stopWords, numTerms, sc)

def topTermsInTopConcepts(svd, numConcepts, numTerms, termIds):
    v = svd.V
    topTerms = []
    arr = v.toArray()
    for i in range(0, numConcepts):
        offs = i * v.numRows
        termWeights = list(enumerate(arr[offs:offs + v.numRows]))
        termSorted =sorted(termWeights,key=itemgetter(0),reverse=True)
        topTerms.append(map(lambda x: (termIds[x[0]], x[1]) ,termSorted[0:numTerms]))
    return topTerms

def topDocsInTopConcepts(svd, numConcepts, numDocs, docIds):
    u = svd.U
    topDocs = []
    for i in range(0, numConcepts):
        docWeights = u.rows.map(lambda r: r[i]).zipWithUniqueId()
        topDocs.append(map(lambda doc: (docIds[doc[1]], doc[0]), docWeights.top(numDocs)))
    return topDocs

if __name__ == "__main__":
    sc = SparkContext(appName="LSA")
    sqlContext = SQLContext(sc)

    k = 100
    numTerms = 50000
    sampleSize = 0.1

    (termDocMatrix, termIds, docIds, idfs) = preprocessing(sampleSize, numTerms, sqlContext)
    termDocMatrix.cache()

    mat = RowMatrix(termDocMatrix)
    svd = computeSVD(mat, k, computeU=True)

    print("Singular values: " + str(svd.s))
    topConceptTerms = topTermsInTopConcepts(svd, 10, 10, termIds)
    topConceptDocs = topDocsInTopConcepts(svd, 10, 10, docIds)
    for termDoc in zip(topConceptTerms, topConceptDocs):
        print("Concept terms: "+ ", ".join(map(lambda x: x[0], termDoc[0])))
        print("Concept docs: " + ", ".join(map(lambda x: x[0], termDoc[1])))
        print('')