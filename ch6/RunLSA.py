import os

os.environ["SPARK_HOME"] = "/Users/Karim/src/spark-2.0.0-bin-hadoop2.6"
os.environ["PYSPARK_PYTHON"] = "/usr/bin/python"

from ParseWikipedia import *
from pyspark import SparkContext
from pyspark.sql import SQLContext


def preprocessing(sampleSize, numTerms, sc):
    pages = readFile("/Users/Karim/Downloads/enwiki-latest-pages-articles1.xml", sc).sample(False, sampleSize, 11L)
    plainText = wikiXmlToPlainText(pages)


if __name__ == "__main__":
    sc = SparkContext(appName="LSA")
    sqlContext = SQLContext(sc)

    k = 100
    numTerms = 50000
    sampleSize = 0.1

    (termDocMatrix, termIds, docIds, idfs) = preprocessing(sampleSize, numTerms, sqlContext)

    df = sqlContext.read.format('com.databricks.spark.xml').options(rowTag='page').load('/Users/Karim/Downloads/enwiki-latest-pages-articles1.xml')
    c = df.select("title").collect()

    for val in c:
        print(val)
