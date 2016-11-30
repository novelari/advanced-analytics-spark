import os

os.environ["SPARK_HOME"] = "/Users/Karim/src/spark-2.0.0-bin-hadoop2.6"
os.environ["PYSPARK_PYTHON"]="/usr/local/bin/python3.5"

import random
from pyspark import SparkContext
from pyspark.mllib.util import MLUtils
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating



class RunRecommender(object):
    def __init__(self, sc):
        self.sc = sc

    def preparation(self, rawUserArtistData, rawArtistData, rawArtistAlias):
        userArtistDF = rawUserArtistData\
            .map(lambda line: line.split(' '))\
            .map(lambda list: (list[0], list[1]))\
            .toDF(("user", "artist"))

        userArtistDF.agg({"user":"min", "user": "max", "artist":"min", "artist": "max"}).show()

        artistByID = self.buildArtistByID(rawArtistData)

        artistAlias = self.buildArtistAlias(rawArtistAlias)

        (badID, goodID) = artistAlias.head
        # artistByID.filter($"id" isin(badID, goodID)).show()

    def buildArtistByID(rawArtistData):
        return rawArtistData\
            .map(lambda x: x.split("\t",1))\
            .filter(lambda artist: artist[0])\
            .map(lambda artist: (int(artist[0]), artist[1].strip()))\
            .toDF(("id", "name"))

    def buildArtistAlias(rawArtistAlias):
        output = rawArtistAlias\
            .map(lambda line: line.split('\t') ) \
            .filter(lambda artist: artist[0])\
            .map(lambda artist: (int(artist[0]), int(artist[1])))\
            .collect()
        return dict(output)

    def buildCounts(rawUserArtistData, bArtistAlias):
        def getArtistID(line):
            (userID, artistID, count) = map(lambda x: int(x), line.split(' '))
            try:
                finalArtistID = bArtistAlias.value[artistID]
            except KeyError:
                finalArtistID = artistID
            return (userID, finalArtistID, count)
        return rawUserArtistData.map(lambda line: getArtistID(line)).toDF(("user", "artist", "count"))




if __name__ == "__main__":
    sc = SparkContext(appName="PythonWordCount")

    base = "file:///Users/Karim/Downloads/profiledata_06-May-2005/"
    rawUserArtistData = sc.textFile(base + "user_artist_data.txt")
    rawArtistData = sc.textFile(base + "artist_data.txt")
    rawArtistAlias = sc.textFile(base + "artist_alias.txt")

    runRecommender = RunRecommender(sc)
    runRecommender.preparation(rawUserArtistData, rawArtistData, rawArtistAlias)
    runRecommender.model(rawUserArtistData, rawArtistData, rawArtistAlias)
    runRecommender.evaluate(rawUserArtistData, rawArtistAlias)
    runRecommender.recommend(rawUserArtistData, rawArtistData, rawArtistAlias)

    sc.stop()
