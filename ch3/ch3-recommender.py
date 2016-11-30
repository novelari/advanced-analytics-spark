import os

os.environ["SPARK_HOME"] = "/Users/Karim/src/spark-2.0.0-bin-hadoop2.6"
os.environ["PYSPARK_PYTHON"]="/usr/local/bin/python3.5"

import random
from operator import itemgetter
from pyspark import SparkContext
from pyspark.mllib.util import MLUtils
from pyspark.sql.functions import col
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating


def buildArtistByID(rawArtistData):
    return rawArtistData \
        .map(lambda x: x.split("\t", 1)) \
        .filter(lambda artist: artist[0]) \
        .map(lambda artist: (int(artist[0]), artist[1].strip()))


def buildArtistAlias(rawArtistAlias):
    return rawArtistAlias \
        .map(lambda line: line.split('\t')) \
        .filter(lambda artist: artist[0]) \
        .map(lambda artist: (int(artist[0]), int(artist[1]))) \
        .collectAsMap()

def preparation(self, rawUserArtistData, rawArtistData, rawArtistAlias):
    userIDStats = rawUserArtistData.map(lambda line: long(line.split(' ')[0])).stats()
    itemIDStats = rawUserArtistData.map(lambda line: long(line.split(' ')[1])).stats()
    print(userIDStats)
    print(itemIDStats)

    artistByID = buildArtistByID(rawArtistData)
    artistAlias = buildArtistAlias(rawArtistAlias)

    (badID, goodID) = artistAlias.head
    print(artistByID.lookup(badID) + " -> " + artistByID.lookup(goodID))

def buildRatings(rawUserArtistData, bArtistAlias):
    def getArtistID(line):
        (userID, artistID, count) = map(lambda x: int(x), line.split(' '))
        try:
            finalArtistID = bArtistAlias.value[artistID]
        except KeyError:
            finalArtistID = artistID
        return Rating(userID, finalArtistID, count)

    return rawUserArtistData.map(lambda line: getArtistID(line))

def model(sc, rawUserArtistData, rawArtistData, rawArtistAlias):
    bArtistAlias = sc.broadcast(buildArtistAlias(rawArtistAlias))
    trainData = buildRatings(rawUserArtistData, bArtistAlias).cache()

    model = ALS.trainImplicit(trainData, 10, 5, 0.01, 1.0)

    trainData.unpersist()
    print(model.userFeatures.mapValues(lambda v: ", ".join(v)).first())

    userID = 2093760

    recommendations = model.recommendProducts(userID, 5)
    for val in recommendations:
        print(val)
    recommendedProductIDs = map(lambda rec: rec.product, recommendations)


    #get specific user data
    rawArtistsForUser = rawUserArtistData\
        .map(lambda x: x.split(' '))\
        .filter(lambda x: x[0] == userID)

    #map artist id to int
    existingProducts = rawArtistsForUser.map(lambda x: int(x[1])).collect()

    artistByID = buildArtistByID(rawArtistData)

    existingArtists = artistByID.filter(lambda artist: artist.id in existingProducts).collect()
    for val in existingArtists:
        print(val)

    recommendedArtists = artistByID.filter(lambda artist: artist.id in recommendedProductIDs).collect()
    for val in recommendedArtists:
        print(val)

    unpersist(model)

def areaUnderCurve(positiveData, bAllItemIDs, predictFunction):
    positiveUserProducts = positiveData.map(lambda r: (r.user, r.product))
    positivePredictions = predictFunction(positiveUserProducts).groupBy(lambda r: r.user)

    def f2(allItemIDs):
        def f3(userID, posItemIDs):
            negative = []
            i = 0
            while i<len(allItemIDs) and len(negative) < len(posItemIDs):
                randomIdx = int(random.choice(range(0, len(allItemIDs))))
                itemID = allItemIDs[randomIdx]
                if(itemID not in posItemIDs):
                    negative.append(itemID)
                i+=1
            map(lambda itemID: (userID,itemID),negative)
        return f3


    def f1(userIDAndPosItemIDs):
        allItemIDs = bAllItemIDs.value
        map(f2(allItemIDs),userIDAndPosItemIDs)


    negativeUserProducts = positiveUserProducts\
        .groupByKey()\
        .mapParitions(f1)\
        .flatMap(lambda t: t)

    negativePredictions = predictFunction(negativeUserProducts).groupBy(lambda x: x.user)

    def f4(positiveRatings, negativeRatings):
        correct = long(0)
        total = long(0)
        for positive in positiveRatings :
            for negative in negativeRatings :
                if(positive.rating > negative.rating):
                    correct += 1
                total += 1
        return float(correct)/total

    return positivePredictions.join(negativePredictions).map(f4).mean


def predictMostListened(sc, train):
    listenCount = train.map(lambda r: (r.product, r.rating)).reduceByKey(lambda a,b: a + b).collectAsMap()
    bListenCount = sc.broadcast(listenCount)
    def predict(allData):
        def getListenCount(bListenCount, product):
            try:
                count = bListenCount.value[product]
            except KeyError:
                count = 0.0
            return count
        allData.map(lambda data: Rating(data[0], data[1], getListenCount(bListenCount, data[1]) ))
    return predict


def evaluate(sc, rawUserArtistData, rawArtistAlias):
    bArtistAlias = sc.broadcast(buildArtistAlias(rawArtistAlias))
    allData = buildRatings(rawUserArtistData, bArtistAlias)
    (trainData, cvData) = allData.randomSplit(weights=[0.9, 0.1])
    trainData.cache()
    cvData.cache()
    allItemIDs = allData.map(lambda item: item.product).distinct().collect()
    bAllItemIDs = sc.broadcast(allItemIDs)
    mostListenedAUC = areaUnderCurve(cvData, bAllItemIDs, predictMostListened(sc, trainData))
    print(mostListenedAUC)

    evaluations = []

    for rank in [10,50]:
        for lambda_val in [1.0, 0.001]:
            for alpha in [1.0, 40.0]:
                model = ALS.trainImplicit(trainData, rank, 10, lambda_val, alpha)
                auc = areaUnderCurve(cvData, bAllItemIDs, model.predict)
                unpersist(model)
                evaluations.append(((rank, lambda_val, alpha), auc))

    sorted(evaluations, key=itemgetter(1), reverse=True)
    for val in evaluations:
        print(val)

    trainData.unpersist()
    cvData.unpersist()

def recommend(sc, rawUserArtistData, rawArtistData, rawArtistAlias):
    bArtistAlias = sc.broadcast(buildArtistAlias(rawArtistAlias))
    allData = buildRatings(rawUserArtistData, bArtistAlias).cache()
    model = ALS.trainImplicit(allData, 50, 10, 1.0, 40.0)
    allData.unpersist()

    userID = 2093760
    recommendations = model.recommendProducts(userID, 5)
    recommendedProductIDs = map(lambda rec: rec.product, recommendations)

    artistByID = buildArtistByID(rawArtistData)

    recommendedArtists = artistByID.filter(lambda artist: artist.id in recommendedProductIDs).collect()
    for val in recommendedArtists:
        print(val)

    someUsers = allData.map(lambda item: item.user).distinct().take(100)
    someRecommendations = map(lambda userId: model.recommendProducts(id, 5),someUsers)
    formattedRecommendations = map(lambda recs: str(recs.head.user) + " -> " + ", ".join( map(lambda x: x.product) ),someRecommendations)
    for val in formattedRecommendations:
        print(val)

    unpersist(model)

def unpersist(model):
    model.userFeatures.unpersist()
    model.productFeatures.unpersist()

if __name__ == "__main__":
    sc = SparkContext(appName="PythonWordCount")

    base = "file:///Users/Karim/Downloads/profiledata_06-May-2005/"
    rawUserArtistData = sc.textFile(base + "user_artist_data.txt")
    rawArtistData = sc.textFile(base + "artist_data.txt")
    rawArtistAlias = sc.textFile(base + "artist_alias.txt")

    preparation(rawUserArtistData, rawArtistData, rawArtistAlias)
    model(sc, rawUserArtistData, rawArtistData, rawArtistAlias)
    evaluate(sc, rawUserArtistData, rawArtistAlias)
    recommend(sc, rawUserArtistData, rawArtistData, rawArtistAlias)