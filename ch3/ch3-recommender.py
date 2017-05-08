import os
import sys

os.environ["SPARK_HOME"] = "/opt/spark"#"/Users/Karim/src/spark-2.0.0-bin-hadoop2.6"
os.environ["SPARK_DRIVER_MEMORY"] = "6g"
os.environ["PYSPARK_PYTHON"]="/usr/bin/python"
#sys.path.append("/opt/spark/python/")
#sys.path.append("/opt/spark/python/lib/py4j-0.9-src.zip")

from random import randrange
from operator import itemgetter
from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, Rating

def representsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def buildArtistByID(rawArtistData):
    '''
        - convert ther rawArtistData into tuples of (artistID, artistName)
        - filter all bad lines
    '''
    return rawArtistData \
        .map(lambda x: x.split("\t", 1)) \
        .filter(lambda artist: artist[0] and representsInt(artist[0])) \
        .map(lambda artist: (int(artist[0]), artist[1].strip()))

def buildArtistAlias(rawArtistAlias):
    '''
        - convert ther rawArtistData into tuples of (aliasID, artistID)
        - filter all bad lines
    '''
    return rawArtistAlias \
        .map(lambda line: line.split('\t')) \
        .filter(lambda artist: artist[0] and representsInt(artist[0])) \
        .map(lambda artist: (int(artist[0]), int(artist[1]))) \
        .collectAsMap()

def preparation(rawUserArtistData, rawArtistData, rawArtistAlias):
    userIDStats = rawUserArtistData.map(lambda line: long(line.split(' ')[0])).stats()
    itemIDStats = rawUserArtistData.map(lambda line: long(line.split(' ')[1])).stats()
    print(userIDStats)
    print(itemIDStats)

    artistByID = buildArtistByID(rawArtistData)
    artistAlias = buildArtistAlias(rawArtistAlias)

    (badID, goodID) = artistAlias.items()[0]
    print(''.join(artistByID.lookup(badID)) + " -> " + ''.join(artistByID.lookup(goodID)))

def buildRatings(rawUserArtistData, bArtistAlias):
    def getArtistRating(line):
        (userID, artistID, count) = map(lambda x: int(x), line.split(' '))
        try:
            finalArtistID = bArtistAlias.value[artistID]
        except KeyError:
            finalArtistID = artistID
        return Rating(userID, finalArtistID, count)

    return rawUserArtistData.map(lambda line: getArtistRating(line))

def model(sc, rawUserArtistData, rawArtistData, rawArtistAlias):
    bArtistAlias = sc.broadcast(buildArtistAlias(rawArtistAlias))
    trainData = buildRatings(rawUserArtistData, bArtistAlias).cache()
    model = ALS.trainImplicit(ratings=trainData, rank=10, iterations=5, lambda_=0.01, alpha=1.0)

    trainData.unpersist()
    print(model.userFeatures().mapValues(lambda v: ", ".join( map(lambda x: str(x),v) )).first())

    userID = 2093760

    recommendations = model.recommendProducts(userID, 5)
    for val in recommendations:
        print(val)
    recommendedProductIDs = map(lambda rec: rec.product, recommendations)

    #get specific user data
    rawArtistsForUser = rawUserArtistData\
        .map(lambda x: x.split(' '))\
        .filter(lambda x: int(x[0]) == userID)

    #map artist id to int
    existingProducts = rawArtistsForUser.map(lambda x: int(x[1])).collect()

    artistByID = buildArtistByID(rawArtistData)

    existingArtists = artistByID.filter(lambda artist: artist[0] in existingProducts).collect()
    for val in existingArtists:
        print(val)

    recommendedArtists = artistByID.filter(lambda artist: artist[0] in recommendedProductIDs).collect()
    for val in recommendedArtists:
        print(val)

    unpersist(model)

def areaUnderCurve(positiveData, bAllItemIDs, predictFunction):
    positiveUserProducts = positiveData.map(lambda r: (r.user, r.product))
    positivePredictions = predictFunction(positiveUserProducts).groupBy(lambda r: r.user)

    def f2(allItemIDs):
        def f3(rec):
            negative = []
            i = 0
            while i<len(allItemIDs) and len(negative) < len(rec[1]):
                randomIdx = randrange(0,len(allItemIDs))
                itemID = allItemIDs[randomIdx]
                if itemID not in rec[1]:
                    negative.append(itemID)
                i+=1
            return map(lambda itemID: (rec[0],itemID),negative)
        return f3

    def f1(userIDAndPosItemIDs):
        allItemIDs = bAllItemIDs.value
        return map(f2(allItemIDs),userIDAndPosItemIDs)

    negativeUserProducts = positiveUserProducts\
        .groupByKey()\
        .mapPartitions(f1)\
        .flatMap(lambda t: t)

    negativePredictions = predictFunction(negativeUserProducts).groupBy(lambda x: x.user)

    def f4(ratings):
        positiveRatings = ratings[0]
        negativeRatings = ratings[1]
        correct = long(0)
        total = long(0)
        for positive in positiveRatings :
            for negative in negativeRatings :
                if(positive.rating > negative.rating):
                    correct += 1
                total += 1
        return float(correct)/total

    return positivePredictions.join(negativePredictions).values().map(f4).mean()


def areaUnderCurve2(positiveData, bAllItemIDs, predictFunction):

    #
    #
    # def (all, a2):
    #     a2 has n items
    #     all has m >>> n items and a2 included in all
    #     return randome a1, of n random elements no in a2 from all
    #

    def pos2neg(rec):
        allItemIDs = bAllItemIDs.value
        userID = rec[0]
        positiveItems = rec[1]
        negativeItems = []
        i = 0
        while i<len(allItemIDs) and len(negativeItems) < len(positiveItems):
            randomIdx = randrange(0,len(allItemIDs))
            itemID = allItemIDs[randomIdx]
            if itemID not in positiveItems:
                negativeItems.append(itemID)
            i+=1

        return (userID, negativeItems)

    positiveUserProducts = positiveData.map(lambda r: (r.user, r.product))
    groupedPositiveUserProducts = positiveUserProducts.groupByKey()
    groupedNegativeUserProducts = groupedPositiveUserProducts.map(lambda pos :pos2neg(pos))
    negativeUserProducts = groupedNegativeUserProducts.flatMapValues(lambda x:x);


    positivePredictions = predictFunction(positiveUserProducts).groupBy(lambda r: r.user)
    negativePredictions = predictFunction(negativeUserProducts).groupBy(lambda x: x.user)

    posAndNegRatingsJoined =  positivePredictions.join(negativePredictions).values()

    def probabilityOfTruePositive(ratings):
        positiveRatings = ratings[0]
        negativeRatings = ratings[1]
        correct = long(0)
        total = long(0)

        #correct positive must be higher thatn any negative.
        for positive in positiveRatings :
            for negative in negativeRatings :
                if(positive.rating > negative.rating):
                    correct += 1
                total += 1
        return float(correct)/total

    return posAndNegRatingsJoined.map(probabilityOfTruePositive).mean()

def predictMostListened(sc, train):
    def predict(allData):
        listenCount = train.map(lambda r: (r.product, r.rating)).reduceByKey(lambda a, b: a + b).collectAsMap()
        bListenCount = sc.broadcast(listenCount)
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
    mostListenedAUC = areaUnderCurve2(cvData, bAllItemIDs, predictMostListened(sc, trainData))
    print(mostListenedAUC)

    # evaluations = []
    #
    # for rank in [10,50]:
    #     for lambda_val in [1.0, 0.001]:
    #         for alpha in [1.0, 40.0]:
    #             model = ALS.trainImplicit(trainData, rank, 10, lambda_val, alpha)
    #             auc = areaUnderCurve(cvData, bAllItemIDs, model.predict)
    #             unpersist(model)
    #             evaluations.append(((rank, lambda_val, alpha), auc))
    #
    # sorted(evaluations, key=itemgetter(1), reverse=True)
    # for val in evaluations:
    #     print(val)

    trainData.unpersist()
    cvData.unpersist()

def recommend(sc, rawUserArtistData, rawArtistData, rawArtistAlias):
    bArtistAlias = sc.broadcast(buildArtistAlias(rawArtistAlias))
    allData = buildRatings(rawUserArtistData, bArtistAlias).cache()
    model = ALS.trainImplicit(ratings=allData, rank=50, iterations=10, lambda_=1.0, alpha=40.0)

    allData.unpersist()

    userID = 2093760
    recommendations = model.recommendProducts(userID, 5)
    recommendedProductIDs = map(lambda rec: rec.product, recommendations)

    artistByID = buildArtistByID(rawArtistData)

    recommendedArtists = artistByID.filter(lambda artist: artist[0] in recommendedProductIDs).collect()
    for val in recommendedArtists:
        print(val)

    someUsers = allData.map(lambda item: item.user).distinct().take(100)
    someRecommendations = map(lambda userId: model.recommendProducts(userId, 5),someUsers)
    formattedRecommendations = map(lambda recs: str(recs[0].user) + " -> " + ", ".join( map(lambda x: str(x.product), recs) ),someRecommendations)
    for val in formattedRecommendations:
        print(val)

    unpersist(model)

def unpersist(model):
    model.userFeatures().unpersist()
    model.productFeatures().unpersist()

if __name__ == "__main__":
    from pyspark import SparkConf;
    conf = SparkConf();
    conf.set("spark.driver.memory", "6g")
    conf.set("spark.executer.memory", "6g")
    sc = SparkContext(appName="recommender", conf =conf)


    base = "/Users/sameh/Dropbox/Docs/NU/Courses/2016-SPRING_CIT-691-Bigdata-II/cit652/AdvAns-data/ch03/data/"
    rawUserArtistData = sc.textFile(base + "user_artist_data.txt").cache()
    rawArtistData = sc.textFile(base + "artist_data.txt").cache()
    rawArtistAlias = sc.textFile(base + "artist_alias.txt").cache()

    #preparation(rawUserArtistData, rawArtistData, rawArtistAlias)

    model(sc, rawUserArtistData, rawArtistData, rawArtistAlias)

    #evaluate(sc, rawUserArtistData, rawArtistAlias)

    #recommend(sc, rawUserArtistData, rawArtistData, rawArtistAlias)
