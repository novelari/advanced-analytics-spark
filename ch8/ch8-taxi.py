import os
import json

os.environ["SPARK_HOME"] = "/Users/Karim/src/spark-2.0.0-bin-hadoop2.6"
os.environ["PYSPARK_PYTHON"]="/usr/bin/python"

from datetime import datetime
from operator import itemgetter
from itertools import chain, imap
from shapely.geometry import shape, Point

from pyspark import SparkContext
from pyspark.rdd import portable_hash
from pyspark.statcounter import StatCounter

epoch = datetime.utcfromtimestamp(0)

def flatmap(f, items):
    return chain.from_iterable(imap(f, items))

def sliding(num, l):
    return [l[i:i + num] for i in xrange(len(l) - (num-1))]

def getMillis(time):
    return (time - epoch).total_seconds() * 1000.0

def parse(fields):
    license = fields[1]
    pickupTime = datetime.strptime(fields[5], '%Y-%m-%d %H:%M:%S')
    dropoffTime = datetime.strptime(fields[6], '%Y-%m-%d %H:%M:%S')
    try:
        pickupLoc = Point(float(fields[10]), float(fields[11]))
        dropoffLoc = Point(float(fields[12]), float(fields[13]))
    except ValueError:
        pickupLoc = Point(0.0, 0.0)
        dropoffLoc= Point(0.0, 0.0)
    trip = {'pickupTime':pickupTime, 'dropoffTime':dropoffTime, 'pickupLoc':pickupLoc, 'dropoffLoc':dropoffLoc}
    return (license, trip)

def hours(trip):
    d= trip['dropoffTime'] - trip['pickupTime']
    return (d.days*24) + (d.seconds/3600)

def hasZero(trip):
    zero = Point(0.0, 0.0)
    return (zero == trip["pickupLoc"] or zero == trip["dropoffLoc"])

def split(t1, t2):
    d = t2['pickupTime'] - t1['pickupTime']
    return ((d.days*24) + (d.seconds/3600)) >= 4

def secondaryKeyFunc(trip):
    return getMillis(trip["pickupTime"])

def partitioner(n):
    def partitioner_(x):
        return portable_hash(x[0]) % n
    return partitioner_

def groupByKeyAndSortValues(rdd, secondaryKeyFunc, splitFunc):
    presess = rdd.map(lambda x: ((x[0], secondaryKeyFunc(x[1])), x[1]) )
    numPartitions = presess.getNumPartitions()
    return presess\
        .repartitionAndSortWithinPartitions(partitionFunc=partitioner(numPartitions))\
        .mapPartitions(lambda partition: groupSorted(partition, splitFunc))

def groupSorted(it, splitFunc):
    cur={'lic': None, 'trips': []}
    def mapper(x):
        lic = x[0][0]
        trip = x[1]
        if(lic != cur['lic'] or splitFunc(cur['trips'][-1], trip)):
            result = (cur['lic'], cur['trips'])
            cur['lic'] = lic
            cur['trips'] = [trip]
            if(len(result[1]) == 0):
                return None
            else:
                return result
        else:
            cur['trips'].append(trip)
            return None
    m = list(map(mapper, it))
    m.append((cur['lic'], cur['trips']))
    return filter(lambda f: f is not None, m)


if __name__ == "__main__":
    sc = SparkContext(appName="taxi")

    taxiRaw = sc.textFile("file:///Users/Karim/Downloads/trip_data_1-1.csv")

    taxiParsed = taxiRaw\
        .map(lambda line: line.split(','))\
        .filter(lambda fields: len(fields) == 14 and fields[0] != "medallion")\
        .map(parse)
    taxiParsed.cache()

    hoursCount = taxiParsed.values().map(hours).countByValue().items()

    sortedHoursCount = sorted(hoursCount, key=itemgetter(0), reverse=False)

    for val in sortedHoursCount:
        print(val)

    def goodHour(hrs):
        return 0 <= hrs and hrs < 3

    taxiClean = taxiParsed.filter(lambda x: goodHour( hours(x[1]) )).cache()

    taxiParsed.unpersist()

    with open('/Users/Karim/PycharmProjects/AAspark/ch8/nyc-boroughs.geojson', 'r') as f:
        geo = json.load(f)

    features = geo['features']
    for f in features:
        f["shape"] = shape(f['geometry'])

    areaSortedFeatures = sorted(features, key=lambda f: (int(f['properties']["boroughCode"]), -f["shape"].area), reverse=False)

    bFeatures = sc.broadcast(areaSortedFeatures)

    def borough(trip):
        for f in bFeatures.value:
            if f['shape'].contains(trip["dropoffLoc"]):
                return str(f['properties']["borough"])
        return None

    boroughCount = taxiClean.values().map(borough).countByValue().items()

    for val in boroughCount:
        print(val)

    taxiDone = taxiClean.filter(lambda x: not hasZero(x[1])).cache()

    boroughCount = taxiDone.values().map(borough).countByValue().items()

    for val in boroughCount:
        print(val)

    sessions = groupByKeyAndSortValues(taxiDone, secondaryKeyFunc, split).cache()

    def boroughDuration(t1, t2):
        b = borough(t1)
        d = t2["pickupTime"] - t1["dropoffTime"]
        return (b, d)

    def mapper(trips):
        iter = sliding(2, trips)
        viter = filter(lambda x: len(x) == 2, iter)
        return map(lambda p: boroughDuration(p[0], p[1]), viter)

    boroughDurations = sessions.values().flatMap(mapper).cache()

    boroughDurationsCount = boroughDurations.values().map(lambda d: (d.days*24) + (d.seconds/3600)).countByValue().items()

    sortedBoroughDurationsCount = sorted(boroughDurationsCount, key=itemgetter(1), reverse=True)

    for val in sortedBoroughDurationsCount:
        print(val)

    taxiDone.unpersist()

    def stats(d):
        s = StatCounter()
        return s.merge(((d.days*24*3600) + d.seconds))

    boroughCollected = boroughDurations\
        .filter(lambda x: ((x[1].days*24*3600) + x[1].seconds) >= 0)\
        .mapValues(stats)\
        .reduceByKey(lambda a, b: a.mergeStats(b))\
        .collect()

    for val in boroughCollected:
        print(val)

    boroughDurations.unpersist()
