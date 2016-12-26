import os, datetime, math, random
import matplotlib.pyplot as plt
import numpy as np

os.environ["SPARK_HOME"] = "/Users/Karim/src/spark-2.0.0-bin-hadoop2.6"
os.environ["PYSPARK_PYTHON"] = "/usr/bin/python"

from KernelDensity import KernelDensity
from sklearn.linear_model import LinearRegression
from scipy import stats
from pyspark import SparkContext
from pyspark.rdd import RDD

def frange(x, y, jump):
  while x < y:
    yield round(x, 5)
    x += jump

def readStocksAndFactors(prefix):
    start = datetime.date(2009, 10, 23)
    end = datetime.date(2014, 10, 23)
    rawStocks = filter(lambda stock: len(stock) >= 260*5+10,readHistories(prefix + "stocks/"))
    trimmedStocks = map(lambda stock: trimToRegion(stock, start, end),rawStocks)
    stocks = map(lambda stock: fillInHistory(stock, start, end), trimmedStocks)
    factorsPrefix = prefix + "factors/"
    factors1 = map(lambda file: readInvestingDotComHistory(factorsPrefix + file),["crudeoil.tsv", "us30yeartreasurybonds.tsv"])
    factors2 = map(lambda file: readYahooHistory(factorsPrefix + file),["SNP.csv", "NDX.csv"])
    trimmedFactors = map(lambda stock: trimToRegion(stock, start, end), factors1 + factors2)
    factors = map(lambda stock: fillInHistory(stock, start, end), trimmedFactors)
    stockReturns = map(twoWeekReturns, stocks)
    factorReturns = map(twoWeekReturns, factors)
    return (stockReturns, factorReturns)

def readHistories(dir):
    files = os.listdir(dir)
    return map(lambda f: readYahooHistory(dir+f), files)

def readYahooHistory(file):
    with open(file) as f:
        lines = f.readlines()
    if "html" in lines[0]:
        return []
    def lineMapper(line):
        cols = line.split(',')
        date = datetime.datetime.strptime(cols[0], '%Y-%m-%d').date()
        value = float(cols[1])
        return (date, value)
    formattedLines = map(lineMapper, lines[1:])
    formattedLines.reverse()
    return formattedLines

def readInvestingDotComHistory(file):
    with open(file) as f:
        lines = f.readlines()
    def lineMapper(line):
        cols = line.split('    ')
        date = datetime.datetime.strptime(cols[0], '%b %d, %Y').date()
        value = float(cols[1])
        return (date, value)
    formattedLines = map(lineMapper, lines)
    formattedLines.reverse()
    return formattedLines

def trimToRegion(history, start, end):
    trimmed = filter(lambda his: his[0] >= start and his[0] <= end, history)
    if(trimmed[0][0] != start):
        trimmed = [(start, trimmed[0][1])] + trimmed
    if (trimmed[-1][0] != end):
        trimmed = trimmed + [(end, trimmed[-1][1])]
    return trimmed

def fillInHistory(history, start, end):
    cur = history
    filled = []
    curDate = start
    while (curDate < end):
        if(len(cur[1:])> 0 and cur[1][0] == curDate):
            cur = cur[1:]
        filled.append((curDate, cur[0][1]))
        curDate += datetime.timedelta(days=1)
        if (curDate.weekday()+1 > 5):
            curDate += datetime.timedelta(days=2)
    return filled

def twoWeekReturns(history):
    mappedHistory = []
    for (idx, val) in enumerate(history):
        if (idx + 10 <= len(history)):
            window = history[idx: idx + 10]
            next = window[-1][1]
            prev = window[0][1]
            mappedHistory.append((next - prev) / prev)
    return mappedHistory

def plotDistribution(samples):
    return plotDistributionRDD(samples) if isinstance(samples, RDD) else plotDistributionList(samples)

def plotDistributionList(samples):
    max_s = max(samples)
    min_s = min(samples)
    domain = list(frange(min_s, max_s, (max_s - min_s) / 100.0))
    densities = KernelDensity(samples).estimate(domain)
    plt.plot(domain, densities, 'ro')
    plt.xlabel("Two Week Return ($)")
    plt.ylabel("Density")
    plt.show()

def plotDistributionRDD(samples):
    stats = samples.stats()
    min_s = stats.min()
    max_s = stats.max()
    domain = list(frange(min_s, max_s, (max_s - min_s) / 100.0))
    densities = KernelDensity(samples).estimate(domain)
    plt.plot(domain, densities, 'ro')
    plt.xlabel("Two Week Return ($)")
    plt.ylabel("Density")
    plt.show()

def factorMatrix(histories):
    mat = [0] * len(histories[0])
    for (i, val) in enumerate(histories[0]):
        mat[i] = map(lambda h: h[i] , histories)
    return mat

def featurize(factorReturns):
    squaredReturns = map(lambda x: math.copysign(1, x) * x * x, factorReturns)
    squareRootedReturns = map(lambda x: math.copysign(1,x) * math.sqrt(abs(x)), factorReturns)
    return squaredReturns + squareRootedReturns + factorReturns

def computeFactorWeights(stocksReturns, factorFeatures):
    linear = map(lambda s: linearModel(s, factorFeatures), stocksReturns)
    return map(lambda l: list(l.coef_), linear)

def linearModel(instrument, factorMatrix):
    lm = LinearRegression()
    return lm.fit(factorMatrix, instrument)

def computeTrialReturns(stocksReturns, factorsReturns, sc, baseSeed, numTrials, parallelism):
    factorMat = factorMatrix(factorsReturns)
    factorCov = np.cov(factorsReturns)
    factorMeans = map(lambda factor: sum(factor)/len(factor), factorsReturns)
    factorFeatures = map(featurize,factorMat)
    factorWeights = computeFactorWeights(stocksReturns, factorFeatures)
    bInstruments = sc.broadcast(factorWeights)
    seeds = range(baseSeed, baseSeed + parallelism)
    seedRdd = sc.parallelize(seeds, parallelism)
    return seedRdd.flatMap(lambda s: trialReturns(s, numTrials / parallelism, bInstruments.value, factorMeans, factorCov))

def trialReturns(seed, numTrials, instruments, factorMeans, factorCovariances):
    np.random.seed(seed)
    trialReturns = [0] * numTrials
    for i in range(0, numTrials):
        trialFactorReturns = np.random.multivariate_normal(factorMeans,factorCovariances)
        trialFeatures = featurize(list(trialFactorReturns))
        trialReturns[i] = trialReturn(trialFeatures, instruments)
    return trialReturns

def trialReturn(trial, instruments):
    totalReturn = 0.0
    for instrument in instruments:
        totalReturn += instrumentTrialReturn(instrument, trial)
    return totalReturn / len(instruments)

def instrumentTrialReturn(instrument, trial):
    instrumentTrialReturn = instrument[0]
    i = 0
    while (i < len(trial)):
        instrumentTrialReturn += trial[i] * instrument[i]
        i += 1
    return instrumentTrialReturn

def fivePercentVaR(trials):
    topLosses = trials.takeOrdered(max(trials.count() / 20, 1))
    return topLosses[-1]

def fivePercentCVaR(trials):
    topLosses = trials.takeOrdered(max(trials.count() / 20, 1))
    return sum(topLosses) / len(topLosses)

def bootstrappedConfidenceInterval(trials, computeStatistic, numResamples, pValue):
    stats = sorted(map(lambda i: computeStatistic(trials.sample(True, 1.0)), range(0,numResamples)))
    lowerIndex = int(numResamples * pValue / 2 - 1)
    upperIndex = int(math.ceil(numResamples * (1 - pValue / 2)))
    return (stats[lowerIndex], stats[upperIndex])

def countFailures(stocksReturns, valueAtRisk):
    failures = 0
    for i in range(0, len(stocksReturns[0])):
        loss = sum(map(lambda x: x[i], stocksReturns))
        if(loss < valueAtRisk):
            failures += 1
    return failures

def kupiecTestStatistic(total, failures, confidenceLevel):
    failureRatio = float(failures)/ total
    logNumer = (total - failures) * math.log1p(-confidenceLevel) * failures * math.log(confidenceLevel)
    logDenom = (total - failures) * math.log1p(-failureRatio) + failures * math.log(failureRatio)
    return -2 * (logNumer - logDenom)

def kupiecTestPValue(stocksReturns, valueAtRisk, confidenceLevel):
    failures = countFailures(stocksReturns, valueAtRisk)
    total = len(stocksReturns[0])
    testStatistic = kupiecTestStatistic(total, failures, confidenceLevel)
    return 1 - stats.chi2.cdf(testStatistic, 1)

if __name__ == "__main__":
    sc = SparkContext(appName="VaR")
    (stocksReturns, factorsReturns) = readStocksAndFactors("/Users/Karim/Downloads/VaR-Data/")
    plotDistribution(factorsReturns[2])
    plotDistribution(factorsReturns[3])
    numTrials = 10000000
    parallelism = 1000
    baseSeed = 1001L
    trials = computeTrialReturns(stocksReturns, factorsReturns, sc, baseSeed, numTrials,parallelism)
    trials.cache()
    valueAtRisk = fivePercentVaR(trials)
    conditionalValueAtRisk = fivePercentCVaR(trials)
    print("VaR 5%: " + str(valueAtRisk))
    print("CVaR 5%: " + str(conditionalValueAtRisk))
    varConfidenceInterval = bootstrappedConfidenceInterval(trials, fivePercentVaR, 100, 0.05)
    cvarConfidenceInterval = bootstrappedConfidenceInterval(trials, fivePercentCVaR, 100, 0.05)
    print("VaR confidence interval: " + str(varConfidenceInterval))
    print("CVaR confidence interval: " + str(cvarConfidenceInterval))
    print("Kupiec test p-value: " + str(kupiecTestPValue(stocksReturns, valueAtRisk, 0.05)))
    plotDistribution(trials)
