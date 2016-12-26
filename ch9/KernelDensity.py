import math

from pyspark.statcounter import StatCounter
from pyspark.rdd import RDD


def aggregate(zeroValue, seqOp, combOp, list):
    vals = map(lambda obj: seqOp(zeroValue, obj),list)
    return reduce(combOp, vals, zeroValue)

class KernelDensity(object):

    def __init__(self, samples):
        self.samples = samples

    def chooseBandwidthList(self):
        stddev = StatCounter(self.samples).stdev()
        return 1.06 * stddev * math.pow(len(self.samples), -.2)

    def chooseBandwidthRDD(self):
        stats = self.samples.stats()
        return 1.06 * stats.stdev() * math.pow(stats.count(), -.2)

    def chooseBandwidth(self):
        return self.chooseBandwidthRDD() if isinstance(self.samples, RDD) else self.chooseBandwidthList()

    def estimate(self, evaluationPoints):
        stddev = self.chooseBandwidth()
        logStandardDeviationPlusHalfLog2Pi = math.log(stddev) + 0.5 * math.log(2 * math.pi)
        zero = ([0.0] * len(evaluationPoints), 0)
        if isinstance(self.samples, RDD):
            (points, count) = self.samples.aggregate(zero,
                                   lambda x, y: KernelDensity.__mergeSingle(x, y, evaluationPoints, stddev, logStandardDeviationPlusHalfLog2Pi),
                                   lambda x1, x2: KernelDensity.__combine(x1, x2, evaluationPoints))
        else:
            (points, count) = aggregate(zero,lambda x, y: KernelDensity.__mergeSingle(x, y, evaluationPoints, stddev, logStandardDeviationPlusHalfLog2Pi),lambda x1, x2: KernelDensity.__combine(x1, x2, evaluationPoints),self.samples)
        i = 0
        while (i < len(points)):
            points[i] /= count
            i += 1

        return points

    @staticmethod
    def __mergeSingle(x, y, evaluationPoints, standardDeviation, logStandardDeviationPlusHalfLog2Pi):
        i = 0
        while (i < len(evaluationPoints)):
            x[0][i] += KernelDensity.__normPdf(y, standardDeviation, logStandardDeviationPlusHalfLog2Pi, evaluationPoints[i])
            i += 1
        return (x[0], i)

    @staticmethod
    def __combine(x, y, evaluationPoints):
        i = 0
        while (i < len(evaluationPoints)):
            x[0][i] += y[0][i]
            i += 1
        return (x[0], x[1] + y[1])

    @staticmethod
    def __normPdf(mean, standardDeviation, logStandardDeviationPlusHalfLog2Pi, x):
        x0 = x - mean
        x1 = x0 / standardDeviation
        return math.exp(-0.5 * x1 * x1 - logStandardDeviationPlusHalfLog2Pi)