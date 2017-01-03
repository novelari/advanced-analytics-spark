import math
from pyspark.mllib.linalg import Vectors

def readFile(path, sc):
    df = sc.read.format('com.databricks.spark.xml').options(rowTag='page') \
        .load(path)
    return df

def pageIsRedirect(text):
    return text.find("#REDIRECT") != -1 or text.find("#redirect") != -1

def pageIsArticle(ns):
    return ns == 0

def pageIsInvalid(title):
    return title.find("(disambiguation)") != -1

def wikiXmlToPlainText(pages):
    return pages.select("title", "ns", "revision.text._VALUE")\
        .rdd\
        .map(lambda page: (page[0], page[1], page[2]) )\
        .filter(lambda page: not (pageIsRedirect(page[2][0:50]) or not pageIsArticle(page[1]) or pageIsInvalid(page[0])) )\
        .map(lambda page: (page[0], page[2]))

def loadStopWords(path):
    with open(path) as f:
        lines = f.readlines()
    return map(lambda l: l.replace("\n", ""), lines)

def plainTextToLemmas(text, stopWords, pipeline):
    doc = pipeline.parse_doc(text)
    lemmas = []
    for sentance in doc['sentences']:
        for lemma in sentance['lemmas']:
            if(len(lemma)>2 and lemma not in stopWords and isOnlyLetters(lemma) ):
                lemmas.append(lemma.lower())
    return lemmas

def isOnlyLetters(str):
    i = 0
    while(i < len(str)):
        if(not str[i].isalpha()):
            return False
        i += 1
    return True

def documentTermMatrix(docs, stowWords, numTerms, sc):

    def foldleft(acc, key):
        if (key not in acc):
            acc[key] = 0
        acc[key] += 1
        return acc

    docTermFreqs = docs.mapValues(lambda terms: reduce(foldleft , terms, {}))
    docTermFreqs.cache()

    docIds = docTermFreqs.map(lambda x: x[0]).zipWithUniqueId().map(lambda x: (x[1], x[0])).collectAsMap()

    docFreqs = documentFrequenciesDistributed(docTermFreqs.map(lambda x: x[1]), numTerms)
    print("Number of terms: "+ str(len(docFreqs)))
    saveDocFreqs("docfreqs.tsv", docFreqs)

    numDocs = len(docIds)

    idfs = inverseDocumentFrequencies(docFreqs, numDocs)

    termIds = dict(enumerate(idfs.keys()))
    idTerms = {y:x for x,y in termIds.iteritems()}

    bIdfs = sc.broadcast(idfs).value
    bIdTerms = sc.broadcast(idTerms).value

    def termFreqsMapper(termFreqs):
        docTotalTerms = sum(termFreqs.values())
        termFreqsFiltered = filter(lambda tf: tf[0] in bIdTerms ,termFreqs.items())
        termScores = map(lambda tf: (bIdTerms[tf[0]], bIdfs[tf[0]] * termFreqs[tf[0]] / docTotalTerms ) , termFreqsFiltered)
        return Vectors.sparse(len(bIdTerms), termScores)

    vecs = docTermFreqs.map(lambda x: x[1]).map(termFreqsMapper)

    return (vecs, termIds, docIds, idfs)


def documentFrequenciesDistributed(docTermFreqs, numTerms):
    docFreqs = docTermFreqs.flatMap(lambda doc: doc.keys()).map(lambda term:(term, 1)).reduceByKey(lambda tc1,tc2: tc1 + tc2, 15)
    return docFreqs.sortBy(lambda tc: tc[1], False).take(numTerms)

def saveDocFreqs(path, docFreqs):
    ps = open(path, 'w')
    for docFreq in docFreqs:
        ps.write(str(docFreq[0]) + "\t" + str(docFreq[1]))
    ps.close()

def inverseDocumentFrequencies(docFreqs, numDocs):
    inv = map(lambda tc: (tc[0], math.log(float(numDocs)/ tc[1])) ,docFreqs)
    return dict(inv)