def readFile(path, sc):
    df = sc.read.format('com.databricks.spark.xml').options(rowTag='page') \
        .load('/Users/Karim/Downloads/enwiki-latest-pages-articles1.xml')
    return df

def pageIsRedirect(text):
    return text.find("#REDIRECT") != -1 or text.find("#redirect") != -1

def pageIsArticle(ns):
    return ns == 0

def pageIsInvalid(title):
    return title.find("(disambiguation)") != -1

def wikiXmlToPlainText(pages):
    pages.select("title", "ns", "revision.text._VALUE")\
        .rdd\
        .map(lambda page: (page[0], page[1], page[2]) )\
        .filter(lambda page: not (pageIsRedirect(page[2][0:50]) or not pageIsArticle(page[1]) or pageIsInvalid(page[0])) )\
        .map(lambda page: (page[0], page[2]))