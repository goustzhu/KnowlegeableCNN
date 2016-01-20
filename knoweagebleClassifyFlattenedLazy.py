# coding=utf-8
# -*- coding: utf-8 -*-
import codecs
import numpy as np
from codecs import decode
import string
import theano
class CorpusReader:
    def __init__(self, minDocSentenceNum, minSentenceWordNum, dataset=None, labelset=None):
        self.minDocSentenceNum = minDocSentenceNum
        self.minSentenceWordNum = minSentenceWordNum
        # Load labels
        if(labelset is not None):
            self.labels, self.labelsList = loadLabels(labelset)
            print "labels: ", len(self.labels)
        else:
            self.labels = None
        # Load documents
        if(dataset is not None):
            self.docs, self.docIdList = loadDocuments(dataset, "GBK")
            print "document: ", len(self.docs)
        else:
            self.docs = None
        
        # Load stop words
        if CorpusReader.stopwords is None:
            CorpusReader.stopwords = loadStopwords("data/stopwords", "GBK")
            print "stop words: ", len(self.stopwords)
        
        # Load w2v model data from file
        if CorpusReader.w2vDict is None:
#             CorpusReader.w2vDict = loadW2vModel("data/w2vFlat")
            CorpusReader.w2vDict = loadW2vModel("data/word2vec_flat_big")
            print "w2v model contains: ", len(self.w2vDict)
        
    labels = None
    docs = None
    stopwords = None
    w2vDict = None
    # print "(570, 301)"
    minDocSentenceNum = 0
    minSentenceWordNum = 0
    __maxDocSentenceNum = 398
    __maxSentenceWordNum = 69
    __wordDim = 200
    __zeroWordVector = [0] * __wordDim
    __zeroSentenceMatrix = [__zeroWordVector] * __maxSentenceWordNum
    
    def getMaxDocSentenceNum(self):
        return self.__maxDocSentenceNum
    
    def getMaxSentenceWordNum(self):
        return self.__maxSentenceWordNum
    
    def getDocNum(self):
        return len(self.labels)
    
    def getDim(self):
        return self.__wordDim
    
    def __sentence2Matrix(self, wordList):
        sentenceMatrix = map(lambda word: (self.w2vDict[word], word) if (word in self.w2vDict) else None, wordList)
        
        sentenceMatrix = filter(lambda item: not item is None, sentenceMatrix)
        
        sentenceWordNum = len(sentenceMatrix)
        if(sentenceWordNum < self.minSentenceWordNum):
            return None
        
        sentenceMatrix, wordList = zip(*sentenceMatrix)
        
        sentenceMatrix = list(sentenceMatrix)
        wordList = list(wordList)
        return (sentenceMatrix, sentenceWordNum, wordList)
    
    def __doc2Matrix(self, e):
        (docIdStr, label) = e
        wordList = self.docs[docIdStr]
        mapL = lambda e: e[1] if (u"\u3000" in e[0] or u"。" in e[0] or u"，" in e[0] or u"." in e[0] or u"," in e[0]) else None
        t = map(mapL, zip(wordList, range(len(wordList))))
        t = filter(None, t)
        t = [-1] + t + [len(wordList)]
        m = map(lambda i:  self.__sentence2Matrix(wordList[i[0] + 1:i[1]]) if(i[1] - i[0] > self.minSentenceWordNum + 1) else None , zip(t[:-1], t[1:]))
        m = filter(lambda item: not item is None, m)
        
        if(len(m) == 0):
            return None
        docMatrix, sentenceWordNum, wordListList = zip(*m)
        docMatrix = list(docMatrix)
        sentenceWordNum = list(sentenceWordNum)
        wordListList = list(wordListList)
        
        docSentenceNum = len(docMatrix)
        if(docSentenceNum < self.minDocSentenceNum):
            return None
        
        # Merge the sentence embedding into a holistic list.
        docMatrix = reduce(add, docMatrix, [])
        
        return (docMatrix, docSentenceNum, sentenceWordNum, docIdStr, label, wordListList)
    
    def __getDataMatrix(self, scope):
        scope[1] = np.min([scope[1], len(self.labels)])
        if(scope[0] < 0 or scope[0] >= scope[1]):
            return None
#         batch = self.labelsList[scope[0]:scope[1]]
        batch = self.labels.items()[scope[0]:scope[1]]
        docInfo = map(self.__doc2Matrix, batch)
        
        if(len(docInfo) == 0):
            print "Lost doc: ", self.labels.items()[scope[0]:scope[1]]
            return None
        
        docInfo = filter(None, docInfo)
        if(len(docInfo) == 0):
            return None
        
        docMatrixes, docSentenceNums, sentenceWordNums, ids, labels, wordListList = zip(*docInfo)
        
        # Merge the sentence embedding into a holistic list.
        docMatrixes = reduce(add, docMatrixes, [])
        sentenceWordNums = reduce(add, sentenceWordNums, [])
        wordListList = reduce(add, wordListList, [])
        
        docSentenceNums = [0] + list(docSentenceNums)
        sentenceWordNums = [0] + sentenceWordNums
        
        docSentenceNums = np.cumsum(docSentenceNums)
        sentenceWordNums = np.cumsum(sentenceWordNums)
        
        #   print docSentenceNums
        #   print sentenceWordNums
        return (docMatrixes, docSentenceNums, sentenceWordNums, ids, labels)
    
    def __getDataMatrixNoLabel(self, scope):
        scope[1] = np.min([scope[1], len(self.docs)])
        if(scope[0] < 0 or scope[0] >= scope[1]):
            return None
        
        ids = self.docIdList[scope[0]:scope[1]]
        batch = zip(ids, ids)
        docInfo = map(self.__doc2Matrix, batch)
        
        if(len(docInfo) == 0):
            print "Lost doc: ", self.labels.items()[scope[0]:scope[1]]
            return None
        
        docInfo = filter(None, docInfo)
        if(len(docInfo) == 0):
            return None
        
        docMatrixes, docSentenceNums, sentenceWordNums, ids, _, wordListList = zip(*docInfo)
        
        # Merge the sentence embedding into a holistic list.
        docMatrixes = reduce(add, docMatrixes, [])
        sentenceWordNums = reduce(add, sentenceWordNums, [])
        wordListList = reduce(add, wordListList, [])
        
        print
        
        docSentenceNums = [0] + list(docSentenceNums)
        sentenceWordNums = [0] + sentenceWordNums
        
        docSentenceNums = np.cumsum(docSentenceNums)
        sentenceWordNums = np.cumsum(sentenceWordNums)
        
        return (docMatrixes, docSentenceNums, sentenceWordNums, ids, wordListList)
    
    def __findBoarder(self, docSentenceCount, sentenceWordCount):
        maxDocSentenceNum = np.max(docSentenceCount)
        maxSentenceWordNum = np.max(np.max(sentenceWordCount))
        return maxDocSentenceNum, maxSentenceWordNum
    
    # Only positive scope numbers are legal.
    def getCorpus(self, scope):
        # print "(398, 69)"
        # docMatrixes, docSentenceNums, sentenceWordNums, labels = self.__getDataMatrix(scope)
        if(self.labels is not None):
            corpusInfo = self.__getDataMatrix(scope)
        else:
            corpusInfo = self.__getDataMatrixNoLabel(scope)
#         print len(corpusInfo[0])
#         print self.__findBoarder(corpusInfo[1],corpusInfo[2])
        return corpusInfo

def add(a, b):
    return a + b

def loadDocuments(filename, charset="utf-8"):
    f = open(filename, "r")
    docList = dict()
    docIdList = list()
    for line0 in f:
        try:
            line = decode(line0, charset, 'ignore')
        except:
            continue
        tokens = line.split("\t")
        idStr = tokens[0]
        title = tokens[1]
        content = tokens[2]
        docList[idStr] = getWords(title) + getWords(content)
        docIdList.append(idStr)
    f.close()
    return (docList, docIdList)

def getWords(wordsStr):
    def dealword(word):
        if(":" in word):
            word = word[:word.index(":")]
        return word
    t = filter(lambda word: len(word) > 1 and ":" in word, wordsStr.split(" "))
    return map(dealword, t)

def loadLabels(filename, charset="utf-8"):
    f = codecs.open(filename, "r", charset, "ignore")
    labels = dict()
    labelsList = list()
    for line in f:
        if(not "\t" in line):
            continue
        (k, v) = line.split("\t")
        labels[k] = string.atof(v)
        labelsList.append((k, v))
    f.close()
    return (labels, labelsList)

def loadStopwords(filename, charset="utf-8"):
    f = codecs.open(filename, "r", charset, "ignore")
    d = set()
    for line in f :
        d.add(line)
    f.close()
    return d

def loadW2vModel(filename, charset="utf-8"):
    f = codecs.open(filename, "r", charset)
    d = dict()
    for line in f :
        data = line.split(" ")
        word = data[0]
        vec = [string.atof(s) for s in data[1:]]
        d[word] = np.array(vec, dtype=theano.config.floatX)
    f.close()
    return d
