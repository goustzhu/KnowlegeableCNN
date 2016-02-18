# coding=utf-8
# -*- coding: utf-8 -*-
import codecs
import numpy as np
from codecs import decode
import string
import theano
import re
class CorpusReader:
    def __init__(self, minDocSentenceNum, minSentenceWordNum):
        self.minDocSentenceNum = minDocSentenceNum
        self.minSentenceWordNum = minSentenceWordNum
        
        self.__pos_vectors = dict()
        for pos, index in self.__posDict.items():
            tmp = [0] * len(self.__posDict)
            tmp[index] = 1
            self.__pos_vectors[pos] = tmp
        
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
    __wordDim = 200
    __zeroWordVector = [0] * __wordDim
    
    __posDict = {"Ag":0, "Bg":1, "Dg":2, "Mg":3, "Ng":4, "Qg":5, "Rg":6, "Tg":7, "Vg":8, "Yg":9, "a":10, "ad":11,
            "an":12, "b":13, "c":14, "d":15, "e":16, "email":17, "f":18, "h":19, "i":20, "j":21, "k":22, "l":23,
            "m":24, "n":25, "nr":26, "nrf":27, "nrg":28, "ns":29, "nt":30, "nx":31, "nz":32, "o":33, "p":34, "q":35,
             "r":36, "s":37, "t":38, "tele":39, "u":40, "v":41, "vd":42, "vn":43, "w":44, "www":45, "x":46, "y":47, "z":48}
    p = re.compile(r'\s*')
    def getDim(self):
        return self.__wordDim
    
    def __sentence2Matrix(self, wordList):
        sentenceMatrix = map(lambda word: (self.w2vDict[word], word[0]) if (word in self.w2vDict) else None, wordList)
        sentenceMatrix = filter(lambda item: not item is None, sentenceMatrix)
        
        sentenceWordNum = len(sentenceMatrix)
        if(sentenceWordNum < self.minSentenceWordNum):
            return None
        
        sentenceMatrix, wordList = zip(*sentenceMatrix)
        
        sentenceMatrix = list(sentenceMatrix)
        wordList = list(wordList)
        
        return (sentenceMatrix, sentenceWordNum, wordList)
    
    def __doc2Matrix(self, content):
        wordList = content
        mapL = lambda e: e[1] if (u"\u3000" in e[0] or u"。" in e[0] or u"？" in e[0] or u"！" in e[0] or  u"." in e[0] or u"?" in e[0] or u"!" in e[0]) else None
        t = map(mapL, zip(wordList, range(len(wordList))))
        t = filter(None, t)
        t = [-1] + t + [len(wordList)]
        m = map(lambda i:  self.__sentence2Matrix(wordList[i[0] + 1:i[1]]) if(i[1] - i[0] > self.minSentenceWordNum + 1) else None , zip(t[:-1], t[1:]))
        m = filter(lambda item: not item is None, m)
        
        if(len(m) == 0):
            return None
        docMatrix, sentenceWordNum, wordListList = zip(*m)
        docMatrix = list(docMatrix)
        wordListList = list(wordListList)
        sentenceWordNum = list(sentenceWordNum)
        
        docSentenceNum = len(docMatrix)
        if(docSentenceNum < self.minDocSentenceNum):
            return None
        
        # Merge the sentence embedding into a holistic list.
        docMatrix = reduce(add, docMatrix, [])
        
        return (docMatrix, docSentenceNum, sentenceWordNum, wordListList)
    
    def __getDataMatrixNoLabel(self, content):
        docInfo = self.__doc2Matrix(content);
        
        if docInfo is None:
            return None
#         docInfo = filter(None, docInfo)
#         if(len(docInfo) == 0):
#             return None
#         
        docMatrixes, docSentenceNums, sentenceWordNums, wordListList = zip(docInfo)
        
        # Merge the sentence embedding into a holistic list.
        docMatrixes = reduce(add, docMatrixes, [])
        sentenceWordNums = reduce(add, sentenceWordNums, [])
        wordListList = reduce(add, wordListList, [])
        
        docSentenceNums = [0] + list(docSentenceNums)
        sentenceWordNums = [0] + sentenceWordNums
        
        docSentenceNums = np.cumsum(docSentenceNums)
        sentenceWordNums = np.cumsum(sentenceWordNums)
        
        return (docMatrixes, docSentenceNums, sentenceWordNums, wordListList)
    
    def __findBoarder(self, docSentenceCount, sentenceWordCount):
        maxDocSentenceNum = np.max(docSentenceCount)
        maxSentenceWordNum = np.max(np.max(sentenceWordCount))
        return maxDocSentenceNum, maxSentenceWordNum
    
    # Only positive scope numbers are legal.
    def getCorpus(self, content):
        content = self.p.sub(' ', content)
        content = loadDocuments(content)
        corpusInfo = self.__getDataMatrixNoLabel(content)
        return corpusInfo

def add(a, b):
    return a + b

def loadDocuments(content):
    wordData = getWords(content)
    return wordData

def getWords(wordsStr):
    def dealword(word):
        word = word.strip()
        return word
    t = filter(lambda word: len(word) >= 1, wordsStr.split(" "))
    return map(dealword, t)


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
