from theano import tensor as T, printing
import theano
import theano.tensor.signal.downsample as downsample
import theano.tensor.signal.conv as conv
import numpy

class DocEmbeddingNN:
    
    def __init__(self,
                          corpus,
                          docSentenceCount,
                          sentenceWordCount,
                          rng,
                          wordEmbeddingDim,
                          sentenceLayerNodesNum=2,
                          sentenceLayerNodesSize=(2, 2),
                          docLayerNodesNum=2,
                          docLayerNodesSize=(2, 3),
                          datatype=theano.config.floatX,
                          sentenceW = None,
                          sentenceB = None,
                          docW = None,
                          docB = None):
        self.__wordEmbeddingDim = wordEmbeddingDim
        self.__sentenceLayerNodesNum = sentenceLayerNodesNum
        self.__sentenceLayerNodesSize = sentenceLayerNodesSize
        self.__docLayerNodesNum = docLayerNodesNum
        self.__docLayerNodesSize = docLayerNodesSize
        self.__WBound = 0.2
        self.__MAXDIM = 10000
        self.__datatype = datatype
        
        # For  DomEmbeddingNN optimizer.
#         self.shareRandge = T.arange(maxRandge)
        
        # Get sentence layer W
        if sentenceW is None:
            sentenceW = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-self.__WBound, high=self.__WBound, size=(self.__sentenceLayerNodesNum, self.__sentenceLayerNodesSize[0], self.__sentenceLayerNodesSize[1])),
                    dtype=datatype
                ),
                borrow=True
            )
        # Get sentence layer b
        if sentenceB is None:
            sentenceB0 = numpy.zeros((sentenceLayerNodesNum,), dtype=datatype)
            sentenceB = theano.shared(value=sentenceB0, borrow=True)
        
        # Get doc layer W
        if docW is None:
            docW = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-self.__WBound, high=self.__WBound, size=(self.__docLayerNodesNum, self.__docLayerNodesSize[0], self.__docLayerNodesSize[1])),
                    dtype=datatype
                ),
                borrow=True
            )
        # Get doc layer b
        if docB is None:
            docB0 = numpy.zeros((docLayerNodesNum,), dtype=datatype)
            docB = theano.shared(value=docB0, borrow=True)
        
        self.output, _ = theano.scan(fn=self.__dealWithOneDoc,
                    non_sequences=[corpus, sentenceWordCount, docW, docB, sentenceW, sentenceB],
                     sequences=[dict(input=docSentenceCount, taps=[-1, -0])],
                     strict=True)
        self.sentenceW = sentenceW
        self.sentenceB = sentenceB
        self.docW = docW
        self.docB = docB
        self.params = [self.sentenceW, self.sentenceB, self.docW, self.docB]
        self.outputDimension = self.__docLayerNodesNum * \
                                                  (self.__sentenceLayerNodesNum * (self.__wordEmbeddingDim - self.__sentenceLayerNodesSize[1] + 1) - self.__docLayerNodesSize[1] + 1)
   
    def __dealWithOneDoc(self, DocSentenceCount0, oneDocSentenceCount1, docs, oneDocSentenceWordCount, docW, docB, sentenceW, sentenceB):
#         t = T.and_((shareRandge < oneDocSentenceCount1 + 1),  (shareRandge >= DocSentenceCount0)).nonzero()
        oneDocSentenceWordCount = oneDocSentenceWordCount[DocSentenceCount0:oneDocSentenceCount1 + 1]
        
        sentenceResults, _ = theano.scan(fn=self.__dealWithSentence,
                            non_sequences=[docs, sentenceW, sentenceB],
                             sequences=[dict(input=oneDocSentenceWordCount, taps=[-1, -0])],
                             strict=True)
        
#         p = printing.Print('docPool')
#         docPool = p(docPool)
#         p = printing.Print('sentenceResults')
#         sentenceResults = p(sentenceResults)
#         p = printing.Print('doc_out')
#         doc_out = p(doc_out)
        doc_out = conv.conv2d(input=sentenceResults, filters=docW)
        docPool = downsample.max_pool_2d(doc_out, (self.__MAXDIM, 1), mode="average_exc_pad", ignore_border=False)
        docOutput = T.tanh(docPool + docB.dimshuffle([0, 'x', 'x']))
        doc_embedding = docOutput.flatten(1)
        return doc_embedding
    
    def __dealWithSentence(self, sentenceWordCount0, sentenceWordCount1, docs, sentenceW, sentenceB):
#         t = T.and_((shareRandge < sentenceWordCount1), (shareRandge >= sentenceWordCount0)).nonzero()
        sentence = docs[sentenceWordCount0:sentenceWordCount1]
        
        sentence_out = conv.conv2d(input=sentence, filters=sentenceW)
        sentence_pool = downsample.max_pool_2d(sentence_out, (self.__MAXDIM, 1), mode="average_exc_pad", ignore_border=False)
        
        sentence_output = T.tanh(sentence_pool + sentenceB.dimshuffle([0, 'x', 'x']))
        sentence_embedding = sentence_output.flatten(1)
        return sentence_embedding
