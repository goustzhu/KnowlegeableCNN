from theano import tensor as T, printing
import theano
import theano.tensor.signal.downsample as downsample
import theano.tensor.signal.conv as conv
import numpy

class DocNN:
    def __init__(self,
                          corpus,
                          tokenCount,
                          rng,
                          embeddingDim,
                          nodesNum=2,
                          nodesSize=(2, 2),
                          datatype=theano.config.floatX,
                          sentenceW=None,
                          sentenceB=None,
                          W=None,
                          B=None,
                          pooling_mode="average_exc_pad"):
        self.__embeddingDim = embeddingDim
        self.__nodesNum = nodesNum
        self.__nodesSize = nodesSize
        self.__WBound = 0.2
        self.__MAXDIM = 10000
        self.__datatype = datatype
        self.__pooling_mode = pooling_mode
        
        # For  DomEmbeddingNN optimizer.
#         self.shareRandge = T.arange(maxRandge)
        
        # Get sentence layer W
        if W is None:
            W = theano.shared(
                numpy.asarray(
                    rng.uniform(low=-self.__WBound, high=self.__WBound, size=(nodesNum, nodesSize[0], nodesSize[1])),
                    dtype=datatype
                ),
                borrow=True
            )
        # Get sentence layer b
        if B is None:
            B0 = numpy.zeros((nodesNum,), dtype=datatype)
            B = theano.shared(value=B0, borrow=True)
        
        self.output, _ = theano.scan(fn=self.__dealWithSentence,
                    non_sequences=[corpus, W, B],
                     sequences=[dict(input=tokenCount, taps=[-1, -0])],
                     strict=True)
        
        self.W = W
        self.B = B
        self.params = [self.W, self.B]
        self.outputDimension = nodesNum * (embeddingDim - nodesSize[1] + 1)
   
    def __dealWithSentence(self, wc0, wc1, docs, W, B):
        sentence = docs[wc0:wc1]
        
        sentence_out = conv.conv2d(input=sentence, filters=W)
        sentence_pool = downsample.max_pool_2d(sentence_out, (self.__MAXDIM, 1), mode=self.__pooling_mode, ignore_border=False)
        
        sentence_output = T.tanh(sentence_pool + B.dimshuffle([0, 'x', 'x']))
        sentence_embedding = sentence_output.flatten(1)
        return sentence_embedding
