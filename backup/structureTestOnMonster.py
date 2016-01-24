from theano import tensor as T, printing
import theano
import numpy
from mlp import HiddenLayer
from logistic_sgd import LogisticRegression
from DocEmbeddingNN import DocEmbeddingNN
# from DocEmbeddingNNPadding import DocEmbeddingNN
from knoweagebleClassifyFlattened import CorpusReader
import cPickle

def work():
	print "Started!"
	
	print "Loading data."
	cr = CorpusReader(minDocSentenceNum=5, minSentenceWordNum=5, dataset="data/split", labelset="data/traindataset2zgb")
	docMatrixes, docSentenceNums, sentenceWordNums, labels = transToTensor(cr.getCorpus([0, 12]))
	
# 	valid_cr = CorpusReader(minDocSentenceNum=5, minSentenceWordNum=5, dataset="data/valid/split", labelset="data/valid/label.txt")
	validDocMatrixes, validDocSentenceNums, validSentenceWordNums, validLabels = transToTensor(cr.getCorpus([800, 870]))
	print "Data loaded."
	
	
	learning_rate = 0.1
	docSentenceCount = T.vector("docSentenceCount")
	sentenceWordCount = T.vector("sentenceWordCount")
	corpus = T.matrix("corpus")
	docLabel = T.ivector('docLabel') 
	index = T.lscalar("index")
	rng = numpy.random.RandomState(23455)
	batchSize = 1
	mr =numpy.max([len(docMatrixes.get_value()), len(validDocMatrixes.get_value())])
	n_batches = (len(docSentenceNums.get_value()) -1 ) / batchSize
	
	print "Train set size is ", len(docMatrixes.get_value())
	print "Validating set size is ", len(validDocMatrixes.get_value())
	print "Batch size is ", batchSize
	print "Number of training batches  is ", n_batches
	
	# for list-type data
	layer0 = DocEmbeddingNN(corpus, docSentenceCount, sentenceWordCount, rng, wordEmbeddingDim=200, \
													 maxRandge=mr, \
													 sentenceLayerNodesNum=100, \
													 sentenceLayerNodesSize=[5, 200], \
													 docLayerNodesNum=100, \
													 docLayerNodesSize=[3, 100])
	# for padding data
# 	layer0 = DocEmbeddingNN(corpus, docSentenceCount, sentenceWordCount, rng, corpus_shape=(batchSize, cr.getMaxDocSentenceNum(), cr.getMaxSentenceWordNum(), cr.getDim()), \
# 													 maxRandge=mr, \
# 													 sentenceLayerNodesNum=100, \
# 													 sentenceLayerNodesSize=5, \
# 													 docLayerNodesNum=200, \
# 													 docLayerNodesSize=3)

	layer1 = HiddenLayer(
		rng,
		input=layer0.output,
		n_in=layer0.outputDimension,
		n_out=100,
		activation=T.tanh
	)
	
	layer2 = LogisticRegression(input=layer1.output, n_in=100, n_out=2)

	error = layer2.errors(docLabel)
	cost = layer2.negative_log_likelihood(docLabel)
	
	# construct the parameter array.
	params = layer2.params + layer1.params + layer0.params
	
	# Load the parameters last time, optionally.
	loadParamsVal(params)

	grads = T.grad(cost, params)

	updates = [
		(param_i, param_i - learning_rate * grad_i)
		for param_i, grad_i in zip(params, grads)
	]
	
	
	print "Compiling computing graph."
	
	valid_model = theano.function(
 		[],
 		[cost, error],
 		givens={
						corpus: validDocMatrixes,
						docSentenceCount: validDocSentenceNums,
						sentenceWordCount: validSentenceWordNums,
						docLabel: validLabels
				}
 	)
	# for list-type data
	train_model = theano.function(
 		[index],
 		[cost, error],
 		updates=updates,
 		givens={
						corpus: docMatrixes,
						docSentenceCount: docSentenceNums[index * batchSize: (index + 1) * batchSize + 1],
						sentenceWordCount: sentenceWordNums,
						docLabel: labels[index * batchSize: (index + 1) * batchSize]
					}
 	)
	
	# for padding data
	# 	train_model = theano.function(
	# 		[corpus, docLabel],
	# 		[cost, error],
	# 		updates=updates,
	# 	)
	print "Compiled."
	
	print "Start to train."
	epoch = 0
	n_epochs = 200
	ite = 0
	
	# ####Validate the model####
	costNum, errorNum = valid_model()
	print "Valid current model:"
	print "Cost: ", costNum
	print "Error: ", errorNum
	
	while (epoch < n_epochs):
		epoch = epoch + 1
		#######################
		for i in range(n_batches):
			# for list-type data
			costNum, errorNum = train_model(i)
			ite = ite + 1
			# for padding data
# 			costNum, errorNum = train_model(docMatrixes, labels)
# 			del docMatrixes, docSentenceNums, sentenceWordNums, labels
			# print ".", 
			if(ite % 1 == 0):
				print
				print "@iter: ", ite
				print "Cost: ", costNum
				print "Error: ", errorNum
				
		# Validate the model
		costNum, errorNum = valid_model()
		print "Valid current model:"
		print "Cost: ", costNum
		print "Error: ", errorNum
		
		# Save model
		print "Saving parameters."
		saveParamsVal(params)
		print "Saved."
	print "All finished!"
	
def saveParamsVal(params):
	with open("model/scnn.model", 'wb') as f:  # open file with write-mode
		for para in params:
			cPickle.dump(para.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)  # serialize and save object

def loadParamsVal(params):
	try:
		with open("model/scnn.model", 'rb') as f:  # open file with write-mode
			for para in params:
				para.set_value(cPickle.load(f), borrow=True)
	except:
		pass
	
def transToTensor(data):
	toReturn = list()
	for i in xrange(len(data)):
		if(i==3):
			t = numpy.int32
		else:
			t = theano.config.floatX
		toReturn.append(theano.shared(
            numpy.array(
                data[i],
                dtype=t
            ),
            borrow=True
        ))
	return toReturn
if __name__ == '__main__':
	work()
	
"""
d = [
            [
                 [
                    [2, 2, 3, 4],
                    [1, 2, 3, 4],
                    [3, 1, 2, 3],
                    [6, 4, 2, 1],
                    [0, 0, 0, 0]
                ],
                 [
                    [4, 3, 2, 1],
                    [4, 6, 9, 2],
                    [6, 6, 3, 1],
                    [2, 5, 2, 9],
                    [3, 2, 1, 7]
                ]
             ],
            [
                     [
                        [9, 8, 7, 6],
                        [5, 4, 3, 2],
                        [1, 9, 8, 7],
                        [6, 5, 4, 3],
                        [0, 0, 0, 0]
                    ],
                     [
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]
                    ]
                 ]
            ]
"""
