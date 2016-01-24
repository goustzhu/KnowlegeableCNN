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
# 	valid_cr = CorpusReader(minDocSentenceNum=5, minSentenceWordNum=5, dataset="data/valid/split", labelset="data/valid/label.txt")
	validDocMatrixes, validDocSentenceNums, validSentenceWordNums, validLabels = cr.getCorpus([800, 870])
	print "Data loaded."
	
	mr = numpy.max([cr.getMaxDocSentenceNum(), cr.getMaxDocSentenceNum()])
	learning_rate = 0.01
	docSentenceCount = T.fvector("docSentenceCount")
	sentenceWordCount = T.fmatrix("sentenceWordCount")
	corpus = T.ftensor4("corpus")
	docLabel = T.ivector('docLabel') 
	rng = numpy.random.RandomState(23455)
	batchSize = 40
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
	
	# for list-type data
 	train_model = theano.function(
 		[corpus, docSentenceCount, sentenceWordCount, docLabel],
 		[cost, error],
 		updates=updates,
 	)
 	
	valid_model = theano.function(
 		[corpus, docSentenceCount, sentenceWordCount, docLabel],
 		[cost, error]
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
	
	while (epoch < n_epochs):
		epoch = epoch + 1
		# ####Validate the model####
		# for list-type data
		costNum, errorNum = valid_model(validDocMatrixes, validDocSentenceNums, validSentenceWordNums, validLabels)
		# for padding data
		# 	costNum, errorNum = train_model(docMatrixes, labels)
		print "Valid current model:"
		print "Cost: ", costNum
		print "Error: ", errorNum
		#######################
		for i in range(1000):
			
# 			if(i * batchSize >= 800):
# 				break
			
			docInfo = cr.getCorpus([i * batchSize, numpy.min([ (i + 1) * batchSize, 800])])
			if(docInfo is None):
				break
			ite = ite + 1
			
			docMatrixes, docSentenceNums, sentenceWordNums, labels = docInfo
			# for list-type data
			costNum, errorNum = train_model(docMatrixes, docSentenceNums, sentenceWordNums, labels)
			
			# for padding data
# 			costNum, errorNum = train_model(docMatrixes, labels)
			del docMatrixes, docSentenceNums, sentenceWordNums, labels
			# print ".", 
			if(ite % 1 == 0):
				print
				print "@iter: ", ite
				print "Cost: ", costNum
				print "Error: ", errorNum

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
