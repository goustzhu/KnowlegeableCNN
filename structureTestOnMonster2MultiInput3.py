from theano import tensor as T, printing
import theano
import numpy
from mlp import HiddenLayer
from logistic_sgd import LogisticRegression
# from DocNN import DocNN
from DocEmbeddingNN import DocEmbeddingNN
# from DocEmbeddingNNPadding import DocEmbeddingNN
from knoweagebleClassifyFlattenedLazy import CorpusReader
import cPickle
import os
import string
import sys

from sklearn.metrics import roc_curve, auc

def work(mode, data_name, test_dataname, pooling_mode="average_exc_pad"):
	print "mode: ", mode
	print "data_name: ", data_name
	print "pooling_mode: ", pooling_mode
	print "Started!"
	rng = numpy.random.RandomState(23455)
	docSentenceCount = T.ivector("docSentenceCount")
	sentenceWordCount = T.ivector("sentenceWordCount")
	corpus = T.matrix("corpus")
	docLabel = T.ivector('docLabel') 
	
	
	# for list-type data
	layer0 = DocEmbeddingNN(corpus, docSentenceCount, sentenceWordCount, rng, \
													wordEmbeddingDim=249, \
													 sentenceLayerNodesNum=50, \
													 sentenceLayerNodesSize=[5, 249], \
													 docLayerNodesNum=10, \
													 docLayerNodesSize=[3, 50],
													 pooling_mode=pooling_mode)

	layer1 = HiddenLayer(
		rng,
		input=layer0.output,
		n_in=layer0.outputDimension,
		n_out=10,
		activation=T.tanh
	)
	
	layer2 = LogisticRegression(input=layer1.output, n_in=10, n_out=2)

	# construct the parameter array.
	params = layer2.params + layer1.params + layer0.params
	
	# Load the parameters last time, optionally.
	
# 	data_name = "car"
	
	para_path = "data/" + data_name + "/model/multi_input_mergeinput" + pooling_mode + ".model"
	traintext = "data/" + data_name + "/train/text"
	trainlabel = "data/" + data_name + "/train/label"
	testtext = "data/" + test_dataname + "/test/text"
	testlabel = "data/" + test_dataname + "/test/label"
	
	
	loadParamsVal(para_path, params)

	if(mode == "train"  or mode == "test"):
		learning_rate = 0.1
		error = layer2.errors(docLabel)
		cost = layer2.negative_log_likelihood(docLabel)
		
		grads = T.grad(cost, params)
	
		updates = [
			(param_i, param_i - learning_rate * grad_i)
			for param_i, grad_i in zip(params, grads)
		]
		
		print "Loading test data."
		cr_test = CorpusReader(minDocSentenceNum=5, minSentenceWordNum=5, dataset=testtext, labelset=testlabel)
		validDocMatrixes, validDocSentenceNums, validSentenceWordNums, validIds, validLabels, _, validPosList = cr_test.getCorpus([0, 1000])
		
# 		print "Right answer: "
# 		print zip(validIds, validLabels)
		
		validDocMatrixes = numpy.column_stack((validDocMatrixes, validPosList))
		validDocMatrixes = transToTensor(validDocMatrixes, theano.config.floatX)
# 		validPosList = transToTensor(validPosList, theano.config.floatX)
		validDocSentenceNums = transToTensor(validDocSentenceNums, numpy.int32)
		validSentenceWordNums = transToTensor(validSentenceWordNums, numpy.int32)
		validLabels = transToTensor(validLabels, numpy.int32)
		print "Data loaded."
		
		valid_model = theano.function(
	 		[],
	 		[cost, error, layer2.y_pred, docLabel, T.transpose(layer2.p_y_given_x)[1]],
	 		givens={
							corpus: validDocMatrixes,
							docSentenceCount: validDocSentenceNums,
							sentenceWordCount: validSentenceWordNums,
							docLabel: validLabels
					},
			allow_input_downcast=True
	 	)
		
		# ####Validate the model####
		costNum, errorNum, pred_label, real_label, pred_prob = valid_model()
		print "Valid current model:"
		print "Cost: ", costNum
		print "Error: ", errorNum
# 		print "Valid Pred: ", pred_label
# 		print "pred_prob: ", pred_prob
		
		fpr, tpr, _ = roc_curve(real_label, pred_prob)
		if mode == "test":
			print "tpr_all: ", tpr
			print "fpr_all: ", fpr
		roc_auc = auc(fpr, tpr)
		print "data_name: ", data_name
		print "test_dataname: ", test_dataname
		print "ROC: ", roc_auc
		
		fpr, tpr, threshold = roc_curve(real_label, pred_label)
		
		index_of_one = list(threshold).index(1)
		ar = (tpr[index_of_one] + 1 - fpr[index_of_one]) / 2
		print "TPR: ", tpr[index_of_one]
		print "FPR: ", fpr[index_of_one]
		print "AR: ", ar
		print "threshold: ", threshold[index_of_one]
		if mode == "test":
			valid_model.free()
			return errorNum, roc_auc, tpr[index_of_one], fpr[index_of_one], ar
		
		print "Loading train data."
		cr_train = CorpusReader(minDocSentenceNum=5, minSentenceWordNum=5, dataset=traintext, labelset=trainlabel)
		docMatrixes, docSentenceNums, sentenceWordNums, ids, labels, _, posList = cr_train.getCorpus([0, 100000])
		
# 		print "Right answer: "
# 		print zip(ids, labels)
		
		
		docMatrixes = numpy.column_stack((docMatrixes, posList))
		docMatrixes = transToTensor(docMatrixes, theano.config.floatX)
# 		posList = transToTensor(posList, theano.config.floatX)
		docSentenceNums = transToTensor(docSentenceNums, numpy.int32)
		sentenceWordNums = transToTensor(sentenceWordNums, numpy.int32)
		labels = transToTensor(labels, numpy.int32)
		
	# 	valid_cr = CorpusReader(minDocSentenceNum=5, minSentenceWordNum=5, dataset="data/valid/split", labelset="data/valid/label.txt")
		print
		index = T.lscalar("index")
		batchSize = 10
		n_batches = (len(docSentenceNums.get_value()) - 1) / batchSize + 1
		print
		print "Train set size is ", len(docMatrixes.get_value())
		print "Validating set size is ", len(validDocMatrixes.get_value())
		print "Batch size is ", batchSize
		print "Number of training batches  is ", n_batches
		
		print "Compiling computing graph."
		
		# for list-type data
		train_model = theano.function(
	 		[index],
	 		[cost, error, layer2.y_pred, docLabel],
	 		updates=updates,
	 		givens={
							corpus: docMatrixes,
							docSentenceCount: docSentenceNums[index * batchSize: (index + 1) * batchSize + 1],
							sentenceWordCount: sentenceWordNums,
							docLabel: labels[index * batchSize: (index + 1) * batchSize],
						},
			allow_input_downcast=True
	 	)
		
		print "Compiled."
		print "Start to train."
		epoch = 0
		n_epochs = 10
		ite = 0
			
		while (epoch < n_epochs):
			epoch = epoch + 1
			#######################
			for i in range(n_batches):
				# for list-type data
				print ".",
				costNum, errorNum, pred_label, real_label = train_model(i)
				print ".",
				ite = ite + 1
				# for padding data
	# 			costNum, errorNum = train_model(docMatrixes, labels)
	# 			del docMatrixes, docSentenceNums, sentenceWordNums, labels
				# print ".", 
				if(ite % 10 == 0):
					print
					print "@iter: ", ite
					print "Cost: ", costNum
					print "Error: ", errorNum
					
			# Validate the model
			costNum, errorNum, pred_label, real_label, pred_prob = valid_model()
			print "Valid current model:"
			print "Cost: ", costNum
			print "Error: ", errorNum
# 			print "pred_prob: ", pred_prob
# 			print "Valid Pred: ", pred_label
			
			fpr, tpr, _ = roc_curve(real_label, pred_prob)
			roc_auc = auc(fpr, tpr)
			print "data_name: ", data_name
			print "test_dataname: ", test_dataname
			print "ROC: ", roc_auc
			
			fpr, tpr, threshold = roc_curve(real_label, pred_label)
			index_of_one = list(threshold).index(1)
			print "TPR: ", tpr[index_of_one]
			print "FPR: ", fpr[index_of_one]
			print "AR: ", (tpr[index_of_one] + 1 - fpr[index_of_one]) / 2
			print "threshold: ", threshold[index_of_one]
			# Save model
			print "Saving parameters."
			saveParamsVal(para_path, params)
			print "Saved."
		valid_model.free()
		train_model.free()
	elif(mode == "deploy"):
		print "Compiling computing graph."
		output_model = theano.function(
	 		[corpus, docSentenceCount, sentenceWordCount],
	 		[layer2.y_pred]
	 	)
		print "Compiled."
		cr = CorpusReader(minDocSentenceNum=5, minSentenceWordNum=5, dataset="data/train_valid/split")
		count = 21000
		while(count <= 21000):
			docMatrixes, docSentenceNums, sentenceWordNums, ids = cr.getCorpus([count, count + 100])
			docMatrixes = numpy.matrix(
			            docMatrixes,
			            dtype=theano.config.floatX
			        )
			docSentenceNums = numpy.array(
			            docSentenceNums,
			            dtype=numpy.int32
			        )
			sentenceWordNums = numpy.array(
			            sentenceWordNums,
			            dtype=numpy.int32
			        )
			print "start to predict."
			pred_y = output_model(docMatrixes, docSentenceNums, sentenceWordNums)
			print "End predicting."
			print "Writing resfile."
	# 		print zip(ids, pred_y[0])
			f = file("data/test/res/res" + str(count), "w")
			f.write(str(zip(ids, pred_y[0])))
			f.close()
			print "Written." + str(count)
			count += 100
	
def saveParamsVal(path, params):
	with open(path, 'wb') as f:  # open file with write-mode
		for para in params:
			cPickle.dump(para.get_value(), f, protocol=cPickle.HIGHEST_PROTOCOL)  # serialize and save object

def loadParamsVal(path, params):
	if(not os.path.exists(path)):
		return None
	try:
		with open(path, 'rb') as f:  # open file with write-mode
			for para in params:
				para.set_value(cPickle.load(f), borrow=True)
	except:
		pass
	
def transToTensor(data, t):
	return theano.shared(
        numpy.array(
            data,
            dtype=t
        ),
        borrow=True
    )
	
if __name__ == '__main__':
	mode = sys.argv[1]
	results = dict()
	if mode == "testall":
		pooling_mode = ["average_exc_pad", "max"]
		train_model_list = ["car", "finance", "house", "cfh_all"]
# 		train_model_list = ["cfh_all"]
		valid_model_list = ["car", "finance", "house", "cfh_all"]
		indicate_name = ["errorRate", "acc", "roc", "tpr", "fpr", "ar"]
		for pm in pooling_mode:
			for train_model in train_model_list:
				for valid_model in valid_model_list:
					print "--------This is \"%s\"->\"%s\" by %s----------" % (train_model, valid_model, pm)
					errorRate, roc, tpr, fpr, ar = work(mode="test", data_name=train_model, test_dataname=valid_model, pooling_mode=pm)
					results["%s->%s by %s" % (train_model, valid_model, pm)] = (errorRate, 1 - errorRate, roc, tpr, fpr, ar) 
					print
		for i in xrange(len(indicate_name)):
			print indicate_name[i]
			print "train/test\t", string.join(valid_model_list, "\t") 
			for pm in pooling_mode:
				for train_model in train_model_list:
					print "%s_%s" % (train_model, pm),
					for valid_model in valid_model_list:
						print "\t", results["%s->%s by %s" % (train_model, valid_model, pm)][i],
					print
	else:
		work(mode=sys.argv[1], data_name=sys.argv[2], test_dataname=sys.argv[3], pooling_mode=sys.argv[4])
	print "All finished!"
