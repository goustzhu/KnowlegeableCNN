from theano import tensor as T, printing
import theano
import numpy
from mlp import HiddenLayer
from logistic_sgd_lazy import LogisticRegression
from DocEmbeddingNN import DocEmbeddingNN
# from DocEmbeddingNNPadding import DocEmbeddingNN
from knoweagebleClassifyFlattenedLazy import CorpusReader
import cPickle
import os

import math
import sys

from sklearn.metrics import roc_curve, auc

def work(mode, data_name, test_dataname, pooling_mode="average_exc_pad"):
	print "mode: ", mode
	print "data_name: ", data_name
	print "pooling_mode: ", pooling_mode
	print "Started!"
	
	data_names = data_name.split(":")
	data_count = len(data_names)
	print "Train dataset:"
	for i in xrange(data_count):
		print "%d: %s" % (i, data_names[i])
		
	print "Test dataset:"
	test_data_names = test_dataname.split(":")
	test_data_count = len(test_data_names)
	for i in xrange(test_data_count):
		print "%d: %s" % (i, test_data_names[i])
	
	if test_data_count != data_count:
		raise Exception("The amount of test and train dataset must be the same.")
	
	rng = numpy.random.RandomState(23455)
	docSentenceCount = T.ivector("docSentenceCount")
	sentenceWordCount = T.ivector("sentenceWordCount")
	corpus = T.matrix("corpus")
	docLabel = T.ivector('docLabel')
	
	hidden_layer_w = None
	hidden_layer_b = None
	logistic_layer_w = None
	logistic_layer_b = None
	layer0 = list()
	layer1 = list()
	layer2 = list()
	local_params = list()
	# for list-type data
	for i in xrange(data_count):
		layer0.append(DocEmbeddingNN(corpus, docSentenceCount, sentenceWordCount, rng, wordEmbeddingDim=200, \
														 sentenceLayerNodesNum=50, \
														 sentenceLayerNodesSize=[5, 200], \
														 docLayerNodesNum=10, \
														 docLayerNodesSize=[3, 50],
														 pooling_mode=pooling_mode))

		layer1.append(HiddenLayer(
			rng,
			input=layer0[i].output,
			n_in=layer0[i].outputDimension,
			n_out=10,
			activation=T.tanh,
			W=hidden_layer_w,
			b=hidden_layer_b
		))
		
# 		hidden_layer_w = layer1[i].W
# 		hidden_layer_b = layer1[i].b
	
		layer2.append(LogisticRegression(input=layer1[i].output, n_in=10, n_out=2, W=logistic_layer_w, b=logistic_layer_b))
		logistic_layer_w = layer2[i].W
		logistic_layer_b = layer2[i].b
		
		local_params.append(layer0[i].params + layer1[i].params)
	
	share_params = list(layer2[0].params)
	# construct the parameter array.
	params = list(layer2[0].params)
	
	for i in xrange(data_count):
		params += layer1[0].params + layer0[i].params
		
	
# 	data_name = "car"
	
	para_path = "data/" + data_name + "/log_model/" + pooling_mode + ".model"
	traintext = ["data/" + data_names[i] + "/train/text"  for i in xrange(data_count)]
	trainlabel = ["data/" + data_names[i] + "/train/label"  for i in xrange(data_count)]
	testtext = ["data/" + test_data_names[i] + "/test/text"  for i in xrange(data_count)]
	testlabel = ["data/" + test_data_names[i] + "/test/label"  for i in xrange(data_count)]
	
	# Load the parameters last time, optionally.
	loadParamsVal(para_path, params)

	if(mode == "train" or mode == "test"):
		train_model = list()
		valid_model = list()
		print "Loading train data."
		batchSize = 10
		share_learning_rate = 0.01
		local_learning_rate = 0.1
		n_batches = list()
		
		print "Loading test data."
		
		for i in xrange(data_count):
			cr_train = CorpusReader(minDocSentenceNum=5, minSentenceWordNum=5, dataset=traintext[i], labelset=trainlabel[i])
			docMatrixes, docSentenceNums, sentenceWordNums, ids, labels, _, _ = cr_train.getCorpus([0, 100000])
			
			docMatrixes = transToTensor(docMatrixes, theano.config.floatX)
			docSentenceNums = transToTensor(docSentenceNums, numpy.int32)
			sentenceWordNums = transToTensor(sentenceWordNums, numpy.int32)
			labels = transToTensor(labels, numpy.int32)
			
			index = T.lscalar("index")
			
			n_batches.append((len(docSentenceNums.get_value())  - 1 - 1) / batchSize + 1)
			print "Dataname: %s" % data_names[i]
			print "Train set size is ", len(docMatrixes.get_value())
			print "Batch size is ", batchSize
			print "Number of training batches  is ", n_batches[i]
			error = layer2[i].errors(docLabel)
			cost = layer2[i].negative_log_likelihood(docLabel)
			
		
			share_grads = T.grad(cost, share_params)
			share_updates = [
				(param_i, param_i - share_learning_rate * grad_i)
				for param_i, grad_i in zip(share_params, share_grads)
			]
			
			grads = T.grad(cost, local_params[i])
			local_updates = [
				(param_i, param_i - local_learning_rate * grad_i)
				for param_i, grad_i in zip(local_params[i], grads)
			]
			updates = share_updates + local_updates
			print "Compiling train computing graph."
			if mode == "train":
				train_model.append(theano.function(
			 		[index],
			 		[cost, error, layer2[i].y_pred, docLabel],
			 		updates=updates,
			 		givens={
									corpus: docMatrixes,
									docSentenceCount: docSentenceNums[index * batchSize: (index + 1) * batchSize + 1],
									sentenceWordCount: sentenceWordNums,
									docLabel: labels[index * batchSize: (index + 1) * batchSize]
								}
		 		))
			print "Compiled."
			
			print "Load test dataname: %s" % test_data_names[i]
			cr_test = CorpusReader(minDocSentenceNum=5, minSentenceWordNum=5, dataset=testtext[i], labelset=testlabel[i])
			validDocMatrixes, validDocSentenceNums, validSentenceWordNums, validIds, validLabels, _, _ = cr_test.getCorpus([0, 1000])
			validDocMatrixes = transToTensor(validDocMatrixes, theano.config.floatX)
			validDocSentenceNums = transToTensor(validDocSentenceNums, numpy.int32)
			validSentenceWordNums = transToTensor(validSentenceWordNums, numpy.int32)
			validLabels = transToTensor(validLabels, numpy.int32)
			print "Validating set size is ", len(validDocMatrixes.get_value())
			print "Data loaded."
			
			print "Compiling test computing graph."
			valid_model.append(theano.function(
		 		[],
		 		[cost, error, layer2[i].y_pred, docLabel, T.transpose(layer2[i].p_y_given_x)[1]],
		 		givens={
								corpus: validDocMatrixes,
								docSentenceCount: validDocSentenceNums,
								sentenceWordCount: validSentenceWordNums,
								docLabel: validLabels
						}
		 	))
			print "Compiled."
			costNum, errorNum, pred_label, real_label, pred_prob = valid_model[i]()
			print "Valid current model :", data_names[i]
			print "Cost: ", costNum
			print "Error: ", errorNum
	 		
			fpr, tpr, _ = roc_curve(real_label, pred_prob)
			roc_auc = auc(fpr, tpr)
			print "data_name: ", data_name
			print "ROC: ", roc_auc
			fpr, tpr, threshold = roc_curve(real_label, pred_label)
			if 1 in threshold:
				index_of_one = list(threshold).index(1)
				print "TPR: ", tpr[index_of_one]
				print "FPR: ", fpr[index_of_one]
				print "threshold: ", threshold[index_of_one]

		if mode == "test":
			return

		print "Start to train."
		epoch = 0
		n_epochs = 10
		ite = 0
		
		# ####Validate the model####
# 		for dataset_index in xrange(data_count):
# 			costNum, errorNum, pred_label, real_label, pred_prob = valid_model[dataset_index]()
# 			print "Valid current model :", data_names[dataset_index]
# 			print "Cost: ", costNum
# 			print "Error: ", errorNum
# 	 		
# 			fpr, tpr, _ = roc_curve(real_label, pred_prob)
# 			roc_auc = auc(fpr, tpr)
# 			print "data_name: ", data_name
# 			print "ROC: ", roc_auc
# 			fpr, tpr, threshold = roc_curve(real_label, pred_label)
# 			index_of_one = list(threshold).index(1)
# 			print "TPR: ", tpr[index_of_one]
# 			print "FPR: ", fpr[index_of_one]
# 			print "threshold: ", threshold[index_of_one]
			
		while (epoch < n_epochs):
			epoch = epoch + 1
			#######################
			for i in range(max(n_batches)):
				for dataset_index in xrange(data_count):
					if i >= n_batches[dataset_index]:
						continue
					# for list-type data
					print "dataset_index: %d, i: %d" %(dataset_index, i)
					costNum, errorNum, pred_label, real_label = train_model[dataset_index](i)
					ite = ite + 1
					# for padding data
					if(ite % 10 == 0):
						print
						print "Dataset name: ", data_names[dataset_index]
						print "@iter: ", ite
						print "Cost: ", costNum
						print "Error: ", errorNum
						
			# Validate the model
			for dataset_index in xrange(data_count):
				costNum, errorNum, pred_label, real_label, pred_prob = valid_model[dataset_index]()
				print "Valid current model :", data_names[dataset_index]
				print "Cost: ", costNum
				print "Error: ", errorNum
		 		
				fpr, tpr, _ = roc_curve(real_label, pred_prob)
				roc_auc = auc(fpr, tpr)
				print "data_name: ", data_name
				print "ROC: ", roc_auc
					
				fpr, tpr, threshold = roc_curve(real_label, pred_label)
				index_of_one = list(threshold).index(1)
				print "TPR: ", tpr[index_of_one]
				print "FPR: ", fpr[index_of_one]
				print "threshold: ", threshold[index_of_one]
			# Save model
			print "Saving parameters."
			saveParamsVal(para_path, params)
			print "Saved."
# 	elif(mode == "deploy"):
# 		print "Compiling computing graph."
# 		output_model = theano.function(
# 	 		[corpus, docSentenceCount, sentenceWordCount],
# 	 		[layer2.y_pred]
# 	 	)
# 		print "Compiled."
# 		cr = CorpusReader(minDocSentenceNum=5, minSentenceWordNum=5, dataset="data/train_valid/split")
# 		count = 21000
# 		while(count <= 21000):
# 			docMatrixes, docSentenceNums, sentenceWordNums, ids = cr.getCorpus([count, count + 100])
# 			docMatrixes = numpy.matrix(
# 			            docMatrixes,
# 			            dtype=theano.config.floatX
# 			        )
# 			docSentenceNums = numpy.array(
# 			            docSentenceNums,
# 			            dtype=numpy.int32
# 			        )
# 			sentenceWordNums = numpy.array(
# 			            sentenceWordNums,
# 			            dtype=numpy.int32
# 			        )
# 			print "start to predict."
# 			pred_y = output_model(docMatrixes, docSentenceNums, sentenceWordNums)
# 			print "End predicting."
# 			print "Writing resfile."
# 	# 		print zip(ids, pred_y[0])
# 			f = file("data/test/res/res" + str(count), "w")
# 			f.write(str(zip(ids, pred_y[0])))
# 			f.close()
# 			print "Written." + str(count)
# 			count += 100
	
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
	work(mode=sys.argv[1], data_name=sys.argv[2], test_dataname=sys.argv[3], pooling_mode=sys.argv[4])
	print "All finished!"
