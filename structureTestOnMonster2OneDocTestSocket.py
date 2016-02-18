# coding=utf-8
# -*- coding: utf-8 -*-
from theano import tensor as T, printing
import theano
import numpy
from mlp import HiddenLayer
from logistic_sgd import LogisticRegression
from DocEmbeddingNNOneDoc import DocEmbeddingNNOneDoc
# from DocEmbeddingNNPadding import DocEmbeddingNN
from knoweagebleClassifyFlattenedLazyOneDoc import CorpusReader
import cPickle
import socket
import string
import codecs
def compileFun(model_name, dataset_name, pooling_mode):
	print "model_name: ", model_name
	print "dataset_name: ", dataset_name
	print "pooling_mode: ", pooling_mode
	print "Started!"
	rng = numpy.random.RandomState(23455)
	sentenceWordCount = T.ivector("sentenceWordCount")
	corpus = T.matrix("corpus")
# 	docLabel = T.ivector('docLabel') 
	
	# for list-type data
	layer0 = DocEmbeddingNNOneDoc(corpus, sentenceWordCount, rng, wordEmbeddingDim=200, \
													 sentenceLayerNodesNum=100, \
													 sentenceLayerNodesSize=[5, 200], \
													 docLayerNodesNum=100, \
													 docLayerNodesSize=[3, 100],
													 pooling_mode=pooling_mode)

	layer1_output_num = 100
	layer1 = HiddenLayer(
		rng,
		input=layer0.output,
		n_in=layer0.outputDimension,
		n_out=layer1_output_num,
		activation=T.tanh
	)
	
	layer2 = LogisticRegression(input=layer1.output, n_in=100, n_out=2)

	cost = layer2.negative_log_likelihood(1 - layer2.y_pred)
		
	# calculate sentence sentence_score
	sentence_grads = T.grad(cost, layer0.sentenceResults)
	sentence_score = T.diag(T.dot(sentence_grads, T.transpose(layer0.sentenceResults)))
	
	# construct the parameter array.
	params = layer2.params + layer1.params + layer0.params
	
	# Load the parameters last time, optionally.
	model_path = "data/" + dataset_name + "/" + model_name + "/" + pooling_mode + ".model"
	loadParamsVal(model_path, params)
	print "Compiling computing graph."
	output_model = theano.function(
 		[corpus, sentenceWordCount],
 		[layer2.y_pred, sentence_score]
 	)
	
	print "Compiled."
	return output_model
	
def loadParamsVal(path, params):
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
	output_model = compileFun(model_name="model_100,100,100,100,parameters", dataset_name="cfh_all", pooling_mode="average_exc_pad")
	cr = CorpusReader(minDocSentenceNum=5, minSentenceWordNum=5)
	
	# content = u"你好 吗 今天 晚饭 吃 什么 菜 呢 晚饭 吃 什么 菜 呢 晚饭 吃 什么 菜 呢。你好 吗 今天 晚饭 吃 什么 菜 呢 晚饭 吃 什么 菜 呢 晚饭 吃 什么 菜 呢。你好 吗 今天 晚饭 吃 什么 菜 呢 晚饭 吃 什么 菜 呢 晚饭 吃 什么 菜 呢。你好 吗 今天 晚饭 吃 什么 菜 呢 晚饭 吃 什么 菜 呢 晚饭 吃 什么 菜 呢。你好 吗 今天 晚饭 吃 什么 菜 呢 晚饭 吃 什么 菜 呢 晚饭 吃 什么 菜 呢。你好 吗 今天 晚饭 吃 什么 菜 呢 晚饭 吃 什么 菜 呢 晚饭 吃 什么 菜 呢。你好 吗 今天 晚饭 吃 什么 菜 呢 晚饭 吃 什么 菜 呢 晚饭 吃 什么 菜 呢。你好 吗 今天 晚饭 吃 什么 菜 呢 晚饭 吃 什么 菜 呢 晚饭 吃 什么 菜 呢。"
	# start server.	
	
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	sock.bind(('localhost', 7788))
	sock.listen(5)
	print "Start to listen."
	while True:
		conn, addr = sock.accept()
		print conn
		try:
# 			conn.settimeout(100)
			buf = conn.recv(100000)
			content = codecs.decode(buf, "utf-8")
			info = cr.getCorpus(content)
			if info is None:
				conn.send("Error: need more valid snippets.\n")
			else:
				docMatrixes, _, sentenceWordNums, sentences = info
				docMatrixes = numpy.matrix(
				            docMatrixes,
				            dtype=theano.config.floatX
				        )
				sentenceWordNums = numpy.array(
				            sentenceWordNums,
				            dtype=numpy.int32
				        )
				info = output_model(docMatrixes, sentenceWordNums)
				pred_y = info[0]
				g = info[1]
				score_sentence_list = zip(g, sentences)
	# 			print score_sentence_list
				toSend = "%d\n" % pred_y
				toSend += "%d\n" % len(g)
				for score, sentence in score_sentence_list:
					toSend += "%f\t%s\n" % (score, string.join(sentence,""))
				toSend = codecs.encode(toSend,"utf-8","ignore")
				conn.sendall(toSend)
				# 			conn.close()
		except socket.timeout:
			print "time out!"
	

		
	print "All finished!"
