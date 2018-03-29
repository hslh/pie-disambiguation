#!/usr/bin/env python2
# -*- coding:utf-8 -*- 

'''Run and evaluate the PIE sense disambiguation system'''

import pickle, os, datetime
import config
import read_corpus
import most_frequent_sense
import canonical_form
import cohesion_graph
import evaluate
from pie import PIE

if __name__ == '__main__':
	print 'Hello!'	
	# Read in the PIE Corpus
	pie_corpus = read_corpus.read_PIE_Corpus(config.PIE_DIR, config.BNC_DIR, config.CONTEXT_SIZE, config.WORK_DIR, config.NO_CACHE)
	print pie_corpus[10].__dict__
	print 'Read {0} PIEs from PIE Corpus.'.format(len(pie_corpus))
	# Read in the SemEval-2013 Task 5b corpus
	semeval = read_corpus.read_SemEval(config.SEMEVAL_DIR, config.SEMEVAL_CONTEXTS, config.CONTEXT_SIZE, config.WORK_DIR, config.NO_CACHE)
	print semeval[180].__dict__
	print 'Read {0} PIEs from SemEval-2013 Task 5b corpus.'.format(len(semeval))
	# Read in the IDIX corpus
	idix = read_corpus.read_IDIX(config.IDIX_DIR, config.BNC_DIR, config.CONTEXT_SIZE, config.WORK_DIR, config.DATA_DIR, config.NO_CACHE)
	print idix[1].__dict__
	print 'Read {0} PIEs from IDIX corpus.'.format(len(idix))
	# Read in the VNC-Tokens dataset
	vnc = read_corpus.read_VNC(config.VNC_DIR, config.BNC_DIR, config.CONTEXT_SIZE, config.WORK_DIR, config.DATA_DIR, config.NO_CACHE)
	print vnc[10].__dict__
	print 'Read {0} PIEs from VNC-Tokens dataset.'.format(len(vnc))
	# Combine datasets
	combined_data = vnc + idix + semeval + pie_corpus
	print 'Dataset contains {0} PIEs in total.'.format(len(combined_data))
	# Filter out PIEs without binary labels
	combined_data = [pie for pie in combined_data if pie.binary_label]
	print '{0} PIEs left after filtering out PIEs without binary labels.'.format(len(combined_data))
	# Split data into development and test set
	dev_corpora = ['VNC-dev', 'VNC-skewed', 'SemEval-lexsample-train', 'SemEval-allwords-train', 'SemEval-lexsample-dev', 'SemEval-allwords-dev', 'IDIX-single', 'PIE-train', 'PIE-dev']
	test_corpora = ['VNC-test', 'IDIX-double', 'SemEval-lexsample-test', 'SemEval-allwords-test', 'PIE-test']
	dev_data = []
	test_data = []
	for pie in combined_data:
		if pie.corpus in dev_corpora:
			pie.split = 'dev'
			dev_data.append(pie)
		elif pie.corpus in test_corpora:
			pie.split = 'test'
			test_data.append(pie)
	print '{0} PIEs in development set, {1} PIEs in test set'.format(len(dev_data), len(test_data))
	# Run disambiguation method
	if config.METHOD == 'mfs':
		combined_data = most_frequent_sense.most_frequent_sense(combined_data, config.TEST)
	elif config.METHOD == 'cform':
		combined_data = canonical_form.canonical_form(combined_data, config.TEST, config.CONTEXT_SIZE, config.DATA_DIR)
	elif config.METHOD == 'cg':
		combined_data = cohesion_graph.cohesion_graph(combined_data, config.TEST, config.EMB_DIR, config.GRAPH_SIZE, config.GRAPH_POS, config.GRAPH_PIE, config.GRAPH_SET, config.GRAPH_LEMMA, config.GRAPH_GLOVE, config.GRAPH_INTRA, config.GRAPH_TYPE, config.GRAPH_DEFINITIONS)
	print 'Classified {0} of {1} as idiomatic.'.format(len([x for x in combined_data if x.predicted_label == 'i']), len([x for x in combined_data if x.predicted_label]))
	# Run evaluation?
	user_input = unicode(raw_input("\nRun evaluation? (y/n): "), 'utf-8')
	if user_input.lower() == 'y':
		combined_data = evaluate.evaluate(combined_data, config.TEST, 15)
	# Save predictions?
	user_input = unicode(raw_input("\nSave predictions to file? (y/n): "), 'utf-8')
	if user_input.lower() == 'y':
		pickle.dump(combined_data, open(os.path.join(config.WORK_DIR, 'predictions_{0}.pickle'.format('{:%Y-%m-%d-%H-%M-%S}'.format(datetime.datetime.now()))), 'wb'))
