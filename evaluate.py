#!/usr/bin/env python2
# -*- coding:utf-8 -*-

'''Evaluates PIE sense disambiguation systems' output'''

from collections import Counter

def evaluate(data, use_test_data, top_n):
	'''
	Evaluates PSD predictions, outputs per-class accuracy,
	P/R/F1 on idiomatic sense, macro-accuracy, and macro-F1.
	Per-class scores for top_n most frequent types only.
	'''

	# Split data and filter out PIEs without binary labels
	evaluation_data = [PIE for PIE in data if (use_test_data and PIE.split == 'test') or (not use_test_data and PIE.split == 'dev')]
	evaluation_data = [PIE for PIE in evaluation_data if PIE.binary_label]
	# Get types for per-type scoring
	PIE_types = set([PIE.pie_type for PIE in evaluation_data])
	PIE_type_counts = Counter(PIE.pie_type for PIE in evaluation_data)
	# Assign true/false positives/negatives
	for PIE in evaluation_data:
		if PIE.binary_label == PIE.predicted_label:
			if PIE.binary_label == 'i':
				PIE.classification = 'tp'
			if PIE.binary_label == 'l':
				PIE.classification = 'tn'
		else:
			if PIE.binary_label == 'i':
				PIE.classification = 'fn'
			if PIE.binary_label == 'l':
				PIE.classification = 'fp'
	# Count true/false positives/negatives
	total_true = float(sum([PIE.classification in ['tp', 'tn'] for PIE in evaluation_data]))
	total_tp = float(sum([PIE.classification == 'tp' for PIE in evaluation_data]))
	total_fp = float(sum([PIE.classification == 'fp' for PIE in evaluation_data]))
	total_fn = float(sum([PIE.classification == 'fn' for PIE in evaluation_data]))
	# Get overall micro-accuracy and micro-F1
	micro_accuracy = total_true/float(len(evaluation_data))*100.	
	micro_precision = total_tp/(total_tp+total_fp)*100.
	micro_recall = total_tp/(total_tp+total_fn)*100.
	micro_f1 = 2. * (micro_precision * micro_recall) / (micro_precision + micro_recall)
	# Get per-type scores
	# NB: define precision and recall as 100%, if denominator is 0
	scores_per_type = {PIE_type: {'accuracy': 0., 'precision': 0., 'recall': 0., 'f1': 0.} for PIE_type in PIE_types}
	for PIE_type in PIE_types:
		total_true = float(sum([PIE.classification in ['tp', 'tn'] for PIE in evaluation_data if PIE.pie_type == PIE_type]))
		scores_per_type[PIE_type]['accuracy'] = total_true/float(PIE_type_counts[PIE_type])*100.
		total_tp = float(sum([PIE.classification == 'tp' for PIE in evaluation_data if PIE.pie_type == PIE_type]))
		total_fp = float(sum([PIE.classification == 'fp' for PIE in evaluation_data if PIE.pie_type == PIE_type]))
		total_fn = float(sum([PIE.classification == 'fn' for PIE in evaluation_data if PIE.pie_type == PIE_type]))
		try:
			scores_per_type[PIE_type]['precision'] = total_tp/(total_tp+total_fp)*100.
		except ZeroDivisionError:
			scores_per_type[PIE_type]['precision'] = 100.
		try:			
			scores_per_type[PIE_type]['recall'] = total_tp/(total_tp+total_fn)*100.
		except ZeroDivisionError:
			scores_per_type[PIE_type]['recall'] = 100.			
		try:
			scores_per_type[PIE_type]['f1'] = 2. * (scores_per_type[PIE_type]['precision'] * scores_per_type[PIE_type]['recall']) / (scores_per_type[PIE_type]['precision'] + scores_per_type[PIE_type]['recall'])
		except ZeroDivisionError:
			scores_per_type[PIE_type]['f1'] = 0.
	# Get macro-averages
	# NB: macro-average F1 is not harmonic mean of macro-average P and macro-average R
	macro_accuracy = sum(scores_per_type[PIE_type]['accuracy'] for PIE_type in scores_per_type)/len(PIE_types)
	macro_precision = sum(scores_per_type[PIE_type]['precision'] for PIE_type in scores_per_type)/len(PIE_types)
	macro_recall = sum(scores_per_type[PIE_type]['recall'] for PIE_type in scores_per_type)/len(PIE_types)
	macro_f1 = sum(scores_per_type[PIE_type]['f1'] for PIE_type in scores_per_type)/len(PIE_types)
	# Print output to screen
	print '### EVALUATION SCORES ###'
	print 'Macro-Accuracy: {0:.2f}\nMicro-Accuracy: {1:.2f}'.format(macro_accuracy, micro_accuracy)
	print '-'*25
	print 'Macro-Precision: {0:.2f}\nMacro-Recall: {1:.2f}\nMacro-F1: {2:.2f}'.format(macro_precision, macro_recall, macro_f1)
	print 'Micro-Precision: {0:.2f}\nMicro-Recall: {1:.2f}\nMicro-F1: {2:.2f}'.format(micro_precision, micro_recall, micro_f1)
	# Print for copying to results file
	print '\n{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}'.format(macro_accuracy, micro_accuracy, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1)
	print '\n### SCORES PER TYPE ###'
	print '{0}\tFreq.\tPrec.\tRecall\tF1\tAccuracy'.format('PIE Type' + 15*' ')
	print '-'*65
	top_n_types = PIE_type_counts.most_common(top_n)
	for top_type in top_n_types:
		scores = scores_per_type[top_type[0]]
		type_with_padding = top_type[0][:23] + (23-len(top_type[0][:23]))*' '
		print '{0}\t{1:d}\t{2:.2f}\t{3:.2f}\t{4:.2f}\t{5:.2f}'.format(type_with_padding, top_type[1], scores['precision'], scores['recall'], scores['f1'], scores['accuracy'])

	return data
