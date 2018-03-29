#!/usr/bin/env python2
# -*- coding:utf-8 -*-

'''Most frequent sense baseline for PIE sense disambiguation'''

from collections import Counter

def most_frequent_sense(data, use_test_data):
	'''
	Calculates most frequent sense in development data and 
	applies that to evaluation data	
	'''

	# Get most frequent label from development data
	dev_labels = [PIE.binary_label for PIE in data if PIE.split == 'dev']
	most_frequent_label = Counter(dev_labels).most_common(1)[0][0]
	# Classify evaluation data
	for PIE in data:
		if (use_test_data and PIE.split == 'test') or (not use_test_data and PIE.split == 'dev'):
			PIE.predicted_label = most_frequent_label

	return data
