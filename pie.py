#!/usr/bin/env python2
# -*- coding:utf-8 -*-

'''Defines the PIE class'''

class PIE:
	'Class for PIE instances'
	
	PIE_counter = 0
	
	def __init__(self, corpus, pie_type, sense_label, binary_label, context, context_untokenized, offsets):
		self.id = PIE.PIE_counter
		PIE.PIE_counter += 1
		self.corpus = corpus
		self.split = ''
		self.pie_type = pie_type
		self.sense_label = sense_label # Original label
		self.binary_label = binary_label # Label, normalised to literal 'l', idiomatic 'i', or other ''
		self.predicted_label = ''
		self.classification = '' # False/true positive/negative
		self.context = context # List of tokenized sentences, where middle sentence contains the PIE
		self.context_untokenized = context_untokenized # Same, but untokenized
		self.offsets = offsets # Character offsets of content words of the PIE in middle sentence of tokenized context
		
	def __str__(self):
		print str(self.__dict__)
		
