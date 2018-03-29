#!/usr/bin/env python2
# -*- coding:utf-8 -*-

'''Cohesion graph-based classifier for PIE sense disambiguation'''

import re
import os
import numpy
import time
import spacy
import en_core_web_md as spacy_model
import copy
from scipy.spatial.distance import cosine as cosine_distance
from definitions import definition_mapping # Dictionary of idiom defintions

# Global variables to capture some dataset-level descriptive statistics
num_empty_graphs = 0 # Number of 'graphs' without edges
num_no_context_graphs = 0 # Number of graphs without context words
num_no_pie_graphs = 0 # Number of graphs without component words
total_similarity_difference = 0.0
backoff_counter = [0,0] # Literal empty, figurative empty

class Edge:
	'Class for edges of the graph'
	
	def __init__(self, words, in_PIE):
		self.words = words # Tuple of words connected by the edge
		self.in_PIE = in_PIE # Is this edge connected to a PIE component word?
		self.similarity = None # Similarity between the words, i.e. the edge weight
		
	def __str__(self):
		return str(self.__dict__)

def cohesion_graph(data, use_test_data, embeddings_dir, graph_size, graph_pos, graph_pie, graph_set, graph_lemma, graph_glove, graph_intra, graph_type, graph_definitions):
	'''
	Construct cohesion graphs for each PIE and predict sense labels
	'''
	
	# Descriptive statistics
	global num_empty_graphs, num_no_pie_graphs, num_no_context_graphs, total_similarity_difference, backoff_counter
	# Load PoS-tagger
	print 'Loading PoS-tagger...'
	time_0 = time.time()
	pos_tagger = spacy_model.load(disable = ['ner', 'parser'])
	print 'Done! Took {0:.2f} seconds'.format(time.time() - time_0)
	# Load embeddings
	print 'Loading embeddings...'
	time_0 = time.time()
	embeddings = {}
	embeddings_file = 'glove.6B.{0}d.txt'.format(graph_glove)
	if graph_glove == 840:
		embeddings_file = 'glove.840B.300d.txt'
	with open(os.path.join(embeddings_dir, embeddings_file)) as f:
		for line in f:
			split = line.strip().split(' ')
			word = unicode(split[0], 'utf-8')
			embedding = numpy.array([float(n) for n in split[1:]])
			embeddings[word] = embedding
	# For unknown words, take average of all embeddings
	embeddings['UNK'] = numpy.mean(numpy.array(embeddings.values()), axis = 0)
	print 'Done! Took {0:.2f} seconds'.format(time.time() - time_0)
	# For each PIE, construct the graph and predict the label
	print 'Constructing graphs and predicting labels...'
	# Keep track of words occurring in graphs
	words = []
	time_0 = time.time()
	for PIE in data:
		if (use_test_data and PIE.split == 'test') or (not use_test_data and PIE.split == 'dev'):
			# Use literalisations
			if graph_definitions:
				# Create regular graph
				graph_literal = construct_graph(PIE, pos_tagger, graph_size, graph_pos, graph_pie, graph_set, graph_lemma, graph_intra, graph_definitions)
				words += list(set([item for sublist in [x.words for x in graph_literal] for item in sublist]))
				graph_literal = weight_graph(graph_literal, embeddings)
				# Create PIE where PIE is replaced by its figurative sense definition, and a graph of that
				PIE_figurative = modify_PIE(PIE, definition_mapping)
				graph_figurative = construct_graph(PIE_figurative, pos_tagger, graph_size, graph_pos, graph_pie, graph_set, graph_lemma, graph_intra, graph_definitions)
				words += list(set([item for sublist in [x.words for x in graph_figurative] for item in sublist]))
				graph_figurative = weight_graph(graph_figurative, embeddings)
				# Use contrasting prediction function
				PIE.predicted_label = predict_label_contrast(graph_literal, graph_figurative, graph_type)
			else:
				graph = construct_graph(PIE, pos_tagger, graph_size, graph_pos, graph_pie, graph_set, graph_lemma, graph_intra, graph_definitions)
				words += list(set([item for sublist in [x.words for x in graph] for item in sublist]))
				graph = weight_graph(graph, embeddings)
				PIE.predicted_label = predict_label(graph, graph_type)
	# Count OOV words
	words = list(set(words))
	oov_counter = sum([word.lower() not in embeddings for word in words])
	# Print descriptive statistics
	print '\nDone! Took {0:.2f} seconds'.format(time.time() - time_0)
	print 'Total number of graphs: {0}'.format(len([x for x in data if x.predicted_label]))
	print 'Empty graphs: {0}. No-PIE-graphs: {1}. No-context-graphs: {2}.'.format(num_empty_graphs, num_no_pie_graphs, num_no_context_graphs)
	if graph_definitions:
		print 'Backoffs: {0} (empty literal), {1} (empty figurative)'.format(backoff_counter[0], backoff_counter[1])
	print 'Found {0} words'.format(len(words))
	print 'Of which {0} are OOV'.format(oov_counter)
	print 'Total similarity difference: {0:.2f}'.format(total_similarity_difference)

	return data
	
def construct_graph(PIE, pos_tagger, graph_size, graph_pos, graph_pie, graph_set, graph_lemma, graph_intra, graph_definitions):
	'''
	Construct cohesion graph by selecting words to be included,
	return list of unweighted Edges
	'''

	graph = []
	PIE_sentence_index = len(PIE.context)/2
	# Take right number of sentences for sentence-length contexts
	if graph_size == '0s':
		context = PIE.context[PIE_sentence_index:PIE_sentence_index + 1]
		PIE_sentence_index = 0
	elif graph_size[-1] == 's':
		num_extra_sentences = int(graph_size[:-1])
		context = PIE.context[PIE_sentence_index - num_extra_sentences : PIE_sentence_index + num_extra_sentences + 1]
		PIE_sentence_index = num_extra_sentences
	# Take all sentences for word-length contexts
	elif graph_size[-1] == 'w':
		context = PIE.context
	# Tag context
	tagged_context = pos_tag(pos_tagger, context)
	# Find token indices of PIE component words in sentence, using final character offsets
	PIE_tokens = []
	end_offsets = [offset[1] for offset in PIE.offsets]
	# Find the character index of the start of the PIE sentence, while dealing with empty pre-contexts
	sentence_start_index = tagged_context[len(' '.join(context[:PIE_sentence_index]).split())].idx
	for token in tagged_context:
		if token.idx - sentence_start_index + len(token) in end_offsets:
			PIE_tokens.append(token.i)
	if len(PIE_tokens) != len(PIE.offsets):
		# Sometimes (some) PIE tokens are not found, either because of garbage like '\xa3' in the context, 
		# or because instances are joined by dashes, which then do not turn into tokens. 
		if not PIE_tokens:
			print 'No PIE tokens found, no graph constructed!'
			return []
		else:
			print 'Some PIE tokens not found!'
	# Select only certain PoS and optionally always select PIE component words regardless
	if graph_pie:
		content_tokens = [token for token in tagged_context if token.pos_ in graph_pos or token.i in PIE_tokens]
	else:
		content_tokens = [token for token in tagged_context if token.pos_ in graph_pos]
	# Filter out pronouns (Spacy tags possessive pronouns as adjectives)
	content_tokens = [token for token in content_tokens if token.lemma_ != u'-PRON-']
	# Filter out placeholder words like someone and something
	content_tokens = [token for token in content_tokens if token.lemma not in [u'something', u'someone']]
	# Limit number of context tokens, always allow context tokens that occur in-between PIE component words
	if graph_size[-1] == 'w':
		max_context_tokens = int(graph_size[:-1])
		pre_context_tokens = [token for token in content_tokens if token.i < PIE_tokens[0]][-max_context_tokens:]
		PIE_context_tokens = [token for token in content_tokens if token.i >= PIE_tokens[0] and token.i <= PIE_tokens[-1]]
		post_context_tokens = [token for token in content_tokens if token.i > PIE_tokens[-1]][:max_context_tokens]
		context_tokens = pre_context_tokens + PIE_context_tokens + post_context_tokens
	else:
		context_tokens = content_tokens
	# Filter out context words that are duplicates of PIE component words
	if graph_set:
		context_token_set = []
		for token_1 in content_tokens:
			context_token_set.append(token_1)
			if token_1.i not in PIE_tokens:
				for token_2 in content_tokens:
					if token_2.i in PIE_tokens:
						if (not graph_lemma and token_1.lower_ == token_2.lower_) or (graph_lemma and token_1.lemma_ == token_2.lemma_):
							del context_token_set[-1]
							break
		context_tokens = context_token_set			
	# Construct graph
	for idx_1 in range(len(context_tokens)):
		for idx_2 in range(idx_1 + 1, len(context_tokens)):
			# Exclude edges between PIE component words
			if graph_intra:
				if context_tokens[idx_1].i in PIE_tokens and context_tokens[idx_2].i in PIE_tokens:
					continue
			# Use lemma or tokens to build graph
			if graph_lemma:
				words = (context_tokens[idx_1].lemma_, context_tokens[idx_2].lemma_)
			else:
				words = (context_tokens[idx_1].text, context_tokens[idx_2].text)
			in_PIE = context_tokens[idx_1].i in PIE_tokens or context_tokens[idx_2].i in PIE_tokens
			graph.append(Edge(words, in_PIE))

	return graph
	
def weight_graph(graph, embeddings):
	'''
	Weight graph edges by similarity between words
	'''

	for edge in graph:
		try: 
			embedding_0 = embeddings[edge.words[0].lower()]
		except KeyError:
			embedding_0 = embeddings['UNK']
		try:
			embedding_1 = embeddings[edge.words[1].lower()]
		except KeyError:
			embedding_1 = embeddings['UNK']
		edge.similarity = 1.0 - cosine_distance(embedding_0, embedding_1)
		
	return graph
	
def predict_label(graph, graph_type):
	'''
	Predict label based on connectivity change with and without PIE
	'''
	
	# Descriptive statistics
	global num_empty_graphs, num_no_pie_graphs, num_no_context_graphs, total_similarity_difference
	# If graph is empty (i.e. only 0 or 1 content words in sentence including PIE component words), label idiomatic
	if not graph:
		num_empty_graphs += 1
		return 'i'
	# If no PIE component words in the graph, label idiomatic
	if not [edge for edge in graph if edge.in_PIE]:
		num_no_pie_graphs += 1
		return 'i'
	# If no context words in the graph, label idiomatic
	if not [edge for edge in graph if not edge.in_PIE]:
		num_no_context_graphs += 1
		return 'i'
	# Use original connectivity measure
	if graph_type == 'original' or re.match('diff', graph_type):
		# Get similarities
		avg_similarity_with = numpy.mean([edge.similarity for edge in graph])
		avg_similarity_without = numpy.mean([edge.similarity for edge in graph if not edge.in_PIE])
		if graph_type == 'original':
			# Label is idiomatic when similarity is higher or equal without PIE
			if avg_similarity_with >= avg_similarity_without:
				total_similarity_difference += (avg_similarity_with - avg_similarity_without)
				return 'l'
			else:
				total_similarity_difference += (avg_similarity_without - avg_similarity_with)
				return 'i'
		else:
			sign = graph_type[4:5]
			difference = float(graph_type[5:])
			# Label is idiomatic when similarity is higher or equal without PIE, with a minimal difference requirement
			if sign == '-':
				avg_similarity_without -= difference
			elif sign == '+':
				avg_similarity_without += difference
			if avg_similarity_with >= avg_similarity_without:
				return 'l'
			else:
				return 'i'

	# Use only top-N connections between PIE and context words, and between context words
	if re.match('top[0-9]+', graph_type):
		n = int(graph_type[3:])
		top_PIE_similarities = sorted([edge.similarity for edge in graph if edge.in_PIE], reverse = True)[:n]
		top_context_similarities = sorted([edge.similarity for edge in graph if not edge.in_PIE], reverse = True)[:n]
		# Label is literal when PIE has stronger connections to context than context within itself
		if numpy.mean(top_PIE_similarities) >= numpy.mean(top_context_similarities):
			return 'l'
		else:
			return 'i'
			
def modify_PIE(PIE, definition_mapping):
	'''
	Take a PIE and replace its component words with its 
	figurative sense definition and adjust offsets
	'''

	# Create new PIE
	modified_PIE = copy.deepcopy(PIE)
	# Get definition
	try:
		definition = definition_mapping[modified_PIE.pie_type]
	except KeyError:
		print 'No definition found for PIE {0} '.format(modified_PIE.pie_type.encode('utf-8'))
	# Get sentence to modify
	sentence_index = len(modified_PIE.context)/2
	sentence = modified_PIE.context[sentence_index]
	# Remove PIE span and insert sense definition there
	initial_offset = modified_PIE.offsets[0][0]
	final_offset = modified_PIE.offsets[-1][-1]
	modified_sentence = sentence[:initial_offset] + definition + sentence[final_offset:]
	# Create new offsets
	new_offsets = []
	for word in definition.split(' '):
		new_offsets.append([initial_offset, initial_offset + len(word)])
		initial_offset += len(word) + 1
	# Replace originals
	modified_PIE.context[sentence_index] = modified_sentence
	modified_PIE.offsets = new_offsets

	return modified_PIE
			
def predict_label_contrast(graph_literal, graph_figurative, graph_type):
	'''
	Predict label based on connectivity in original graph 
	and graph containing idiom definition
	'''
	
	# Descriptive statistics
	global num_empty_graphs, num_no_pie_graphs, num_no_context_graphs, total_similarity_difference, backoff_counter
	# If graph is empty (i.e. only 0 or 1 content words in sentence including PIE component words), label idiomatic
	if not graph_literal or not graph_figurative:
		num_empty_graphs += 1
		return 'i'
	# If no PIE component words in either graph, label idiomatic
	# If only one graph is empty, back off to single-graph prediction function, compare with and without
	if not [edge for edge in graph_literal if edge.in_PIE] and not [edge for edge in graph_figurative if edge.in_PIE]:
		num_no_pie_graphs += 1
		return 'i'
	elif not [edge for edge in graph_figurative if edge.in_PIE]:
		backoff_counter[1] += 1
		return predict_label(graph_literal, graph_type)
	elif not [edge for edge in graph_literal if edge.in_PIE]:
		backoff_counter[0] += 1
		# Inverse labels
		if predict_label(graph_figurative, graph_type) == 'i':
			return 'l'
		else:
			return 'i'
	# If no context words in the graph, label idiomatic
	if not [edge for edge in graph_literal if not edge.in_PIE] or not [edge for edge in graph_figurative if not edge.in_PIE]:
		num_no_context_graphs += 1
		return 'i'
	# With two well-formed graphs, compare average similarity (connectivity)
	avg_similarity_literal = numpy.mean([edge.similarity for edge in graph_literal])
	avg_similarity_figurative = numpy.mean([edge.similarity for edge in graph_figurative])
	total_similarity_difference += abs(avg_similarity_figurative - avg_similarity_literal)
	if graph_type == 'original':
		# Label is idiomatic when connectivity is higher or equal with figurative sense definition
		if avg_similarity_figurative >= avg_similarity_literal:
			return 'i'
		else:
			return 'l'
	elif re.match('diff', graph_type):
		sign = graph_type[4:5]
		difference = float(graph_type[5:])
		# Label is idiomatic when connectivity is higher or equal with figurative sense definition, with a minimal difference requirement
		if sign == '-':
			avg_similarity_figurative -= difference
		elif sign == '+':
			avg_similarity_figurative += difference
		if avg_similarity_figurative >= avg_similarity_literal:
			return 'i'
		else:
			return 'l'

def pos_tag(pos_tagger, sentences):
	'''
	Take list of sentences, return Spacy Doc that preserves
	original tokenization and sentence split
	'''

	# Normalize quotes, ‘ ’ ❛ ❜ to ', and “ ” ❝ ❞ to ", Spacy doesn't process them well
	sentences = [re.sub(u'‘|’|❛|❜', u"'", sentence) for sentence in sentences]
	sentences = [re.sub(u'“|”|❝|❞', u'"', sentence) for sentence in sentences]
	# Find sentence boundaries
	sentence_lengths = [len(sentence.split()) for sentence in sentences]
	sentence_starts = [sum(sentence_lengths[:i+1]) for i in range(len(sentence_lengths))]
	# Make Doc
	doc = spacy.tokens.Doc(pos_tagger.vocab, ' '.join(sentences).split())
	# Set sentence boundaries
	for token in doc:
		if token.i in sentence_starts:
			token.is_sent_start = True
		else:
			token.is_sent_start = False
	# Do actual tagging
	doc = pos_tagger.tagger(doc)

	return doc
