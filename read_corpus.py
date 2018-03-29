#!/usr/bin/env python2
# -*- coding:utf-8 -*-

'''Read in various PIE corpora into a list of PIE objects'''

import os, copy, re, time, json
import cPickle as pickle
from bs4 import BeautifulSoup
from pie import PIE

def get_sentences_from_BNC(bnc_dir, sentences):
	'''
	Extracts sentences from the British National Corpus. 
	Takes in a dictionary of {'document_id': {sentence_number: ''}} format, 
	returns dictionary of {'document_id': {sentence_number: 'sentence'}} format.
	Using regex rather than parsing the XML is not pretty, but much faster and lighter.
	'''
	
	time_0 = time.time()
	print 'Getting {0} sentences from BNC...'.format(sum(len(doc) for doc in sentences.values()))
	for document_id in sentences:
		document_filename = os.path.join(bnc_dir, document_id + '.xml')
		with open(document_filename, 'r') as document_file:
			# Get whole document as string to circumvent new-line inconsistencies in BNC
			doc_string = ''.join([line.strip() for line in document_file])
			ss = re.findall('<s n=".*?</s>', doc_string)
			for s in ss:
				try:
					found_sentence_number = int(re.findall(r'<s n=".*?"', s)[0][6:-1])
				except ValueError:
					# Some sentences have numbers like 1343_1. Skip those sentences. 
					continue
				if found_sentence_number in sentences[document_id]:
					raw_words = re.findall('(?<=>).*?(?=<)', s)
					words = []
					for raw_word in raw_words:
						# Deal with special XML characters
						if '&' in raw_word:
							raw_word = re.sub('&amp;', '&', raw_word)
							raw_word = re.sub('&gt;', '>', raw_word)							
							raw_word = re.sub('&lt;', '<', raw_word)
						words.append(raw_word)
					sentence_tokenized = unicode(' '.join([word.strip() for word in words if word.strip()]), 'utf-8')
					sentence_untokenized = unicode(''.join(words), 'utf-8')
					sentences[document_id][found_sentence_number] = (sentence_tokenized, sentence_untokenized)
	print 'Done! Took {0:.2f} seconds'.format(time.time() - time_0)
	
	return sentences
	
def convert_VNC_offsets(sentence, offsets):
	'''
	VNC offsets are on encoded strings, convert them to 
	offsets on unicode strings via token offsets
	'''
	
	new_offsets = []
	sentence_encoded = sentence.encode('utf-8')
	# Get token offsets
	token_offsets = []
	character_count = 0
	token_count = 0 
	for token in sentence_encoded.split():
		for offset in offsets:
			if character_count == offset[0]:
				token_offsets.append(token_count)
				break
		character_count += len(token)
		character_count += 1
		token_count += 1
	# Get character offsets on unicode string
	character_count = 0
	for idx, token in enumerate(sentence.split()):
		if idx in token_offsets:
			new_offsets.append([character_count, character_count + len(token)])
		character_count += len(token)
		character_count += 1

	return new_offsets

def read_VNC(vnc_dir, bnc_dir, context_size, work_dir, data_dir, no_cache):
	'''
	Reads in the VNC-Tokens Dataset
	'''
	
	# If possible, read in VNC from cached json-file
	cache_filename = os.path.join(work_dir, 'vnc-c{0}.pickle'.format(context_size))
	if not no_cache and os.path.exists(cache_filename):
		vnc = pickle.load(open(cache_filename, 'r'))
		return vnc
		
	# Load offset mapping
	offset_mapping = json.load(open(os.path.join(data_dir, 'VNC_offset_mapping.json')))

	# Define VNC split
	pie_types_dev = ['blow trumpet', 'find foot', 'get nod', 'hit road', 'hit roof', 'kick heel', 'lose head', 'make face', 'make pile', 'pull leg', 'pull plug', 'pull weight', 'see star', 'take heart']
	pie_types_test = ['blow top', 'blow whistle', 'cut figure', 'get sack', 'get wind', 'have word', 'hit wall', 'hold fire', 'lose thread', 'make hay', 'make hit', 'make mark', 'make scene', 'pull punch']
	pie_types_skewed = ['blow smoke', 'bring luck', 'catch attention', 'catch death', 'catch imagination', 'get drift', 'give notice', 'give sack', 'have fling', 'have future', 'have misfortune', 'hold fort', 'hold horse', 'hold sway', 'keep tab', 'kick habit', 'lay waste', 'lose cool', 'lose heart', 'lose temper', 'make fortune', 'move goalpost', 'set fire', 'take root', 'touch nerve']
				
	vnc = [] # List of PIE objects
	sentence_numbers = {}
	# Only a verb and noun are given, normalize that to make for more canonical dictionary forms
	pie_type_mappings = {'blow top': 'blow one\'s top', 'blow trumpet': 'blow one\'s own trumpet', 'blow whistle': 'blow the whistle', 'catch attention':  'catch someone\'s attention', 'cut figure': 'cut a figure', 'find foot': 'find one\'s feet', 'get drift': 'get the drift', 'get nod': 'get the nod', 'get sack': 'get the sack', 'give sack': 'give someone the sack', 'have fling': 'have a fling', 'have future': 'have a future', 'have misfortune': 'have the misfortune', 'have word': 'have a word', 'hit road': 'hit the road', 'hit roof': 'hit the roof', 'hit wall': 'hit the wall', 'hold fire': 'hold your fire', 'hold fort': 'hold the fort', 'hold horse': 'hold your horse', 'keep tab': 'keep tabs', 'kick habit': 'kick the habit', 'kick heel': 'kick one\'s heels', 'lose cool': 'lose one\'s cool', 'lose head': 'lose one\'s head', 'lose temper': 'lose one\'s temper', 'lose thread': 'lose the thread', 'make face': 'make a face', 'make fortune': 'make a fortune', 'make hit': 'make a hit', 'make mark': 'make one\'s mark', 'make pile': 'make a pile', 'make scene': 'make a scene', 'move goalpost': 'move the goalposts', 'pull leg': 'pull someone\'s leg', 'pull plug': 'pull the plug', 'pull punch': 'pull your punches', 'pull weight': 'pull one\'s weight', 'see star': 'see stars', 'touch nerve': 'touch a nerve'}
	
	# Go through file, collect sentences to extract from BNC
	vnc_filename = os.path.join(vnc_dir, 'VNC-Tokens.csv')
	with open(vnc_filename, 'r') as vnc_file:
		for line in vnc_file:
			split = line.strip().split(' ')
			bnc_document = split[2]
			if bnc_document not in sentence_numbers:
				sentence_numbers[bnc_document] = {}
			bnc_sentence = int(split[3])
			context_sentences = range(bnc_sentence - context_size, bnc_sentence + context_size + 1)
			for context_sentence in context_sentences:
				sentence_numbers[bnc_document][context_sentence] = ('', '') # Tuple of tokenized and untokenized sentence
			
	# Get actual BNC sentences	
	bnc_sentences = get_sentences_from_BNC(bnc_dir, sentence_numbers)
	
	# Go through file, create PIE objects and add to list
	with open(vnc_filename, 'r') as vnc_file:
		for line in vnc_file:
			split = line.strip().split(' ')
			sense_label = split[0]
			# Get binary label
			binary_label = ''
			if sense_label == 'L':
				binary_label = 'l'
			elif sense_label == 'I':
				binary_label = 'i'
			# Get PIE type and split in subcorpora
			pie_type = ' '.join(split[1].split('_'))
			if pie_type in pie_types_dev:
				corpus = 'VNC-dev'
			elif pie_type in pie_types_test:
				corpus = 'VNC-test'
			elif pie_type in pie_types_skewed:
				corpus = 'VNC-skewed'
			# Get BNC info
			bnc_document = split[2]
			bnc_sentence = int(split[3])
			# Get context
			context_sentences = range(bnc_sentence - context_size, bnc_sentence + context_size + 1)
			context = [bnc_sentences[bnc_document][context_sentence][0] for context_sentence in context_sentences]
			context_untokenized = [bnc_sentences[bnc_document][context_sentence][1] for context_sentence in context_sentences]
			# There are 3 cases for which the context (combined document and sentence no.) does not exist in the BNC, skip those
			if context[context_size] == '':
				continue
			# Get offsets
			offsets = offset_mapping[pie_type][bnc_document][str(bnc_sentence)]
			offsets = convert_VNC_offsets(context[context_size], offsets)
			# Normalize PIE type
			if pie_type in pie_type_mappings:
				pie_type = pie_type_mappings[pie_type]
			# Create PIE and add to list
			new_pie = PIE(corpus, pie_type, sense_label, binary_label, context, context_untokenized, offsets)
			vnc.append(new_pie)
			
	# Cache read-in corpus to json-file
	pickle.dump(vnc, open(cache_filename, 'w'))
	
	return vnc
	
def read_IDIX(idix_dir, bnc_dir, context_size, work_dir, data_dir, no_cache):
	'''
	Reads in the IDIX Corpus
	'''

	# If possible, read in IDIX from cached json-file
	cache_filename = os.path.join(work_dir, 'idix-c{0}.pickle'.format(context_size))
	if not no_cache and os.path.exists(cache_filename):
		idix = pickle.load(open(cache_filename, 'r'))
		return idix
				
	idix = [] # List of PIE objects
	sentence_numbers = {}
	
	# Subdirectories
	single_annotated = os.path.join(idix_dir, 'single_data')
	double_annotated = os.path.join(idix_dir, 'double_data')
	
	# Load offset mapping
	offset_mapping = json.load(open(os.path.join(data_dir, 'IDIX_offset_mapping.json')))

	# Get sentence numbers from single and double annotated data
	file_paths_single = sorted([os.path.join(single_annotated, filename) for filename in os.listdir(single_annotated) if re.search('_XK.xml$', filename)])
	file_paths_double = sorted([os.path.join(double_annotated, filename) for filename in os.listdir(double_annotated) if re.search('_pg.xml$', filename)])
	
	for file_path in file_paths_double + file_paths_single:
		print 'Processing {0}...'.format(file_path)
		parsed_xml = BeautifulSoup(open(file_path), 'lxml-xml')
		for finding in parsed_xml.find_all('finding'):
			# Get document id
			bnc_document = finding['file']
			bnc_document = '{0}/{1}/{2}'.format(bnc_document[:1], bnc_document[:2], bnc_document[:3])
			if bnc_document not in sentence_numbers:
				sentence_numbers[bnc_document] = {}
			# Get sentence number
			bnc_sentence = int(finding['sid'])
			context_sentences = range(bnc_sentence - context_size, bnc_sentence + context_size + 1)
			for context_sentence in context_sentences:
				sentence_numbers[bnc_document][context_sentence] = ('', '')  # Tuple of tokenized and untokenized sentence
				
	# Get actual BNC sentences	
	bnc_sentences = get_sentences_from_BNC(bnc_dir, sentence_numbers)

	# Go through files, create PIE objects and add to list
	for file_path in file_paths_double + file_paths_single:
		print 'Processing {0}...'.format(file_path)
		corpus = 'IDIX-single'
		# Get labels of annotator XK as well
		if file_path in file_paths_double:
			file_path_XK = re.sub('_pg', '_XK', file_path)
			parsed_xml = BeautifulSoup(open(file_path_XK), 'lxml-xml')
			labels_XK = [finding['label'] for finding in parsed_xml.find_all('finding')]
			corpus = 'IDIX-double'
		parsed_xml = BeautifulSoup(open(file_path), 'lxml-xml')	
		pie_type = file_path.split('/')[-1].split('_')[0] # XML sometimes lacks type info, so take from filename
		for idx, finding in enumerate(parsed_xml.find_all('finding')):
			sense_label = finding['label']
			# Ignore false extractions (i.e. non-PIEs)
			if sense_label == 'f':
				continue
			# Ignore double annotated PIEs with disagreement
			if file_path in file_paths_double and sense_label != labels_XK[idx]:
				continue
			# Get binary label
			binary_label = ''
			if sense_label in ['n', 'n1', 'n2']:
				binary_label = 'i'
			elif sense_label == 'l':
				binary_label = 'l'
			# Get document id
			bnc_document = finding['file']
			bnc_document = '{0}/{1}/{2}'.format(bnc_document[:1], bnc_document[:2], bnc_document[:3])
			# Get sentence number
			bnc_sentence = int(finding['sid'])
			context_sentences = range(bnc_sentence - context_size, bnc_sentence + context_size + 1)
			context = [bnc_sentences[bnc_document][context_sentence][0] for context_sentence in context_sentences]
			context_untokenized = [bnc_sentences[bnc_document][context_sentence][1] for context_sentence in context_sentences]
			# Get offsets
			offsets = offset_mapping[pie_type][finding['file']][finding['sid']][finding['count']]
			# Create PIE and add to list, unless it's a false extraction
			new_pie = PIE(corpus, pie_type, sense_label, binary_label, context, context_untokenized, offsets)
			idix.append(new_pie)	

	# Filter out duplicates
	idix = list(set(idix))
	
	# Cache read-in corpus to json-file
	pickle.dump(idix, open(cache_filename, 'w'))
	
	return idix
	
def read_PIE_Corpus(pie_dir, bnc_dir, context_size, work_dir, no_cache):
	'''
	Reads in the PIE Corpus
	'''

	# If possible, read from cached json-file
	cache_filename = os.path.join(work_dir, 'pie-corpus-c{0}.pickle'.format(context_size))
	if not no_cache and os.path.exists(cache_filename):
		pie_corpus = pickle.load(open(cache_filename, 'r'))
		return pie_corpus

	pie_corpus = [] # List of PIE objects
	sentence_numbers = {}
	
	# Go through file, collect sentences to extract from BNC
	pie_corpus_parts = ['train', 'dev', 'test']
	pie_corpus_filenames = [os.path.join(pie_dir, 'PIE_annotations_type_{0}_v2.json'.format(part)) for part in pie_corpus_parts]
	pie_corpus_files = [open(pie_corpus_filename, 'r') for pie_corpus_filename in pie_corpus_filenames]
	for pie_corpus_file in pie_corpus_files:
		pie_corpus_json = json.load(pie_corpus_file)
		for pie in pie_corpus_json:
			# Get document id
			bnc_document = pie['document_id']
			bnc_document = '{0}/{1}/{2}'.format(bnc_document[:1], bnc_document[:2], bnc_document[:3])
			if bnc_document not in sentence_numbers:
				sentence_numbers[bnc_document] = {}		
			# Get sentence number
			bnc_sentence = int(pie['sentence_number'])
			context_sentences = range(bnc_sentence - context_size, bnc_sentence + context_size + 1)
			for context_sentence in context_sentences:
				sentence_numbers[bnc_document][context_sentence] = ('', '') # Tuple of tokenized and untokenized sentence
			
	# Get actual BNC sentences	
	bnc_sentences = get_sentences_from_BNC(bnc_dir, sentence_numbers)
	
	# Go through file, create PIE objects and add to list
	pie_corpus_files = [open(pie_corpus_filename, 'r') for pie_corpus_filename in pie_corpus_filenames]
	for pie_corpus_file in pie_corpus_files:
		if 'train' in pie_corpus_file.name:
			corpus = 'PIE-train'
		elif 'dev' in pie_corpus_file.name:
			corpus = 'PIE-dev'
		elif 'test' in pie_corpus_file.name:
			corpus = 'PIE-test'
		pie_corpus_json = json.load(pie_corpus_file)
		for pie in pie_corpus_json:
			if pie['PIE_label'] == 'y':
				sense_label = pie['sense_label']
				# Get binary label
				binary_label = ''
				if sense_label == 'n':
					binary_label = 'l'
				elif sense_label == 'y':
					binary_label = 'i'
				pie_type = pie['idiom']
				bnc_document = pie['document_id']
				bnc_document = '{0}/{1}/{2}'.format(bnc_document[:1], bnc_document[:2], bnc_document[:3])
				bnc_sentence = int(pie['sentence_number'])
				context_sentences = range(bnc_sentence - context_size, bnc_sentence + context_size + 1)
				context = [bnc_sentences[bnc_document][context_sentence][0] for context_sentence in context_sentences]
				context_untokenized = [bnc_sentences[bnc_document][context_sentence][1] for context_sentence in context_sentences]
				offsets = pie['offsets']
				# Create PIE and add to list
				new_pie = PIE(corpus, pie_type, sense_label, binary_label, context, context_untokenized, offsets)
				pie_corpus.append(new_pie)

	# Cache read-in corpus to json-file
	pickle.dump(pie_corpus, open(cache_filename, 'w'))

	return pie_corpus
	
def read_SemEval(semeval_dir, semeval_contexts, context_size, work_dir, no_cache):
	'''
	Reads in the SemEval-2013 Task 5b corpus
	'''
	
	# If possible, read in SemEval from cached json-file
	cache_filename = os.path.join(work_dir, 'semeval-c{0}.pickle'.format(context_size))
	if not no_cache and os.path.exists(cache_filename):
		semeval = pickle.load(open(cache_filename, 'r'))
		return semeval
		
	semeval = [] # List of PIE objects
	# The verbs and nouns in dictionary forms in SemEval are lemmatized, normalize that to make forms more canonical
	pie_type_mappings = {'anything go': 'anything goes', 'behind bar': 'behind bars', 'be leave hold the baby': 'be left holding the baby', 'carve in stone': 'carved in stone', 'deer in the headlight': 'deer in the headlights', 'have kitten': 'have kittens', 'small potato': 'small potatoes'}

	# Read in mapping of SemEval IDs to re-extracted contexts from ukWaC
	context_mapping = json.load(open(semeval_contexts, 'r'))
	# Go through files, create PIE objects and add to list
	for filename in sorted(os.listdir(semeval_dir)):
		if re.search('^subtask5b_en_.*.txt$', filename):
			file_path = os.path.join(semeval_dir, filename)
			with open(file_path, 'r') as f:
				for line in f:
					split = line.strip().split('\t')
					corpus = 'SemEval-' + '-'.join(filename[:-4].split('_')[2:4])
					pie_type = split[1]
					if pie_type in pie_type_mappings:
						pie_type = pie_type_mappings[pie_type]
					sense_label = split[2]
					# Get binary label
					binary_label = ''
					if sense_label == 'figuratively':
						binary_label = 'i'
					elif sense_label == 'literally':
						binary_label = 'l'
					original_context = unicode(split[3], 'utf-8')
					PIE_form = re.sub(r'</?b>', '', re.findall('<b>.*</b>', original_context)[0])
					# Get context
					try:
						context_tokenized = context_mapping[split[0]][3][20 - context_size : 20 + context_size + 1]
					except KeyError:
						# For 3 cases, no context was found, exclude these
						continue
					# 'untokenize' text to some extent
					context_untokenized = [re.sub(' ([,.?:!\)])', r'\1', sentence) for sentence in context_tokenized]
					context_untokenized = [re.sub('([\(]) ', r'\1', sentence) for sentence in context_untokenized]
					context_untokenized = [re.sub(" (n't|'s|'re|'ve|'m|'d|'ll)", r'\1', sentence) for sentence in context_untokenized]
					# Get offsets of PIE by finding PIE form in re-extracted context
					offsets = []
					for match in re.finditer(PIE_form, context_tokenized[context_size]):
						start = match.start()
						end = match.end()
						counter = 0
						for char in context_tokenized[context_size][start:end]:
							counter += 1
							if char == ' ':
								offsets.append([start, start + counter - 1])
								start = start + counter
								counter = 0
						offsets.append([start, end])
						break
					# Filter out offsets of determiners
					filtered_offsets = []
					for offset_pair in offsets:
						if not context_tokenized[context_size][offset_pair[0]:offset_pair[1]].lower() in ['a', 'an', 'the']:
							filtered_offsets.append(offset_pair)
					# Create PIE and add to list
					new_pie = PIE(corpus, pie_type, sense_label, binary_label, context_tokenized, context_untokenized, filtered_offsets)
					semeval.append(new_pie)

	# Cache read-in corpus to json-file
	pickle.dump(semeval, open(cache_filename, 'w'))

	return semeval
