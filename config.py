#!/usr/bin/env python2
# -*- coding:utf-8 -*-

'''Sets parameters, parses and validates command-line arguments'''

import argparse, os, re

# Parameters
WORK_DIR = './working'
EXT_DIR = './ext'
DATA_DIR = './data'
BNC_DIR = os.path.join(EXT_DIR, 'BNC')
VNC_DIR = os.path.join(EXT_DIR, 'VNC')
IDIX_DIR = os.path.join(EXT_DIR, 'IDIX')
SEMEVAL_DIR = os.path.join(EXT_DIR, 'SemEval')
PIE_DIR = os.path.join(EXT_DIR, 'PIE_Corpus')
EMB_DIR = os.path.join(EXT_DIR, 'glove')
SEMEVAL_CONTEXTS = os.path.join(DATA_DIR, 'mapping_SemEval_ukWaC_with_context_xml.json')

# Read in arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--context', type = int, metavar = 'N', default = 0, help = 'Number of context sentences to extract around the sentence containing the PIE, on each side.')
parser.add_argument('-m', '--method', type = str, metavar = 'mfs|cform|cg', default = 'mfs', help = "Method to use for disambiguation. Options are most frequent sense (mfs), canonical form (cform), and cohesion graph (cg).")
parser.add_argument('-gs', '--graph-size', type = str, metavar = '{0-9}+{ws}', default = '0s', help = "Amount of context to use for the cohesion graph. Can be a number of words or sentences. '0w/s' will yield only the PIE (sentence), '1w/s' one word/sentence of relevant context on both sides of the PIE (sentence).")
parser.add_argument('-gp', '--graph-pos', type = str, metavar = 'n,v,a,r,p,P', default = 'n,v', help = "Which PoS to include in cohesion graph, noun/verb/adjective/adverb/preposition/proper noun.")
parser.add_argument('-gk', '--graph-keep-pie', action = 'store_true', help = "Always keep all PIE component words in the graph, regardless of PoS.")
parser.add_argument('-gd', '--graph-no-duplicates', action = 'store_true', help = "Exclude context tokens/lemmata which are duplicates of component words.")
parser.add_argument('-gl', '--graph-lemma', action = 'store_true', help = "Build graph with lemmata instead of tokens.")
parser.add_argument('-ge', '--graph-embedding-size', type = int, metavar = '50|100|200|300|840', default = 50, help = "Which dimensionality GloVe word embeddings to use. All are on 6B tokens, except 840, which is 300-dimensional on 840B tokens.")
parser.add_argument('-gi', '--graph-no-internal', action = 'store_true', help = "Exclude edges between PIE component words from graph.")
parser.add_argument('-gm', '--graph-measure', type = str, metavar = 'original|topN|diff+-X.X', default = 'original', help = "Which graph connectivity measure to use for label prediction. TopN works best with the '-gi' flag.")
parser.add_argument('-gf', '--graph-definitions', action = 'store_true', help = "Use figurative sense definitions to create contrasting graphs.")
parser.add_argument('-t', '--test', action = 'store_true', help = "Run and evaluate on test set, instead of development set.")
parser.add_argument('-nc', '--no-cache', action = 'store_true', help = "Read corpora from file, do not use cache.")
args = parser.parse_args()

# Validate and store arguments
CONTEXT_SIZE = args.context
TEST = args.test
NO_CACHE = args.no_cache
METHOD = args.method.lower()
if args.method.lower() not in ['mfs', 'cform', 'cg']:
	raise ValueError('{0} is not a valid method.'.format(args.method))

if re.match('[0-9]+[ws]', args.graph_size):
	GRAPH_SIZE = args.graph_size
else:
	raise ValueError("No valid context window argument provided. Should be of the format [0-9]+[ws].")	

# Check if sufficient context is extracted
if re.match('[0-9]+[s]', args.graph_size):
	if int(args.context) < int(args.graph_size[:-1]):
		raise ValueError("Insufficient context extracted to build the graph.")
if re.match('[0-9]+[w]', args.graph_size):
	if (3 * int(args.context)) < int(args.graph_size[:-1]):
		print "Extracted context size might not be large enough to build the graph."

# Convert PoS labels to Spacy PoS
GRAPH_POS = args.graph_pos.split(',')
mapping = {'n': 'NOUN', 'v': 'VERB', 'a': 'ADJ', 'r': 'ADV', 'p': 'ADP', 'P': 'PROPN'}
GRAPH_POS  = [mapping[pos] for pos in GRAPH_POS]

GRAPH_PIE = args.graph_keep_pie
GRAPH_SET = args.graph_no_duplicates
GRAPH_LEMMA = args.graph_lemma

if args.graph_embedding_size in [50, 100, 200, 300, 840]:
	GRAPH_GLOVE = args.graph_embedding_size
else:
	raise ValueError('{0} is not a valid embedding size.'.format(args.graph_embedding_size))

GRAPH_INTRA = args.graph_no_internal

if re.match('original|top[0-9]+|diff[+\-][0-9]\.[0-9]+', args.graph_measure):
	GRAPH_TYPE = args.graph_measure
else:
	raise ValueError('{0} is not a valid graph measure.'.format(args.graph_measure))

GRAPH_DEFINITIONS = args.graph_definitions
