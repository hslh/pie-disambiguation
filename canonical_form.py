#!/usr/bin/env python2
# -*- coding:utf-8 -*-

'''Baseline that classifies all PIE instances in a canonical form as idiomatic'''

import json, re, os

def canonical_form(data, use_test_data, context_size, data_dir):
	'''
	Apply canonical form baseline
	'''

	canonical_forms = get_canonical_forms(data, use_test_data, data_dir)
	# Classify everything in canonical form as idiomatic
	for PIE in data:
		if (use_test_data and PIE.split == 'test') or (not use_test_data and PIE.split == 'dev'):
			PIE_sentence = PIE.context[context_size]
			PIE_forms = canonical_forms[PIE.pie_type]
			PIE.predicted_label = 'l'
			for PIE_form in PIE_forms:
				if re.search(PIE_form, PIE_sentence, flags = re.I): # Ignore case
					PIE.predicted_label = 'i'
					break

	return data

def get_canonical_forms(data, use_test_data, data_dir):
	'''
	Retrieves canonical forms for PIE types from dictionaries
	'''
	
	canonical_forms = {}
	# Load dictionaries, lower-case and join together
	ue = json.load(open(os.path.join(data_dir, 'idiom_list_ue_final.json'), 'r'))
	wiktionary = json.load(open(os.path.join(data_dir, 'idiom_list_wiktionary_final.json'), 'r'))
	combined_dictionary = list(set([entry.lower() for entry in ue + wiktionary]))
	# Get set of PIE types
	PIE_types = list(set([PIE.pie_type for PIE in data if (use_test_data and PIE.split == 'test') or (not use_test_data and PIE.split == 'dev')]))
	# Select canonical forms from dictionary by matching content words
	for PIE_type in PIE_types:
		canonical_forms[PIE_type] = [PIE_type]
		words = PIE_type.split(' ')
		words = [word for word in words if word not in ['a', 'the', 'an']]
		for entry in combined_dictionary:
			match = True
			for word in words:
				if not re.search(word, entry, flags = re.I):
					match = False
					break # Check next entry if one of the words is not found
			if match:
				canonical_forms[PIE_type].append(entry)
		# Deal with placeholder words (one's, someone's, etc.)
		canonical_forms[PIE_type] = expand_indefinite_pronouns(canonical_forms[PIE_type])
		# Filter out duplicates
		canonical_forms[PIE_type] = list(set(canonical_forms[PIE_type]))

	return canonical_forms
	
def expand_indefinite_pronouns(canonical_forms):
	'''
	When one's,someone's, someone, your or something occurs in a canonical
	form, add canonical forms with personal pronouns added in. Don't 
	expand 'one', because it is too ambiguous.
	'''
	
	expanded_canonical_forms = []
	possessive_pronouns = ['my', 'your', 'his', 'her', 'its', 'our', 'their']
	objective_pronouns = ['me', 'you', 'him', 'her', 'us', 'them', 'it']

	for canonical_form in canonical_forms:
		expanded_canonical_forms.append(canonical_form)
		# Add possessive pronouns only
		if re.search("\\b(one's|your)\\b", canonical_form):
			for possessive_pronoun in possessive_pronouns:
				expanded_canonical_form = re.sub("\\b(one's|your)\\b", possessive_pronoun, canonical_form)
				expanded_canonical_forms.append(expanded_canonical_form)
		# Add possessive pronouns and a wildcard for other words
		elif re.search("\\bsomeone's\\b", canonical_form):
			for possessive_pronoun in possessive_pronouns + [unicode("\w+'s", 'utf-8')]:
				expanded_canonical_form = re.sub("\\bsomeone's\\b", possessive_pronoun, canonical_form)
				expanded_canonical_forms.append(expanded_canonical_form)
		# Add objective pronouns and a wildcard for other words
		elif re.search("\\bsomeone\\b", canonical_form):
			for objective_pronoun in objective_pronouns + [unicode("\w+", 'utf-8')]:
				expanded_canonical_form = re.sub("\\bsomeone\\b", objective_pronoun, canonical_form)
				expanded_canonical_forms.append(expanded_canonical_form)
		# Add 'it' and 'them' and a wildcard for other words
		elif re.search("\\bsomething\\b", canonical_form):
			for objective_pronoun in ['it', 'them', unicode("\w+", 'utf-8')]:
				expanded_canonical_form = re.sub("\\bsomething\\b", objective_pronoun, canonical_form)
				expanded_canonical_forms.append(expanded_canonical_form)

	return expanded_canonical_forms
