import os, sys
import pandas as pd
from Bio.SubsMat import MatrixInfo
from Bio.Alphabet.IUPAC import IUPACProtein
from itertools import product
import tensorflow as tf
from sklearn.model_selection import train_test_split

from modules import tcrpair_config

# Functions to process data in order to make it usable for tcrpair_train.py

def generate_enc_dict(matrix: str):
	enc_dict = {x : list() for x in IUPACProtein.letters}
	if matrix == 'onehot':
		for i, x in enumerate(IUPACProtein.letters):
			enc_dict[x] = [0] * len(IUPACProtein.letters)
			enc_dict[x][i] = 1
		return enc_dict
	elif matrix == 'blosum50':
		encmatrix = MatrixInfo.blosum50
	elif matrix == 'blosum62':
		encmatrix = MatrixInfo.blosum62
	elif matrix == 'blosum75':
		encmatrix = MatrixInfo.blosum75
	elif matrix == 'blosum85':
		encmatrix = MatrixInfo.blosum85
	elif matrix == 'blosum100':
		encmatrix = MatrixInfo.blosum100
	else:
		sys.exit('[ERROR] {} is not a valid encoding matrix.'.format(matrix))
	for x, y in product(IUPACProtein.letters, repeat=2):
		if (x, y) in encmatrix:
			enc_dict[x].append(encmatrix[(x, y)])
		else:
			enc_dict[x].append(encmatrix[(y, x)])
	return enc_dict
### FUNCTION END ###

def encode_sequences_df(df: pd.DataFrame, encoding: str, *fields):
	"""
	:return: df with fields.enc encoded by substitution matrix
	:return: list of unique values used for encoding
	"""
	vocab_str = str()
	enc_dict = generate_enc_dict(encoding)
	enc_fields = list()
	for field in fields[0]:
		df['{}.enc'.format(field)] = df[field].apply(lambda x : [enc_dict[y] for y in list(x)])
		enc_fields.append('{}.enc'.format(field))
		vocab_str = ''.join([vocab_str, str(df[field].sum())])
	return df, set(vocab_str)
### FUNCTION END ###

def concatenate_and_pad(alpha: list, beta: list, maxlen = None):
	if maxlen is None:
		return tf.keras.preprocessing.sequence.pad_sequences([alpha + beta], padding='post')
	else:
		return tf.keras.preprocessing.sequence.pad_sequences([alpha + beta], padding='post', maxlen=maxlen)
### FUNCTION END ###

def make_negative_dataset(df: pd.DataFrame, seed: int = 10, part: bool = False):
	""" Generates a negative dataset of TCR pairs of chains specific for different antigens, HLA is not considered
		(we assume a subset with only one HLA allele """
	if part is False:
		alpha_cols = [x for x in tcrpair_config.alpha_cols + tcrpair_config.antigen_cols if not x.startswith('part')]
		beta_cols = [x for x in tcrpair_config.beta_cols + tcrpair_config.antigen_cols if not x.startswith('part')]
	else:
		alpha_cols = tcrpair_config.alpha_cols + tcrpair_config.antigen_cols
		beta_cols = tcrpair_config.beta_cols + tcrpair_config.antigen_cols
	# Split dataset by alpha and beta, preserve antigen information and shuffle
	alpha_df = df[alpha_cols].sample(
		frac=1, random_state=seed).reset_index(drop=True)
	beta_df  = df[beta_cols].sample(
		frac=1, random_state=int(seed/2)).reset_index(drop=True)

	neg_df = pd.merge(alpha_df, beta_df, left_index=True, right_index=True)
	# Select only those TCR pairs with mismatching epitope
	neg_df = neg_df.loc[neg_df['antigen.epitope_x'] != neg_df['antigen.epitope_y']].reset_index(drop=True)
	def select_epitope(dfrow):
		""" Returns x suffix epitope for odd columns and y suffix epitope for even columns """
		if dfrow.name%2 == 0:
			return dfrow['antigen.epitope_y']
		else:
			return dfrow['antigen.epitope_x']
	### FUNCTION END ###
	neg_df['antigen.epitope'] = neg_df.apply(select_epitope, axis=1)
	return neg_df.drop(labels=[x for x in neg_df.columns if len(x.split('_')) > 1], axis=1)
### FUNCTION END ###

def generate_datasets(pos_df: pd.DataFrame, neg_df: pd.DataFrame, dt: str):
	""" Function to generate datasets for training/evaluating consisting of different datatypes (CDR3, full, etc.) """
	# Generate encoded datasets
	df = pd.concat([
		pd.DataFrame({
			'{}.alpha.enc'.format(dt) : pos_df['{}.alpha.enc'.format(dt)].values,
			'{}.beta.enc'.format(dt)  : pos_df['{}.beta.enc'.format(dt)].values,
			'recognition'   		  : [1] * len(pos_df)
		}), pd.DataFrame({
			'{}.alpha.enc'.format(dt) : neg_df['{}.alpha.enc'.format(dt)].values,
			'{}.beta.enc'.format(dt)  : neg_df['{}.beta.enc'.format(dt)].values,
			'recognition'    		  : [0] * len(neg_df)
		})
	]).reset_index(drop=True)
	# Concatenate sequences
	def concat_lists(dfrow):
		return dfrow['{}.alpha.enc'.format(dt)] + dfrow['{}.beta.enc'.format(dt)]
	df['concatseqs'] = df.apply(concat_lists, axis=1)
	# Return padded sequences, labels, max input length
	return tf.keras.preprocessing.sequence.pad_sequences(
				df['concatseqs'].values, padding='post', value=tcrpair_config.mask_value
	), df['recognition'].values, max(len(l) for l in df['concatseqs'].values)
### FUNCTION END ###

def split_data(xdata, ydata):
	return train_test_split(xdata, ydata, test_size=0.2, random_state=10)
### FUNCTION END ###

def store_validation_data(pos_df: pd.DataFrame, neg_df: pd.DataFrame):
	pos_df.insert(0, 'recognition', 1)
	neg_df.insert(0, 'recognition', 0)
	df = pos_df.append(neg_df)
	X_train_df, X_test_df, y_train, y_test = split_data(df[['antigen.epitope', 'antigen.gene', 'antigen.species',
				   'cdr3.alpha', 'cdr3.beta',
				   'j.alpha', 'j.beta', 'v.alpha', 'v.beta', 'mhc.a']], df['recognition'])
	X_test_df.insert(len(X_test_df.columns), 'recognition', y_test)
	X_test_df.to_csv(
		os.path.join(
			tcrpair_config.val_path, 'firstvalidationdata_df.tsv.gz'
		), compression='gzip', sep='\t', index=False
	)
### FUNCTION END ###
