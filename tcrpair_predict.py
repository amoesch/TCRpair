import os, sys
import argparse
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from Bio.Alphabet.IUPAC import IUPACProtein

from modules import tcrpair_config, mlmodels, mldata, tcr_reconstruction

def main():
	parser = argparse.ArgumentParser(description='TCRpair: Prediction of TCR pairing')
	parser.add_argument('-m',
						choices=['cdr3', 'part_3', 'part_5', 'part_7', 'part_9', 'part_11', 'part_13', 'part_15', 'full'],
						required=True,
						help='Model input type. Everything except [cdr3] requires V and J allele annotation')
	parser.add_argument('-a',
						type=str,
						required=True,
						help='Path to input textfile. There are two possibilities:\n'
							 '1) Has to be either tab or comma separated and has to contain the following columns:\n'
							 'cdr3.alpha, cdr3.beta and v.alpha, j.alpha, v.beta and j.beta if -d is not [cdr3].\n'
							 '2) Has to be MiXCR >= v2 output for alpha chain. In this case, -b is required.')
	parser.add_argument('-b',
						type=str,
						required=False,
						help='MiXCR >=2 output for beta chain. Only required if the corresponding alpha chain file is given for -a.')
	parser.add_argument('-o',
						type=str,
						required=True,
						help='Path to output file.')
	parser.add_argument('-c',
						type=int,
						required=False,
						default=100,
						help='Minimum clone count for all chains in MiXCR output to be used by TCRpair. Default = 100.')
	parser.add_argument('-f',
						type=float,
						required=False,
						default=0.5,
						help='Difference in clone fraction between alpha and beta chains in MiXCR output to be mapped as pair.'
							 'TCRpair will predict functional pairing only for mapped pairs. Default = 0.5.')

	args = parser.parse_args()
	modelprefix = args.m
	afilepath   = args.a
	bfilepath   = args.b
	outputfile  = args.o
	ccount		= args.c
	cfraction   = args.f

	#############################################################

	dt = modelprefix.split('_')[0]
	vdjdbcols = ['cdr3.alpha', 'cdr3.beta']
	mixcrcols = ['aaSeqCDR3', 'bestVHit', 'bestJHit', 'cloneCount', 'cloneFraction']
	if dt != 'cdr3':
		vdjdbcols = vdjdbcols + ['v.alpha', 'j.alpha', 'v.beta', 'j.beta']

	try:
		# Load data
		df = pd.read_csv(afilepath, sep='\t|,', engine='python')

		# Identify input type
		def check_columns(cols, subcols):
			if len([x for x in subcols if x not in cols]) == 0:
				return True
			else:
				return False
		if check_columns(df.columns.to_list(), mixcrcols) is True:
			if bfilepath is None:
				sys.exit('[ERROR] Beta chain input file required for MiXCR input.')
			else:
				b_df = pd.read_csv(bfilepath, sep='\t')
				# Find pairs for prediction
				ca_list = list()
				cb_list = list()
				va_list = list()
				ja_list = list()
				vb_list = list()
				jb_list = list()
				for a in df.index:
					if df.loc[a, 'cloneCount'] >= ccount:
						for b in b_df.index:
							if b_df.loc[b, 'cloneCount'] >= ccount and\
									abs(df.loc[a, 'cloneFraction'] - b_df.loc[b, 'cloneFraction']) <= cfraction:
								ca_list.append(df.loc[a, 'aaSeqCDR3'])
								cb_list.append(b_df.loc[b, 'aaSeqCDR3'])
								va_list.append(df.loc[a, 'bestVHit'])
								ja_list.append(df.loc[a, 'bestJHit'])
								vb_list.append(b_df.loc[b, 'bestVHit'])
								jb_list.append(b_df.loc[b, 'bestJHit'])
				df = pd.DataFrame({'cdr3.alpha' : ca_list,
								   'cdr3.beta' : cb_list,
					               'v.alpha' : va_list,
				                   'j.alpha' : ja_list,
				                   'v.beta' : vb_list,
				                   'j.beta' : jb_list})
		elif check_columns(df.columns.to_list(), vdjdbcols) is False:
			sys.exit('[ERROR] Input type not identified, please check -h for requirements.')

		# Filter for non-amino acid sequences
		def aacheck(seq):
			check = True
			for s in seq:
				if s not in IUPACProtein.letters:
					check = False
			return check
		for i in df.index:
			if aacheck(df.loc[i, ['cdr3.alpha']][0]) is False or aacheck(df.loc[i, ['cdr3.beta']][0]) is False:
				df.drop(i, inplace=True)

		# Reconstruct sequences (if necessary)
		if dt != 'cdr3':
			df = tcr_reconstruction.add_full_tcr_to_df(df, 'VDJdb', dropnanalleles=True)
			if dt == 'part':
				df = tcr_reconstruction.add_partial_tcr_to_df(df, int(modelprefix.split('_')[1]))

		# Encode
		df, _ = mldata.encode_sequences_df(df, 'blosum62', ['{}.alpha'.format(dt), '{}.beta'.format(dt)])
		def concat_lists(dfrow):
			return dfrow['{}.alpha.enc'.format(dt)] + dfrow['{}.beta.enc'.format(dt)]
		df['concatseqs'] = df.apply(concat_lists, axis=1)

		# Load model
		for modelname in os.listdir(tcrpair_config.models_path):
			if modelname.startswith(modelprefix):
				print('Loading model {}'.format(modelprefix))
				model = mlmodels.load_model(modelname)
				maxlen = model.layers[0].input_shape[1]

				# Predict
				print('Predicting')
				x_test = tf.keras.preprocessing.sequence.pad_sequences(
					df['concatseqs'].values, padding='post', maxlen=maxlen, value=tcrpair_config.mask_value
				)
				df['predicted.likelihood'] = model.predict(x_test)
				df['predicted.functionality'] = df['predicted.likelihood'].round()

				df.to_csv(outputfile, sep='\t', index=False,
						  columns=vdjdbcols + ['predicted.likelihood', 'predicted.functionality'])
		print('Done')
	except IOError as e:
		print(e)
### FUNCTION END ###

if __name__ == '__main__':
	main()
