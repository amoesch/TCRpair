import os
import argparse
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from modules import tcrpair_config, mlmodels, mldata, tcr_reconstruction

def main():
	parser = argparse.ArgumentParser(description='TCRpair: Prediction of TCR pairing')
	parser.add_argument('-m',
						choices=['one', 'two'],
						required=True,
						help='Predict likelihoods for every TCR [one] or compare alpha/beta TCR combinations [two]. '
							 'The latter requires two TCRs with the same identifier (first row) in input.')
	parser.add_argument('-d',
						choices=['cdr3', 'part_3', 'part_5', 'part_7', 'part_9', 'part_11', 'part_13', 'part_15', 'full'],
						required=True,
						help='Model input type. Everything except [cdr3] requires V and J allele annotation')
	parser.add_argument('-f',
						type=str,
						required=True,
						help='File path to input textfile. Has to be CSV and has to contain the following columns:\n'
							 'name, cdr3.alpha, cdr3.beta and v.alpha, j.alpha, v.beta and j.beta if -d is not [cdr3].')

	args = parser.parse_args()
	predictmode = args.m
	modelprefix = args.d
	filepath    = args.f

	#############################################################

	dt = modelprefix.split('_')[0]
	cols = ['name', 'cdr3.alpha', 'cdr3.beta']
	if dt != 'cdr3':
		cols = cols + ['v.alpha', 'j.alpha', 'v.beta', 'j.beta']

	try:
		# Load data
		df = pd.read_csv(filepath, usecols=cols)

		# Reconstruct sequences (if necessary)
		if dt != 'cdr3':
			df = tcr_reconstruction.add_full_tcr_to_df(df, 'VDJdb', dropnnanalleles=False)
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
				x_test = tf.keras.preprocessing.sequence.pad_sequences(
					df['concatseqs'].values, padding='post', maxlen=maxlen, value=tcrpair_config.mask_value
				)
				df['predicted.likelihood'] = model.predict(x_test)

				# Print output
				if predictmode == 'one':
					df['predicted.functionality'] = df['predicted.likelihood'].round()
					print(df.loc[:, ['name', 'cdr3.alpha', 'cdr3.beta', 'predicted.functionality', 'predicted.likelihood']].to_string())
				elif predictmode == 'two':
					nm_df_list = list()
					af_df_list = list()
					bf_df_list = list()
					lf_df_list = list()
					ln_df_list = list()
					for name in df['name'].unique():
						tmp_df = df.loc[df['name'] == name]
						higher_lh = tmp_df['predicted.likelihood'].max()
						nm_df_list.append(name)
						af_df_list.append(tmp_df.loc[tmp_df['predicted.likelihood'] == higher_lh, ['cdr3.alpha']].values[0][0])
						bf_df_list.append(tmp_df.loc[tmp_df['predicted.likelihood'] == higher_lh, ['cdr3.beta']].values[0][0])
						lf_df_list.append(higher_lh)
						ln_df_list.append(tmp_df.loc[tmp_df['predicted.likelihood'] != higher_lh, ['predicted.likelihood']].values[0][0])
					print(pd.DataFrame(
						{'name' : nm_df_list,
						 'cdr3.alpha.functional' : af_df_list,
						 'cdr3.beta.functional' : bf_df_list,
						 'predicted.likelihood' : lf_df_list,
						 'predicted.likelihood.nonfunctional' : ln_df_list}).to_string())
	except IOError as e:
		print(e)
### FUNCTION END ###

if __name__ == '__main__':
	main()
