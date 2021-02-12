import os
import pandas as pd
import tensorflow as tf
from sklearn import metrics
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from modules import tcrpair_config, mldata, mlmodels, tcr_reconstruction

def main():

	mn_list = list()
	hla_list = list()
	auc_list = list()

	for modelname in os.listdir(tcrpair_config.models_path):
		# Load model
		model = mlmodels.load_model(modelname)
		maxlen = model.layers[0].input_shape[1]
		dt = modelname.split('_')[0]

		# Load data
		test_df = pd.read_csv(
			os.path.join(tcrpair_config.td_path, 'trainingdata_df.tsv.gz'), sep='\t', index_col=None, low_memory=False
		)

		for hla in ['HLA-A*03:01', 'HLA-A*11:01', 'HLA-B*08:01']:

			pos_df = test_df.loc[test_df['mhc.a'].str.startswith(hla)]

			seq_cols = [x for x in tcrpair_config.seq_cols if not x.startswith('part')]
			part = False
			if dt == 'part':
				# Add partial sequences
				pos_df = tcr_reconstruction.add_partial_tcr_to_df(pos_df, int(modelname.split('_')[1]))
				seq_cols = tcrpair_config.seq_cols
				part = True
			neg_df = mldata.make_negative_dataset(pos_df, part=part)

			pos_df, _ = mldata.encode_sequences_df(pos_df, 'blosum62', seq_cols)
			neg_df, _ = mldata.encode_sequences_df(neg_df, 'blosum62', seq_cols)

			def concat_lists(dfrow):
				return dfrow['{}.alpha.enc'.format(dt)] + dfrow['{}.beta.enc'.format(dt)]
			pos_df['concatseqs'] = pos_df.apply(concat_lists, axis=1)
			neg_df['concatseqs'] = neg_df.apply(concat_lists, axis=1)

			pos_df['predicted.functionality'] = model.predict(
				tf.keras.preprocessing.sequence.pad_sequences(
					pos_df['concatseqs'].values, padding='post', maxlen=maxlen, value=tcrpair_config.mask_value
				)
			).round()
			neg_df['predicted.functionality'] = model.predict(
				tf.keras.preprocessing.sequence.pad_sequences(
					neg_df['concatseqs'].values, padding='post', maxlen=maxlen, value=tcrpair_config.mask_value
				)
			).round()

			auc = metrics.roc_auc_score(
				[1] * len(pos_df) + [0] * len(neg_df),
				pos_df['predicted.functionality'].to_list() + neg_df['predicted.functionality'].to_list()
			)
			mn_list.append(modelname)
			hla_list.append(hla)
			auc_list.append(auc)

	results_df = pd.DataFrame(
		{'modelname' : mn_list,
		'hla' : hla_list,
		'auc' : auc_list}
	)
	print(results_df.pivot(index='modelname', columns='hla', values='auc'))

### FUNCTION END ###

if __name__ == '__main__':
	main()

