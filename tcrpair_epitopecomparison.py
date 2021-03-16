import os
import pandas as pd
import tensorflow as tf
from sklearn import metrics
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import seaborn as sns
import matplotlib.pyplot as plt

from modules import tcrpair_config, mldata, mlmodels, tcr_reconstruction

def main():

	# Load data
	test_df = pd.read_csv(
		os.path.join(tcrpair_config.val_path, 'firstvalidationdata_df.tsv.gz'), sep='\t', index_col=None, low_memory=False
	)
	test_df = tcr_reconstruction.add_full_tcr_to_df(test_df, 'VDJdb', dropnanalleles=False)

	test_epitopes = ['GILGFVFTL', 'NLVPMVATV', 'GLCTLVAML']

	epi_comp = {x : list() for x in ['model', 'epitope', 'n', 'auc']}

	nettcr_comp = {x : list() for x in [
		'epitope', 'model', 'tcrpair auc', 'nettcr auc',
		'tcrpair precision', 'nettcr precision', 'tcrpair recall', 'nettcr recall', 'all pos tcrs', 'nettcr train pos tcrs']}

	for modelname in os.listdir(tcrpair_config.models_path):
		if modelname.split('_')[1] in ['5', '7', '9', '11']:
			# Load model
			model = mlmodels.load_model(modelname)
			maxlen = model.layers[0].input_shape[1]
			dt = modelname.split('_')[0]

			for epitope in test_df['antigen.epitope'].unique():

				ep_df = test_df.loc[test_df['antigen.epitope'] == epitope]
				if len(ep_df) >= 10:
					# # Save NetTCR testdata -> needs to be used as input on https://services.healthtech.dtu.dk/service.php?NetTCR-2.0
					# if epitope in test_epitopes and dt == 'cdr3':
					# 	ep_df.loc[ep_df['recognition'] == 0].to_csv(
					# 		os.path.join(tcrpair_config.ep_path, '{}_neg.tsv'.format(epitope)),
					# 		index=False, sep='\t', columns=['cdr3.alpha', 'cdr3.beta'], header=False)
					# 	ep_df.loc[ep_df['recognition'] == 1].to_csv(
					# 		os.path.join(tcrpair_config.ep_path, '{}_pos.tsv'.format(epitope)),
					# 		index=False, sep='\t', columns=['cdr3.alpha', 'cdr3.beta'], header=False)
					seq_cols = [x for x in tcrpair_config.seq_cols if not x.startswith('part')]
					if dt == 'part':
						# Add partial sequences
						ep_df = tcr_reconstruction.add_partial_tcr_to_df(ep_df, int(modelname.split('_')[1]))
						seq_cols = tcrpair_config.seq_cols

					ep_df, _ = mldata.encode_sequences_df(ep_df, 'blosum62', seq_cols)
					def concat_lists(dfrow):
						return dfrow['{}.alpha.enc'.format(dt)] + dfrow['{}.beta.enc'.format(dt)]
					ep_df['concatseqs'] = ep_df.apply(concat_lists, axis=1)

					ep_df['predicted.functionality'] = model.predict(
						tf.keras.preprocessing.sequence.pad_sequences(
							ep_df['concatseqs'].values, padding='post', maxlen=maxlen, value=tcrpair_config.mask_value
						)
					).round()

					auc = metrics.roc_auc_score(ep_df['recognition'].to_list(), ep_df['predicted.functionality'].to_list())

					epi_comp['model'].append(modelname)
					epi_comp['epitope'].append(epitope)
					epi_comp['n'].append(len(ep_df))
					epi_comp['auc'].append(round(auc, 2))

					# Compare NetTCR results
					if epitope in test_epitopes:
						net_pos_df = pd.read_csv(os.path.join(tcrpair_config.ep_path, '{}_pos_nettcr'.format(epitope)), sep='\t')
						net_neg_df = pd.read_csv(os.path.join(tcrpair_config.ep_path, '{}_neg_nettcr'.format(epitope)), sep='\t')
						nettcr_train_df = pd.concat([pd.read_csv(os.path.join(tcrpair_config.ep_path, 'train_ab_90_alphabeta.csv')),
													 pd.read_csv(os.path.join(tcrpair_config.ep_path, 'train_ab_95_alphabeta.csv'))])
						nettcr_train_df.drop(columns=['partition'], inplace=True)
						nettcr_train_df['CDR3a'] = nettcr_train_df['CDR3a'].apply(lambda x: ''.join(['C', x, 'F']))
						nettcr_train_df['CDR3b'] = nettcr_train_df['CDR3b'].apply(lambda x: ''.join(['C', x, 'F']))

						pos_merge_df = pd.merge(
							nettcr_train_df.loc[(nettcr_train_df['peptide'] == epitope) & (nettcr_train_df['binder'] == 1)], net_pos_df,
							on=['CDR3a', 'CDR3b']
						).drop_duplicates()

						net_pos_df['predicted.functionality'] = net_pos_df['prediction'].round()
						net_neg_df['predicted.functionality'] = net_neg_df['prediction'].round()
						nettcr_comp['epitope'].append(epitope)
						nettcr_comp['model'].append(modelname)
						nettcr_pred = net_pos_df['predicted.functionality'].to_list() + net_neg_df['predicted.functionality'].to_list()
						nettcr_true = [1] * len(net_pos_df) + [0] * len(net_neg_df)
						tcrpair_true = ep_df['recognition']
						tcrpair_pred = ep_df['predicted.functionality'].to_list()
						nettcr_comp['tcrpair auc'].append(round(metrics.roc_auc_score(tcrpair_true, tcrpair_pred), 2))
						nettcr_comp['nettcr auc'].append(round(metrics.roc_auc_score(nettcr_true, nettcr_pred), 2))
						nettcr_comp['tcrpair precision'].append(round(metrics.precision_score(tcrpair_true, tcrpair_pred), 2))
						nettcr_comp['nettcr precision'].append(round(metrics.precision_score(nettcr_true, nettcr_pred), 2))
						nettcr_comp['tcrpair recall'].append(round(metrics.recall_score(tcrpair_true, tcrpair_pred), 2))
						nettcr_comp['nettcr recall'].append(round(metrics.recall_score(nettcr_true, nettcr_pred), 2))
						nettcr_comp['all pos tcrs'].append(len(net_pos_df))
						nettcr_comp['nettcr train pos tcrs'].append(len(pos_merge_df))

	pd.DataFrame(nettcr_comp).set_index(['model', 'epitope']).to_csv(os.path.join(tcrpair_config.tables_path, 'nettcr_comparison.tsv'), sep='\t')
	pd.DataFrame(epi_comp).set_index(['model', 'epitope']).to_csv(os.path.join(tcrpair_config.tables_path, 'epitope_comparison.tsv'), sep='\t')

### FUNCTION END ###

if __name__ == '__main__':
	main()

