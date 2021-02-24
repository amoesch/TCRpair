import os
import pandas as pd
import tensorflow as tf
from sklearn import metrics
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import seaborn as sns
import matplotlib.pyplot as plt

from modules import tcrpair_config, mldata, mlmodels, tcr_reconstruction

def main():

	mn_list = list()
	ep_list = list()
	auc_list = list()

	# Load data
	test_df = pd.read_csv(
		os.path.join(tcrpair_config.val_path, 'firstvalidationdata_df.tsv.gz'), sep='\t', index_col=None, low_memory=False
	)
	test_df = tcr_reconstruction.add_full_tcr_to_df(test_df, 'VDJdb', dropnanalleles=False)

	for modelname in os.listdir(tcrpair_config.models_path):
		# Load model
		model = mlmodels.load_model(modelname)
		maxlen = model.layers[0].input_shape[1]
		dt = modelname.split('_')[0]

		for epitope in test_df['antigen.epitope'].unique():

			ep_df = test_df.loc[test_df['antigen.epitope'] == epitope]
			if len(ep_df) >= 10:
				# # Save NetTCR testdata
				# if epitope == 'GILGFVFTL':
				# 	ep_df.loc[ep_df['recognition'] == 0].to_csv(
				# 		os.path.join(tcrpair_config.gil_path, 'nettcr_testdata_neg.tsv'),
				# 		index=False, sep='\t', columns=['cdr3.alpha', 'cdr3.beta'], header=False)
				# 	ep_df.loc[ep_df['recognition'] == 1].to_csv(
				# 		os.path.join(tcrpair_config.gil_path, 'nettcr_testdata_pos.tsv'),
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

				mn_list.append(modelname)
				ep_list.append(epitope)
				auc_list.append(auc)

				# Compare NetTCR results
				# if epitope == 'GILGFVFTL':
				# 	net_pos_df = pd.read_csv(os.path.join(tcrpair_config.gil_path, 'nettcr_predictions_pos.csv'), sep='\t')
				# 	net_neg_df = pd.read_csv(os.path.join(tcrpair_config.gil_path, 'nettcr_predictions_neg.csv'), sep='\t')
				# 	net_recall = metrics.recall_score([1] * len(net_pos_df), [round(x) for x in net_pos_df['prediction'].to_list()])
				# 	i = 0
				# test1 = metrics.roc_auc_score(ep_df['recognition'], ep_df['predicted.functionality'].to_list())
				# test2 = metrics.roc_auc_score([1] * len(net_pos_df) + [0] * len(net_neg_df),
				# 							 net_pos_df['prediction'].to_list() + net_neg_df['prediction'].to_list())

	results_df = pd.DataFrame(
		{'model' : mn_list,
		 'epitope' : ep_list,
		 'auc' : auc_list}
	)
	results_df = results_df.pivot(index='model', columns='epitope', values='auc')
	results_df['order'] = results_df.index
	results_df['model'] = results_df['order'].apply(lambda x: '_'.join(x.split('_')[0:-1]))
	results_df['order'] = results_df['order'].apply(lambda x: x.split('-')[1])
	results_df.sort_values(by=['order'], inplace=True)
	results_df.drop(columns=['order'], inplace=True)
	results_df.set_index('model', inplace=True)
	sns.heatmap(results_df, cmap='YlGnBu')
	plt.savefig(
		os.path.join(tcrpair_config.plots_path, 'epitope_comparison.png'), bbox_inches='tight', dpi=350
	)
	plt.close()

### FUNCTION END ###

if __name__ == '__main__':
	main()

