import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from modules import tcrpair_config, mlmodels, mldata, tcr_reconstruction

def main():

	# Load data
	multichains_df = pd.read_csv(os.path.join(tcrpair_config.val_path, 'coexpr_tcrs.csv'))
	# TCR reconstruction (uses VDJdb column names)
	multichains_df = tcr_reconstruction.add_full_tcr_to_df(multichains_df, 'VDJdb', dropnanalleles=False)

	for modelname in os.listdir(tcrpair_config.models_path):
		print('Loading model {}'.format(modelname))
		# Load model
		model = mlmodels.load_model(modelname)
		maxlen = model.layers[0].input_shape[1]
		dt = modelname.split('_')[0]
		if dt == 'part':
			# Add partial sequences
			multichains_df = tcr_reconstruction.add_partial_tcr_to_df(multichains_df, int(modelname.split('_')[1]))
		# Encode
		multichains_df, _ = mldata.encode_sequences_df(
			multichains_df, 'blosum62', ['{}.alpha'.format(dt), '{}.beta'.format(dt)]
		)
		def concat_lists(dfrow):
			return dfrow['{}.alpha.enc'.format(dt)] + dfrow['{}.beta.enc'.format(dt)]
		multichains_df['concatseqs'] = multichains_df.apply(concat_lists, axis=1)
		x_test = tf.keras.preprocessing.sequence.pad_sequences(
			multichains_df['concatseqs'].values, padding='post', maxlen=maxlen, value=tcrpair_config.mask_value
		)
		multichains_df['predicted likelihood'] = model.predict(x_test)

		count = 0
		for name in multichains_df['name'].unique():
			if multichains_df.loc[
				(multichains_df['name'] == name) & (multichains_df['recognition'] == 1), ['predicted likelihood']
			].values[0][0] > multichains_df.loc[
				(multichains_df['name'] == name) & (multichains_df['recognition'] == 0), ['predicted likelihood']
			].values[0][0]:
				count += 1
		print('{} of {}'.format(count, len(multichains_df['name'].unique())))

		sns.set_style('whitegrid')
		markers = {0 : 'v', 1 : '^'}
		ax = sns.scatterplot(data=multichains_df, x='name', y='predicted likelihood',
						hue='recognition', style='recognition', markers=markers, s=150)
		ax.set(xlabel='name of TCR')
		new_labels = [x.replace('alpha', 'α') for x in multichains_df['name'].unique()]
		new_labels = [x.replace('beta', 'β') for x in new_labels]
		new_labels = [x.replace('_', ' ') for x in new_labels]
		ax.set(xticklabels=new_labels)
		plt.xticks(rotation=90)
		plt.savefig(
			os.path.join(tcrpair_config.pr_plots_path, '{}_multichains_comp.png'.format(modelname)), bbox_inches='tight', dpi=350
		)
		plt.close()
### FUNCTION END ###

if __name__ == '__main__':
	main()
