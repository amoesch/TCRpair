import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from modules import mlmodels, mldata, tcr_reconstruction, tcrpair_config

np.random.seed(10)

def var_importance(model, maxlen, x):
	effects_list = list()
	orig_out = model.predict(x)
	for i in range(maxlen):
		new_x = x.copy()
		perturbation = np.random.normal(0.0, 1.0, size=(new_x.shape[0], 20))
		new_x[:, i, :] = new_x[:, i, :] + perturbation
		perturbed_out = model.predict(new_x)
		effects_list.append(((orig_out - perturbed_out) ** 2).mean() ** 0.5)
	return effects_list
### FUNCTION END ###

def main():

	enc = 'blosum62'
	datasets = ['cdr3', 'part_9', 'part_11']
	max_flank = 11

	dn_df_list = list()
	ps_df_list = list()
	ef_df_list = list()

	td_pos_df = pd.read_csv(
		os.path.join(tcrpair_config.td_path, 'trainingdata_df.tsv.gz'), sep='\t', index_col=None, low_memory=False
	)
	td_pos_df = td_pos_df.loc[td_pos_df['mhc.a'].str.startswith('HLA-A*02:01')]

	for datasetname in datasets:
		flank = 0
		for modelname in os.listdir(tcrpair_config.models_path):
			if modelname.startswith(datasetname):
				model = mlmodels.load_model(modelname)
				if datasetname.startswith('part'):
					flank = int(modelname.split('_')[1])
					# Add partial sequences
					td_pos_df = tcr_reconstruction.add_partial_tcr_to_df(td_pos_df, flank)
					td_neg_df = mldata.make_negative_dataset(td_pos_df, part=True)
					seq_cols = tcrpair_config.seq_cols
				else:
					td_neg_df = mldata.make_negative_dataset(td_pos_df, part=False)
					seq_cols = [x for x in tcrpair_config.seq_cols if not x.startswith('part')]
				td_pos_df, _ = mldata.encode_sequences_df(td_pos_df, enc, seq_cols)
				td_neg_df, _ = mldata.encode_sequences_df(td_neg_df, enc, seq_cols)
				xdata, ydata, _ = mldata.generate_datasets(td_pos_df, td_neg_df, datasetname.split('_')[0])
				X_train, X_test, y_train, y_test = mldata.split_data(xdata, ydata)

				effect_values = var_importance(model, model.layers[0].input_shape[1], X_test)
				dn_df_list = dn_df_list + [datasetname] * len(effect_values)
				ps_df_list = ps_df_list + [x for x in range(1 + max_flank - flank, model.layers[0].input_shape[1] + max_flank - flank + 1)]
				ef_df_list = ef_df_list + effect_values

	plot_df = pd.DataFrame({'dataset' : dn_df_list, 'amino acid position' : ps_df_list, 'perturbation effect' : ef_df_list})
	sns.set_style('whitegrid')
	sns.catplot(data=plot_df.loc[plot_df['amino acid position'] < 30], x='amino acid position', y='perturbation effect', hue='dataset', kind='bar', height=5, aspect=2, legend=False)
	plt.xticks(rotation=90)
	plt.legend(loc='upper left')
	plt.savefig(os.path.join(tcrpair_config.plots_path, 'perturbation_effects.png'), bbox_inches='tight', dpi=350)
	plt.close()

if __name__ == "__main__":
	main()
