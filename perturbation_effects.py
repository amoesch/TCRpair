import os
import numpy as np
import pandas as pd
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from modules import mlmodels, mldata, tcr_reconstruction, tcrpair_config

np.random.seed(10)

def var_importance(model, maxlen, x, cdr3len, fl):
	regions = {y : 0.0 for y in ['cdr3', 'cdr3cons', 'cdr3var', 'v', 'j']}
	positions = [(fl, fl + cdr3len), (fl, fl + 2), (fl + 2, fl + cdr3len), (0, fl), (fl + cdr3len, maxlen)]
	for region, pos in zip(regions.keys(), positions):
		effects_list = list()
		orig_out = model.predict(x)
		for i in range(pos[0], pos[1]):
			new_x = x.copy()
			perturbation = np.random.normal(0.0, 1.0, size=(new_x.shape[0], 20))
			new_x[:, i, :] = new_x[:, i, :] + perturbation
			perturbed_out = model.predict(new_x)
			effects_list.append(((orig_out - perturbed_out) ** 2).mean() ** 0.5)
		regions[region] = np.mean(effects_list)
	return regions
### FUNCTION END ###

def main():

	enc = 'blosum62'
	datasets = ['part_5', 'part_9', 'part_7', 'part_11']

	mn_df_list = list()
	rg_df_list = list()
	pe_df_list = list()
	cl_df_list = list()

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
					dt = 'part'
					flank = int(modelname.split('_')[1])
					# Add partial sequences
					td_pos_df = tcr_reconstruction.add_partial_tcr_to_df(td_pos_df, flank)
					td_neg_df = mldata.make_negative_dataset(td_pos_df, part=True)
					seq_cols = tcrpair_config.seq_cols
				else:
					dt = 'cdr3'
					td_neg_df = mldata.make_negative_dataset(td_pos_df, part=False)
					seq_cols = [x for x in tcrpair_config.seq_cols if not x.startswith('part')]
				pert_df = pd.concat([td_pos_df, td_neg_df])
				pert_df['cdr3.length'] = pert_df['cdr3.alpha'].apply(lambda x: len(x))
				for cdr3len in [12, 13, 14, 15]:
					tmp_df, _ = mldata.encode_sequences_df(pert_df.loc[pert_df['cdr3.length'] == cdr3len], enc, seq_cols)
					def concat_lists(dfrow):
						return dfrow['{}.alpha.enc'.format(dt)] + dfrow['{}.beta.enc'.format(dt)]
					tmp_df['concatseqs'] = tmp_df.apply(concat_lists, axis=1)
					region_pert = var_importance(
						model,
						model.layers[0].input_shape[1],
						tf.keras.preprocessing.sequence.pad_sequences(
							tmp_df['concatseqs'].values, padding='post', maxlen=model.layers[0].input_shape[1],
							value=tcrpair_config.mask_value
						),
						cdr3len,
						flank
					)
					rg_keys = list(region_pert.keys())
					mn_df_list = mn_df_list + [datasetname] * len(region_pert)
					rg_df_list = rg_df_list + rg_keys
					pe_df_list = pe_df_list + [region_pert[x] for x in rg_keys]
					cl_df_list = cl_df_list + [cdr3len] * len(region_pert)
	results_df = pd.DataFrame({'model' : mn_df_list, 'cdr3 length' : cl_df_list, 'region' : rg_df_list, 'score difference' : pe_df_list})
	results_df.groupby(['model', 'region']).mean().to_csv(os.path.join(tcrpair_config.tables_path, 'perturbation_effects.tsv'), sep='\t')

if __name__ == "__main__":
	main()
