import pandas as pd
import os

from modules import tcrpair_config

# Load data
td_pos_df = pd.read_csv(
	os.path.join(tcrpair_config.td_path, 'trainingdata_df.tsv.gz'), sep='\t', index_col=None, low_memory=False
)

for hla in td_pos_df['mhc.a'].unique():
	size = len(td_pos_df.loc[td_pos_df['mhc.a'] == hla])
	if size > 200:
		print(hla,
			  size,
			  round(len(td_pos_df.loc[td_pos_df['mhc.a'] == hla]['cdr3.alpha'].unique()) / size, 2),
			  round(len(td_pos_df.loc[td_pos_df['mhc.a'] == hla]['cdr3.beta'].unique()) / size, 2),
			  len(td_pos_df.loc[td_pos_df['mhc.a'] == hla]['antigen.epitope'].unique()))
