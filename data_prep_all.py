import os
import pandas as pd

from modules import tcrpair_config

def group_by(df: pd.DataFrame, field: str):
	""" Group TCRs of df by field """
	group_df = df.groupby(field).count()
	group_df['TCR.fraction'] = group_df['cdr3.alpha'].apply(lambda x: x/len(df))
	return group_df.sort_values('TCR.fraction').reset_index().rename({'index' : field})
### FUNCTION END ###

vdjdb_df = pd.read_csv(
	os.path.join(
		tcrpair_config.vdjdb_df_path, 'vdjdb_full_df.tsv.gz'
	), sep='\t', index_col=None, low_memory=False).assign(origin=lambda x: 'vdjdb'
)
print('Original size of VDJdb: {}'.format(len(vdjdb_df)))

iedb_df = pd.read_csv(
	os.path.join(
		tcrpair_config.iedb_df_path, 'iedb_full_df.tsv'
	), sep='\t', index_col=None, low_memory=False).assign(origin=lambda x: 'iedb'
)
print('Original size of IEDB: {}'.format(len(iedb_df)))

iedb_reconstructed_df = iedb_df.loc[
		(iedb_df['Chain 1 Full Sequence'].isnull()) & (iedb_df['Chain 2 Full Sequence'].isnull())
	][[x for x in iedb_df.columns if x in vdjdb_df.columns]]
print('Size of IEDB with reconstructed sequences: {}'.format(len(iedb_reconstructed_df)))

# Concatenate dataframes
td_df = pd.concat([vdjdb_df, iedb_reconstructed_df], sort=True).reset_index(drop=True)
print('Size of combined trainig data: {}'.format(len(td_df)))

# Mark duplicates
td_df['full.duplicate'] = td_df.duplicated(subset=['full.alpha', 'full.beta', 'antigen.epitope', 'mhc.a'], keep=False)

# Drop IEDB duplicates
td_df = td_df.drop(labels=td_df.loc[(td_df['full.duplicate']) & (td_df['origin'] == 'iedb')].index)
# Rename origin of VDJdb duplicates
td_df.loc[td_df['full.duplicate'], 'origin'] = 'vdjdb/iedb'
print('Size of combined trainig data w/o duplicates: {}'.format(len(td_df)))

# Export data
td_df.to_csv(
	os.path.join(tcrpair_config.td_path, 'trainingdata_df.tsv.gz'), compression='gzip', sep='\t', index=False)
