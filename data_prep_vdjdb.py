import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from modules import tcrpair_config, tcr_reconstruction

def load_vdjdb_paired():
	""" Loads VDJdb data and filters for paired sequences """
	# Load VDJdb full which contains paired data in one row
	vdjdb_df = pd.read_csv(os.path.join(tcrpair_config.vdjdb_data_path, 'vdjdb_full.txt'), sep='\t', low_memory=False)
	# Filter
	vdjdb_df.dropna(subset=['cdr3.alpha', 'cdr3.beta'], inplace=True)
	vdjdb_df = vdjdb_df.loc[vdjdb_df['species'] == 'HomoSapiens']
	return vdjdb_df.reset_index(drop=True)
### FUNCTION END ###

def main():

	vdjdb_df = load_vdjdb_paired()
	print('VDJdb size unfiltered: {}'.format(len(vdjdb_df)))

	# Filter for unique alpha/beta pairings and epitopes
	vdjdb_df = vdjdb_df.drop_duplicates(subset=['cdr3.alpha', 'cdr3.beta', 'antigen.epitope', 'mhc.a'], keep='first')
	print('VDJdb without duplicates: {}'.format(len(vdjdb_df)))

	# Reconstruct full TCR sequences
	vdjdb_df = tcr_reconstruction.add_full_tcr_to_df(vdjdb_df, 'VDJdb', dropnnanalleles=True)

	# Store final df without entries without full sequences
	vdjdb_df = vdjdb_df.dropna(subset=['full.alpha', 'full.beta'])
	print('VDJdb with reconstructed sequences: {}'.format(len(vdjdb_df)))
	vdjdb_df.to_csv(
		os.path.join(tcrpair_config.vdjdb_df_path, 'vdjdb_full_df.tsv.gz'),
		compression='gzip', sep='\t', index=False, columns=vdjdb_df.keys())

if __name__ == "__main__":
	main()

