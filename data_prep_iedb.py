import os
import pandas as pd

from modules import tcrpair_config, tcr_reconstruction

def main():

	iedb_df = pd.read_csv(os.path.join(tcrpair_config.iedb_data_path, 'tcell_receptor_table_export_1574439161.csv'), low_memory=False)

	# Filter for paired alpha/beta chains
	iedb_df = iedb_df.loc[(iedb_df['Chain 1 Type'] == 'alpha') & (iedb_df['Chain 2 Type'] == 'beta')]

	# Use curated CDR3 sequences and calculated if no curated is available
	iedb_df['cdr3.alpha'] = iedb_df['Chain 1 CDR3 Curated'].fillna(iedb_df['Chain 1 CDR3 Calculated'])
	iedb_df['cdr3.beta']  = iedb_df['Chain 2 CDR3 Curated'].fillna(iedb_df['Chain 2 CDR3 Calculated'])

	# Use only calculated gene notations, curated contains errors (like &nbsp; and weird annotation)
	iedb_df['v.alpha'] = iedb_df['Calculated Chain 1 V Gene']
	iedb_df['v.beta']  = iedb_df['Calculated Chain 2 V Gene']
	iedb_df['j.alpha'] = iedb_df['Calculated Chain 1 J Gene']
	iedb_df['j.beta']  = iedb_df['Calculated Chain 2 J Gene']

	# Match VDJdb nomenclature
	iedb_df.rename({'Description': 'antigen.epitope',
					'Antigen'    : 'antigen.description',
					'Organism'   : 'antigen.species'}, axis=1, inplace=True)

	# Filter for nan
	iedb_df.dropna(subset=['cdr3.alpha', 'cdr3.beta'], inplace=True)
	print('IEDB size unfiltered (paired): {}'.format(len(iedb_df)))
	# Filter for non-aa characters in CDR3 sequences
	iedb_df = iedb_df.loc[
		(iedb_df['cdr3.alpha'].str.match('^[A-Z]+$') == True) & (iedb_df['cdr3.beta'].str.match('^[A-Z]+$') == True)
	]
	# Filter for unclear/PTM epitopes
	iedb_df = iedb_df.loc[(iedb_df['antigen.epitope'].str.match('^[A-Z]+$') == True)].reset_index(drop=True)
	print('IEDB without weird CDR3s and epitopes without sequence/with PTMs: {}'.format(len(iedb_df)))

	# Full TCR sequence reconstruction
	iedb_df = tcr_reconstruction.add_full_tcr_to_df(iedb_df, 'IEDB', dropnnanalleles=False)
	iedb_df = iedb_df.dropna(subset=['full.alpha', 'full.beta'])
	print('IEDB with both reconstructed alpha and beta CDR3: {}'.format(len(iedb_df)))

	# Add HLA restriction informationd T
	hlas_df = pd.read_csv(
		os.path.join(tcrpair_config.iedb_data_path, 'mhc_ligand_full.zip'),
		header=1, usecols=['Description', 'Allele Name', 'Reference IRI']
	).rename(columns={'Description' : 'antigen.epitope', 'Allele Name' : 'mhc.a'}).drop_duplicates()

	# Add reference ID column from IRI column
	hlas_df['Reference ID'] = hlas_df['Reference IRI'].apply(lambda x: int(x.split('/')[-1]))
	# Chop off mutant annotaion of MHCs
	hlas_df['mhc.a'] = hlas_df['mhc.a'].apply(lambda x: x.split(' ')[0])
	hlas_df = hlas_df.drop_duplicates()

	# Merge HLA allele information to iedb_df
	iedb_df = pd.merge(
		iedb_df, hlas_df, how='left', on=['Reference ID', 'antigen.epitope']
	).dropna(subset=['mhc.a'])

	# Filter for duplicates
	iedb_df = iedb_df.drop_duplicates(subset=['full.alpha', 'full.beta', 'antigen.epitope', 'mhc.a'], keep='first')
	print('IEDB without duplicates: {}'.format(len(iedb_df)))

	# Identify Receptor/Epitope pairs with putative multiple HLA restrictions
	multihla_df = iedb_df.groupby('Receptor ID')['antigen.epitope'].value_counts().reset_index(name='count')
	tocheck_df = pd.merge(
		iedb_df, multihla_df.loc[multihla_df['count'] > 1], how='right', on=['Receptor ID', 'antigen.epitope'])
	with open(os.path.join(tcrpair_config.iedb_path, 'iedb_reference_multiplemhcspertcr.txt'), 'w') as fh:
		for ref_id in tocheck_df['Reference ID'].unique():
			fh.write(''.join(['http://www.iedb.org/details_v3.php?type=reference&id=', str(ref_id), '\n']))

	iedb_df.to_csv(
		os.path.join(tcrpair_config.iedb_df_path, 'iedb_full_df.tsv'), sep='\t', index=False, columns=iedb_df.keys())

if __name__ == '__main__':
	main()
