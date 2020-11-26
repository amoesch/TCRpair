import re
import os
import pandas as pd
from Bio import SeqIO
from Bio.Alphabet import IUPAC

from modules import tcrpair_config, helpers

def load_imgt_fasta():
	""" Loads all IMGT FASTA sequences in one dataframe """
	allele_df_list = list()
	aaseq_df_list  = list()
	for locus in ['AV', 'BV', 'AJ', 'BJ']:
		fastafile = 'TR{}.fasta'.format(locus)
		for seq_record in SeqIO.parse(os.path.join(tcrpair_config.imgt_data_path, fastafile), 'fasta', IUPAC.protein):
			ref_allele = seq_record.id.split('|')[1]
			allele_df_list.append(ref_allele)
			aaseq_df_list.append(str(seq_record.seq))
	return pd.DataFrame({'allele' : allele_df_list,
						 'aaseq'  : aaseq_df_list})
### FUNCTION END ###

def fetch_sequence(allele: str, imgt_df: pd.DataFrame):
	""" Fetches nucleotide sequence from IMGT data """
	c = None
	s = None
	if re.search('A', allele):   c = 'A'
	elif re.search('B', allele): c = 'B'
	if re.search('V', allele):   s = 'V'
	elif re.search('J', allele): s = 'J'
	if c is None or s is None:
		if  allele != 'nan':
			print('[ERROR] {} not a valid allele.'.format(allele))
	else:
		if allele in imgt_df['allele'].unique():
			return imgt_df.loc[imgt_df['allele'] == allele, 'aaseq'].values[0]
		else:
			for a in imgt_df['allele'].unique():
				if a.startswith(allele) and a.endswith('01'):
					return imgt_df.loc[imgt_df['allele'] == a, 'aaseq'].values[0]
			#print('[ERROR] No amino acid sequence could be found for allele {}'.format(allele))
			return None
### FUNCTION END ###

def find_sequence(cdr3seq: str, chain: str, locus: str, imgt_df: pd.DataFrame):
	""" Find best matching sequence for CDR3 """
	best_score  = 0.0
	best_seq    = ''
	for aaseq in imgt_df.loc[
		(imgt_df['allele'].str.startswith('TR{}{}'.format(chain, locus))) &
		(imgt_df['aaseq'].str.contains('\*') == False),
		'aaseq'
	].unique():
		alnlist = helpers.make_alignment(aaseq, cdr3seq)
		if alnlist:
			aln = alnlist[0]
			if aln[2] > best_score:
				best_score = aln[2]
				best_seq = aaseq
	if best_score >= 5.0:
		return best_seq
	else:
		return None
### FUNCTION END ###

def get_allele_sequences(df: pd.DataFrame, dropnan):
	imgt_df = load_imgt_fasta()
	def fetch_seqs(dfrow: pd.Series, chain: str, locus: str):
		""" Take a df row, IMGT alleles and return matching allele aa sequence """
		assert chain in tcrpair_config.chains;
		'[ERROR] {} not a valid argument for chain'.format(chain)
		assert locus in ['V', 'J'];
		'[ERROR] {} not a valid argument for locus'.format(locus)
		seq = fetch_sequence(str(dfrow['{}.{}'.format(locus.lower(), tcrpair_config.chains[chain])]), imgt_df)
		if seq is None:
			seq = find_sequence(str(dfrow['cdr3.{}'.format(tcrpair_config.chains[chain])]), chain, locus, imgt_df)
		return seq
	### FUNCTION END ###

	# Store V and J sequences in separate columns (use closest matching sequence if no allele name is given)
	df['vseq.alpha'] = df.apply(fetch_seqs, axis=1, chain='A', locus='V')
	df['jseq.alpha'] = df.apply(fetch_seqs, axis=1, chain='A', locus='J')
	df['vseq.beta']  = df.apply(fetch_seqs, axis=1, chain='B', locus='V')
	df['jseq.beta']  = df.apply(fetch_seqs, axis=1, chain='B', locus='J')
	# Filter for sequences that could not be retrieved
	if dropnan is True:
		df.dropna(subset=['vseq.alpha', 'jseq.alpha', 'vseq.beta', 'jseq.beta'], inplace=True)
	return df
### FUNCTION END ###
