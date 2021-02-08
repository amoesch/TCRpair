import pandas as pd

from modules import tcrpair_config, helpers
import imgt

def tcr_seq_assembly(cdr3seq: str, vseq: str, jseq: str):
	""" Reconstruct TCR sequence from CDR3, V and J segment """
	seq_okay = True
	# V segment
	n = 10 # Last n aa from V that are going to be used for the alignment
	s = 3.0 # Minimum score for alignment
	alnv = helpers.make_alignment(vseq[-n:], cdr3seq)[0]
	if alnv[2] < s:
		# Check if alignment makes sense, in this case we also allow a score of one less
		if alnv[2] < s - 1 and not alnv[0].endswith('---') and alnv[1].startswith('---'):
			seq_okay = False
	# J segment
	alnj = helpers.make_alignment(jseq, cdr3seq)[0]
	if alnj[2] < s:
		seq_okay = False
	else:
		jseq = jseq[-alnj[1].count('-'):]
	if seq_okay is True:
		vseq = vseq[:(len(vseq)-n+alnv[3])]
		return ''.join([vseq, cdr3seq, jseq])
	else:
		return None
### FUNCTION END ###

def tcr_seq_extraction(givenfulltcr, vseq, jseq, cdr3seq):
	""" Extract full TCR sequence from given Full TCR  Sequence (IEDB) to make it match IMGT start/end definitions
		if this fails, use the tcr_seq_assembly function
	"""
	# Check if we have a full sequence and vseq/jseq we can use for reconstruction
	if type(givenfulltcr) is str and type(vseq) is str and type(jseq) is str:
		startalnlist = helpers.make_alignment(givenfulltcr, vseq)
		endalnlist   = helpers.make_alignment(givenfulltcr, jseq)
		if len(startalnlist) > 0 and len(endalnlist) > 0:
			startpos = startalnlist[0][3]
			endpos = endalnlist[0][4]
			return ''.join([vseq[:startpos], givenfulltcr[:endpos]])
		else:
			if type(cdr3seq) is str:
				return tcr_seq_assembly(cdr3seq, vseq, jseq)
			else:
				return None
	elif type(cdr3seq) is str and type(vseq) is str and type(jseq) is str:
		return tcr_seq_assembly(cdr3seq, vseq, jseq)
	else:
		return None
### FUNCTION END ####

def reconstruct_tcr(dfrow: pd.Series, datasetname: str, chain: str):
	""" Take a df row, IMGT sequences and return reoncstructed TCR sequence through TCR reconstruction function """
	assert chain in tcrpair_config.chains; '[ERROR] {} not a valid argument for chain.'.format(chain)
	assert datasetname in ['VDJdb', 'IEDB']; '[ERROR] {} not a valid dataset name.'.format(datasetname)
	if datasetname == 'VDJdb':
		return tcr_seq_assembly(
			dfrow['cdr3.{}'.format(tcrpair_config.chains[chain])],
			dfrow['vseq.{}'.format(tcrpair_config.chains[chain])],
			dfrow['jseq.{}'.format(tcrpair_config.chains[chain])])
	else:
		if chain == 'A':
			chainnum = '1'
		else:
			chainnum = '2'
		return tcr_seq_extraction(
			dfrow['Chain {} Full Sequence'.format(chainnum)],
			dfrow['vseq.{}'.format(tcrpair_config.chains[chain])],
			dfrow['jseq.{}'.format(tcrpair_config.chains[chain])],
			dfrow['cdr3.{}'.format(tcrpair_config.chains[chain])])
### FUNCTION END ###

def add_full_tcr_to_df(df: pd.DataFrame, datasetname: str, dropnanalleles: False):
	df = imgt.get_allele_sequences(df, dropnanalleles)
	print('TCRs with allele sequences in IMGT: {}'.format(len(df)))
	df['full.alpha'] = df.apply(reconstruct_tcr, axis=1, datasetname=datasetname, chain='A')
	print('No alpha reconstruction possible: {}'.format(len(df.loc[df['full.alpha'].isna()])))
	df['full.beta']  = df.apply(reconstruct_tcr, axis=1, datasetname=datasetname, chain='B')
	print('No beta reconstruction possible: {}'.format(len(df.loc[df['full.beta'].isna()])))
	return df
### FUNCTION END ###


def add_partial_tcr_to_df(df: pd.DataFrame, n: int):
	""" n -> number of amino acids flanking the CDR3 region """
	def chop_full_tcr(dfrow: pd.Series, m: int, chain: str):
		full = dfrow['full.{}'.format(tcrpair_config.chains[chain])]
		cdr3 = dfrow['cdr3.{}'.format(tcrpair_config.chains[chain])]
		return full[full.find(cdr3)-m:full.find(cdr3)+len(cdr3)+m]
	### FUNCTION END ###
	df['part.alpha'] = df.apply(chop_full_tcr, axis=1, m=n, chain='A')
	df['part.beta']  = df.apply(chop_full_tcr, axis=1, m=n, chain='B')
	return df
### FUNCTION END ###
