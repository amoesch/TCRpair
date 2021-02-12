import os
import argparse
import pandas as pd

from modules import tcrpair_config, mldata, mlmodels, tcr_reconstruction

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', choices=['cdr3', 'full', 'part'], required=True, help='Type of data')
	parser.add_argument('-a', type=str, required=False, default='HLA-A*02:01', help='HLA allele')
	parser.add_argument('-e', type=int, required=True, help='Number of epochs')
	parser.add_argument('-b', type=int, default=50, help='Batch size')
	parser.add_argument('-c', choices=['onehot', 'blosum50', 'blosum62', 'blosum75', 'blosum85', 'blosum100'],
						required=True, help='Encoding')
	parser.add_argument('-u', type=int, required=True, help='LSTM units')
	parser.add_argument('-r', type=float, default=0.2, help='Dropout rate')
	parser.add_argument('-n', type=int, default=3, help='Number of CDR3 flanking amino acids for partial model')

	args = parser.parse_args()
	datasetname  = args.d
	hla_allele   = args.a
	epochs       = args.e
	batch_size   = args.b
	enc			 = args.c
	units        = args.u
	dropout_rate = args.r
	flank_aa     = args.n

	#############################################################

	# Load data
	td_pos_df = pd.read_csv(
		os.path.join(tcrpair_config.td_path, 'trainingdata_df.tsv.gz'), sep='\t', index_col=None, low_memory=False
	)

	if hla_allele is not None:
		td_pos_df = td_pos_df.loc[td_pos_df['mhc.a'].str.startswith(hla_allele)]
		assert len(td_pos_df) >= 1000, '[ERROR] Size of training dataset is not sufficient ({})'.format(len(td_pos_df))

	if datasetname == 'part':
		fulldatasetname = '_'.join([datasetname, str(flank_aa)])
		# Add partial sequences
		td_pos_df = tcr_reconstruction.add_partial_tcr_to_df(td_pos_df, flank_aa)
		td_neg_df = mldata.make_negative_dataset(td_pos_df, part=True)
		seq_cols = tcrpair_config.seq_cols
	else:
		fulldatasetname = datasetname
		td_neg_df = mldata.make_negative_dataset(td_pos_df, part=False)
		seq_cols = [x for x in tcrpair_config.seq_cols if not x.startswith('part')]

	# Encode sequences
	td_pos_df, vocab_pos = mldata.encode_sequences_df(td_pos_df, enc, seq_cols)
	td_neg_df, vocab_neg = mldata.encode_sequences_df(td_neg_df, enc, seq_cols)
	vocab = vocab_pos.union(vocab_neg)

	xdata, ydata, input_length = mldata.generate_datasets(td_pos_df, td_neg_df, datasetname)

	input_size = (input_length, len(vocab))

	mlmodels.single_input_model(xdata, ydata,
								epochs, batch_size, input_size, units, dropout_rate, fulldatasetname)
### FUNCTION END ###

if __name__ == '__main__':
	main()

