from Bio import pairwise2

def make_alignment(seq1, seq2):
	return pairwise2.align.localmd(seq1, seq2, 1, -2, -1, -0.1, -1, -0.1, one_alignment_only=True)
### FUNCTION END ###
