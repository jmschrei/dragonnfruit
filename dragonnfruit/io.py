# io.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch
import pandas

from bpnetlite.io import one_hot_encode

class DataGenerator(torch.utils.data.Dataset):
	"""A data generator for dragonnfruit inputs. Adapted from bpnet-lite.

	This generator takes in a set of sequences and output signals 
	and will return a single element with random jitter and reverse-complement 
	augmentation applied. Because the data is single-cell where the output
	signals differ across cells, each returned element is a random locus
	in a random cell. 

	A conceptual difference between this DataGenerator and the one implemented
	in bpnet-lite is that the bpnet-lite one assumes that you can extract all
	loci into an array. Here, because there are hundreds of thousands of peaks
	and potentially thousands of cells, it is actually more efficient to 

	Parameters
	----------
	sequences: dict of torch.tensors, shape=(n, 4), dtype=torch.float32
		A dictionary of the nucleotide sequences to use.

	signals: dict of torch.tensors, shape=(n, n_cells), dtype=torch.float32
		A dictionary of the cell signals

	loci: str or pandas.DataFrame
		A set of loci to use.
	
	cell_states: 
	"""

	def __init__(self, sequence, signal, loci_file, neighbors, cell_states, 
		read_depths, trimming, window, chroms, reverse_complement=True, 
		max_jitter=128, random_state=None):
		self.trimming = trimming
		self.window = window
		self.chroms = chroms
		self.reverse_complement = reverse_complement
		self.max_jitter = max_jitter
		self.random_state = numpy.random.RandomState(random_state)

		self.signal = {chrom: signal[chrom] for chrom in chroms}
		self.sequence = {chrom: sequence[chrom] for chrom in chroms}
		self.neighbors = neighbors
		self.cell_states = cell_states
		self.read_depths = read_depths

		if not isinstance(loci_file, list):
			loci_file = [loci_file]

		loci = []
		names = ['chrom', 'start', 'end']
		for filename in loci_file:
			loci_ = pandas.read_csv(filename, sep='\t', usecols=[0, 1, 2], 
				header=None, index_col=False, names=names)
			loci_['mid'] = (loci_['end'] - loci_['start']) // 2 + loci_['start']
			loci_ = loci_[numpy.isin(loci_['chrom'], chroms)]
			loci_ = loci_.sample(frac=1).reset_index(drop=True) # shuffle
			loci.append(loci_)

		self.loci = pandas.concat(loci).sort_index().reset_index(drop=True) # interleave
		print("Loci: {}".format(self.loci.shape[0]))

	def __len__(self):
		return self.loci.shape[0]

	def __getitem__(self, idx):
		chrom, _, _, mid = self.loci.iloc[idx]

		mid += self.random_state.randint(-self.max_jitter, self.max_jitter+1)
		start, end = mid - self.window // 2, mid + self.window // 2

		j = numpy.random.randint(self.cell_states.shape[0])
		n = self.neighbors[j]

		X = self.sequence[chrom][start:end].T.astype('float32')
		y = self.signal[chrom][:, start+self.trimming:end-self.trimming]
		y = numpy.array(y[n].sum(axis=0))[0]

		c = self.cell_states[j]
		r = self.read_depths[j]

		if self.reverse_complement and idx % 2 == 0:
			X = X[::-1][:, ::-1].copy()
			y = y[::-1].copy()

		X = torch.from_numpy(X)
		y = torch.from_numpy(y)
		c = torch.from_numpy(c)
		r = torch.from_numpy(r)
		return X, y, c, r
	