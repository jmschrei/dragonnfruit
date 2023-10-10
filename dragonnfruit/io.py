# io.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import time
import numpy
import scipy
import torch
import pandas
import pyfaidx

from bpnetlite.io import one_hot_encode
from joblib import Parallel, delayed

def extract_fasta(filename, chroms, n_jobs=-1):
	"""Extract and one-hot encode chromosomes from a FASTA file in parallel.

	This function will take in a FASTA file and a set of chromosomes and, in
	parallel, will one-hot encode the chromosomes and return a dictionary
	with the extracted values.


	Parameters
	----------
	filename: str
		The name of a FASTA file.

	chroms: list or tuple
		A set of chromosomes to return one-hot encodings for.

	n_jobs: int, optional
		The number of jobs to process in parallel.



	Returns
	-------
	ohes: dict
		A dictionary where the keys are the chromosomes and the values are the
		one-hot encodings.
	"""

	fa = pyfaidx.Fasta(filename)

	f = delayed(one_hot_encode)
	ohes = Parallel(n_jobs=n_jobs)(f(fa[chrom][:].seq.upper()) 
		for chrom in chroms)

	return {chrom: ohe for chrom, ohe in zip(chroms, ohes)}



def _extract_example(self, chrom, mid, cell_idx, idx):
	"""An internal function for extracting a single example.

	This function will extract an example from a given position in a given
	cell and handle adding jitter, creating the dynamic pseudobulk, and
	potentially reverse complementing the sequence. It will return the
	one-hot encoded sequence, signal, cell representation, and read depth
	of that cell.

	Note that this function returns a *single* example and that the data
	generators below handle the creation of batches by repeatedly calling this
	function and concatenating the examples.


	Parameters
	----------
	self: torch.utils.data.Dataset
		This is one of the data generators defined below. Although they may
		differ in how loci and cells are selected, they share the same logic
		for how to extract the inputs given a cell and location.

	chrom: str
		The chromosome name to extract from.

	mid: int
		The middle position to extract a window around.

	cell_idx: int
		The integer index of the cell to operate on.

	idx: int
		The index being generated. Necessary when reverse complementing every
		even sequence.


	Returns
	-------
	X: torch.Tensor, shape=(4, 2114)
		The one-hot encoded sequence

	y: torch.Tensor, shape=(1000,)
		The signal to be predicted

	c: torch.Tensor, shape=(50,)
		The cell representation to be extracted.

	r: torch.Tensor, shape=(1,)
		The read depth of that particular cell.
	"""

	start, end = mid - self.window // 2, mid + self.window // 2
	neighbs = self.neighbors[cell_idx]

	X = self.sequence[chrom][start:end].T.astype('float32')
	y = self.signal[chrom][:, start+self.trimming:end-self.trimming]
	y = numpy.array(y[neighbs].sum(axis=0))[0]

	c = self.cell_states[cell_idx]
	r = self.read_depths[cell_idx]

	if self.reverse_complement and idx % 2 == 0:
		X = X[::-1][:, ::-1].copy()
		y = y[::-1].copy()

	X = torch.from_numpy(X)
	y = torch.from_numpy(y)
	c = torch.from_numpy(c)
	r = torch.from_numpy(r)
	return X, y, c, r


class LocusGenerator(torch.utils.data.Dataset):
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
		read_depths, trimming, window, chroms, n_cells_per_locus=1, 
		reverse_complement=True, max_jitter=128, random_state=None):
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
		self.n_cells_per_locus = n_cells_per_locus

		if not isinstance(loci_file, list):
			loci_file = [loci_file]

		loci = []
		names = ['chrom', 'start', 'end']
		for filename in loci_file:
			loci_ = pandas.read_csv(filename, sep='\t', usecols=[0, 1, 2], 
				header=None, index_col=False, names=names)
			loci_['mid'] = (loci_['end'] - loci_['start']) // 2 + loci_['start']
			loci_ = loci_[numpy.isin(loci_['chrom'], chroms)]
			loci_ = loci_.sample(frac=1, random_state=self.random_state)
			loci_ = loci_.reset_index(drop=True)
			loci.append(loci_)

		self.loci = pandas.concat(loci).sort_index().reset_index(drop=True) # interleave

	def __len__(self):
		return self.loci.shape[0]

	def __getitem__(self, idx):
		chrom, _, _, mid = self.loci.iloc[idx // self.n_cells_per_locus]
		mid += self.random_state.randint(-self.max_jitter, self.max_jitter+1)
		cell_idx = self.random_state.randint(self.cell_states.shape[0])
		return _extract_example(self, chrom, mid, cell_idx, idx)

	
class GenomewideGenerator(torch.utils.data.Dataset):
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

	def __init__(self, sequence, signal, neighbors, cell_states, 
		read_depths, trimming, window, chroms, cells_per_loci=1, 
		reverse_complement=True, random_state=None):
		self.trimming = trimming
		self.window = window
		self.chroms = chroms
		self.cells_per_loci = cells_per_loci
		self.reverse_complement = reverse_complement
		self.random_state = numpy.random.RandomState(random_state)

		self.signal = {chrom: signal[chrom] for chrom in chroms}
		self.sequence = {chrom: sequence[chrom] for chrom in chroms}
		self.neighbors = neighbors
		self.cell_states = cell_states
		self.read_depths = read_depths
		self._lengths = numpy.array([seq.shape[0] for seq in self.sequence.values()])

	def __len__(self):
		return sum(self._lengths)

	def __getitem__(self, idx):
		c_idx = numpy.random.choice(len(self._lengths), 
			p=self._lengths / self._lengths.sum())
		chrom = self.chroms[c_idx]

		mid = numpy.random.randint(10000, self._lengths[c_idx]-10000)
		cell_idx = numpy.random.randint(self.cell_states.shape[0])
		return _extract_example(self, chrom, mid, cell_idx, idx)
