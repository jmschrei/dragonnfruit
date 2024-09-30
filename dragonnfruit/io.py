# io.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import time
import numpy
import scipy
import torch
import pandas
import pyfaidx

from tangermeme.utils import one_hot_encode
from joblib import Parallel, delayed


def extract_fasta(filename, chroms):
	"""Extract and one-hot encode chromosomes from a FASTA file.

	This function will take in a FASTA file and a set of chromosomes and will 
	one-hot encode the chromosomes and return a dictionary with the extracted 
	values. Importantly, this function will return numpy arrays rather than 
	torch tensors for easier slicing in subsequent functions.


	Parameters
	----------
	filename: str
		The name of a FASTA file.

	chroms: list or tuple
		A set of chromosomes to return one-hot encodings for.


	Returns
	-------
	ohes: dict
		A dictionary where the keys are the chromosomes and the values are the
		one-hot encodings. 
	"""

	fa = pyfaidx.Fasta(filename)
	d = {}

	for chrom in chroms:
		seq = fa[chrom][:].seq.upper()
		ohe = one_hot_encode(seq).numpy()
		d[chrom] = ohe

	return d


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

	X = self.sequence[chrom][:, start:end]
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
	"""A data generator that samples across a fixed set of loci.

	This generator takes in genome-wide sequence and signal and a set of loci,
	randomly samples a locus from the loci, and extracts the signal at those
	coordinates. This is different from the data sampler in bpnet-lite because
	that sampler could extract a fixed set of loci and keep them in memory.
	Here, because of the cell axis, it is actually more feasible to keep
	genome-wide signal in memory and sample from it, building dynamic
	pseudobulks on the fly.


	Parameters
	----------
	sequences: dict of numpy.ndarrays, shape=(n, 4), dtype=numpy.int8
		A dictionary of the nucleotide sequences to use. Generally, `n` is the
		size of a chromosome because the entire genome is being loaded into
		memory. This is a numpy array to allow for faster slicing in subsequent
		functions.

	signals: dict of scipy.sparse.csc_matrix, shape=(n, n_cells)
		A dictionary of the cell signals, where `n` is generally the size of
		a chromosome, `n_cells` is the number of cells being modelled, and
		the underlying data is a sparse matrix of counts.

	loci: list of strs
		A list of filenames for files that have bed-formatted coordinates.
		Each file should essentially be a set of coordinates to sample. These
		coordinates are shuffled and interleaved for sampling.

	neighbors: numpy.ndarray, shape=(n_cells, n_neighbors)
		An array of integers where each row is a cell, column i corresponds to
		the i-th nearest neighbor, and the value is the integer index of the
		cell that is that neighbor.
	
	cell_states: numpy.ndarray, shape=(n_cells, n_dims)
		An array of representations for each cell where each row is a cell and
		each dimension is a feature in the representation for that cell.

	read_depths: numpy.ndarray, shape=(n_cells, 1)
		A numpy.ndarray of read depths for each cell. Rather than being the sum of
		counts across the cell, this is usually log2(x+1) of that count, but
		can be whatever the user wants.

	chroms: list of str
		A list of the chromosomes to sample from. All chromosomes in the loci
		files that are not in this list will be filtered out.

	trimming: int, optional
		The number of bp to trim off each side of the input window to get the
		output window. For instance, if the input window is 2114bp and trimming
		is 557, the output window will be 1000bp. Default is 557.

	window: int, optional
		The size of the input window to extract. Default is 2114.

	n_cells_per_locus: int, optional
		The number of cells to sample for each sampled locus. If this number is
		1, cells and loci will be uniformly randomly sampled. If this number is
		above 1, multiple examples in a row will come from the same locus but
		different cells. Default is 1.

	reverse_complement: bool, optional
		Whether to reverse complement half of the examples. Default is True.

	max_jitter: int, optional
		Random uniform values to add to the sampled loci. Sampled values will
		range from -max_jitter to max_jitter. Default is 128.

	random_state: int, optional
		A random seed to use to make the sampling process deterministic. If
		None, process is not deterministic. Default is None.
	"""

	def __init__(self, sequence, signal, loci, neighbors, cell_states, 
		read_depths, chroms, trimming=557, window=2114, n_cells_per_locus=1, 
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

		if not isinstance(loci, list):
			loci = [loci]

		_loci = []
		names = ['chrom', 'start', 'end']
		for filename in loci:
			loci_ = pandas.read_csv(filename, sep='\t', usecols=[0, 1, 2], 
				header=None, index_col=False, names=names)
			loci_['mid'] = (loci_['end'] - loci_['start']) // 2 + loci_['start']
			loci_ = loci_[numpy.isin(loci_['chrom'], chroms)]
			loci_ = loci_.sample(frac=1, random_state=self.random_state)
			loci_ = loci_.reset_index(drop=True)
			_loci.append(loci_)

		self.loci = pandas.concat(_loci).sort_index().reset_index(drop=True)

	def __len__(self):
		return self.loci.shape[0]

	def __getitem__(self, idx):
		chrom, _, _, mid = self.loci.iloc[idx // self.n_cells_per_locus]
		mid += self.random_state.randint(-self.max_jitter, self.max_jitter+1)
		cell_idx = self.random_state.randint(self.cell_states.shape[0])
		return _extract_example(self, chrom, mid, cell_idx, idx)

	
class GenomewideGenerator(torch.utils.data.Dataset):
	"""A data generator that samples across the entire genome.

	This general does not sample from a fixed set of loci but will, rather,
	sample uniformly across the entire genome. So, it is not necessary to
	pass in a set of peaks or even perform peak calling on signal from
	single-cell data to use this generator.


	Parameters
	----------
	sequences: dict of torch.tensors, shape=(n, 4), dtype=torch.float32
		A dictionary of the nucleotide sequences to use. Generally, `n` is the
		size of a chromosome because the entire genome is being loaded into
		memory.

	signals: dict of scipy.sparse.csc_matrix, shape=(n, n_cells)
		A dictionary of the cell signals, where `n` is generally the size of
		a chromosome, `n_cells` is the number of cells being modelled, and
		the underlying data is a sparse matrix of counts.

	neighbors: torch.tensor, shape=(n_cells, n_neighbors)
		A tensor of integers where each row is a cell, column i corresponds to
		the i-th nearest neighbor, and the value is the integer index of the
		cell that is that neighbor.
	
	cell_states: torch.tensor, shape=(n_cells, n_dims)
		A tensor of representations for each cell.

	read_depths: torch.tensor, shape=(n_cells, 1)
		A tensor of read depths for each cell. Rather than being the sum of
		counts across the cell, this is usually log2(x+1) of that count, but
		can be whatever the user wants.

	chroms: list of str
		A list of the chromosomes to sample from. All chromosomes in the loci
		files that are not in this list will be filtered out.

	cell_weights: torch.tensor or None, optional
		A weight to put on the cell for the purpose of sampling. This vector
		will be rescaled into probabilities and so does not need to sum to 1.
		If None, uniformly sample cells. Default is None.

	trimming: int, optional
		The number of bp to trim off each side of the input window to get the
		output window. For instance, if the input window is 2114bp and trimming
		is 557, the output window will be 1000bp. Default is 557.

	window: int, optional
		The size of the input window to extract. Default is 2114.

	n_cells_per_locus: int, optional
		The number of cells to sample for each sampled locus. If this number is
		1, cells and loci will be uniformly randomly sampled. If this number is
		above 1, multiple examples in a row will come from the same locus but
		different cells. Default is 1.

	reverse_complement: bool, optional
		Whether to reverse complement half of the examples. Default is True.

	random_state: int, optional
		A random seed to use to make the sampling process deterministic. If
		None, process is not deterministic. Default is None.
	"""

	def __init__(self, sequence, signal, neighbors, cell_states, 
		read_depths, chroms, cell_weights=None, trimming=557, window=2114,  
		reverse_complement=True, random_state=None):
		self.trimming = trimming
		self.window = window
		self.chroms = chroms
		self.reverse_complement = reverse_complement
		self.random_state = numpy.random.RandomState(random_state)

		if cell_weights is not None:
			cell_weights = numpy.array(cell_weights, dtype=numpy.float32)
			cell_weights = cell_weights / cell_weights.sum() 

		self.cell_weights = cell_weights

		self.signal = {chrom: signal[chrom] for chrom in chroms}
		self.sequence = {chrom: sequence[chrom] for chrom in chroms}
		self.neighbors = neighbors
		self.cell_states = cell_states
		self.read_depths = read_depths
		self._lengths = numpy.array([seq.shape[1] for seq in 
			self.sequence.values()])
		self._chrom_probs = self._lengths / self._lengths.sum()

	def __len__(self):
		return sum(self._lengths)

	def __getitem__(self, idx):
		c_idx = numpy.random.choice(len(self._lengths), p=self._chrom_probs)
		chrom = self.chroms[c_idx]

		mid = numpy.random.randint(10000, self._lengths[c_idx]-10000)
		cell_idx = numpy.random.choice(len(self.cell_states), 
			p=self.cell_weights)
		
		return _extract_example(self, chrom, mid, cell_idx, idx)
