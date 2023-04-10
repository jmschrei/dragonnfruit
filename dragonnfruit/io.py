# io.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import time
import numpy
import scipy
import torch
import pandas

from bpnetlite.io import one_hot_encode


def _extract_example(self, chrom, mid, cell_idx, idx):
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
		print("Loci: {}".format(self.loci.shape[0]))

	def __len__(self):
		return self.loci.shape[0]

	def __getitem__(self, idx):
		chrom, _, _, mid = self.loci.iloc[idx // self.n_cells_per_locus]
		mid += self.random_state.randint(-self.max_jitter, self.max_jitter+1)
		cell_idx = self.random_state.randint(self.cell_states.shape[0])
		return _extract_example(self, chrom, mid, cell_idx, idx)

	
class GWGenerator(torch.utils.data.Dataset):
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
		print(self._lengths)

	def __len__(self):
		return sum(self._lengths)

	def __getitem__(self, idx):
		c_idx = numpy.random.choice(len(self._lengths), 
			p=self._lengths / self._lengths.sum())
		chrom = self.chroms[c_idx]

		mid = numpy.random.randint(10000, self._lengths[c_idx]-10000)
		cell_idx = numpy.random.randint(self.cell_states.shape[0])
		return _extract_example(self, chrom, mid, cell_idx, idx)


class GWBGenerator(torch.utils.data.Dataset):
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
		read_depths, trimming, window, chroms, batch_size=1024, 
		cells_per_loci=1, reverse_complement=True, random_state=None):
		self.trimming = trimming
		self.window = window
		self.chroms = chroms
		self.reverse_complement = reverse_complement
		self.batch_size = batch_size
		self.cells_per_loci = cells_per_loci
		self.random_state = numpy.random.RandomState(random_state)

		print(signal['chr21'].dtype, sequence['chr21'].dtype)

		self.signal = {chrom: signal[chrom] for chrom in chroms}
		self.sequence = {chrom: sequence[chrom] for chrom in chroms}
		self.neighbors = neighbors
		self.cell_states = cell_states
		self.read_depths = read_depths
		self._lengths = numpy.array([seq.shape[0] for seq in self.sequence.values()])
		self._probs = self._lengths / self._lengths.sum()

		print(self._lengths)

	def __len__(self):
		return sum(self._lengths)

	def __getitem__(self, idx):
		tic = time.time()
		X = numpy.empty((self.batch_size, 4, self.window), dtype=numpy.int8)
		y = numpy.empty((self.batch_size, self.window - self.trimming*2), dtype=numpy.float32)
		c = numpy.empty((self.batch_size, self.cell_states.shape[1]), dtype=numpy.float32)
		r = numpy.empty((self.batch_size, 1), dtype=numpy.float32)

		cell_idxs = self.random_state.randint(self.cell_states.shape[0], size=self.batch_size)
		chrom_idxs = self.random_state.choice(len(self._lengths), p=self._probs, size=self.batch_size) 

		nc = self.cells_per_loci
		a = time.time() - tic
		b, d, e = 0, 0, 0

		for i, (cell_idx, chrom_idx) in enumerate(zip(cell_idxs, chrom_idxs)):
			tic = time.time()
			if i % nc == 0:
				chrom = self.chroms[chrom_idx]

				mid = self.random_state.randint(10000, self._lengths[chrom_idx]-10000)
				start, end = mid - self.window // 2, mid + self.window // 2

				lidx = (i // nc) * nc
				X[lidx:lidx+nc] = self.sequence[chrom][start:end].T
				y_ = self.signal[chrom][:, start+self.trimming:end-self.trimming].tocsr()
				b += time.time() - tic

			tic = time.time()
			neighbs = self.neighbors[cell_idx]
			y[i] = numpy.array(y_[neighbs].sum(axis=0))[0]
			d += time.time() - tic

			tic = time.time()
			c[i] = self.cell_states[cell_idx]
			r[i] = self.read_depths[cell_idx]
			e += time.time() - tic

			if self.reverse_complement and idx % 2 == 0:
				X[i] = X[i][::-1][:, ::-1].copy()
				y[i] = y[i][::-1].copy()

		tic = time.time()
		X = torch.from_numpy(X.astype('float32'))
		y = torch.from_numpy(y)
		c = torch.from_numpy(c)
		r = torch.from_numpy(r)
		#print(idx, a, b, d, e, time.time() - tic)
		return X, y, c, r
