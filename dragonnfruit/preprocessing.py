# preprocessing.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import io
import gzip
import h5py
import numpy
import pyBigWig

import collections

from tqdm import tqdm
from scipy.sparse import csc_matrix
from scipy.sparse import save_npz

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors


def reverse_complement(seq):
	rc = ''

	for char in seq.upper():
		if char == 'A':
			rc += 'T'
		elif char == 'C':
			rc += 'G'
		elif char == 'G':
			rc += 'C'
		elif char == 'T':
			rc += 'A'
		else:
			rc += 'N'

	return rc[::-1]


def read_chrom_sizes(filename, chroms=None):
	"""Read a standard <genome>_chrom.sizes file.

	Reads a tab-separated file that contains two columns: the name of the
	chromosome and its length in bp. Returns a dictionary storing the sizes
	as well as a header file for use in creating a bigwig.


	Parameters
	----------
	filename: str
		The name of the file containing chromosome sizes.

	chroms: list or None
		A list of chromosomes to consider. If None, return all chromosomes
		found in the file.


	Returns
	-------
	chrom_sizes: dict
		A dictionary where the keys are chromosome names and the values are the
		size of that chromosome.

	header: list
		The same information as chrom_sizes but ordered according to the
		chroms. If chroms is None, just returns the items.
	"""

	chrom_sizes = {}
	with open(chrom_size_file, "r") as infile:
		for line in infile:
			chrom, n = line.strip("\r\n").split()
			if chroms is None or chrom in chroms:
				chrom_sizes[chrom] = int(n)

	if chroms is None:
		chroms = chrom_sizes.keys()

	header = []
	for chrom in chroms:
		header.append((chrom, chrom_sizes[chrom]))

	return chrom_sizes, header


class MultimodalData(object):
	"""A collection of data sets organized around a shared row axis.

	Based loosly on AnnData, this object organizes single-cell data at
	basepair resolution using sparse matrices to store the reads. This can
	organize a single experiment or multiome data by aligning the rows
	across experiments.


	Parameters
	----------
	X: dict
		datums
	"""

	def __init__(self, chrom_sizes, cell_names=[], chroms=None, verbose=True):
		self.atac = {}
		self.rna = {}

		self.cell_names = cell_names
		self._cell_name_mapping = {}
		for i, name in enumerate(cell_names):
			self._cell_name_mapping[name] = i
			self._cell_name_mapping[reverse_complement(name)] = i

		self.n = len(self.cell_names)

		if isinstance(chrom_sizes, str)
			self.chrom_sizes = read_chrom_sizes(chrom_sizes, chroms=chroms)[0]
		else:
			self.chrom_sizes = chrom_sizes

		self.chroms = list(self.chrom_sizes.keys())

	def add_atac_fragments(self, filenames, add_cells=True, cell_name_prefix='', 
		key='atac', max_fragment_length=1000):
	 	"""Add multiple scATAC-seq fragment files to the data set.

	 	This function will go through and add several filenames to the stored
	 	data, appending cell prefixes as necessary.


	 	Parameters
	 	----------

	 	"""

		# Read the file and add reads to the start and the end of the paired-end read
		X = {chrom: {'row': [], 'col': [], 'val': []} for chrom in chroms}

		# Add read depths vector
		read_depths = collections.defaultdict(int)

		for filename in filenames:
			with gzip.open(filename, "r") as infile:
				for i, line in tqdm(enumerate(infile)):
					line = line.decode('UTF-8')
					if line.startswith("#"):
						continue

					chrom, start, end, name, _ = line.split()
					start, end = int(start), int(end)
					name = cell_name_prefix + str(name)

					# Make sure that the cell is in the inclusion list if provided
					if name not in self._cell_name_mapping:
						if add_cells == False:
							continue

						name_rc = reverse_complement(name)

						self.cell_names.append(name)
						self._cell_name_mapping[name] = self.n
						self._cell_name_mapping[name_rc] = self.n
						self.n += 1

					# Filter by chromosome
					if chrom not in self.chrom_sizes:
						continue

					# Filter reads off the end
					if start > self.chrom_sizes[chrom]:
						raise ValueError("Read {}:{}-{} maps past the end of chrom {}".format(
							chrom, start, end, chrom))

					# Return the index that is consistent with RNA-seq
					cell_idx = self._cell_name_mapping[name]

					# Filter by length
					if end - start >= max_fragment_length:
						continue

					read_depths[cell_idx] += 2

					# The fragment file is +4/-5 corrected and we need +4/-4
					X[chrom]['row'].append(cell_idx)
					X[chrom]['col'].append(start)
					X[chrom]['val'].append(1)

					X[chrom]['row'].append(cell_idx)
					X[chrom]['col'].append(end+1)
					X[chrom]['val'].append(1)

		X_cscs = {}
		for chrom in chroms:
			row = X[chrom]['row']
			col = X[chrom]['col']
			val = X[chrom]['val']
			n, d = len(cell_barcodes), chrom_sizes[chrom]

			# Create and save the sparse matrix
			X_csc = csc_matrix((val, (row, col)), shape=(n, d), dtype='int8')
			X_cscs[chrom] = X_csc

		self.atac['X'] = X_cscs
		self.atac['read_depths'] = numpy.array([read_depths[cell] for cell in 
			self.cell_names])

	def preprocess_atac(self, peaks, n_components=50, n_neighbors=2000):
		"""preprocess"""

		names = 'chrom', 'start', 'end'
		self.atac_peaks = pandas.read_csv(peaks, sep="\t", names=names, 
			usecols=(0, 1, 2))
		
		d = not self.verbose
		atac_peak_counts = numpy.zeros((self.n, peaks.shape[0]))
		for i, (chrom, start, end) in tqdm(peaks.iterrows(), disable=d):
			if chrom not in self.chrom_sizes:
				continue

			peak = self.X['atac'][chrom][:, start:end+1].sum(axis=1)
			peak = numpy.array(peak).flatten()
			atac_peak_counts[:, i] = peak

		# ATAC peak counts
		self.atac['peak_counts'] = atac_peak_counts

		# tf-idf it
		atac_tfidf = TfidfTransformer().fit_transform(atac_peak_counts)
		self.atac['tfidf'] = atac_tfidf

		# PCA
		atac_pca = TruncatedSVD(n_components).fit_transform(atac_tfidf)
		self.atac['pca'] = atac_pca

		# UMAP
		atac_umap = UMAP().fit_transform(atac_pca)
		self.atac['umap'] = atac_umap

		# Neighbors
		nn = NearestNeighbors(n_neighbors=n_neighbors)
		nn.fit(atac_pca)
		neighbors = nn.kneighbors(atac_pca)
		self.atac['neighbors'] = neighbors

	def save(self, filename):
		f = h5py.File(filename, "w")
		f.create_dataset("chroms", data=numpy.array(chroms))
		f.create_dataset("chrom_sizes", data=numpy.array([chrom_sizes[chrom]
			for chrom in self.chroms]))
		f.create_dataset("cell_names", data=numpy.array(self.cell_names))
		f.create_dataset("n", data=numpy.array([self.n]))
		f.create_dataset("verbose", data=numpy.array([self.verbose]))

		atac = f.create_group("atac")
		atac_X = atac.create_group("X")

		for chrom in self.chroms:
			data = self.atac['X'][chrom]

			atac_X.create_dataset(chrom + ".data", data=data.data)
			atac_X.create_dataset(chrom + ".indptr", data=data.indptr)
			atac_X.create_dataset(chrom + ".indices", data=data.indices)

		params = ['read_depths', 'peak_counts', 'tfidf', 'pca', 'umap', 
			'neighbors']

		for param in params:
			atac.create_dataset(param, data=self.atac[param])


	@classmethod
	def load(cls, filename):
		f = h5py.File(filename, "r")
		chroms = f['chroms'][:]
		chrom_sizes = f['chrom_sizes'][:]
		chrom_sizes = {chrom: size for chrom, size in zip(chroms, chrom_sizes)}
		cell_names = f['cell_names'][:]
		n = f['n'][:][0]
		verbose = f['verbose'][:][0]

		data = MultimodalData(chrom_sizes=chrom_sizes, cell_names=cell_names, 
			chroms=chroms, verbose=verbose)
		data['atac']['X'] = {}

		atac = f.atac
		atac_X = f.atac.X

		for chrom in chroms:
			data = f.atac.X[chrom + ".data"][:]
			indptr = f.atac.X[chrom + ".indptr"][:]
			indices = f.atac.X[chrom + '.indices'][:]

			csc = scipy.sparse.csc_matrix((data, indices, indptr),
				shape=(len(cell_names), chrom_sizes[chrom]))

			data['atac']['X'][chrom] = csc

		params = ['read_depths', 'peak_counts', 'tfidf', 'pca', 'umap', 
			'neighbors']

		for param in params:
			data['atac'][param] = f.atac[param][:]





def create_pseudobulks(X_index, row_index, output_filename, chroms=None):
	chrom_set = self.chrom_sizes.keys() if chroms is None else set(chroms)
	header = list(self.chrom_sizes.items())

	# Create the bigwig to save a pseudobulk
	bw = pyBigWig.open(output_filename, "w")
	bw.addHeader(header, maxZooms=0)
	for chrom in chroms:
		X_csc = reads[chrom]

		# Sum across cells to produce the pseudobulk
		X_bulk = X_csc.astype('float32').sum(axis=0)
		X_bulk = numpy.array(X_bulk)[0]

		# Add to the bigwig
		bw.addEntries(chrom, 0, values=X_bulk, step=1, span=1)
		del X_csc

	bw.close()
