# preprocessing.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import gzip
import numpy
import pandas
import pyBigWig

import collections

from tqdm import tqdm
from scipy.sparse import csc_matrix
from scipy.sparse import save_npz

from joblib import Parallel
from joblib import delayed

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
	with open(filename, "r") as infile:
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


def extract_fragments(filename, chrom_sizes=None, include_cells=None,
	exclude_cells=None, cell_name_prefix='', max_fragment_length=1000, 
	start_offset=4, end_offset=-5, idx=None, verbose=True):
	"""aaa

	"""

	# Create a dictionary for the reads
	X = {chrom: {'row': [], 'col': [], 'val': []} for chrom in chrom_sizes}

	# Create a read depths vector
	read_depths = collections.defaultdict(int)

	_cell_names = []
	_cell_name_mapping = {}

	if include_cells is not None:
		for idx, name in enumerate(include_cells):
			name_rc = reverse_complement(name)

			name = cell_name_prefix + name
			name_rc = cell_name_prefix + name_rc

			_cell_name_mapping[name] = len(_cell_names)
			_cell_name_mapping[name_rc] = len(_cell_names)
			_cell_names.append(name)

		include_cells = set(include_cells)

	_excluded_cells = set()
	_included_cells = set()
	_not_included_cells = set()
	_added_cell_count = 0
	_fragment_chrom_filter_count = 0
	_fragment_length_filter_count = 0
	_fragment_count = 0

	with gzip.open(filename, "r") as infile:
		for i, line in tqdm(enumerate(infile), position=idx):
			line = line.decode('UTF-8')
			if line.startswith("#"):
				continue

			# Extract and recast the fragment information
			chrom, start, end, name, _ = line.split()
			start, end = int(start), int(end)
			name = str(name)
			prefix_name = cell_name_prefix + name

			# Filter fragments by chromosome
			if chrom not in chrom_sizes:
				_fragment_chrom_filter_count += 1
				continue

			# Make sure that the cell is in the inclusion list if provided
			if exclude_cells is not None and name in exclude_cells:
				if name not in _excluded_cells:
					_excluded_cells.add(name)
				continue

			if include_cells is not None and name not in include_cells:
				if name not in _not_included_cells:
					_not_included_cells.add(name)
				continue

			# Filter by length
			if (end - start) >= max_fragment_length:
				_fragment_length_filter_count += 1
				continue

			# Filter reads off the end
			if start > chrom_sizes[chrom]:
				raise ValueError("Read {}:{}-{} maps past the end of chrom {}".format(
					chrom, start, end, chrom))

			# Only after all the filters, add a new cell if an explicit
			# inclusion list has not been provided.
			if include_cells is None and prefix_name not in _cell_name_mapping:
				name_rc = reverse_complement(name)
				prefix_name_rc = cell_name_prefix + name_rc

				_cell_names.append(name)
				_cell_name_mapping[prefix_name] = n_cells
				_cell_name_mapping[prefix_name_rc] = n_cells
				
				n_cells += 1
				_added_cell_count += 1
			
			if name not in _included_cells:
				_included_cells.add(name)

			# Return the index that is consistent with RNA-seq
			cell_idx = _cell_name_mapping[prefix_name]

			read_depths[cell_idx] += 2
			_fragment_count += 1

			# The fragment file is +4/-5 corrected and we need +4/-4
			X[chrom]['row'].append(cell_idx)
			X[chrom]['col'].append(start + 4 - start_offset)
			X[chrom]['val'].append(1)

			X[chrom]['row'].append(cell_idx)
			X[chrom]['col'].append(end - end_offset - 5)
			X[chrom]['val'].append(1)

	ni = 0 if include_cells is None else len(include_cells)
	ne = 0 if exclude_cells is None else len(exclude_cells) // 2
	if include_cells is not None and ni != len(_included_cells):
		raise ValueError("Not all cells in inclusion list have been" +
			" observed. Inclusion={}, Observed={}".format(ni,
				len(_included_cells)))

	if verbose:
		print(filename)
		print("Cell included: ", len(_included_cells))
		print("Cells excluded: ", len(_excluded_cells))
		print("Cells not included: ", len(_not_included_cells))
		print("Cells added: ", _added_cell_count)
		print("Cell inclusion list size: ", ni)
		print("Cell exclusion list size: ", ne)
		print(len(_cell_names), len(_cell_name_mapping), len(read_depths))

		print("Frag. included count: ", _fragment_count)
		print("Frag. filtered by chrom: ", _fragment_chrom_filter_count)
		print("Frag. filtered by length: ", _fragment_length_filter_count)
		print()

	return X, read_depths, _cell_names, _cell_name_mapping
	

def fragments_to_sparse(fragments, chrom_sizes, chroms=None, include_cells=None, 
	exclude_cells=None, cell_name_prefixes=None, max_fragment_length=1000, 
	start_offset=4, end_offset=-5, verbose=True):
	"""Add multiple scATAC-seq fragment files to the data set.

	This function will go through and add several filenames to the stored
	data, appending cell prefixes as necessary.


	Parameters
	----------

	"""

	# Turn the exclusion lists into sets with reverse complementing
	n_files = len(fragments)

	if include_cells is None:
		_include_cells = [None for i in range(n_files)]
	else:
		_include_cells = []
		for _include in include_cells:
			if isinstance(_include, str):
				_include = numpy.loadtxt(_include, dtype=str)

			_include_cells.append(_include)


	_exclude_cells = []
	if exclude_cells is None:
		_exclude_cells = [None for i in range(n_files)]
	else:
		_exclude_cells = []
		for _exclude in exclude_cells:
			if isinstance(_exclude, str):
				_exclude = numpy.loadtxt(_exclude, dtype=str).tolist()

			_exclude_cells.append(set(_exclude))

	###

	if cell_name_prefixes is None:
		cell_name_prefixes = [None for i in range(n_files)]

	# Load the chromosome sizes 
	if isinstance(chrom_sizes, str):
		chrom_sizes = read_chrom_sizes(chrom_sizes, chroms=chroms)[0]
	else:
		chrom_sizes = chrom_sizes

	chroms = list(chrom_sizes.keys())

	results = Parallel(n_jobs=-1)(delayed(extract_fragments)(
		filename=filename, 
		chrom_sizes=chrom_sizes, 
		include_cells=_include,
		exclude_cells=_exclude, 
		cell_name_prefix=prefix,
		max_fragment_length=max_fragment_length,
		start_offset=start_offset,
		end_offset=end_offset,
		idx=i,
		verbose=verbose) 
	for i, filename, _include, _exclude, prefix in zip(range(n_files), 
		fragments, _include_cells, _exclude_cells, cell_name_prefixes)
	)

	# Create a dictionary for the reads
	X = {chrom: {'row': [], 'col': [], 'val': []} for chrom in chroms}

	# Create a read depths vector
	read_depths = collections.defaultdict(int)

	_cell_names = []
	_cell_name_mapping = {}
	n_cells = 0

	for X_i, read_depths_i, _cell_names_i, _cell_name_mapping_i in results:
		_cell_names.extend(_cell_names_i)
		for name, idx in _cell_name_mapping_i.items():
			_cell_name_mapping[name] = idx + n_cells

		for idx, count in read_depths_i.items():
			read_depths[idx + n_cells] = count

		for chrom in chroms:
			for idx in X_i[chrom]['row']:
				X[chrom]['row'].append(idx + n_cells)

			X[chrom]['col'].extend(X_i[chrom]['col'])
			X[chrom]['val'].extend(X_i[chrom]['val'])

		print(n_cells, len(_cell_names), len(_cell_names_i), len(_cell_name_mapping_i))

		n_cells += len(_cell_names_i)
		del X_i, read_depths_i, _cell_names_i, _cell_name_mapping_i


	X_cscs = {}
	for chrom in chroms:
		row = X[chrom]['row']
		col = X[chrom]['col']
		val = X[chrom]['val']
		d = chrom_sizes[chrom]

		# Create and save the sparse matrix
		X_csc = csc_matrix((val, (row, col)), shape=(n_cells, d), dtype='int8')
		X_cscs[chrom] = X_csc


	read_depths = numpy.array([read_depths[i] for i in range(n_cells)])
	return X_cscs, read_depths

def preprocess_atac(X, peaks, chroms, n_components=50, n_neighbors=500, 
	verbose=True):
	"""preprocess"""

	names = 'chrom', 'start', 'end'
	peaks = pandas.read_csv(peaks, sep="\t", names=names, 
		usecols=(0, 1, 2))
	
	chroms = set(chroms)
	n = X[list(chroms)[0]].shape[0]
	d = not verbose

	peak_counts = numpy.zeros((n, peaks.shape[0]))
	for i, (chrom, start, end) in tqdm(peaks.iterrows(), disable=d):
		if chrom not in chroms:
			continue

		peak = X[chrom][:, start:end+1].sum(axis=1)
		peak = numpy.array(peak).flatten()
		peak_counts[:, i] = peak

	# tf-idf it
	X_tfidf = TfidfTransformer().fit_transform(peak_counts)
	X_pca = TruncatedSVD(n_components).fit_transform(X_tfidf)
	X_umap = X_pca
	#X_umap = UMAP().fit_transform(X_pca)

	# Neighbors
	nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
	nn.fit(X_pca)
	neighbors = nn.kneighbors(X_pca)
	return X_pca, X_umap, neighbors


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
