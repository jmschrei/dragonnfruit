# preprocessing.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import os
import gzip
import scipy
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

import time


class FlexibleArray():
	"""A wrapper around a numpy array that can expand.

	Classic numpy arrays are fixed in size and so cannot easily be used when
	you do not know the number of elements to store. In contrast, linked lists
	can be used regardless of the number of elements but have a memory overhead
	and use Python's primitive data types, which themselves have significant
	overhead. This thin-wrapper around a numpy array implements the `append` 
	method of a Python list by beginning with an array of a given size and, 
	when the next element would fall off the end of the array, increasing 
	the size of the array. Each expansion requires copying the entire array
	to the new object, incurring a small overhead. There is also a trim method
	to be called at the end which will prune the size down to the actual
	number of stored elements.


	Parameters
	----------
	n_elems: int
		The size of the original numpy array

	step_size: int
		The number of elements to add when the capacity fills up.

	dtype: numpy.dtype, optional
		The dtype of the underlying array. Default is numpy.uint32.
	"""

	def __init__(self, n_elems, step_size, dtype=numpy.uint32):
		self.array = numpy.empty(n_elems, dtype=dtype)
		self.step_size = step_size
		self.n_set_items = 0

	def __len__(self):
		return len(self.array)

	def __getitem__(self, idx):
		return self.array[idx]

	def _expand(self):
		array = numpy.empty(len(self) + self.step_size, dtype=self.array.dtype)
		array[:len(self)] = self.array
		self.array = array

	def append(self, item):
		if self.n_set_items == len(self):
			self._expand()
		
		self.array[self.n_set_items] = item
		self.n_set_items += 1

	def trim(self):
		self.array = self.array[:self.n_set_items]


def reverse_complement(seq):
	"""Reverse complement a DNA sequence.

	Parameters
	----------
	seq: str
		A str composed of the letters 'A', 'C', 'G', and 'T'.


	Returns
	-------
	rc: str
		A string that has been reversed in direction and had the nucleotides
		inverted.
	"""

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


def _extract_fragments(filename, chrom_sizes, include_cells=None,
	exclude_cells=None, cell_name_prefix='', max_fragment_length=1000, 
	start_offset=4, end_offset=-5, tqdm_position=None, verbose=False):
	"""Read a fragments file and filter using some basic criteria.

	This function will take in a single fragment file and return the fragments
	in it after filtering. The filtering can involve only keeping fragments
	from specific cells, excluding fragments from specific cells, and having a
	maximum length the fragment can be. Additionally, this method can correct
	the exact fragment positioning.

	This function is meant for internal use only.


	Parameters
	----------
	filename: str
		The name of the fragment file to be processed.

	chrom_sizes: dict
		A dictionary of chromosome sizes where the keys are the name of the
		chromosomes and the values are the lengths, in basepairs, of that
		chromosome.

	include_cells: list or None, optional
		If a list, contains the barcodes of the only cells to extract fragments
		for. These barcodes are internally reverse complemented. If None,
		use fragments from all cells. Default is None.

	exclude_cells: list or None, optional
		If a list, contains the barcodes of cells whose fragments should be
		excluded. These barcodes are internally reverse complemented. If None,
		do not exclude fragments from any cells. Default is None.

	cell_name_prefix: str, optional
		If your include/exclude lists have prefixes, perhaps because you have
		many fragment files and wanted to specify the file and the cell jointly,
		indicate that prefix here. Default is ''.

	max_fragment_length: int, optional
		The maximum length of fragments to consider. Default is 1000.

	start_offset: int, optional
		The exact recorded positioning of the starts is the recorded start
		of the fragment - 4 + start_offset. When this value is 4, no offset
		is recorded. Default is 4.

	end_offset: int, optional
		The exact recorded positioning of the ends is the recorded end of the
		fragment - end_offset - 5. When this value is 5, no offset is recorded.
		Default is 5.

	tqdm_position: int or None, optional
		When using verbose mode, a progress bar is displayed. When processing
		multiple files, one can display vertically stacked progress bars, one 
		for each file by controlling their positioning with this parameter.
		If None, the progress bar displays at the bottom of the screen. Default 
		is None.

	verbose: bool, optional
		Whether to display a progress bar as fragments are extracted. Default
		is False.


	Returns
	-------
	X: dict
		A nested dictionary where the keys are chromosome names and values
		are dictionary that themselves have the keys 'row', 'col', and 'val'.
		These inner dictionaries are sparse representations, at basepair
		resolution, of where the fragment ends are.

	read_depths: dict
		A dictionary where the keys are the barcodes of the cells and the
		values are the integer number of fragment ends recorded.

	_cell_names: list
		A list of cell barcodes correspond to the 'row' parameter in X.

	_cell_name_mapping: dict
		A dictionary where the keys are cell barcodes, and also the reverse
		complement of the barcodes, and the value is the index in _cell_names
		that the barcode appears. Because this contains cell barcodes and
		reverse complements, it should contain twice as many elements as
		_cell_names.
	"""

	# Create a dictionary for the reads
	X = {
			chrom: {
				'row': FlexibleArray(1000000, 1000000), 
				'col': FlexibleArray(2000000, 2000000)
			} for chrom in chrom_sizes
	}

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
		for i, line in tqdm(enumerate(infile), desc=cell_name_prefix, 
			position=tqdm_position, disable=not verbose, leave=None):
			line = line.decode('UTF-8')
			if line.startswith("#"):
				continue

			# Extract and recast the fragment information
			chrom, start, end, name, _ = line.split()
			name = str(name)

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
			start, end = int(start), int(end)
			if (end - start) >= max_fragment_length:
				_fragment_length_filter_count += 1
				continue

			# Filter reads off the end
			if start > chrom_sizes[chrom]:
				raise ValueError("Read {}:{}-{} maps off chrom {}".format(
					chrom, start, end, chrom))

			# Only after all the filters, add a new cell if an explicit
			# inclusion list has not been provided.
			prefix_name = cell_name_prefix + name
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
			X[chrom]['col'].append(end - end_offset - 5)

	for d in X.values():
		d['row'].trim()
		d['col'].trim()

	ni = 0 if include_cells is None else len(include_cells)
	ne = 0 if exclude_cells is None else len(exclude_cells) // 2
	if include_cells is not None and ni != len(_included_cells):
		raise ValueError("Not all cells in inclusion list have been" +
			" observed. Inclusion={}, Observed={}".format(ni,
				len(_included_cells)))

	if verbose:
		print("\n" + filename)
		print("Cell included: ", len(_included_cells))
		print("Cells excluded: ", len(_excluded_cells))
		print("Cells not included: ", len(_not_included_cells))
		print("Cells added: ", _added_cell_count)
		print("Cell inclusion list size: ", ni)
		print("Cell exclusion list size: ", ne)

		print("Frag. included count: ", _fragment_count)
		print("Frag. filtered by chrom: ", _fragment_chrom_filter_count)
		print("Frag. filtered by length: ", _fragment_length_filter_count)
		print()

	return X, read_depths, _cell_names, _cell_name_mapping
	

def extract_fragments(fragments, chrom_sizes, chroms=None, include_cells=None, 
	exclude_cells=None, cell_name_prefixes=None, max_fragment_length=1000, 
	start_offset=4, end_offset=-5, n_jobs=-1, verbose=True):
	"""Process multiple fragment files into a unified set of sparse matrices.

	This function operates on several fragment files, extracting fragments
	from each one according to file-specific inclusion/exclusion lists, and
	returns a single sparse matrix that combines fragments and cells across
	all files.

	This function will return a unified set of sparse matrices, onee per
	chromosome, where the rows contain cells across all fragment files.


	Parameters
	----------
	fragments: list
		A list of filenames of fragment files.

	chrom_sizes: str or dict
		If a string, will read a chromosome size file into a dictionary where
		the keys are chromosome names and the values are the size, in basepairs,
		of the chromosome. If a dictionary, will assume that format.

	chroms: list or None
		If a list, will only extract fragments from the specified chromosomes.
		If None, will include all chromosomes. Default is None.

	include_cells: list or None, optional
		If a list, assumes that it is a list of lists where the inner lists 
		contain the barcodes of the only cells to extract fragments for, with
		the first list in `include_cells` corresponding to the first fragment
		file, etc. These barcodes are internally reverse complemented. If None,
		use fragments from all cells. Default is None.

	exclude_cells: list or None, optional
		If a list, assumes that it is a list of lists where the inner lists 
		contain the barcodes of cells whose fragments should be excluded for, 
		with the first list in `include_cells` corresponding to the first 
		fragment file, etc. These barcodes are internally reverse complemented. 
		If None, use fragments from all cells. Default is None.

	cell_name_prefixes: list, optional
		A list of string prefixes to put at the beginning of the cell names in
		the include/exclude_cells names, with the i-th prefix in this list
		being added to cells in the i-th include/exclude_cell_names barcodes.
		This option exists for when the include/exclude_cells lists contain a
		prefix indicating which file the cells come from, perhaps indicating
		the time in a timecourse. If None, do not add a prefix. Default is None.

	max_fragment_length: int, optional
		The maximum length of fragments to consider. Default is 1000.

	start_offset: int, optional
		The exact recorded positioning of the starts is the recorded start
		of the fragment - 4 + start_offset. When this value is 4, no offset
		is recorded. Default is 4.

	end_offset: int, optional
		The exact recorded positioning of the ends is the recorded end of the
		fragment - end_offset - 5. When this value is 5, no offset is recorded.
		Default is 5.

	n_jobs: int, optional
		The number of fragment files to process in parallel. If -1, use all
		available cores. Defaut is -1.

	verbose: bool, optional
		Whether to display a progress bar as fragments are extracted. Default
		is False.


	Returns
	-------
	X_cscs: dict
		A dictionary where the keys are chromosome names and the values are
		csc-formatted sparse matrices with the rows being cells across all
		fragment files and the columns being basepairs in the chromosome.

	read_depths: numpy.ndarray
		The total number of fragment ends for each cell. If one were to sum
		X_cscs across basepairs (axis=1) and then sum across the chromosomes,
		they would get this value.
	"""

	# Process the inclusion/exclusion inputs for the next steps 
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


	if cell_name_prefixes is None:
		cell_name_prefixes = [None for i in range(n_files)]


	# Load the chromosome sizes 
	if isinstance(chrom_sizes, str):
		chrom_sizes = read_chrom_sizes(chrom_sizes, chroms=chroms)[0]
	else:
		chrom_sizes = chrom_sizes

	chroms = list(chrom_sizes.keys())


	# Run the extraction function on each of the files. 
	results = Parallel(n_jobs=n_jobs)(delayed(_extract_fragments)(
		filename=filename, 
		chrom_sizes=chrom_sizes, 
		include_cells=_include,
		exclude_cells=_exclude, 
		cell_name_prefix=prefix,
		max_fragment_length=max_fragment_length,
		start_offset=start_offset,
		end_offset=end_offset,
		tqdm_position=i,
		verbose=verbose) 
	for i, filename, _include, _exclude, prefix in zip(range(n_files), 
		fragments, _include_cells, _exclude_cells, cell_name_prefixes)
	)

	# Create a dictionary for the reads
	X = {chrom: {'row': [], 'col': []} for chrom in chroms}

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
			X[chrom]['row'].append(X_i[chrom]['row'].array + n_cells)
			X[chrom]['col'].append(X_i[chrom]['col'].array)
			
		n_cells += len(_cell_names_i)
		del X_i, read_depths_i, _cell_names_i, _cell_name_mapping_i


	X_cscs = {}
	for chrom in chroms:
		row = numpy.repeat(numpy.concatenate(X[chrom]['row']), 2)
		col = numpy.concatenate(X[chrom]['col'])
		val = numpy.ones(len(col), dtype='int8')
		d = chrom_sizes[chrom]

		# Create and save the sparse matrix
		X_csc = csc_matrix((val, (row, col)), shape=(n_cells, d), dtype='int8')
		X_cscs[chrom] = X_csc


	read_depths = numpy.array([read_depths[i] for i in range(n_cells)])
	return X_cscs, read_depths


def extract_cellxpeak(X, peaks, chroms=None, verbose=False):
	"""Take mapped reads and return a cell x peak matrix.

	This function will take in a dictionary of sparse matrices, e.g., the bp
	mapped reads, and a set of peak coordinates, and return an AnnData object
	that is the number of reads mapped at each position.


	Parameters
	----------
	X: dict
		A dictionary of scipy sparse matrices where each key is a chromosome
		with the same name as those in `peaks` and each value is a scipy
		sparse matrix where the rows are cells and the columns are positions
		in the genome.

	peaks: pandas.DataFrame or str
		Either a pandas DataFrame in bed format where the first three columns
		are `chrom`, `start`, `end` (additional columns will be ignored) or
		a string. A string is assumed to be a filename for a bed file and read
		assuming that format.

	chroms: list, tuple, or None, optional
		An iterable of chroms to restrict the analysis to or None. If None,
		uses peaks from all chromosomes. Default is None.

	verbose: bool, optional
		Whether to display progress as peaks are being extracted.


	Returns
	-------
	y: scipy.sparse.csr_matrix
		An AnnData object containing reads for each cell at each peak.
	"""

	if isinstance(peaks, str):
		names = 'chrom', 'start', 'end'
		peaks = pandas.read_csv(peaks, sep="\t", names=names, 
			usecols=(0, 1, 2))
	
	if chroms is None:
		chroms = set(list(numpy.unique(peaks['chrom'])))
	else:
		chroms = set(chroms)

	n = X[list(chroms)[0]].shape[0]
	d = not verbose
	n_peaks = 0

	data, indices, indptr = [], [], [0]
	for i, (chrom, start, end) in tqdm(peaks.iterrows(), disable=d):
		if chrom not in chroms:
			continue

		peak = X[chrom][:, start:end+1]
		n_peaks += 1

		data.append(peak.data)
		indices.append(peak.indices)
		indptr.append(len(peak.data))

	data = numpy.concatenate(data)
	indices = numpy.concatenate(indices)
	indptr = numpy.cumsum(indptr)

	peak_counts = scipy.sparse.csc_matrix((data, indices, indptr), shape=(n, 
		n_peaks), dtype='int8')
	return peak_counts


def preprocess_sparse_atac(X_cscs, peaks, chroms, n_components=50, 
	n_neighbors=500, n_jobs=-1, verbose=True):
	"""Preprocess the sparse fragments into all the needed files.

	Given a matrix of representations `X`, run TF-IDF to reweight positions
	and cells based on read depth, run PCA to calculate a low-dimensional 
	embedding, calculate the nearest neighbors in that low-dimensional
	embedding.


	Parameters
	----------
	X_cscs: dict
		A dictionary where the keys are chromosome names and the values are
		csc-formatted sparse matrices with the rows being cells across all
		fragment files and the columns being basepairs in the chromosome.

	peaks: pandas.DataFrame or str
		Either a pandas DataFrame in bed format where the first three columns
		are `chrom`, `start`, `end` (additional columns will be ignored) or
		a string. A string is assumed to be a filename for a bed file and read
		assuming that format.

	chroms: list, tuple, or None, optional
		An iterable of chroms to restrict the analysis to or None. If None,
		uses peaks from all chromosomes. Default is None.

	n_components: int, optional
		The number of dimensions to use PCA to project into. Default is 50.

	n_neighbors: int, optional
		The number of neighbors to calculate for each point. Default is 500.

	n_jobs: int, optional
		The number of jobs to use to calculate nearest neighbors. If -1, use
		all cores. Default is -1.

	verbose: bool, optional
		Whether to display progress as peaks are being extracted.


	Returns
	-------
	peak_counts: numpy.ndarray, shape=(-1, n_peaks)
		A cell x n_peaks matrix, containing the number of reads mapping to
		each peak in each cell.

	X_pca: numpy.ndarray, shape=(-1, n_components)
		A cell x n_components matrix, containing low-dimensional representations
		from PCA for each cell.

	neighbors: numpy.ndarray, shape=(-1, n_neighbors)
		An integer list of neighbors for each cell.
	"""

	peak_counts = extract_cellxpeak(X_cscs, peaks, chroms=chroms, 
		verbose=verbose)

	X_tfidf = TfidfTransformer().fit_transform(peak_counts)
	X_pca = TruncatedSVD(n_components).fit_transform(X_tfidf)

	nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
	neighbors = nn.fit(X_pca).kneighbors(X_pca)[1]
	return peak_counts, X_pca, neighbors


def create_pseudobulks(X_index, row_index, output_filename, chroms=None):
	""" do not use or your computer will die."""

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
