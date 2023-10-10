# models.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import time
import numpy
import torch

from tqdm import tqdm
from bpnetlite.io import one_hot_encode


def marginalize(model, motif, X, cell_states, read_depths=None, spacing=24,
	verbose=False):
	"""Perform in-silico marginalization on background sequences.

	This method will perform in-silico marginalization on the provided
	background sequences given one cell state. This involves first making
	predictions on the original background sequences, inserting the motifs
	into the sequence with the given spacing, and then recalculating these
	predictions given the sequences with the motifs inserted.

	This method takes in a set of background sequences, one cell state, and
	one (set of) motif/s. If you want to marginalize across many cell states, 
	use the `marginalize_cross` method. It will give you the same results but
	with less overhead and code.

	This method works with the full DragoNNFruit model, if read_depths are
	passed in, or with just the accessibility model, if not.


	Parameters
	----------
	model: torch.nn.Module
		Any model with a `predict` class exposed that takes in sequences
		and cell states. Typically a DragoNNFruit model.

	motif: str or list
		A motif or set of motifs to pass in. If string, the motif is added
		to the middle of the sequence. If list, ALL motifs are added to the
		sequence.

	X: torch.Tensor, shape=(-1, 4, 2114)
		A tensor of one-hot encoded sequences to calculate initial
		values for.

	cell_states: torch.tensor, shape=(-1, n_dims)
		The cell states to marginalize over. Dimensions must match those
		expected by the controller model and the number of negative sequences.

	read_depths: torch.tensor, shape=(-1, 1), optional
		If using the full DragoNNFruit model, the read depths to pass in. If
		using only the accessibility model, do not pass in anything. Default is
		None.

	spacing: int, optional
		If providing multiple motifs, specify the spacing.

	verbose: bool, optional
		Whether to print a progress bar while iterating over loci. Default is 
		False.


	Returns
	-------
	y_before: torch.tensor, (-1, -1, 1000)
		Predicted logits for the sequences across cell states before inserting
		the motif.

	y_after: torch.tensor, (-1, -1, 1000)
		Predicted logits for the sequences across cell states after inserting
		the motif.
	"""

	if read_depths is not None:
		y_before = model.predict(X, cell_states, read_depths, 
			verbose=verbose).cpu()
	else:
		y_before = model.predict(X, cell_states, verbose=verbose).cpu()
	
	X_perturb = torch.clone(X)
	alphabet = ['A', 'C', 'G', 'T']

	motif = motif if isinstance(motif, list) else [motif]
	motif_ohes = [torch.from_numpy(one_hot_encode(m, alphabet=alphabet)) 
		for m in motif]
	
	start = X.shape[-1] // 2 - len(motif) // 2
	for motif_ohe in motif_ohes:
		for i in range(len(motif_ohe)):
			if motif_ohe[i].sum() > 0:
				X_perturb[:, :, start+i] = motif_ohe[i]
		
		start += len(motif_ohe) + spacing
	
	if read_depths is not None:
		y_after = model.predict(X_perturb, cell_states, read_depths, 
			verbose=verbose).cpu()
	else:
		y_after = model.predict(X_perturb, cell_states, verbose=verbose).cpu()

	return y_before, y_after


def marginalize_cross(model, motif, X, cell_states, read_depths=None, 
	spacing=24, verbose=False):
	"""Perform in-silico marginalization on background sequences across states.

	This method will perform in-silico marginalization on the provided
	background sequences given many cell states. This involves first making
	predictions on the original background sequences, inserting the motifs
	into the sequence with the given spacing, and then recalculating these
	predictions given the sequences with the motifs inserted.

	This method takes in a set of background sequences, a set of cell states, 
	and one (set of) motif/s. Essentially, this can be thought as running
	the `marginalize` function on each state iteratively.


	Parameters
	----------
	model: torch.nn.Module
		Any model with a `predict` class exposed that takes in sequences
		and cell states. Typically a DragoNNFruit model.

	motif: str or list
		A motif or set of motifs to pass in. If string, the motif is added
		to the middle of the sequence. If list, ALL motifs are added to the
		sequence.

	X: torch.Tensor, shape=(-1, 4, 2114)
		A tensor of one-hot encoded sequences to calculate initial
		values for.

	cell_states: torch.tensor, shape=(-1, n_dims)
		The cell state to marginalize over. Dimensions must match those
		expected by the controller model but does not need to match the
		number of negative sequences.

	read_depths: torch.tensor, shape=(-1, 1), optional
		If using the full DragoNNFruit model, the read depths to pass in. If
		using only the accessibility model, do not pass in anything. Default is
		None.

	spacing: int, optional
		If providing multiple motifs, specify the spacing.

	verbose: bool, optional
		Whether to print a progress bar while iterating over cell states.
		Default is False.


	Returns
	-------
	y_before: torch.tensor, (-1, -1, 1000)
		Predicted logits for the sequences across cell states before inserting
		the motif.

	y_after: torch.tensor, (-1, -1, 1000)
		Predicted logits for the sequences across cell states after inserting
		the motif.
	"""

	y_befores, y_afters = [], []

	for i, cell_state in tqdm(enumerate(cell_states), disable=not verbose):
		cell_state = cell_state.expand(X.shape[0], -1)

		if read_depths is not None:
			read_depth = read_depths[i].expand(X.shape[0], -1)
		else:
			read_depth = None

		y_before, y_after = marginalize(model=model, motif=motif, X=X, 
			cell_states=cell_state, read_depths=read_depth, spacing=spacing)
		y_befores.append(y_before)
		y_afters.append(y_after)

	return torch.stack(y_befores), torch.stack(y_afters)
