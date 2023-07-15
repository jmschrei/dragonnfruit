# models.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import time
import numpy
import torch

from tqdm import tqdm
from bpnetlite.io import one_hot_encode


def marginalize(model, motif, X, cell_states, spacing=24):
	"""Perform in-silico marginalization on background sequences.

	This method will perform in-silico marginalization on the provided
	background sequences given one cell state. This involves first making
	predictions on the original background sequences, inserting the motifs
	into the sequence with the given spacing, and then recalculating these
	predictions given the sequences with the motifs inserted.

	This method takes in a set of background sequences, one cell state, and
	one (set of) motif/s. If you want to marginalize across cell states, use
	the `marginalize_cross` method.


	Parameters
	----------
	model: torch.nn.Module
		Any model with a `predict` class exposed that takes in sequences
		and cell states. Typically a DragoNNFruit model.

	motif: str or list
		A motif or set of motifs to pass in. If string, the motif is added
		to the middle of the sequence. If list, ALL motifs are added to the
		sequence.

	X: torch.Tensor, shape=(1, 4, 2114)
		A tensor of one-hot encoded sequences to calculate initial
		values for.

	cell_state: torch.tensor, shape=(-1, n_dims)
		The cell state to marginalize over. Dimensions must match those
		expected by the controller model and the number of negative sequences.

	spacing: int, optional
		If providing multiple motifs, specify the spacing.


	Returns
	-------
	y_before: torch.tensor, (-1, -1, 1000)
		Predicted logits for the sequences across cell states before inserting
		the motif.

	y_after: torch.tensor, (-1, -1, 1000)
		Predicted logits for the sequences across cell states after inserting
		the motif.
	"""

	n_states = cell_states.shape[0]

	y_before = model.predict(X.expand(n_states, -1, -1), cell_states).cpu()
	X_perturb = torch.clone(X)

	if isinstance(motif, str):
		motif_ohe = one_hot_encode(motif, alphabet=['A', 'C', 'G', 'T'])
		motif_ohe = torch.from_numpy(motif_ohe)

		start = X.shape[-1] // 2 - len(motif) // 2
		for i in range(len(motif)):
			if motif_ohe[i].sum() > 0:
				X_perturb[:, :, start+i] = motif_ohe[i]
				
	elif isinstance(motif, list):
		motif_ohes = [one_hot_encode(m, alphabet=['A', 'C', 'G', 'T']) for m in motif]
		motif_ohes = [torch.from_numpy(m) for m in motif_ohes]
		
		start = X.shape[-1] // 2 - len(motif) // 2
		
		for motif_ohe in motif_ohes:
			for i in range(len(motif_ohe)):
				if motif_ohe[i].sum() > 0:
					X_perturb[:, :, start+i] = motif_ohe[i]
			
			start += len(motif_ohe) + spacing
			
	y_after = model.predict(X_perturb.expand(n_states, -1, -1), cell_states).cpu()
	return y_before, y_after


def marginalize_cross(model, motif, X, cell_states, spacing=24, verbose=False):
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

	for x in tqdm(X, disable=not verbose):
		x = x.unsqueeze(0)

		y_before, y_after = marginalize(model, motif, x, cell_states, spacing)
		y_befores.append(y_before)
		y_afters.append(y_after)

	y_befores = torch.stack(y_befores).permute(1, 0, 2)
	y_afters = torch.stack(y_afters).permute(1, 0, 2)
	return y_befores, y_afters
