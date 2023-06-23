# models.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import time
import numpy
import torch

from bpnetlite.io import one_hot_encode


def marginalize(model, motif, X, cell_state, spacing=24):
	X = torch.clone(X)
	X_perturb = torch.clone(X)
	
	y_before_profile = model.predict(X, cell_state)

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
			
	y_after_profile = model.predict(X_perturb, cell_state)
	return y_before_profile.cpu(), y_after_profile.cpu()

def marginalize_cross(model, motif, X, cell_states, spacing=24):
	y_befores, y_afters = [], []

	for cell_state in cell_states:
		cell_state = cell_state.expand(X.shape[0], -1)

		y_before, y_after = marginalize(model, motif, X, cell_state, spacing)
		y_befores.append(y_before)
		y_afters.append(y_after)

	return torch.stack(y_befores), torch.stack(y_afters)
