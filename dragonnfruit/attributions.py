# attributions.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch

from tqdm import trange
from captum.attr import DeepLiftShap

from bpnetlite.attributions import dinucleotide_shuffle
from bpnetlite.attributions import hypothetical_attributions

class ProfileWrapper(torch.nn.Module):
	"""A wrapper class that returns transformed profiles.

	This class takes in a trained model and returns the weighted softmaxed
	outputs of the first dimension. Specifically, it takes the predicted
	"logits" and takes the dot product between them and the softmaxed versions
	of those logits. This is for convenience when using captum to calculate
	attribution scores.
	
	Parameters
	----------
	model: torch.nn.Module
		A torch model to be wrapped.
	"""

	def __init__(self, model):
		super(ProfileWrapper, self).__init__()
		self.model = model

	def forward(self, X, cell_states):
		logits = self.model(X, cell_states)
		logits = logits - torch.mean(logits, dim=-1, keepdims=True)

		with torch.no_grad():
			l = torch.clone(logits).detach()
			y = torch.exp(l - torch.logsumexp(l, dim=-1, keepdims=True))
			
		return (logits * y).sum(axis=-1, keepdims=True)


def calculate_attributions(model, X, cell_states, n_shuffles=10, batch_size=1, 
	verbose=True):
	wrapper = ProfileWrapper(model.cuda())
	ig = DeepLiftShap(wrapper)

	reference = dinucleotide_shuffle(X, n_shuffles=n_shuffles).cuda()
	reference = reference.type(X.dtype)
	
	X = X.unsqueeze(0).cuda()
	_cell_states = cell_states.cuda()

	attributions = []
	for start in trange(0, len(cell_states), batch_size, disable=not verbose):
		c_ = _cell_states[start:start+batch_size]
		X_ = X.expand(len(c_), -1, -1)
		
		attr = ig.attribute(X_, reference, target=0, 
			additional_forward_args=(c_,), 
			custom_attribution_func=hypothetical_attributions)

		attr = (attr * X_).cpu()
		attributions.append(attr)
	
	attributions = torch.cat(attributions)    
	return attributions

def calculate_attributions_cross(model, X, cell_states, n_shuffles=10, 
	batch_size=1, verbose=True):
	n_loci, n_states = cell_states.shape[0], cell_states.shape[0]
	y_attr = []

	for i in range(n_loci):
		_X = X[i:i+1].expand(n_states, -1, -1)
		_y_attr = calculate_attributions(model, _X, cell_states, 
			n_shuffles=n_shuffles, batch_size=batch_size, verbose=verbose)
		y_attr.append(_y_attr)

	return torch.stack(y_attr).permute(1, 0, 2)
