
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
		l = torch.clone(logits).detach()
		
		y = torch.exp(l - torch.logsumexp(l, dim=-1, keepdims=True))
		return (logits * y).sum(axis=-1, keepdims=True)


def calculate_attributions(model, X, cell_states, n_shuffles=10, batch_size=1):
	wrapper = ProfileWrapper(model.cuda())
	ig = DeepLiftShap(wrapper)

	reference = dinucleotide_shuffle(X, n_shuffles=n_shuffles).cuda()
	reference = reference.type(X.dtype)
	
	X = X.unsqueeze(0).cuda()
	_cell_states = cell_states.cuda()

	attributions = []
	with torch.no_grad():
		for start in trange(0, len(cell_states), batch_size):
			c_ = _cell_states[start:start+batch_size]
			X_ = X.expand(len(c_), -1, -1)
			
			attr = ig.attribute(X_, reference, target=0, 
				additional_forward_args=(c_,), 
				custom_attribution_func=hypothetical_attributions)

			attr = (attr * X_).cpu()
			attributions.append(attr)
	
	attributions = torch.cat(attributions)    
	return attributions