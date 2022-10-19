
# attributions.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
import torch

from tqdm import trange
from captum.attr import DeepLiftShap

from bpnetlite.attributions import dinucleotide_shuffle

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

		y = torch.nn.functional.log_softmax(logits, dim=-1)
		y = torch.exp(y).detach()
		return (logits * y).sum(axis=-1).unsqueeze(-1)


def calculate_attributions(model, X, cell_states, n_shuffles=10, batch_size=1):
	wrapper = ProfileWrapper(model.cuda())
	ig = DeepLiftShap(wrapper)

	reference = dinucleotide_shuffle(X, n_shuffles=n_shuffles).cuda()
	X = X.unsqueeze(0).cuda(non_blocking=True)

	attributions = []
	with torch.no_grad():
		for start in trange(0, len(cell_states), batch_size):
			c_ = cell_states[start:start+batch_size].cuda(non_blocking=True)
			X_ = X.expand(len(c_), -1, -1)
			
			attr = ig.attribute(X_, reference, target=0, 
				additional_forward_args=(c_,))
			attr = (attr * X_).cpu()
			attributions.append(attr)
	
	attributions = torch.cat(attributions)    
	return attributions