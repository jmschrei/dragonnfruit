# attributions.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from tqdm import tqdm
from tqdm import trange

from bpnetlite.attributions import _ProfileLogitScaling
from bpnetlite.attributions import calculate_attributions
from bpnetlite.attributions import create_references
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
		self.scaling = _ProfileLogitScaling()

	def forward(self, X, cell_states):
		logits = self.model(X, cell_states)
		logits = logits - torch.mean(logits, dim=-1, keepdims=True)

		y = self.scaling(logits)
		return y.sum(axis=-1, keepdims=True)


class CountWrapper(torch.nn.Module):
	"""A wrapper class that returns counts from a DragoNNFruit model.

	This class takes in a trained model and returns the logsumexp'd predicted
	logits from a DragoNNFruit model.

	
	Parameters
	----------
	model: torch.nn.Module
		A torch model to be wrapped.
	"""

	def __init__(self, model):
		super(CountWrapper, self).__init__()
		self.model = model

	def forward(self, X, cell_states):
		logits = self.model(X, cell_states)
		return torch.logsumexp(logits, dim=-1, keepdims=True)	


def calculate_attributions_cross(model, X, cell_states, args=None, 
	model_output="profile", attribution_func=hypothetical_attributions, 
	hypothetical=False, algorithm="deepliftshap", references='dinucleotide', 
	n_shuffles=20, batch_size=32, return_references=False, 
	warning_threshold=0.001, print_convergence_deltas=False, verbose=False, 
	random_state=None):
	"""Calculate the attributions crossed with cell states.

	This will calculate the attributions on each sequence for each cell state
	provided. If `args` are provided, it will be crossed with the cell_states
	input.


	Parameters
	----------
	model: torch.nn.Module
		The model to use, either BPNet or one of it's variants.

	X: torch.tensor, shape=(-1, 4, -1)
		A one-hot encoded sequence input to the model.

	args: tuple or None, optional
		Additional arguments to pass into the forward function. If None,
		pass nothing additional in. Default is None.

	model_output: None, "profile" or "count", optional
		If None, then no wrapper is applied to the model. If "profile", wrap 
		the model using ProfileWrapper and calculate attributions with respect 
		to the profile. If "count", wrap the model using CountWrapper and 
		calculate attributions with respect to the count. Default is "profile".

	hypothetical: bool, optional
		Whether to return attributions for all possible characters at each
		position or only for the character that is actually at the sequence.
		Practically, whether to return the returned attributions from captum
		with the one-hot encoded sequence. Default is False.

	algorithm: "deepliftshap" or "ism", optional
		The algorithm to use to calculate attributions. Must be one of
		"deepliftshap", which uses the DeepLiftShap object, or "ism", which
		uses the naive_ism method. Default is "deepliftshap".

	references: "dinucleotide", "freq", "zeros", optional
		The reference to use when algorithm is "deepliftshap". If "dinucleotide"
		generate dinucleotide shuffled sequences. If "freq", set each value to
		0.25. If "zeros", set each value to 0.

	n_shuffles: int, optional
		The number of dinucleotide shuffles to return. Only needed when
		algorithm is "deepliftshap". Default is 10.

	batch_size: int, optional
		The number of attributions to calculate at the same time. This is
		limited by GPU memory. Default is 8.

	return_references: bool, optional
		Whether to return the references that were generated during this
		process.

	warning_threshold: float, optional
		A threshold on the convergence delta that will always raise a warning
		if the delta is larger than it. Normal deltas are in the range of
		1e-6 to 1e-8. Note that convergence deltas are calculated on the
		gradients prior to the attribution_func being applied to them. Default 
		is 0.001. 

	print_convergence_deltas: bool, optional
		Whether to print the convergence deltas for each example when using
		DeepLiftShap. Default is False.

	verbose: bool, optional
		Whether to display a progress bar.

	random_state: int or None or numpy.random.RandomState, optional
		The random seed to use to ensure determinism. If None, the
		process is not deterministic. Default is None. 


	Returns
	-------
	attributions: torch.tensor
		The attributions calculated for each input sequence, with the same
		shape as the input sequences.

	references: torch.tensor, optional
		The references used for each input sequence, with the shape
		(n_input_sequences, n_shuffles, 4, length). Only returned if
		`return_references = True`. 
	"""

	if model_output is None:
		wrapper = model
	elif model_output == "profile":
		wrapper = ProfileWrapper(model)
	elif model_output == "count":
		wrapper = CountWrapper(model)
	else:
		raise ValueError("model_output must be None, 'profile' or 'count'.")

	n_loci, n_states = X.shape[0], cell_states.shape[0]
	refs, attrs = [], []
	for i in range(n_loci):
		_X = X[i:i+1].expand(n_states, -1, -1)
		
		# Calculate references
		if isinstance(references, torch.Tensor):
			_ref = references[i:i+1].to(device)
		else:
			_ref = create_references(X[i:i+1], algorithm=references, 
				n_shuffles=n_shuffles).expand(n_states, -1, -1, -1)

		attr = calculate_attributions(model=wrapper, 
			X=_X, 
			args=(cell_states,) + args if args is not None else (cell_states,), 
			model_output=None, 
			attribution_func=attribution_func,
			hypothetical=hypothetical,
			algorithm=algorithm,
			references=_ref,
			n_shuffles=n_shuffles,
			batch_size=batch_size,
			return_references=return_references,
			warning_threshold=warning_threshold,
			print_convergence_deltas=print_convergence_deltas,
			verbose=verbose,
			random_state=random_state
		)

		if return_references:
			attr, ref = attr
			refs.append(ref)
		attrs.append(attr)


	attrs = torch.stack(attrs).permute(1, 0, 2, 3)
	if return_references:
		refs = torch.stack(refs).permute(1, 0, 2, 3)
		return attrs, refs
	return attrs
