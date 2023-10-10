# models.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import time
import numpy
import torch

from tqdm import trange

from bpnetlite.losses import MNLLLoss
from bpnetlite.losses import log1pMSELoss

from bpnetlite.performance import pearson_corr
from bpnetlite.performance import calculate_performance_measures

from bpnetlite.logging import Logger


@torch.inference_mode()
def trt_predict(model, X, cell_states, read_depths, batch_size=32):
	"""Makes predictions using fixed batch sizes.

	This function takes in a model and makes predictions for an entire data
	set using fixed batch sizes. This is identical to the `.predict` method
	that comes with the methods except that it constructs a final, fixed-size
	batch for the final elements. For instance, if you are using a batch size
	of 32 and there are 40 elements, the second batch here will have size
	32 but only fill in the first 8 elements and then truncate the remaining
	out. This is necessary when using ahead-of-time compilation, such as
	torch2trt, to speed up model inference, because these methods require
	fixed tensor sizes.


	Parameters
	----------
	X: torch.tensor, shape=(-1, 4, 2114)
		A one-hot encoded sequence tensor.

	cell_states: torch.tensor, shape=(-1, n_inputs)
		A tensor representing the state of each cell that is passed into
		the model, e.g. through PCA or LSI.

	read_depths: torch.tensor, shape=(-1, 1)
		A tensor containing the read depth of each cell or aggregated
		across similar cells.

	batch_size: int, optional
		The number of elements to use in each batch. Default is 64.

	dtype: torch datatype
		The data type to c

	Returns
	-------
	y_hat: torch.tensor, shape=(-1, 1000)
		A tensor containing the predicted profiles.
	"""	

	starts = numpy.arange(0, X.shape[0], batch_size)
	ends = starts + batch_size

	y = []

	for start, end in zip(starts, ends):
		if end >= X.shape[0]:
			idx = X.shape[0] - start
			
			_X = torch.zeros(batch_size, X.shape[1], X.shape[2], dtype=X.dtype, 
				device=X.device)
			_X[:idx] = X[start:end]
			
			_cs = torch.zeros(batch_size, cell_states.shape[1], 
				dtype=cell_states.dtype, device=cell_states.device)
			_cs[:idx] = cell_states[start:end]
			
			_rd = torch.zeros(batch_size, read_depths.shape[1], 
				dtype=read_depths.dtype, device=read_depths.device)
			_rd[:idx] = read_depths[start:end]
		else:
			_X = X[start:end]
			_cs = cell_states[start:end]
			_rd = read_depths[start:end]
		
		_y = model(_X, _cs, _rd)
		
		if end >= X.shape[0]:
			idx = X.shape[0] - start
			
			_y = _y[:idx]
		 
		y.append(_y)

	return torch.cat(y)


class DragoNNFruit(torch.nn.Module):
	"""An entire DragoNNFruit model with all components.

	DragoNNFruit is a model for analyzing single cell data sets that include
	an accessibility component. It can be used on scATAC-seq data alone or it
	can be used on multimodal data where one of the components is scATAC-seq.

	Briefly, the model is primarily a ChromBPNet model that uses an external
	factor to modify the parameters of the accessibility model. This external
	factor can be LSI representations from the same scATAC-seq data being
	predicted or it can come from the other modalities, e.g. scRNA-seq, if the
	data is multimodal.

	A DragoNNFruit model is made up of three components: (1) the bias model,
	(2) the dynamic accessibility model, and (3) the dynamic controller. These
	components are defined below, but the first two are based on ChromBPNet.


	Parameters
	----------
	bias: torch.nn.Module 
		This model takes in sequence and outputs the shape one would expect in 
		ATAC-seq data due to Tn5 bias alone. This is usually a BPNet model
		from the bpnet-lite repo that has been trained on GC-matched non-peak
		regions.

	accessibility: torch.nn.Module
		This model takes in sequence and outputs the accessibility one would 
		expect due to the components of the sequence, but also takes in a cell 
		representation which modifies the parameters of the model, hence, 
		"dynamic." This model is usually a DynamicBPNet model, defined below.

	name: str
		The name to prepend when saving the file.
	"""

	def __init__(self, bias, accessibility, name, alpha=1):
		super(DragoNNFruit, self).__init__()

		for parameter in bias.parameters():
			parameter.requires_grad = False

		self.bias = bias
		self.accessibility = accessibility
		self.name = name
		self.alpha = alpha
		self.logger = Logger(["Epoch", "Iteration", "Training Time",
			"Validation Time", "Training MNLL", "Validation MNLL",
			"Validation Profile Correlation", "Validation Count Correlation", 
			"Saved?"], verbose=True)

	def forward(self, X, cell_states, read_depths):
		"""A forward pass through the network.

		This function is usually accessed through calling the model, e.g.
		doing `model(x)`. The method defines how inputs are transformed into
		the outputs through interactions with each of the layers.


		Parameters
		----------
		X: torch.tensor, shape=(-1, 4, 2114)
			A one-hot encoded sequence tensor.

		cell_states: torch.tensor, shape=(-1, n_nodes)
			A tensor representing the state of each cell that is passed into
			the model, e.g. through PCA or LSI.

		read_depths: torch.tensor, shape=(-1, 1)
			A tensor containing the read depth of each cell or aggregated
			across similar cells.


		Returns
		-------
		y_profile: torch.tensor, shape=(-1, 1000)
			The predicted logit profile for each example. Note that this is not
			a normalized value.
		"""

		return read_depths + self.bias(X)[0][:, 0] + self.accessibility(X,
			cell_states)
		
	@torch.no_grad()
	def predict(self, X, cell_states, read_depths, batch_size=16, 
		reduction=None, verbose=False):
		"""A method for making batched predictions.

		This method will take in a large number of cell states and provide
		predictions in a batched manner without storing the gradients. Useful
		for inference step.


		Parameters
		----------
		X: torch.tensor, shape=(-1, 4, 2114)
			A one-hot encoded sequence tensor.

		cell_states: torch.tensor, shape=(-1, n_inputs)
			A tensor representing the state of each cell that is passed into
			the model, e.g. through PCA or LSI.

		read_depths: torch.tensor, shape=(-1, 1)
			A tensor containing the read depth of each cell or aggregated
			across similar cells.

		batch_size: int, optional
			The number of elements to use in each batch. Default is 64.

		reduction: None, 'logsumexp', or 'sum', optional
			Whether to reduce the predictions along the final axis. Useful to
			reduce the memory overhead of making large number of predictions.
			If None, perform no reduction. Default is None.

		verbose: bool, optional
			Whether to display a progress bar across batches. Default is False.


		Returns
		-------
		y_hat: torch.tensor, shape=(-1, 1000) or (-1, 1)
			A tensor containing the predicted profiles.
		"""		

		y_hat = []
		for start in trange(0, len(X), batch_size, disable=not verbose):
			X_batch = X[start:start+batch_size]
			cs_batch = cell_states[start:start+batch_size]
			rd_batch = read_depths[start:start+batch_size]

			y_hat_ = self(X_batch, cs_batch, rd_batch)
			if reduction == 'sum':
				y_hat_ = torch.sum(y_hat_, dim=-1)
			elif reduction == 'logsumexp':
				y_hat_ = torch.logsumexp(y_hat_, dim=-1)

			y_hat.append(y_hat_)

		y_hat = torch.cat(y_hat)
		return y_hat

	@torch.no_grad()
	def predict_cross(self, X, cell_states, read_depths, batch_size=64,
		reduction=None, verbose=False):
		"""A method for making batched predictions on a locus/cell cross.

		This method makes predictions for a cross between loci and cell states.
		Use this method and not `predict` when you want to make predictions
		for all loci for all cell states. Because the cross is constructed
		in batches the amount of memory required is limited and the size
		of the loci and cell states do not need to match in the first
		dimension.


		Parameters
		----------
		X: torch.tensor, shape=(n_loci, 4, 2114)
			A one-hot encoded sequence tensor.

		cell_states: torch.tensor, shape=(n_states, n_inputs)
			A tensor representing the state of each cell that is passed into
			the model, e.g. through PCA or LSI.

		read_depths: torch.tensor, shape=(n_states, 1)
			A tensor containing the read depth of each cell or aggregated
			across similar cells.

		batch_size: int, optional
			The number of elements to use in each batch. Default is 64.

		reduction: None, 'logsumexp', or 'sum', optional
			Whether to reduce the predictions along the final axis. Useful to
			reduce the memory overhead of making large number of predictions.
			If None, perform no reduction. Default is None.

		verbose: bool, optional
			Whether to display a progress bar across states. Default is False.


		Returns
		-------
		y_hat: torch.tensor, shape=(n_loci, n_states, 1000)
			A tensor containing the predicted profiles.
		"""		

		n_states = cell_states.shape[0]
		y_hat = []

		for i in trange(n_states, disable=not verbose):
			_y_hat = self.predict(X, cell_states[i:i+1].expand(X.shape[0], -1), 
				read_depths=read_depths[i:i+1].expand(X.shape[0], -1),
				batch_size=batch_size, reduction=reduction)
			y_hat.append(_y_hat)

		return torch.stack(y_hat)


	def _train_step(self, X, cell_states, read_depths, y, optimizer):
		optimizer.zero_grad()

		with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
			y_hat = self(X, cell_states, read_depths)
			y_hat_ = torch.nn.functional.log_softmax(y_hat.flatten(), dim=-1)
			loss = MNLLLoss(y_hat_, y.flatten())

		loss_ = loss.item()
		loss = loss.backward()

		torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
		optimizer.step()
		return loss_

	def fit(self, training_data, validation_data, optimizer, 
		n_validation_samples=5000, max_epochs=100, batch_size=64, 
		validation_iter=100, verbose=True):
		"""Fit an entire DragoNNFruit model to the data.
		"""

		X_valid, y_valid, c_valid, r_valid = zip(*[validation_data[i]
			for i in range(n_validation_samples)])
		X_valid = torch.stack(X_valid).cuda()
		y_valid = torch.stack(y_valid).type(torch.float32).cuda()
		c_valid = torch.stack(c_valid).cuda()
		r_valid = torch.stack(r_valid).cuda()

		start, best_corr = time.time(), 0
		self.logger.start()
		for epoch in range(max_epochs):
			for i, (X, y, cell_states, read_depths) in enumerate(training_data):
				X = X.cuda()
				y = y.cuda()
				cell_states = cell_states.cuda()
				read_depths = read_depths.cuda()

				train_loss = self._train_step(X, cell_states, read_depths, y,
					optimizer)

				if verbose and i % validation_iter == 0:
					train_time = time.time() - start
					tic = time.time()

					# Make predictions
					y_hat = self.predict(X_valid, c_valid, r_valid, batch_size)
					
					# Calculate MNLL loss
					y_hat_ = torch.nn.functional.log_softmax(y_hat.flatten(), 
						dim=-1)
					valid_loss = MNLLLoss(y_hat_, y_valid.flatten()).item()

					# Calculate count correlation
					y_hat_ = torch.logsumexp(y_hat, dim=-1)
					valid_count_corr = pearson_corr(y_hat_, y_valid.sum(dim=-1))
					valid_count_corr = valid_count_corr.mean().item()

					# Calculate profile correlation
					y_hat = torch.nn.functional.log_softmax(y_hat.flatten(), 
						dim=-1).reshape(*y_hat.shape)
					y_hat = torch.exp(y_hat)
					valid_profile_corr = pearson_corr(y_hat, 
						y_valid).mean().item()

					valid_time = time.time() - tic

					self.logger.add([epoch, i, train_time, valid_time, 
						train_loss, valid_loss, valid_profile_corr, 
						valid_count_corr, valid_profile_corr > best_corr])

					self.logger.save("{}.log".format(self.name))

					if valid_profile_corr > best_corr:
						best_corr = valid_profile_corr
						torch.save(self, "{}.best.torch".format(self.name))

					start = time.time()
					torch.save(self, "{}.{}.torch".format(self.name, epoch))


class DynamicBPNet(torch.nn.Module):
	"""A BPNet model whose weights are modified by an external factor.

	This model has the same architecture as the original BPNet model except
	that parameters can be controlled by some external factor. Specifically,
	in this model, the bias terms are controlled using the external factor,
	and that external factor is the cell state.

	Because the intended application of this is single cell data, the final
	predictions are modified using the read depth.
	

	Parameters
	----------
	controller: torch.nn.Module
		This model takes in the cell representation and outputs a 
		representation that will be fed into the dynamic accessibility model.
		This model is usually a CellStateController model, as defined below.

	n_filters: int, optional
		The number of filters to use in each convolution layer. Default is 128.

	n_layers: int, optional
		The number of residual dilated convolutions sandwiched between the
		first and the last layer. Does not include the size of the first
		and the last layer. Default is 8.

	trimming: int or None, optional
		The amount of trimming to apply to input sequences to go from the size
		of the input to the size of the output predictions. If none, will
		automatically trim to 2 ** n_layers + 37. Default is None.

	conv_bias: bool, optional
		Whether the convolutions should have learned bias terms in addition
		to those learned from the cell state repressentation. Mostly only
		useful when transferring weights from a pre-trained model.
	"""

	def __init__(self, controller, n_filters=128, n_layers=8, trimming=None, 
		conv_bias=False):
		super(DynamicBPNet, self).__init__()

		self.trimming = trimming or 2 ** n_layers + 37
		self.n_filters = n_filters
		self.n_layers = n_layers


		self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=21, padding=10)
		self.irelu = torch.nn.ReLU()
		self.fconv = torch.nn.Conv1d(n_filters, 1, kernel_size=75, padding=37,
			bias=conv_bias)

		self.biases = torch.nn.ModuleList([
			torch.nn.Linear(controller.n_outputs, n_filters) for i in range(
				n_layers)
		])

		self.convs = torch.nn.ModuleList([
			torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, stride=1, 
				dilation=2**i, padding=2**i, bias=conv_bias) for i in range(1, 
					n_layers+1)
		])

		self.relus = torch.nn.ModuleList([
			torch.nn.ReLU() for i in range(n_layers)
		])

		self.controller = controller

	def forward(self, X, cell_states):
		"""A forward pass through the network.

		This function is usually accessed through calling the model, e.g.
		doing `model(x)`. The method defines how inputs are transformed into
		the outputs through interactions with each of the layers.


		Parameters
		----------
		X: torch.tensor, shape=(-1, 4, 2114)
			A one-hot encoded sequence tensor.

		cell_states: torch.tensor, shape=(-1, n_nodes)
			A tensor representing the state of each cell that is passed into
			the model, e.g. through PCA or LSI.


		Returns
		-------
		y_profile: torch.tensor, shape=(-1, 1000)
			The predicted logit profile for each example. Note that this is not
			a normalized value.
		"""

		start, end = self.trimming, X.shape[2] - self.trimming
		
		cell_states = self.controller(cell_states)
		X = self.irelu(self.iconv(X))

		for i in range(self.n_layers):
			X_conv = self.convs[i](X)
			X_bias = self.biases[i](cell_states).unsqueeze(-1)
			
			X = X + self.relus[i](X_conv + X_bias)

		y_profile = self.fconv(X)[:, 0, start:end]
		return y_profile

	@torch.no_grad()
	def predict(self, X, cell_states, batch_size=64, reduction=None, 
		verbose=False):
		"""A method for making batched predictions.

		This method makes predictions for a paired set of loci and cell states.
		Each locus and cell state are considered individually and so the number
		of loci must match the number of cell states even if many predictions
		are being made for the same cell state or locus. In that case, one
		must pass copies of the locus or cell state to ensure that the shapes
		match.


		Parameters
		----------
		X: torch.tensor, shape=(-1, 4, 2114)
			A one-hot encoded sequence tensor.

		cell_states: torch.tensor, shape=(-1, n_inputs)
			A tensor representing the state of each cell that is passed into
			the model, e.g. through PCA or LSI.

		batch_size: int, optional
			The number of elements to use in each batch. Default is 64.

		reduction: None, 'logsumexp', or 'sum', optional
			Whether to reduce the predictions along the final axis. Useful to
			reduce the memory overhead of making large number of predictions.
			If None, perform no reduction. Default is None.

		verbose: bool, optional
			Whether to display a progress bar across batches. Default is False.


		Returns
		-------
		y_hat: torch.tensor, shape=(-1, 1000)
			A tensor containing the predicted profiles.
		"""		

		y_hat = []
		for start in trange(0, len(X), batch_size, disable=not verbose):
			X_batch = X[start:start+batch_size]
			cs_batch = cell_states[start:start+batch_size]

			y_hat_ = self(X_batch, cs_batch)
			if reduction == 'sum':
				y_hat_ = torch.sum(y_hat_, dim=-1)
			elif reduction == 'logsumexp':
				y_hat_ = torch.logsumexp(y_hat_, dim=-1)

			y_hat.append(y_hat_)

		y_hat = torch.cat(y_hat)
		return y_hat

	@torch.no_grad()
	def predict_cross(self, X, cell_states, batch_size=64, reduction=None,
		verbose=False):
		"""A method for making batched predictions on a locus/cell cross.

		This method makes predictions for a cross between loci and cell states.
		Use this method and not `predict` when you want to make predictions
		for all loci for all cell states. Because the cross is constructed
		in batches the amount of memory required is limited and the size
		of the loci and cell states do not need to match in the first
		dimension.


		Parameters
		----------
		X: torch.tensor, shape=(n_loci, 4, 2114)
			A one-hot encoded sequence tensor.

		cell_states: torch.tensor, shape=(n_states, n_inputs)
			A tensor representing the state of each cell that is passed into
			the model, e.g. through PCA or LSI.

		batch_size: int, optional
			The number of elements to use in each batch. Default is 64.

		reduction: None, 'logsumexp', or 'sum', optional
			Whether to reduce the predictions along the final axis. Useful to
			reduce the memory overhead of making large number of predictions.
			If None, perform no reduction. Default is None.

		verbose: bool, optional
			Whether to display a progress bar across states. Default is False.


		Returns
		-------
		y_hat: torch.tensor, shape=(n_loci, n_states, 1000)
			A tensor containing the predicted profiles.
		"""		

		n_states = cell_states.shape[0]
		y_hat = []

		for i in trange(n_states, disable=not verbose):
			_y_hat = self.predict(X, cell_states[i:i+1].expand(X.shape[0], -1), 
				batch_size=batch_size, reduction=reduction)
			y_hat.append(_y_hat)

		return torch.stack(y_hat)

	def load_conv_weights(self, filename, freeze=False):
		"""Load convolution weights from a pretrained ChromBPNet model.

		This function will load the convolution weights from a pre-trained
		model. Specifically, it will load the convolutions from the 
		`accessibility` attribute of the model. Must have the same number
		of layers and filter sizes.


		Parameters
		----------
		filename: str
			The filename of the model to load.

		freeze: bool, optional
			Whether to freeze the layers whose weights are loaded.
		"""

		device = next(self.parameters()).device
		model = torch.load(filename, map_location='cpu').to(device)
		model = model.accessibility

		self.iconv.weight = torch.nn.Parameter(model.iconv.weight)
		self.iconv.weight.requires_grad = not freeze

		self.iconv.bias = torch.nn.Parameter(model.iconv.bias)
		self.iconv.bias.requires_grad = not freeze

		for sconv, mconv in zip(self.convs, model.rconvs):
			sconv.weight = torch.nn.Parameter(mconv.weight)
			sconv.weight.requires_grad = not freeze

			sconv.bias = torch.nn.Parameter(mconv.bias)
			sconv.bias.requires_grad = not freeze

		self.fconv.weight = torch.nn.Parameter(model.fconv.weight)
		self.fconv.weight.requires_grad = not freeze


class CellStateController(torch.nn.Module):
	"""A controller of DynamicBPNet parameters using cell state.

	This component of the DragoNNFruit model concerns taking in cell states
	and transforming them somehow. It is passed into the DynamicBPNet model
	and provides the representations that are used to control the parameters
	of the model. 


	Parameters
	----------
	n_inputs: int
		The number of dimensions in the input cell representation.

	n_nodes: int, optional
		The number of nodes in the internal neural network layers. Default
		is 256.

	n_outputs: int, optional
		The number of dimensions in the output cell representation. Default
		is 64.

	n_layers: int, optional
		The total number of internal layers to apply to the representations,
		not including the first layer that transforms the representations from
		input representations to the internal representations or the last
		layer that transforms from the internal representations to the output
		representations. Default is 0.
	"""

	def __init__(self, n_inputs, n_nodes=256, n_outputs=64, n_layers=0):
		super(CellStateController, self).__init__()

		self.n_inputs = n_inputs
		self.n_nodes = n_nodes
		self.n_outputs = n_outputs
		self.n_layers = n_layers

		self.ifc = torch.nn.Linear(n_inputs, n_nodes)
		self.irelu = torch.nn.ReLU()

		self.fcs = torch.nn.ModuleList([
			torch.nn.Linear(n_nodes, n_nodes) for i in range(n_layers)
		])
		self.relus = torch.nn.ModuleList([
			torch.nn.ReLU() for i in range(n_layers)
		])

		self.ffc = torch.nn.Linear(n_nodes, n_outputs)

	def forward(self, cell_states):
		"""A forward pass through the network.

		This function is usually accessed through calling the model, e.g.
		doing `model(x)`. The method defines how inputs are transformed into
		the outputs through interactions with each of the layers.


		Parameters
		----------
		cell_states: torch.tensor, shape=(-1, n_inputs)
			A tensor representing the state of each cell that is passed into
			the model, e.g. through PCA or LSI.


		Returns
		-------
		cell_states: torch.tensor, shape=(-1, n_outputs)
			A tensor representing the transformed state of each cell.
		"""

		cell_states = self.irelu(self.ifc(cell_states))
		for i in range(self.n_layers):
			cell_states = self.relus[i](self.fcs[i](cell_states))

		cell_states = self.ffc(cell_states)
		return cell_states


	@torch.no_grad()
	def predict(self, cell_states, batch_size=64):
		"""A method for making batched predictions.

		This method will take in a large number of cell states and provide
		predictions in a batched manner without storing the gradients. Useful
		for inference step.


		Parameters
		----------
		cell_states: torch.tensor, shape=(-1, n_inputs)
			A tensor representing the state of each cell that is passed into
			the model, e.g. through PCA or LSI.

		batch_size: int, optional
			The number of elements to use in each batch. Default is 64.


		Returns
		-------
		cell_states: torch.tensor, shape=(-1, n_outputs)
			A tensor representing the transformed state of each cell.
		"""		

		y_hat = []
		for start in range(0, len(cell_states), batch_size):
			cs_batch = cell_states[start:start+batch_size]

			y_hat_ = self(cs_batch)
			y_hat.append(y_hat_)

		y_hat = torch.cat(y_hat)
		return y_hat
