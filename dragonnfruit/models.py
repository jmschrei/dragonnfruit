# dragonnfruit.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import time
import torch

from bpnetlite.losses import MNLLLoss
from bpnetlite.performance import pearson_corr

from .logging import Logger

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

	def __init__(self, bias, accessibility, name):
		super(DragoNNFruit, self).__init__()
		for parameter in bias.parameters():
			parameter.requires_grad = False

		self.bias = bias
		self.accessibility = accessibility
		self.name = name
		self.logger = Logger(["Epoch", "Iteration", "Training Time",
			"Validation Time", "Training MNLL", "Validation MNLL",
			"Validation Correlation"], verbose=True)

		self.read_depths = torch.nn.Linear(1, 1, bias=False)

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


		y_acc = self.accessibility(X, cell_states)
		y_bias = self.bias(X)[0][:,0]
		y_bias = y_bias - torch.mean(y_bias, dim=-1, keepdims=True)
		return y_acc + y_bias + self.read_depths(read_depths)

	@torch.no_grad()
	def predict(self, X, cell_states, read_depths, batch_size=16):
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


		Returns
		-------
		y_hat: torch.tensor, shape=(-1, 1000)
			A tensor containing the predicted profiles.
		"""		

		y_hat = []
		for start in range(0, len(X), batch_size):
			X_batch = X[start:start+batch_size]
			cs_batch = cell_states[start:start+batch_size]
			rd_batch = read_depths[start:start+batch_size]

			y_hat_ = self(X_batch, cs_batch, rd_batch)
			y_hat.append(y_hat_)

		y_hat = torch.cat(y_hat)
		return y_hat

	@torch.no_grad()
	def _evaluate(self, X, cell_states, read_depths, y, y_bias, batch_size):
		y_acc = self.accessibility.predict(X, cell_states, batch_size)
		read_depths = self.read_depths(read_depths)

		y_hat = y_acc + y_bias + read_depths
		y_hat_ = torch.nn.functional.log_softmax(y_hat.flatten(), dim=-1)

		loss = MNLLLoss(y_hat_, y.flatten()).item()

		y_hat = torch.nn.functional.log_softmax(y_hat, dim=-1)
		y_hat = torch.exp(y_hat)

		corr = pearson_corr(y_hat, y).mean().item()
		return loss, corr

	def _train_step(self, X, cell_states, read_depths, y, optimizer):
		optimizer.zero_grad()

		y_hat = self(X, cell_states, read_depths)
		y_hat = torch.nn.functional.log_softmax(y_hat.flatten(), dim=-1)

		loss = MNLLLoss(y_hat, y.flatten())
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
		X_valid = torch.stack(X_valid).cuda(non_blocking=True)
		y_valid = torch.stack(y_valid).cuda(non_blocking=True)
		c_valid = torch.stack(c_valid).cuda(non_blocking=True)
		r_valid = torch.stack(r_valid).cuda(non_blocking=True)

		y_bias_valid = self.bias.predict(X_valid)[0][:, 0].cuda()
		y_bias_valid -= y_bias_valid.mean(dim=1, keepdims=True)

		start, best_corr = time.time(), 0
		self.logger.start()
		for epoch in range(max_epochs):
			for i, (X, y, cell_states, read_depths) in enumerate(training_data):
				X = X.cuda(non_blocking=True).squeeze()
				y = y.cuda(non_blocking=True).squeeze()
				cell_states = cell_states.cuda(non_blocking=True)
				read_depths = read_depths.cuda(non_blocking=True)

				train_loss = self._train_step(X, cell_states, read_depths, y,
					optimizer)

				if verbose and i % validation_iter == 0:
					train_time = time.time() - start
					tic = time.time()

					valid_loss, valid_corr = self._evaluate(X_valid, c_valid, 
						r_valid, y_valid, y_bias_valid, batch_size)

					valid_time = time.time() - tic
					self.logger.add([epoch, i, train_time, valid_time, 
						train_loss, valid_loss, valid_corr])

					if valid_corr > best_corr:
						best_corr = valid_corr
						self.save("best")

					start = time.time()

			self.save(epoch)
			self.logger.save("{}.log".format(self.name))

	def save(self, suffix=""):
		name = "{}.{}.dragonnfruit.torch".format(self.name, suffix)
		dynamic_name = "{}.{}.dynamic.torch".format(self.name, suffix)
		controller_name = "{}.{}.controller.torch".format(self.name, suffix)

		torch.save(self, name)
		torch.save(self.accessibility, dynamic_name)
		torch.save(self.accessibility.controller, controller_name)


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
	n_filters: int, optional
		The number of filters to use in each convolution layer. Default is 128.

	n_layers: int, optional
		The number of residual dilated convolutions sandwiched between the
		first and the last layer. Does not include the size of the first
		and the last layer. Default is 8.

	n_nodes: int, optional
		The dimensionality of the cell state being passed in. Default is 64.

	trimming: int or None, optional
		The amount of trimming to apply to input sequences to go from the size
		of the input to the size of the output predictions. If none, will
		automatically trim to 2 ** n_layers + 37. Default is None.

	controller: torch.nn.Module
		This model takes in the cell representation and outputs a 
		representation that will be fed into the dynamic accessibility model.
		This model is usually a CellStateController model, as defined below.
	"""

	def __init__(self, n_filters=128, n_layers=8, n_nodes=64, trimming=None,
		controller=None):
		super(DynamicBPNet, self).__init__()
		self.trimming = trimming or 2 ** n_layers + 37
		self.n_filters = n_filters
		self.n_layers = n_layers
		self.n_nodes = n_nodes

		self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=21, padding=10)
		self.fconv = torch.nn.Conv1d(n_filters, 1, kernel_size=75, bias=False)

		self.convs = []
		self.biases = []

		for i in range(n_layers):
			bias = torch.nn.Linear(n_nodes, n_filters)
			self.biases.append(bias)

			conv = torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, 
				stride=1, dilation=2**(i+1), padding=2**(i+1), bias=False)
			self.convs.append(conv)

		self.convs = torch.nn.ModuleList(self.convs)
		self.biases = torch.nn.ModuleList(self.biases)
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

		X = torch.nn.ReLU()(self.iconv(X))
		for i in range(self.n_layers):
			#weights = self.convs[i](cell_states_)
			#weights = weights.view(self.n_filters*X.shape[0], self.n_filters, 3)

			X_conv = self.convs[i](X)
			X_bias = self.biases[i](cell_states).unsqueeze(-1)

			#X_conv = X.view(1, -1, X.shape[2])
			#X_conv = torch.nn.functional.conv1d(X_conv, weights,
			#	stride=1, padding=2**(i+1), dilation=2**(i+1), 
			#	groups=X.shape[0])
			#X_conv = X_conv.view(*X.shape)

			X = X + torch.nn.ReLU()(X_conv + X_bias)

		y_profile = self.fconv(X).squeeze()
		y_profile = y_profile[:, start:end]
		return y_profile

	@torch.no_grad()
	def predict(self, X, cell_states, batch_size=64):
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

		batch_size: int, optional
			The number of elements to use in each batch. Default is 64.


		Returns
		-------
		y_hat: torch.tensor, shape=(-1, 1000)
			A tensor containing the predicted profiles.
		"""		

		y_hat = []
		for start in range(0, len(X), batch_size):
			X_batch = X[start:start+batch_size]
			cs_batch = cell_states[start:start+batch_size]

			y_hat_ = self(X_batch, cs_batch)
			y_hat.append(y_hat_)

		y_hat = torch.cat(y_hat)
		return y_hat


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

		fcs = [torch.nn.Linear(n_inputs, n_nodes)]
		for i in range(self.n_layers):
			fc = torch.nn.Linear(n_nodes, n_nodes)
			fcs.append(fc)

		fcs.append(torch.nn.Linear(n_nodes, n_outputs))
		self.fcs = torch.nn.ModuleList(fcs)

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

		for layer in self.fcs:
			cell_states = torch.nn.ReLU()(layer(cell_states))

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
		for start in range(0, len(X), batch_size):
			cs_batch = cell_states[start:start+batch_size]

			y_hat_ = self(cs_batch)
			y_hat.append(y_hat_)

		y_hat = torch.cat(y_hat)
		return y_hat
