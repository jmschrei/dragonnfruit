# dragonnfruit.py
# Author: Jacob Schreiber <jmschreiber91@gmail.com>

import torch

from bpnetlite.losses import MNLLLoss

class DragoNNFruit(torch.nn.Module):
	def __init__(self, bias, accessibility, name):
		bias.eval()

		self.bias = bias
		self.accessibility = accessibility
		self.name = name

	def forward(self, X, cell_state, read_depth):
		y_bias = self.bias(X)[0][:,0]
		y_bias = y_bias - torch.mean(y_bias, dim=-1, keepdims=True)
		return self.accessibility(X, cell_state, read_depth) + y_bias

	def predict(self, X, cell_state, read_depth):
		y_hat = []
		for start in range(0, len(X), batch_size):
			X_batch = X[start:start+batch_size]
			cs_batch = cell_states[start:start+batch_size]
			rd_batch = read_depths[start:start+batch_size]

			y_hat_ = self(X_batch, cs_batch, rd_batch)
			y_hat.append(y_hat_)

		y_hat = torch.cat(y_hat)
		return y_hat

	def fit_generator(self, training_data, optimizer, X_valid=None, 
		y_valid=None, y_bias_valid=None, cell_states_valid=None, 
		read_depths_valid=None, max_epochs=100, batch_size=64, 
		validation_iter=100, verbose=True):
		
		if verbose:
			print("Epoch\tIteration\tTraining Time\tValidation Time\t"
				"Training MLL\tValidation MLLL\tValidation Correlation")

		start = time.time()
		best_corr = 0

		for epoch in range(max_epochs):
			for i, (X, y, cell_state, read_depth) in enumerate(training_data): 
				X = X.cuda(non_blocking=True).squeeze()
				y = y.cuda(non_blocking=True).squeeze()
				cell_state = cell_state.cuda(non_blocking=True)
				read_depth = read_depth.cuda(non_blocking=True)

				y_bias = self.bias(X)[0][:,0]
				y_bias = y_bias - torch.mean(y_bias, dim=-1, keepdims=True)

				optimizer.zero_grad()
				self.accessibility.train()

				y_hat = self.accessibility(X, cell_state, read_depth) + y_bias
				y_hat = torch.nn.functional.log_softmax(y_hat, dim=0)

				loss = MNLLLoss(y_hat.flatten(), y.flatten())
				train_loss = loss.item()
				loss = loss.backward()

				torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
				optimizer.step()

				with torch.no_grad():
					if verbose and i % validation_iter == 0:
						self.accessibility.eval()

						train_time = time.time() - start
						tic = time.time()

						y_hat = self.predict(X_valid, cell_states_valid, 
							read_depths_valid, batch_size=batch_size)
						y_hat += y_bias_valid
						
						y_hat_ = torch.nn.functional.log_softmax(y_hat.flatten(), dim=-1)
						y_hat_ = y_hat.flatten()

						y_valid_ = y_valid.flatten()

						valid_loss = MNLLLoss(y_hat_, y_valid_).item()

						y_hat = torch.nn.functional.log_softmax(y_hat, dim=-1)
						y_hat = torch.exp(y_hat)

						valid_corrs = pearson_corr(y_hat, y_valid).mean()
						valid_time = time.time() - tic

						print("{}\t{}\t{:4.4}\t{:4.4}\t{:6.6}\t{:6.6}\t{:4.4}".format(
							epoch, i, train_time, valid_time, train_loss, valid_loss, 
							valid_corrs))

						start = time.time()

						if valid_corrs > best_corr:
							best_corr = valid_corrs
							torch.save(self, "{}.best.torch".format(self.name))

			torch.save(self, "{}.{}.torch".format(self.name, epoch))


class DynamicBPNet(torch.nn.Module):
	def __init__(self, n_filters=128, n_nodes=256, n_nodes2=64, n_layers=4, 
		trimming=None):
		super(DynamicBPNet, self).__init__()
		self.trimming = trimming or 2 ** n_layers + 37
		self.n_filters = n_filters
		self.n_layers = n_layers

		self.fc1 = torch.nn.Linear(50, n_nodes)
		self.fc2 = torch.nn.Linear(n_nodes, n_nodes)

		self.iconv = torch.nn.Conv1d(4, n_filters, kernel_size=21, padding=10)
		self.fconv = torch.nn.Conv1d(n_filters, 1, kernel_size=75, bias=False)

		self.convs = []
		self.biases = []
		#self.fcs = []
		for i in range(n_layers):
			k = n_filters*n_filters*3

			#conv = torch.nn.Linear(n_nodes2, k)
			conv = torch.nn.Conv1d(n_filters, n_filters, kernel_size=3, 
				stride=1, dilation=2**(i+1), padding=2**(i+1), bias=False)
			self.convs.append(conv)

			bias = torch.nn.Linear(n_nodes, n_filters)
			self.biases.append(bias)

			#fc = torch.nn.Linear(n_nodes, n_nodes2)
			#self.fcs.append(fc)

		self.convs = torch.nn.ModuleList(self.convs)
		self.biases = torch.nn.ModuleList(self.biases)
		#self.fcs = torch.nn.ModuleList(self.fcs)

		self.read_depth = torch.nn.Linear(1, 1, bias=False)

	def forward(self, X, cell_states, read_depths):
		start, end = self.trimming, X.shape[2] - self.trimming
		
		cell_states = torch.nn.ReLU()(self.fc1(cell_states))
		cell_states = torch.nn.ReLU()(self.fc2(cell_states))

		X = torch.nn.ReLU()(self.iconv(X))
		for i in range(self.n_layers):
			#cell_states_ = torch.nn.ReLU()(self.fcs[i](cell_states))

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

		y_profile = self.fconv(X).squeeze() + self.read_depth(read_depths)
		y_profile = y_profile[:, start:end]
		return y_profile

	@torch.no_grad()
	def predict(self, X, cell_states, read_depths, batch_size=64):
		y_hat = []
		for start in range(0, len(X), batch_size):
			X_batch = X[start:start+batch_size]
			cs_batch = cell_states[start:start+batch_size]
			rd_batch = read_depths[start:start+batch_size]

			y_hat_ = self(X_batch, cs_batch, rd_batch)
			y_hat.append(y_hat_)

		y_hat = torch.cat(y_hat)
		return y_hat
