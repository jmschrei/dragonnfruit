![image](https://github.com/jmschrei/dragonnfruit/assets/3916816/10f835ca-8feb-43fd-84d8-630d8018379c)

> **Note**
> IMPORTANT: This repository and it's associated documentation are under active development. Features may change. We are aiming to have a preprint out by the end of <s>August</s> <s>November 2023</s> (I thought I could get this out before going on the job market this cycle. It didn't end up working out. I'm aiming for it to come out by July, as travel and preparation are taking up all my time. Sorry for the delay). 

DragoNNFruit is a method for dissecting the cis- and trans-regulatory factors underlying chromatin accessibility at single-cell and base-pair resolution. At a high level, DragoNNFruit models cis-regulatory sequence using a convolutional neural network whose parameters are dynamically generated from a second network that models trans-regulatory state (from ATAC-seq, RNA-seq, spatial coordinates, etc). Through the inclusion of an explicit model of Tn5 bias, DragoNNFruit’s de-noised predictions reveal TF footprints that cannot be observed from the original, biased, data. Taken together, DragoNNFruit’s capabilities enable the identification of cell type-specific motifs and their higher-order syntax, the prediction of variant effect and footprinting genome-wide, and the tracking of all these across entire single-cell experiments without the need for manual cell clustering and annotation.

By explicitly modeling both cis- and trans-regulatory factors, DragoNNFruit departs from current regulatory modeling approaches such as Enformer, scBasset, and ChromVAR. Neither Enformer nor scBasset explicitly model trans-regulatory factors and so cannot generalize past observed cell states. Further, neither operate at base-pair resolution, limiting their interpretability and predictive power. ChromVAR explicitly models cell state by tracking global TF motif activity across cells, but does not model sequence and so cannot reveal how local syntax affects TF binding at individual loci.

### Installation

`pip install dragonnfruit`

### Usage

#### Training a DragoNNFruit Model

The most flexible way of using DragoNNFruit is through the Python API. Here, a model can be created, trained, used to make predictions, or used to calculate attributions across loci and cell states. 

```python
from dragonnfruit.io import LocusGenerator
from dragonnfruit.io import GenomewideGenerator

from dragonnfruit.models import CellStateController
from dragonnfruit.models import DynamicBPNet
from dragonnfruit.models import DragoNNFruit

import torch

neighbors = numpy.load("neighbors.npy")
cell_states = numpy.load("cell_states.npy")
cell_states = (cell_states - cell_states.mean(axis=0, keepdims=True)) / cell_states.std(axis=0, keepdims=True)

read_depths = numpy.load("read_depths.npy")
read_depths = read_depths[neighbors].sum(axis=1)
read_depths = numpy.log2(read_depths + 1).reshape(-1, 1)

signal, sequence = {}, {}
for chrom in ['chr1', 'chr2']:
	signal[chrom] = scipy.sparse.load_npz("y.{}.npz".format(chrom))
  sequence[chrom] = numpy.load("X.chrom.npy".format(chrom))


X = torch.utils.data.DataLoader(
	GenomewideGenerator(
		sequence=sequence,
		signal=signal,
		neighbors=neighbors,
		cell_states=cell_states,
		read_depths=read_depths,
		trimming=(2114 - 1000) // 2, 
		window=1000, 
		chroms=['chr1'],
		random_state=None),
	pin_memory=True, 
	num_workers=8,
	worker_init_fn=lambda x: numpy.random.seed(x),
	batch_size=128)

X_valid = LocusGenerator(
	sequence=sequence,
	signal=signal,
	loci_file=peak_file,
	neighbors=neighbors,
	cell_states=cell_states,
	read_depths=read_depths,
	trimming=(2114 - 1000) // 2, 
	window=1000,
	chroms=['chr2'],
	random_state=0)

bias_model = torch.load("bias.torch").cuda()

controller = CellStateController(n_inputs=cell_states.shape[-1], n_nodes=1024, 
	n_layers=1, n_outputs=128).cuda()

accessibility_model = DynamicBPNet(n_filters=256, n_layers=8,
	trimming=(2114 - 1000) // 2, controller=controller).cuda()

model = DragoNNFruit(bias_model, accessibility_model, "dragonnfruit").cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

model.fit(X, X_valid, optimizer, n_validation_samples=50324, max_epochs=100, 
	validation_iter=250, batch_size=batch_size)
```

The `fit` method will handle the training of a DragoNNFruit model including the cell state controller and the dynamic BPNet (aka accessibility model) components. During training, a log will be returned with the following form: 

```
Epoch   Iteration       Training Time   Validation Time Training MNLL   Validation MNLL Validation Profile Correlation  Validation Count Correlation    Saved?
0       0       12.519434690475464      12.455398082733154      49810.28125     4573590.0       0.21076039969921112     0.28073418140411377     True
0       250     111.76781487464905      6.667823791503906       41393.359375    4248712.0       0.21249620616436005     0.47288715839385986     True
0       500     112.2810971736908       6.673614501953125       39414.453125    4503676.0       0.2115216702222824      0.47341281175613403     False
0       750     112.34485268592834      6.668895483016968       39264.953125    4170154.0       0.21469004452228546     0.4784104824066162      True
0       1000    112.16604423522949      6.6715662479400635      39661.1484375   4330604.0       0.2135431468486786      0.5013536810874939      False
0       1250    112.28104138374329      6.667646646499634       41895.6015625   4147008.0       0.21612848341464996     0.4790436029434204      True
0       1500    112.10946655273438      6.663406610488892       38563.546875    4202624.0       0.21725130081176758     0.4884423613548279      True
```

When the model achieves a new best in terms of validation profile correlation, the model is saved with the suffix `.best`. i.e., `dragonnfruit.best.torch`. Further, the model is saved at each tick with the epoch number, i.e., `dragonnfruit.0.best`. 

#### Making predictions with a DragoNNFruit model

After training a model we can then make predictions across loci and cell states. The central method is `.predict`, which takes in a paired set of loci, cell states, and read depths.

```python
X = <torch.tensor with shape (n_examples, 4, 2114)>
cell_states = <torch.tensor with shape (n_examples, 50)>
read_depths = <torch.tensor with shape (n_examples, 1)>

y_hat = model.predict(X, cell_states, read_depths)
```

An important note is that the predictions are unnormalized basepair resolution logits. This method will shuttle batches of data to the GPU and predictions back to the GPU so you can make predictions on a number of examples that do not fit in GPU memory. If all you want is count predictions for each example, you can reduce each example using `logsumexp`.

```python
X = <torch.tensor with shape (n_examples, 4, 2114)>
cell_states = <torch.tensor with shape (n_examples, 50)>
read_depths = <torch.tensor with shape (n_examples, 1)>

y_hat = model.predict(X, cell_states, read_depths, reduction='logsumexp')
```

However, because the loci and the cell states must be paired you will probably be copying one of them repeatedly if you need to analyze one locus across all cell states or all loci at a given cell state. If you want to automatically make predictions for all cell states in all loci and only provide unpaired sets, you can use `predict_cross`. Below, `n_loci` and `n_states` do not have to be the same size.

```python
X = <torch.tensor with shape (n_loci, 4, 2114)>
cell_states = <torch.tensor with shape (n_states, 50)>
read_depths = <torch.tensor with shape (n_states, 1)>

y_hat = model.predict(X, cell_states, read_depths, reduction='logsumexp')
```

### Calculating attributions with DragoNNFruit

You can calculate the DeepLIFT/DeepSHAP attributions for DragoNNFruit using the `calculate_attributions` method to analyze a single locus across many cell states.

```python
X = <torch.tensor with shape (4, 2114)>
cell_states = <torch.tensor with shape (n_states, 50)

attributions = calculate_attributions(model, X, cell_states, n_shuffles=10, batch_size=8)
```

Similar to prediction, if you'd like to explain several loci across several cell states, you can use the `calculate_attribution_cross` method.

```python
X = <torch.tensor with shape (n_loci, 4, 2114)>
cell_states = <torch.tensor with shape (n_states, 50)

attributions = calculate_attributions_cross(model, X, cell_states, n_shuffles=10, batch_size=8)
```
