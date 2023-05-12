# DragoNNFruit

![image](https://github.com/jmschrei/dragonnfruit/assets/3916816/00ec6a9b-e8b7-4bd1-9895-3a73b377d6da)

DragoNNFruit is a method for dissecting the cis- and trans-regulatory factors underlying chromatin accessibility at single-cell and base-pair resolution. At a high level, DragoNNFruit models cis-regulatory sequence using a convolutional neural network whose parameters are dynamically generated from a second network that models trans-regulatory state (from ATAC-seq, RNA-seq, spatial coordinates, etc). Through the inclusion of an explicit model of Tn5 bias, DragoNNFruit’s de-noised predictions reveal TF footprints that cannot be observed from the original, biased, data. Taken together, DragoNNFruit’s capabilities enable the identification of cell type-specific motifs and their higher-order syntax, the prediction of variant effect and footprinting genome-wide, and the tracking of all these across entire single-cell experiments without the need for manual cell clustering and annotation.

By explicitly modeling both cis- and trans-regulatory factors, DragoNNFruit departs from current regulatory modeling approaches such as Enformer, scBasset, and ChromVAR. Neither Enformer nor scBasset explicitly model trans-regulatory factors and so cannot generalize past observed cell states. Further, neither operate at base-pair resolution, limiting their interpretability and predictive power. ChromVAR explicitly models cell state by tracking global TF motif activity across cells, but does not model sequence and so cannot reveal how local syntax affects TF binding at individual loci.

### Installation

`pip install dragonnfruit`

### Usage

