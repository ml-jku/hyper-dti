# HyperDTI

[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Pytorch](https://img.shields.io/badge/PyTorch-1.9-red.svg)](https://pytorch.org/get-started/previous-versions/)
![Licence](https://img.shields.io/github/license/ml-jku/hyper-dti)

A HyperNetwork approach to drug-target interaction prediction.

**[Abstract](#abstract)**
| **[Dependencies](#dependencies)**
| **[Data](#data)**
| **[Citation](#citation)**

![plot](hyper-dti.png)

## Abstract

### Robust task-specific adaption of drug-target interaction models

Emma Svensson, Pieter-Jan Hoedt, Sepp Hochreiter, Günter Klambauer

With the rise of new diseases, the fast discovery of drugs decreases the harm done to individuals. To this end, computational methods must be efficiently adaptable to new tasks, e.g. drug targets. HyperNetworks have been established as an effective technique to quickly adapt the parameters of neural networks. Notably, HyperNetwork-based parameter adaption has improved multi-task generalization in various domains, such as personalized federated learning and neural architecture search. In the drug discovery domain, drug-target interaction (DTI) models must be adapted to new drug targets, such as proteins, which constitute descriptions of prediction tasks. Current state-of-the-art Deep Learning architectures apply a few fully-connected layers to concatenated, learned embeddings of the description of the drug target and the molecule. However, these architectures do not have a specific mechanism to adapt the parameters to new targets. In this work, we develop a HyperNetwork approach to adapt the parameters of DTI models. On an established benchmark, our HyperNetwork approach improves the predictive performance of current architectures in several categories. Furthermore, we extend our approach to learn all parameters of a graph neural network as the molecular encoder using a particular weight initialization scheme. The proposed HyperNetwork approach renders DTI models more robust to new tasks and improves predictive performance in low data settings.

## Dependencies

Main requirements are,
- CUDA >= 11.4
- PyTorch >= 1.9
- Pytorch-Lightning >= 1.5 

Additional packages: rdkit, aidd-codebase, chemprop, bio-embeddings

Optionally supports: tensorboard and/or wandb

## Data
Datasets currently supported,
- Benchmark (Lenselink, 2017) derived from [ChEMBL](https://www.ebi.ac.uk/chembl/)

## Citation
Not yet published.

## References
Schmidhuber, J., “Learning to control fast-weight memories: An alternative to dynamic recurrent networks.” Neural Computation, 1992.

Lenselink, E. B., et al. "Beyond the hype: deep neural networks outperform established methods using a ChEMBL bioactivity benchmark set." Journal of cheminformatics 9.1 (2017): 1-14.

Ha, D., et al. “HyperNetworks”. ICLR, 2017.

Chang, O., et al., “Principled weight initialization for hypernetworks.” International Conference on Learning Representations, 2019.

Kim, P. T., et al. "Unsupervised Representation Learning for Proteochemometric Modeling." International Journal of Molecular Sciences 22.23 (2021): 12882.

## Keywords
Drug Discovery, Machine Learning, Drug-target interaction prediction, Zero-shot, HyperNetworks

