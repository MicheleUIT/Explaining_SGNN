# Explainability in subgraphs-enhanced Graph Neural Networks

This repository contains the code of the paper
**[Explainability in subgraphs-enhanced Graph Neural Networks](https://arxiv.org/abs/2209.07926)**

Authors: Michele Guerra, Indro Spinelli, Simone Scardapane, Filippo Maria Bianchi

Abstract:

Recently, subgraphs-enhanced Graph Neural Networks (SGNNs) have been introduced to enhance the expressive power of Graph Neural Networks (GNNs), which was proved to be not higher than the 1-dimensional Weisfeiler-Leman isomorphism test. The new paradigm suggests using subgraphs extracted from the input graph to improve the model's expressiveness, but the additional complexity exacerbates an already challenging problem in GNNs: explaining their predictions. In this work, we adapt PGExplainer, one of the most recent explainers for GNNs, to SGNNs. The proposed explainer accounts for the contribution of all the different subgraphs and can produce a meaningful explanation that humans can interpret. The experiments that we performed both on real and synthetic datasets show that our framework is successful in explaining the decision process of an SGNN on graph classification tasks.

The code is build upon the repository of [Equivariant Subgraph Aggregation Networks (ESAN)](https://github.com/beabevi/ESAN) and we thanks the authors (Beatrice Bevilacqua, Fabrizio Frasca, Derek Lim, Balasubramaniam Srinivasan, Chen Cai, Gopinath Balamurugan, Michael M. Bronstein and Haggai Maron) for making their code accessible to everyone.

## Install

First create a conda environment
```
conda env create -f environment.yml
```
and activate it
```
conda activate subgraph
```

## Train ESAN

Launch `train_esan.py` with the desired configuration.

## Run the explainer

ADD SECTION


## Credits

For attribution in academic contexts, please cite

```
@misc{https://doi.org/10.48550/arxiv.2209.07926,
  doi = {10.48550/ARXIV.2209.07926},
  url = {https://arxiv.org/abs/2209.07926},
  author = {Guerra, Michele and Spinelli, Indro and Scardapane, Simone and Bianchi, Filippo Maria},
  keywords = {Machine Learning (cs.LG), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Explainability in subgraphs-enhanced Graph Neural Networks},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
