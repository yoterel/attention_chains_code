<h2 align="center">Attention (as Discrete-Time Markov) Chains</h2>
<div align="center"> 
  <a href="https://yoterel.github.io/" target="_blank">Yotam Erel*</a>,
  <a href="https://odunkel.github.io" target="_blank">Olaf Dünkel*</a>, 
  <a href="https://rishabhdabral.github.io/" target="_blank">Rishabh Dabral</a>,</span>
  <a href="https://people.mpi-inf.mpg.de/~golyanik/" target="_blank">Vladislav Golyanik</a>,</span>
  <a href="https://people.mpi-inf.mpg.de/~theobalt/" target="_blank">Christian Theobalt</a>,</span>
  <a href="https://www.cs.tau.ac.il/~amberman/" target="_blank">Amit H. Bermano</a></span>
  
*denoted equal contribution
</div>

<br>

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://yoterel.github.io/attention_chains/)
[![Paper](https://img.shields.io/badge/arXiv-PDF-b31b1b)](https://arxiv.org/pdf/2507.17657)

</div>


This is the official implementation of Attention (as Discrete-Time Markov) Chains.

## Usage

### Core Functionality
For a straight-forward implemetation of multi-bounce attention, TokenRank, and lambda-weighting form the paper, see [helpers.py](https://github.com/yoterel/attention_chains_code/blob/main/helpers.py).

### Demos

We provide a demo for DINOv1/2, CLIP, supervised ViT (from transformers library) in [demo.ipynb](https://github.com/yoterel/attention_chains_code/blob/main/demo.ipynb).

#### FLUX Visualization

For visualizing attention with FLUX, run:

`flux.py flux.yml`

You can edit [flux.yml](https://github.com/yoterel/attention_chains_code/blob/main/flux.yml) for tinkering with the results.

*Note: you must have the libraries imported by [flux.py](https://github.com/yoterel/attention_chains_code/blob/main/flux.py) installed in your virtual environment

###

## Todos
- [x] Basic functionality
- [x] Visualization demo for FLUX
- [x] Segmentation demo for FLUX
- [x] Demo for DINOv1/2, ViT, CLIP
- [ ] Reproduction of experiments
