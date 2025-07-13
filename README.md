[Project Page](https://yoterel.github.io/attention_chains/) | [Paper]() | [Supplementary](https://yoterel.github.io/attention_chains/static/pdfs/AttentionChains_supp.pdf)

# Attention Chains

This is the official implementation of Attention (as discrete-time Markov) Chains

## Usage

### Core Functionallity
For a straight-forward implemetation of multi-bounce attention, TokenRank, and lambda-weighting form the paper, see [here](https://github.com/yoterel/attention_chains_code/blob/main/helpers.py).

### FLUX Visualization Demo

For visualizing attention with FLUX, run:

`flux.py flux.yml`

You can edit [flux.yml](https://github.com/yoterel/attention_chains_code/blob/main/flux.yml) for tinkering with the results.

*Note: you must have the libraries imported by [flux.py](https://github.com/yoterel/attention_chains_code/blob/main/flux.py) installed in your virtual environment

###

## Todos
- [x] basic helper functions
- [x] viz demo for flux
- [x] seg demo for flux
- [x] demo for DINOv1/2
- [ ] reproduction of experiments
