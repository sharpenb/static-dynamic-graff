# Static and Dynamic GRAFF: Understanding Convolution on Graphs via Energies

This repository presents a clean implmentation of the paper (with additional features):

[Understanding Convolution on Graphs via Energies](https://arxiv.org/pdf/2206.10991.pdf)<br>
Francesco Di Giovanni, James Rowbottom, Benjamin P. Chamberlain, Thomas Markovich, Michael M. Bronstein<br>
Transactions on Machine Learning Research  (TMLR), 2023.

[Paper](https://arxiv.org/pdf/2206.10991.pdf)

## Setup

```{bash}
conda create -n graff -c pytorch -c conda-forge pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 python=3.9 pip
conda activate graff
conda install pyg -c pyg
pip install ogb
pip install wandb
pip install pytorch_lightning
pip install gdown
pip install seml
pip install dotmap
pip install notebook
python setup.py develop
```

After setup, you can run the models by using `src/models/graff/run.py` (and `src/models/gcn/run.py`). For GRAFF, we make available different parametrizations of the matrices W, Omega, and Q defined in the papers. We also propse two parametrizations of GRAFF steps wher all layers share the same parameters (static) or have independent parameters (dynamic). Input parameters of methods should be self-explanatory. 

## Track ML code execution with Weight&Biases

- Install Weight&Biases.
- Authorize your API key using `wandb login`.
- Create and use the porject id in Weight&Biases.

For more detail see [official instructions](https://wandb.ai/quickstart/pytorch).

## Manage experiments with seml or ray-tune

For more detail on seml see [official instructions](https://github.com/TUM-DAML/seml).

For more detail on ray-tune see [official instructions](https://docs.ray.io/en/latest/tune/index.html).

## Acknowledgements

Thanks to Francesco Di Giovanni for discussing advanced parametrization ideas for GRAFF.
