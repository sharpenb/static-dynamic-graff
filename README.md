# Dynamic GRAFF

## Setup

```{bash}
conda create -n dynamic_graff -c pytorch -c conda-forge pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 python=3.9 pip
conda activate dynamic_graff
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

## Track ML code execution with Weight&Biases

- Install Weight&Biases.
- Authorize your API key using `wandb login`.
- Create and use the porject id in Weight&Biases.

For more detail see [official instructions](https://wandb.ai/quickstart/pytorch).

## Manage experiments with seml or ray-tune

For more detail on seml see [official instructions](https://github.com/TUM-DAML/seml).

For more detail on ray-tune see [officialt instructions](https://docs.ray.io/en/latest/tune/index.html).
