# README

## Requirements

* python 3.7
* `pip install -r utils/requirements.txt`

## Configs

See `config.py` for exhaustive set of config options. 

Create a new function to overwrite and add it to `named_configs`. 

## Train Script

Run as:
`python train.py with xyz_config {system=""} {other flags}`

For a multi-gpu version:

`python -m torch.distributed.launch --nproc_per_node=3 --use_env=True train.py with xyz_config distdataparallel=True {other flags}`

## Val Script

Run as:
`python train.py with xyz_config {system=""} {other flags}`