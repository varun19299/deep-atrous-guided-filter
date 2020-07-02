# README

Our submission to the ECCV 2020 [Under Display Camera Challenge (UDC)](https://rlq-tod.github.io/challenge2.html).

## Requirements

* python 3.7
* pytorch 1.5+
* Use `pip install -r utils/requirements.txt` for the remaining

## Configs

See `config.py` for exhaustive set of config options. 

Create a new function to overwrite and add it to `named_configs`. 

## Train Script

Run as:
`python train.py with xyz_config {system=""} {other flags}`

For a multi-gpu version (we use pytorch's `distributed-data-parallel`):

`python -m torch.distributed.launch --nproc_per_node=3 --use_env=True train.py with xyz_config distdataparallel=True {other flags}`

## Val Script

Run as:
`python train.py with xyz_config {system=""} {other flags}`

Useful Flags:

* `save_mat`: Dumps mat file in `outputs/args.exp_name/test_latest/`. Used for submitting to CodeLabs.
* `self_ensemble`: Use self-ensembling. Ops may be found in `utils/self_ensembling.py`.

See config.py for exhaustive set of arguments (under `base_config`).
