# README

Our submission to the ECCV 2020 [Under Display Camera Challenge (UDC)](https://rlq-tod.github.io/challenge2.html).

## Requirements

* python 3.7
* pytorch 1.5+
* Use `pip install -r utils/requirements.txt` for the remaining

## Directory Setup

Create the following symbolic links (assume `path_to_root_folder/` is `~/udc_net`):

* Data folder: `ln -s /data_dir/ ~/udc_net`
* Runs folder: `ln -s /runs_dir/ ~/udc_net`
* Ckpts folder: `ln -s /ckpt_dir/ ~/udc_net`
* Outputs folder: `ln -s /output_dir/ ~/udc_net`

### Organisation 

**Data folder**: Each subfolder contains a data split.

```shell
|-- Poled_train
|   |-- HQ
|   |-- |-- 101.png
|   |-- |-- 102.png
|   |-- |-- 103.png
|   `-- LQ
|-- Poled_val
|   `-- LQ
```

Splits: 
* Poled_{train,val}: Poled acquired images, HQ (glass), LQ (Poled) pairs.
* Toled_{train,val}: Toled acquired images, HQ (glass), LQ (Toled) pairs.
* Sim_{train,val}: our simulated set.
* DIV2K_{train,val}: source images used for Poled, Toled in monitor acquisition. Used to train sim networks.

**Outputs folder**: Val, test dumps under various experiment names.

```shell
outputs
|-- guided-filter-l1-tanh-pixelshuffle
|   |-- test_latest
|   |-- test_latest_self_ensemble
|   |-- val_latest
|   `-- val_latest_self_ensemble
|-- guided-filter-l1-tanh-pixelshuffle-5x5
|   |-- test_latest
    |   |-- 9.png
    |   |-- readme.txt
    |   `-- results.mat
|   `-- val_latest
        |-- 99.png
        |-- 9.png
        `-- metrics.txt
```

**Ckpts folder**: Ckpts under various experiment names. We store every 64th epoch, and every 5 epochs prior for model snapshots. This is mutable under `config.py`.

```shell
ckpts
|-- guided-filter-l1-tanh-pixelshuffle-gca-5x5-improved-ECA
|   |-- Epoch_126_model_latest.pth
|   |-- Epoch_190_model_latest.pth
|   |-- Epoch_62_model_latest.pth
|   `-- model_latest.pth
|-- guided-filter-l1-tanh-pixelshuffle-gca-5x5-improved-FFA
|   `-- model_latest.pth
```

**Runs folder:** Tensorboard event files under various experiment names.

```shell
runs
|-- guided-filter-l1-tanh-pixelshuffle
|   |-- events.out.tfevents.1592369530.jarvis.26208.0
|-- guided-filter-l1-tanh-pixelshuffle-5x5
|   |-- events.out.tfevents.1592719979.jarvis.37079.0
```


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
