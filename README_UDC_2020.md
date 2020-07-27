# README

Our submission to the ECCV 2020 [Under Display Camera Challenge (UDC)](https://rlq-tod.github.io/challenge2.html).

## Requirements

* python 3.7
* pytorch 1.5+
* Use `pip install -r utils/requirements.txt` for the remaining

## Downloading Data and Checkpoints

We shall consider this folder (`udc_net`) as the root folder.

* Download data from [here](https://drive.google.com/drive/folders/1-2NDQXTZvPaoHIjVRobgENlW1qocbRZX?usp=sharing), place under `data`.
* Download ckpts from [here](https://drive.google.com/drive/folders/1eX8Rhj5_M0vi8HYRX8zKKp_h5PIh2Vec?usp=sharing),  place under `ckpts`.
* Download outputs from [here](https://drive.google.com/drive/folders/1fCr2dxiymYnb-COqPIGkjZprY0yj7Rew?usp=sharing),  place under `outputs`.

For data, download `Poled_{train,val,test}`, `Toled_{train,val,test}` folders. You can also use symlinks.

The ckpt and output folders contain all our experiments, but you only need to download the configurations given below. 

## Configurations in Submission

POLED:
* `final_poled_sim_actual` (with self ensembling)
* `final_poled_sim_actual` (with self and model ensembling)
* `final_poled_sim_actual_aug`

TOLED:
* `final_toled_sim_actual`
* `final_toled_sim_actual`
* `final_toled_sim_actual_aug`

For experiment names (which correspond to the ckpt and output folder names), replace "_" by "-" in the config name.

While using the `aug` configs you must download corresponding `sim_actual` output folder (val and test folders atleast).

## Reproduce Results

### From checkpoints

`python val.py with xyz_config {system=""} {save_mat=True} {self_ensemble=True}`

`xyz` could be `final_poled_sim_actual`, `final_poled_sim_actual_aug` etc.

Useful Flags:

* `save_mat`: Dumps mat file in `outputs/exp_name/test_latest/`. Used for submitting to CodaLab.
* `self_ensemble`: Use self-ensembling. Ops may be found in `utils/self_ensembling.py`.
* `model_ensemble`: Use model-ensembling. Averaged mat file will be found in `outputs/exp_name/val_model_ensemble.../results.mat`.

See config.py for exhaustive set of arguments (under `base_config`).


### From Scratch

Ensure checkpoint under `ckpts/exp_name` is removed.

`python train.py with xyz_config`

`xyz` could be `final_poled`, `final_poled_sim_actual` etc.

For a multi-gpu version (we use pytorch's `distributed-data-parallel`):

```bash
export NUM_GPUS=4 # 4 GPUs
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS --use_env train.py with xyz_config distdataparallel=True {other flags}
```

For `_aug` configs, replace `train.py` with `train_aug.py` and ensure that you download corresponding `sim_actual` output folder (val and test folders atleast).

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
* DIV2K: source images used for train Poled, Toled in monitor acquisition. Used to train sim networks.

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

## Other Configs

See `config.py` for exhaustive set of config options. 

Create a new function to overwrite and add it to `named_configs`. 

## Using Symlinks

You can alternatively create the following symbolic links (assume `path_to_root_folder/` is `udc_net`):

* Data folder: `ln -s /data_dir/ ~/udc_net`
* Runs folder: `ln -s /runs_dir/ ~/udc_net`
* Ckpts folder: `ln -s /ckpt_dir/ ~/udc_net`
* Outputs folder: `ln -s /output_dir/ ~/udc_net`