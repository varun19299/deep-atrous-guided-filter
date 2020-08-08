# Deep Atrous Guided Filter

Our submission to the [Under Display Camera Challenge (UDC)](https://rlq-tod.github.io/challenge2.html) at ECCV 2020. We placed **2nd** and **5th** on the POLED and TOLED tracks respectively!

[Project Page] [Paper] [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/varun19299/deep-atrous-guided-filter/blob/master/demo_DAGF.ipynb)

![Method Diagram](figs/fig_2.png)

Official implementation of our ECCVW 2020 paper, "Deep Atrous Guided Filter for Image Restoration in Under Display Cameras",
[Varun Sundar<sup>*</sup>](mailto:vsundar4@wisc.edu), [Sumanth Hedge<sup>*</sup>](mailto:sumanth@smail.iitm.ac.in), [Divya K Raman](mailto:), [Kaushik Mitra](mailto:kmitra@ee.iitm.ac.in).
Indian Institute of Technology Madras, * denotes equal contribution.

## Quick Collab Demo

If you want to experiment with Deep Atrous Guided Filter (DAGF), we recommend you get started with the [collab notebook](https://colab.research.google.com/github/varun19299/deep-atrous-guided-filter/blob/master/demo_DAGF.ipynb). It exposes the core aspects of our method, while abstracting away minor details and helper functions.

It requires no prior setup, and contains a demo for both POLED and TOLED measurements.

If you're unfamiliar with Under Display Cameras, they are a new imaging system for smartphones, where the camera is mounted right under the display. This makes truly bezel-free displays possible, and opens up a bunch of other applications. You can read more here.

## Get Started

If you would like to reproduce **all** our experiments presented in the paper, head over to the [experiments](https://github.com/varun19299/deep-atrous-guided-filter/tree/experiments) branch. For a concise version with just our final models, you may continue here.


You'll need to install the following:

* python 3.7+
* pytorch 1.5+
* Use `pip install -r utils/requirements.txt` for the remaining

## Data


Download the required folder and place it under the `data/` directory. The train and val splits contain both low-quality measurements (`LQ` folder) and high-quality groudtruth (`HQ` folder). The test set contains only measurements currently.

We also provide our simulated dataset, based on training a shallow version of DAGF with CoBi loss.

## Configs and Checkpoints

We use [sacred] to handle config parsing, with the following command-line invocation:

```bash
python train{val}.py with config_name {other flags} -p
```

Various configs available:

Download the required checkpoint folder and place it under `ckpts/`. 

Further, see `config.py` for exhaustive set of config options. To add a config, create a new function in `config.py and add it to `named_configs`. 

## Directory Setup

Create the following symbolic links (assume `path_to_root_folder/` is `~/udc_net`):

* Data folder: `ln -s /data_dir/ ~/udc_net`
* Runs folder: `ln -s /runs_dir/ ~/udc_net`
* Ckpts folder: `ln -s /ckpt_dir/ ~/udc_net`
* Outputs folder: `ln -s /output_dir/ ~/udc_net`

### High Level Organisation 

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

## Train Script

Run as:
`python train.py with xyz_config{other flags}`

For a multi-gpu version (we use pytorch's `distributed-data-parallel`):

`python -m torch.distributed.launch --nproc_per_node=3 --use_env train.py with xyz_config distdataparallel=True {other flags}`

## Val Script

Run as:
`python val.py with xyz_config {other flags}`

Useful Flags:

* `self_ensemble`: Use self-ensembling. Ops may be found in `utils/self_ensembling.py`.

See config.py for exhaustive set of arguments (under `base_config`).

## Citation

If you find our work useful in your research, please cite:

```

```

## Contact

Feel free to mail us if you have any questions!