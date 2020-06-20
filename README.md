# README

## Requirements

* python 3.7
* `pip install -r utils/requirements.txt`
* Clone `PerceptualSimilarity` (LPIPS) from [here](https://github.com/richzhang/PerceptualSimilarity).
* Modify `PerceptualSimilarity/models/__init__.py` and related files to load from the root itself.

ie... change this 
```python
from models import dist_model
```
to 

```python
from PerceptualSimilarity.models import dist_model
```

etc.

Or just clone `varun19299/PerceptualSimilarity`.

## Configs

See `config.py` for exhaustive set of config options. Create a YAML file for testing variants. Refer to the `configs/` folder for more.

## Train Script


## Train Script DDP

python -m torch.distributed.launch --nproc_per_node=3 --use_env=True train_ddp.py with xyz_config {other flags}



## Val Script

## Test Script

## Streamlit Server

## Misc

### Rsync Checkpoints Selectively

To copy from source to dest:
`rsync -rtauvc --exclude-from 'utils/checkpoint_excluder.txt' source_path_to_models/ckpts_phase_mask_Feb_2020_size_384/exp-name/ dest_path_to_models/ckpts_phase_mask_Feb_2020_size_384/exp-name/`

Command to be executed from root of this folder. Else, mention the path to `utils/checkpoint_excluder.txt` in absolute sense. Note: trailing slashes in both source_path and dest_path are important, which is typical of rsync.

To sync source to dest:
`rsync -rtauvc --delete --exclude-from 'utils/checkpoint_excluder.txt' source_path_to_models/ckpts_phase_mask_Feb_2020_size_384/exp-name/ dest_path_to_models/ckpts_phase_mask_Feb_2020_size_384/exp-name/`

See [this](https://www.jveweb.net/en/archives/2010/11/synchronizing-folders-with-rsync.html) for more information.

## References

1. Learned Reconstructions for mask based lensless imaging: [paper](learned reconstructions for practical mask-based lensless imaging), [website](https://waller-lab.github.io/LenslessLearning/).
2. Diffuser Cam: [paper], [website](https://waller-lab.github.io/DiffuserCam/index.html)
3. LPIPS: [paper](https://arxiv.org/abs/1801.03924), [code](https://github.com/richzhang/PerceptualSimilarity).
