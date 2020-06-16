"""
Convention

ours/naive-fft-(fft_h-fft_w)-learn-(learn_h-learn_w)-meas-(meas_h-meas_w)-kwargs

* Phlatcam: 1518 x 2012 (post demosiacking)
* Flatcam: 512 x 640 (post demosiacking)
* Diffusercam: 270 x 480 (post demosiacking, downsize by 4)
"""
from pathlib import Path
import torch


def base_config():
    exp_name = "ours"
    system = "CFI"
    assert system in ["CFI", "FPM", "Jarvis", "Varun"]

    # ---------------------------------------------------------------------------- #
    # Directories
    # ---------------------------------------------------------------------------- #

    if system == "CFI":
        image_dir = Path("/mnt/ssd/udc/")
        output_dir = Path("output") / exp_name
        ckpt_dir = Path("ckpts")  # Checkpoints saved to ckpt_dir / exp_name
        run_dir = Path("runs")  # Runs saved to run_dir / exp_name

    elif system == "FPM":
        image_dir = Path("/media/salman/udc/")
        output_dir = Path("output") / exp_name
        ckpt_dir = Path(
            "/media/salman/udc/ckpts"
        )  # Checkpoints saved to ckpt_dir / exp_name
        run_dir = Path("/media/salman/udc/runs")  # Runs saved to run_dir / exp_name

    elif system == "FPM":
        image_dir = Path("/media/data/salman/udc/")
        output_dir = Path("output") / exp_name
        ckpt_dir = Path(
            "/media/data/salman/udc/ckpts"
        )  # Checkpoints saved to ckpt_dir / exp_name
        run_dir = Path(
            "/media/data/salman/udc/runs"
        )  # Runs saved to run_dir / exp_name

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    train_source_dir = image_dir / "Poled" / "LQ"
    train_target_dir = image_dir / "Poled" / "HQ"

    val_source_dir = None
    val_target_dir = None

    test_source_dir = image_dir / "Poled_val" / "LQ"

    static_val_image = "1.npy"
    static_test_image = "1.npy"

    image_height = 1024
    image_width = 2048

    batch_size = 8
    num_threads = batch_size  # parallel workers

    # ---------------------------------------------------------------------------- #
    # Train Configs
    # ---------------------------------------------------------------------------- #
    # Schedules
    num_epochs = 512 - 1

    learning_rate = 3e-4

    # Betas for AdamW.
    # We follow https://arxiv.org/pdf/1704.00028
    beta_1 = 0.9  # momentum
    beta_2 = 0.999

    lr_scheduler = "cosine"  # or step

    # Cosine annealing
    T_0 = 1
    T_mult = 2

    # Step lr
    step_size = 2

    # saving models
    save_filename_G = "model.pth"
    save_filename_D = "D.pth"

    save_filename_latest_G = "model_latest.pth"
    save_filename_latest_D = "D_latest.pth"

    # save a copy of weights every x epochs
    save_copy_every_epochs = 10

    # the number of iterations (default: 10) to print at
    log_interval = 20

    # run val or test only every x epochs
    val_test_epoch_interval = 5

    # ---------------------------------------------------------------------------- #
    # Model: See models/get_model.py for registry
    # ---------------------------------------------------------------------------- #

    model = "guided-filter"

    use_spectral_norm = False

    gan_type = "NSGAN"  # or RAGAN
    assert gan_type in ["NSGAN", "RAGAN"]
    use_patch_gan = False

    normaliser = "group_norm"
    assert normaliser in ["batch_norm", "instance_norm", "group_norm", "layer_norm"]
    num_groups = 8 if normaliser == "group_norm" else None

    # ---------------------------------------------------------------------------- #
    # Loss
    # ---------------------------------------------------------------------------- #
    lambda_adversarial = 0.0
    lambda_perception = 0.0
    lambda_image = 1  # l1

    resume = True
    finetune = False  # Wont load loss or epochs

    # ---------------------------------------------------------------------------- #
    # Inference Args
    # ---------------------------------------------------------------------------- #
    inference_mode = "latest"
    assert inference_mode in ["latest", "best"]

    # ---------------------------------------------------------------------------- #
    # Distribution Args
    # ---------------------------------------------------------------------------- #
    # choose cpu or cuda:0 device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    lpips_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataparallel = False
    device_list = None


def hdrnet():
    exp_name = "hdrnet"

    model = "hdrnet"  # We wont use fft though


def guided_filter():
    exp_name = "guided-filter"

    model = "guided-filter"  # We wont use fft though


def guided_filter_l1():
    exp_name = "guided-filter-l1"

    model = "guided-filter"  # We wont use fft though


def guided_filter_l1_tanh():
    exp_name = "guided-filter-l1-tanh"

    model = "guided-filter"  # We wont use fft though


def guided_filter_l1_percep_adv():
    exp_name = "guided-filter-l1-percep-adv"

    model = "guided-filter"  # We wont use fft though

    num_epochs = 512 - 1

    batch_size = 2

    log_interval = 30

    lpips_device = "cuda:1" if torch.cuda.is_available() else "cpu"

    lambda_adversarial = 0.6
    lambda_perception = 1.2
    lambda_image = 1  # l1


named_configs = [
    hdrnet,
    guided_filter,
    guided_filter_l1,
    guided_filter_l1_tanh,
    guided_filter_l1_percep_adv,
]


def initialise(ex):
    ex.config(base_config)
    for named_config in named_configs:
        ex.named_config(named_config)
    return ex
