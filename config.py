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
    assert system in ["CFI", "FPM", "Jarvis", "Varun", "CapSys"]

    # ---------------------------------------------------------------------------- #
    # Directories
    # ---------------------------------------------------------------------------- #

    test_glob_pattern = "*.png"
    if system == "CFI":
        image_dir = Path("/mnt/ssd/udc/")
        output_dir = Path("output") / exp_name
        ckpt_dir = Path("ckpts")  # Checkpoints saved to ckpt_dir / exp_name
        run_dir = Path("runs")  # Runs saved to run_dir / exp_name
        test_image_dir = None

    if system == "FPM":
        image_dir = Path("/media/salman/udc/")
        output_dir = Path("output") / exp_name
        ckpt_dir = Path(
            "/media/salman/udc/ckpts"
        )  # Checkpoints saved to ckpt_dir / exp_name
        run_dir = Path(
            "/media/salman/udc/runs"
        )  # Runs saved to run_dir / exp_name
        test_image_dir = None

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    train_source_dir = image_dir / "Poled" / "LQ"
    train_target_dir = image_dir / "Poled" / "HQ"

    val_source_dir = None
    val_target_dir = None

    test_source_dir = image_dir / "Poled_val" / "LQ"

    # PSF
    wiener_mat = Path("data/poled_wiener_R.npy")

    static_val_image = "1.npy"
    static_test_image = "1.npy"

    image_height = 1024
    image_width = 2048

    batch_size = 6
    num_threads = batch_size  # parallel workers

    # ---------------------------------------------------------------------------- #
    # Train Configs
    # ---------------------------------------------------------------------------- #
    # Schedules
    fft_epochs = 0
    pretrain_epochs = 500
    num_epochs = 500

    learning_rate = 3e-4
    fft_learning_rate = 4e-10
    # Betas for AdamW. We follow https://arxiv.org/pdf/1704.00028
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
    save_filename_FFT = "FFT.pth"
    save_filename_D = "D.pth"

    save_filename_latest_G = "model_latest.pth"
    save_filename_latest_FFT = "FFT_latest.pth"
    save_filename_latest_D = "D_latest.pth"

    log_interval = 20  # the number of iterations (default: 10) to print at

    # ---------------------------------------------------------------------------- #
    # Model
    # ---------------------------------------------------------------------------- #
    # See models/get_model.py for registry
    model = "hdrnet-fft"

    use_spectral_norm = False
    pixelshuffle_ratio = 1  # Used for train_pixel_shuffle

    gan_type = "NSGAN"  # or RAGAN
    assert gan_type in ["NSGAN", "RAGAN"]
    use_patch_gan = False

    normaliser = "group_norm"
    assert normaliser in ["batch_norm", "instance_norm", "group_norm", "layer_norm"]
    num_groups = 8 if normaliser == "group_norm" else None

    # ---------------------------------------------------------------------------- #
    # Loss
    # ---------------------------------------------------------------------------- #
    lambda_adversarial = 0.6
    lambda_contextual = 0.0
    lambda_perception = 1.2  # 0.006
    lambda_image = 1  # mse
    lambda_lpips = 0.0
    lambda_ssim = 0.0
    lambda_sobel = 0.0
    lambda_forward = 0.0
    lambda_interm = 0.0

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


"""
9th March 2020
Instead of 1518 x 2012, 1280 x 1408, no psf pad
"""


def ours_pixelshuffle():
    exp_name = "ours_pixelshuffle"

    model = "unet-pixelshuffle-fft"
    pixelshuffle_ratio = 2


def hdrnet():
    exp_name = "hdrnet"

    model = "hdrnet-fft"  # We wont use fft though


def guided_filter():
    exp_name = "guided-filter"

    model = "guided-filter"  # We wont use fft though


def guided_filter_l1():
    exp_name = "guided-filter-l1"

    model = "guided-filter"  # We wont use fft though


named_configs = [ours_pixelshuffle, hdrnet, guided_filter, guided_filter_l1]


def initialise(ex):
    ex.config(base_config)
    for named_config in named_configs:
        ex.named_config(named_config)
    return ex
