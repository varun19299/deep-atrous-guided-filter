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
        dump_dir = Path(".")

    elif system == "FPM":
        image_dir = Path("/media/salman/udc/")
        dump_dir = image_dir

    elif system == "Jarvis":
        image_dir = Path("/media/data/salman/udc/")
        dump_dir = image_dir

    output_dir = dump_dir / "outputs" / exp_name
    ckpt_dir = dump_dir / "ckpts"  # Checkpoints saved to ckpt_dir / exp_name
    run_dir = dump_dir / "runs"  # Runs saved to run_dir / exp_name

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    train_source_dir = image_dir / "Poled" / "LQ"
    train_target_dir = image_dir / "Poled" / "HQ"

    val_source_dir = None
    val_target_dir = None

    test_source_dir = image_dir / "Poled_val" / "LQ"

    static_val_image = "1.png"
    static_test_image = "1.png"

    image_height = 1024
    image_width = 2048

    batch_size = 8
    num_threads = batch_size  # parallel workers

    # augment
    do_augment = True

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
    save_copy_every_epochs = 128

    # the number of iterations (default: 10) to print at
    log_interval = 20

    # run val or test only every x epochs
    val_test_epoch_interval = 5

    # ----------------------------------------------------------------------------  #
    # Val / Test Configs
    # ---------------------------------------------------------------------------- #

    # Self ensemble
    self_ensemble = False

    # Save mat file
    save_mat = False

    inference_mode = "latest"
    assert inference_mode in ["latest", "best"]

    # ---------------------------------------------------------------------------- #
    # Model: See models/get_model.py for registry
    # ---------------------------------------------------------------------------- #

    model = "guided-filter"
    CAN_layers = 5

    use_spectral_norm = False
    pixelshuffle_ratio = 1

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
    # Distribution Args
    # ---------------------------------------------------------------------------- #
    # choose cpu or cuda:0 device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    lpips_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dataparallel = False
    device_list = None


def guided_filter_l1_tanh():
    exp_name = "guided-filter-l1-tanh"

    model = "guided-filter"  # We wont use fft though


def guided_filter_l1_tanh_pixelshuffle():
    exp_name = "guided-filter-l1-tanh-pixelshuffle"

    batch_size = 9
    CAN_layers = 21

    do_augment = False

    model = "guided-filter-pixelshuffle"  # We wont use fft though
    pixelshuffle_ratio = 2

    dataparallel = True
    device_list = [0, 1, 2]


def guided_filter_l1_tanh_pixelshuffle_eca():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-eca"

    batch_size = 9
    CAN_layers = 21

    do_augment = False
    num_epochs = 1024 - 1

    model = "guided-filter-pixelshuffle-eca"  # We wont use fft though
    pixelshuffle_ratio = 2

    dataparallel = True
    device_list = [0, 1, 2]


def guided_filter_l1_tanh_pixelshuffle_gca():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-gca"

    batch_size = 9

    do_augment = False

    model = "guided-filter-pixelshuffle-gca"  # We wont use fft though
    pixelshuffle_ratio = 2

    dataparallel = True
    device_list = [0, 1, 2]


def guided_filter_l1_tanh_pixelshuffle_sim():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-sim"

    batch_size = 9
    CAN_layers = 21

    do_augment = True

    model = "guided-filter-pixelshuffle"  # We wont use fft though
    pixelshuffle_ratio = 2

    dataparallel = True
    device_list = [0, 1, 2]
    num_epochs = 128 - 1
    finetune = False

    val_test_epoch_interval = 1
    save_copy_every_epochs = 32
    log_interval = 80

    system = "CFI"
    assert system in ["CFI", "FPM", "Jarvis", "Varun"]

    # ---------------------------------------------------------------------------- #
    # Directories
    # ---------------------------------------------------------------------------- #

    if system == "CFI":
        image_dir = Path("/mnt/ssd/udc/")
        dump_dir = Path(".")

    elif system == "FPM":
        image_dir = Path("/media/salman/udc/")
        dump_dir = image_dir

    elif system == "Jarvis":
        image_dir = Path("/media/data/salman/udc/")
        dump_dir = image_dir

    output_dir = dump_dir / "outputs" / exp_name
    ckpt_dir = dump_dir / "ckpts"  # Checkpoints saved to ckpt_dir / exp_name
    run_dir = dump_dir / "runs"  # Runs saved to run_dir / exp_name

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    train_source_dir = image_dir / "DIV2K_train" / "LQ"
    train_target_dir = image_dir / "DIV2K_train" / "HQ"

    val_source_dir = image_dir / "DIV2K_val" / "LQ"
    val_target_dir = image_dir / "DIV2K_val" / "HQ"

    test_source_dir = None


def guided_filter_l1_tanh_pixelshuffle_augment():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-augment"

    batch_size = 9
    CAN_layers = 21

    do_augment = True

    finetune = True
    num_epochs = 128 - 1

    model = "guided-filter-pixelshuffle"  # We wont use fft though
    pixelshuffle_ratio = 2

    dataparallel = True
    device_list = [0, 1]


def guided_filter_l1_tanh_pixelshuffle_inverse():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-inverse"

    batch_size = 6
    CAN_layers = 15
    do_augment = True

    model = "guided-filter-pixelshuffle"  # We wont use fft though
    pixelshuffle_ratio = 2

    dataparallel = True
    device_list = [0, 1]

    num_epochs = 256 - 1

    system = "CFI"
    assert system in ["CFI", "FPM", "Jarvis", "Varun"]

    # ---------------------------------------------------------------------------- #
    # Directories
    # ---------------------------------------------------------------------------- #

    if system == "CFI":
        image_dir = Path("/mnt/ssd/udc/")
        dump_dir = Path(".")

    elif system == "FPM":
        image_dir = Path("/media/salman/udc/")
        dump_dir = image_dir

    elif system == "Jarvis":
        image_dir = Path("/media/data/salman/udc/")
        dump_dir = image_dir

    output_dir = dump_dir / "outputs" / exp_name
    ckpt_dir = dump_dir / "ckpts"  # Checkpoints saved to ckpt_dir / exp_name
    run_dir = dump_dir / "runs"  # Runs saved to run_dir / exp_name

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    train_source_dir = image_dir / "Poled" / "HQ"
    train_target_dir = image_dir / "Poled" / "LQ"

    val_source_dir = None
    val_target_dir = None

    # test_source_dir = image_dir / "DIV2K_train" / "HQ"
    test_source_dir = image_dir / "DIV2K_val" / "HQ"


def guided_filter_l1_tanh_pixelshuffle_toled():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-toled"

    batch_size = 6
    CAN_layers = 21

    do_augment = False
    num_epochs = 1024 - 1

    model = "guided-filter-pixelshuffle"  # We wont use fft though
    pixelshuffle_ratio = 2

    dataparallel = True
    device_list = [0, 1]

    system = "CFI"
    assert system in ["CFI", "FPM", "Jarvis", "Varun"]

    # ---------------------------------------------------------------------------- #
    # Directories
    # ---------------------------------------------------------------------------- #

    if system == "CFI":
        image_dir = Path("/mnt/ssd/udc/")
        dump_dir = Path(".")

    elif system == "FPM":
        image_dir = Path("/media/salman/udc/")
        dump_dir = image_dir

    elif system == "Jarvis":
        image_dir = Path("/media/data/salman/udc/")
        dump_dir = image_dir

    output_dir = dump_dir / "outputs" / exp_name
    ckpt_dir = dump_dir / "ckpts"  # Checkpoints saved to ckpt_dir / exp_name
    run_dir = dump_dir / "runs"  # Runs saved to run_dir / exp_name

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    train_source_dir = image_dir / "Toled" / "LQ"
    train_target_dir = image_dir / "Toled" / "HQ"

    val_source_dir = None
    val_target_dir = None

    test_source_dir = image_dir / "Toled_val" / "LQ"


named_configs = [
    guided_filter_l1_tanh,
    guided_filter_l1_tanh_pixelshuffle,
    guided_filter_l1_tanh_pixelshuffle_eca,
    guided_filter_l1_tanh_pixelshuffle_gca,
    guided_filter_l1_tanh_pixelshuffle_sim,
    guided_filter_l1_tanh_pixelshuffle_augment,
    guided_filter_l1_tanh_pixelshuffle_inverse,
    guided_filter_l1_tanh_pixelshuffle_toled,
]


def initialise(ex):
    ex.config(base_config)
    for named_config in named_configs:
        ex.named_config(named_config)
    return ex
