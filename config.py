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
    use_SIREN = False
    use_ECA = False

    pixelshuffle_ratio = 1

    guided_map_kernel_size = 1
    guided_map_channels = 16

    # Discriminator
    gan_type = "NSGAN"  # or RAGAN
    assert gan_type in ["NSGAN", "RAGAN"]
    use_patch_gan = False
    use_spectral_norm = False
    normaliser = "group_norm"
    assert normaliser in ["batch_norm", "instance_norm", "group_norm", "layer_norm"]
    num_groups = 8 if normaliser == "group_norm" else None

    # ---------------------------------------------------------------------------- #
    # Loss
    # ---------------------------------------------------------------------------- #
    lambda_adversarial = 0.0
    lambda_perception = 0.0
    lambda_image = 1  # l1
    lambda_ms_ssim = 0.0

    resume = True
    finetune = False  # Wont load loss or epochs

    # ---------------------------------------------------------------------------- #
    # Distribution Args
    # ---------------------------------------------------------------------------- #
    # choose cpu or cuda:0 device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    lpips_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    distdataparallel = False


def guided_filter_l1_tanh():
    exp_name = "guided-filter-l1-tanh"

    model = "guided-filter"


def guided_filter_l1_tanh_pixelshuffle():
    exp_name = "guided-filter-l1-tanh-pixelshuffle"

    batch_size = 3
    CAN_layers = 21

    do_augment = False

    model = "guided-filter-pixelshuffle"
    pixelshuffle_ratio = 2


def guided_filter_l1_tanh_pixelshuffle_5x5():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-5x5"

    batch_size = 3
    CAN_layers = 21

    # Finetune it
    num_epochs = 64 - 1

    guided_map_kernel_size = 5
    guided_map_channels = 24

    do_augment = True

    model = "guided-filter-pixelshuffle"
    pixelshuffle_ratio = 2


def guided_filter_l1_tanh_pixelshuffle_5x5_ms_ssim():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-5x5-ms-ssim"

    batch_size = 2
    num_epochs = 1024 - 1
    do_augment = True

    CAN_layers = 21
    guided_map_kernel_size = 5
    guided_map_channels = 24
    model = "guided-filter-pixelshuffle"
    pixelshuffle_ratio = 2

    lambda_image = 1 - 0.84
    lambda_ms_ssim = 0.84

    # Cosine annealing
    T_0 = 64
    T_mult = 2


def guided_filter_l1_tanh_pixelshuffle_siren():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-siren"

    batch_size = 3
    CAN_layers = 21

    do_augment = True

    model = "guided-filter-pixelshuffle"
    pixelshuffle_ratio = 2
    use_SIREN = True


def guided_filter_l1_tanh_pixelshuffle_eca():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-eca"

    batch_size = 2
    CAN_layers = 21

    do_augment = False
    num_epochs = 1024 - 1

    model = "guided-filter-pixelshuffle"
    use_ECA = True
    pixelshuffle_ratio = 2


def guided_filter_l1_tanh_pixelshuffle_gca():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-gca"

    batch_size = 2
    do_augment = True

    model = "guided-filter-pixelshuffle-gca"
    pixelshuffle_ratio = 2

    num_threads = batch_size * 2
    log_interval = 25

    val_test_epoch_interval = 6


def guided_filter_l1_tanh_pixelshuffle_gca_sim():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-gca-sim"

    batch_size = 2
    do_augment = True
    num_epochs = 16 - 1

    model = "guided-filter-pixelshuffle-gca"
    pixelshuffle_ratio = 2

    num_threads = batch_size * 2
    log_interval = 25

    val_test_epoch_interval = 1
    save_copy_every_epochs = 6

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

    train_source_dir = image_dir / "Sim_train" / "POLED"
    train_target_dir = image_dir / "Sim_train" / "Glass"

    val_source_dir = image_dir / "Sim_val" / "POLED"
    val_target_dir = image_dir / "Sim_val" / "Glass"

    test_source_dir = None


def guided_filter_l1_tanh_pixelshuffle_gca_sim_actual():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-gca-sim-actual"

    batch_size = 2
    do_augment = True

    num_epochs = 64 - 1
    model = "guided-filter-pixelshuffle-gca"
    pixelshuffle_ratio = 2

    num_threads = batch_size * 2
    log_interval = 25

    val_test_epoch_interval = 6


def guided_filter_l1_tanh_pixelshuffle_gca_5x5():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-gca-5x5"

    batch_size = 2
    do_augment = True

    # Finetune it
    num_epochs = 64 - 1

    model = "guided-filter-pixelshuffle-gca"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 5
    guided_map_channels = 24

    num_threads = batch_size * 2
    log_interval = 25

    val_test_epoch_interval = 6


def guided_filter_l1_tanh_pixelshuffle_gca_5x5_improved():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-gca-5x5-improved-ssim"

    batch_size = 2
    do_augment = True
    num_epochs = 1024 - 1

    # Model args
    model = "guided-filter-pixelshuffle-gca-improved"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 5
    guided_map_channels = 24

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6

    # Cosine annealing
    T_0 = 64
    T_mult = 2


def guided_filter_l1_tanh_pixelshuffle_sim():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-sim"

    batch_size = 3
    CAN_layers = 21

    do_augment = True

    model = "guided-filter-pixelshuffle"
    pixelshuffle_ratio = 2

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

    batch_size = 4
    CAN_layers = 21

    do_augment = True

    finetune = True
    num_epochs = 128 - 1

    model = "guided-filter-pixelshuffle"
    pixelshuffle_ratio = 2


def guided_filter_l1_tanh_pixelshuffle_forward_glass():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-forward-glass-conv"

    CAN_layers = 21
    do_augment = True

    model = "guided-filter"

    batch_size = 3
    num_epochs = 127 - 1

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

    train_source_dir = image_dir / "DIV2K" / "GT_train_aligned"
    train_target_dir = image_dir / "Poled" / "HQ"

    val_source_dir = None
    val_target_dir = None

    # test_source_dir = image_dir / "Sim_train" / "GT"
    test_source_dir = image_dir / "Sim_val" / "GT"


def guided_filter_l1_tanh_pixelshuffle_forward_poled():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-forward-poled-conv"

    batch_size = 3
    CAN_layers = 15
    do_augment = True

    model = "guided-filter"

    num_epochs = 127 - 1

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

    train_source_dir = image_dir / "DIV2K" / "GT_train_aligned"
    train_target_dir = image_dir / "Poled" / "LQ"

    val_source_dir = None
    val_target_dir = None

    # test_source_dir = image_dir / "Sim_train" / "GT"
    test_source_dir = image_dir / "Sim_val" / "GT"


def guided_filter_l1_tanh_pixelshuffle_toled():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-toled"

    batch_size = 3
    CAN_layers = 21

    do_augment = False
    num_epochs = 1024 - 1

    model = "guided-filter-pixelshuffle"
    pixelshuffle_ratio = 2

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
    guided_filter_l1_tanh_pixelshuffle_5x5,
    guided_filter_l1_tanh_pixelshuffle_5x5_ms_ssim,
    guided_filter_l1_tanh_pixelshuffle_sim,
    guided_filter_l1_tanh_pixelshuffle_siren,
    guided_filter_l1_tanh_pixelshuffle_eca,
    guided_filter_l1_tanh_pixelshuffle_gca,
    guided_filter_l1_tanh_pixelshuffle_gca_sim,
    guided_filter_l1_tanh_pixelshuffle_gca_sim_actual,
    guided_filter_l1_tanh_pixelshuffle_gca_5x5,
    guided_filter_l1_tanh_pixelshuffle_gca_5x5_improved,
    guided_filter_l1_tanh_pixelshuffle_augment,
    guided_filter_l1_tanh_pixelshuffle_forward_glass,
    guided_filter_l1_tanh_pixelshuffle_forward_poled,
    guided_filter_l1_tanh_pixelshuffle_toled,
]


def initialise(ex):
    ex.config(base_config)
    for named_config in named_configs:
        ex.named_config(named_config)
    return ex
