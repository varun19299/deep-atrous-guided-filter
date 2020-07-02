from pathlib import Path
import torch


def base_config():
    exp_name = "ours"
    system = "CFI"
    assert system in ["CFI", "FPM", "Jarvis", "Varun", "Genesis"]

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

    elif system == "Genesis":
        image_dir = Path("data/")
        dump_dir = Path("/mnt/vol_b/udc/")

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
    train_random_patch = False  # extract patches

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
    lambda_CoBi_RGB = 0.0  # https://arxiv.org/pdf/1905.05169.pdf

    cobi_rgb_patch_size = 8
    cobi_rgb_stride = 8

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


def guided_filter_l1_tanh_pixelshuffle_gca_5x5_atrous():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-gca-5x5-atrous-increasing"

    batch_size = 2
    do_augment = True
    num_epochs = 960

    # Model args
    model = "guided-filter-pixelshuffle-gca-atrous"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 5
    guided_map_channels = 24

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    # image_height = 512
    # image_width = 1024

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def guided_filter_l1_tanh_pixelshuffle_gca_5x5_improved():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-gca-5x5-improved"

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
    save_copy_every_epochs = 64

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def guided_filter_l1_tanh_pixelshuffle_gca_5x5_improved_ms_ssim_perceptual():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-gca-5x5-improved-ms-ssim-perceptual"

    batch_size = 1
    do_augment = True
    num_epochs = 64 + 128 + 256

    # Model args
    model = "guided-filter-pixelshuffle-gca-improved"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 5
    guided_map_channels = 24

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    # Loss
    lambda_image = 1 - 0.84
    lambda_ms_ssim = 0.84
    lambda_perception = 1.2

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-5


def guided_filter_l1_tanh_pixelshuffle_gca_5x5_improved_sim():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-gca-5x5-improved-sim"

    batch_size = 2
    do_augment = True
    num_epochs = 64

    # Model args
    model = "guided-filter-pixelshuffle-gca-improved"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 5
    guided_map_channels = 24

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

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-5


def guided_filter_l1_tanh_pixelshuffle_gca_5x5_improved_sim_actual():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-gca-5x5-improved-sim-actual"

    batch_size = 2
    do_augment = True
    num_epochs = 128

    # Model args
    model = "guided-filter-pixelshuffle-gca-improved"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 5
    guided_map_channels = 24

    num_threads = batch_size * 2
    log_interval = 25

    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    # Cosine annealing
    T_0 = 128
    T_mult = 2

    learning_rate = 3e-5


def guided_filter_l1_tanh_pixelshuffle_gca_5x5_improved_FFA():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-gca-5x5-improved-FFA"

    batch_size = 2
    do_augment = True
    num_epochs = 960

    # Model args
    model = "guided-filter-pixelshuffle-gca-improved-FFA"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 5
    guided_map_channels = 24

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def guided_filter_l1_tanh_pixelshuffle_gca_5x5_improved_FFA_sim():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-gca-5x5-improved-FFA-sim"

    batch_size = 2
    do_augment = True
    num_epochs = 16 + 32 + 64

    # Model args
    model = "guided-filter-pixelshuffle-gca-improved-FFA"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 5
    guided_map_channels = 24

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 16

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

    # Cosine annealing
    T_0 = 16
    T_mult = 2

    learning_rate = 3e-4


def guided_filter_l1_tanh_pixelshuffle_gca_5x5_improved_ECA():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-gca-5x5-improved-ECA"

    batch_size = 2
    do_augment = True
    num_epochs = 960

    # Model args
    model = "guided-filter-pixelshuffle-gca-improved-FFA"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 5
    guided_map_channels = 24
    use_ECA = True

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def guided_filter_l1_tanh_pixelshuffle_gca_5x5_improved_FFA_deeper():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-gca-5x5-improved-FFA-deeper"

    batch_size = 1
    do_augment = True
    num_epochs = 960

    # Model args
    model = "guided-filter-pixelshuffle-gca-improved-FFA-deeper"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 5
    guided_map_channels = 24

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def guided_filter_l1_tanh_pixelshuffle_gca_5x5_improved_contextual():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-gca-5x5-improved-contextual"

    batch_size = 2
    do_augment = True
    num_epochs = 960 - 1
    learning_rate = 3e-5

    # Model args
    model = "guided-filter-pixelshuffle-gca-improved"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 5
    guided_map_channels = 24

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-5

    # Loss
    lambda_image = 0.0  # l1
    lambda_CoBi_RGB = 1.0

    cobi_rgb_patch_size = 8
    cobi_rgb_stride = 8


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
    exp_name = (
        "guided-filter-l1-tanh-pixelshuffle-forward-glass-contextual-patch-8-stride-8"
    )

    CAN_layers = 7
    do_augment = True

    model = "guided-filter-pixelshuffle"
    pixelshuffle_ratio = 2

    batch_size = 2
    num_epochs = 255 - 1

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

    # Data
    train_source_dir = image_dir / "DIV2K" / "GT_train_aligned"
    train_target_dir = image_dir / "Poled" / "HQ"

    val_source_dir = None
    val_target_dir = None

    test_source_dir = image_dir / "Sim_train" / "GT"
    # test_source_dir = image_dir / "Sim_val" / "GT"

    # Loss
    lambda_image = 0.0  # l1
    lambda_CoBi_RGB = 1.0


def guided_filter_l1_tanh_pixelshuffle_forward_poled():
    exp_name = (
        "guided-filter-l1-tanh-pixelshuffle-forward-poled-contextual-patch-8-stride-8"
    )

    batch_size = 3
    CAN_layers = 7
    do_augment = True

    model = "guided-filter-pixelshuffle"
    pixelshuffle_ratio = 2

    num_epochs = 255 - 1

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

    test_source_dir = image_dir / "Sim_train" / "GT"
    # test_source_dir = image_dir / "Sim_val" / "GT"

    # Loss
    lambda_image = 0.0  # l1
    lambda_CoBi_RGB = 1.0


named_configs = [
    guided_filter_l1_tanh,
    guided_filter_l1_tanh_pixelshuffle,
    guided_filter_l1_tanh_pixelshuffle_5x5,
    guided_filter_l1_tanh_pixelshuffle_5x5_ms_ssim,
    guided_filter_l1_tanh_pixelshuffle_gca_5x5_atrous,
    guided_filter_l1_tanh_pixelshuffle_gca_5x5_improved,
    guided_filter_l1_tanh_pixelshuffle_gca_5x5_improved_ms_ssim_perceptual,
    guided_filter_l1_tanh_pixelshuffle_gca_5x5_improved_sim,
    guided_filter_l1_tanh_pixelshuffle_gca_5x5_improved_sim_actual,
    guided_filter_l1_tanh_pixelshuffle_gca_5x5_improved_FFA,
    guided_filter_l1_tanh_pixelshuffle_gca_5x5_improved_FFA_sim,
    guided_filter_l1_tanh_pixelshuffle_gca_5x5_improved_ECA,
    guided_filter_l1_tanh_pixelshuffle_gca_5x5_improved_FFA_deeper,
    guided_filter_l1_tanh_pixelshuffle_gca_5x5_improved_contextual,
    guided_filter_l1_tanh_pixelshuffle_augment,
    guided_filter_l1_tanh_pixelshuffle_forward_glass,
    guided_filter_l1_tanh_pixelshuffle_forward_poled,
]


def initialise(ex):
    ex.config(base_config)
    for named_config in named_configs:
        ex.named_config(named_config)
    return ex
