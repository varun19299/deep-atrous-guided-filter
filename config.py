from pathlib import Path
import torch

from ablation_config import ablative_configs
from utils.self_ensemble import ensemble_ops


def base_config():
    exp_name = "ours"

    # ---------------------------------------------------------------------------- #
    # Directories
    # ---------------------------------------------------------------------------- #

    image_dir = Path("data")
    output_dir = Path("outputs") / exp_name
    ckpt_dir = Path("ckpts")  # Checkpoints saved to ckpt_dir / exp_name
    run_dir = Path("runs")  # Runs saved to run_dir / exp_name

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #
    use_source_npy = False  # Use it to load source images directly as CHW
    train_source_dir = image_dir / "Poled_train" / "LQ"
    train_target_dir = image_dir / "Poled_train" / "HQ"

    val_source_dir = image_dir / "Poled_val" / "LQ"
    val_target_dir = image_dir / "Poled_val" / "HQ"

    test_source_dir = image_dir / "Poled_test" / "LQ"

    static_val_image = "1.png"
    static_test_image = "1.png"

    if use_source_npy:
        static_val_image = "1.npy"
        static_test_image = "1.npy"

    image_height = 1024
    image_width = 2048

    batch_size = 8
    num_threads = batch_size  # parallel workers
    train_random_patch = False  # extract patches

    # augment
    do_augment = True

    # crop patches
    use_patches = False
    crop_height = None
    crop_width = None

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
    T_0 = 64
    T_mult = 2

    # Step lr
    step_size = 2

    # saving models
    save_filename_G = "model.pth"
    save_filename_D = "D.pth"

    save_filename_latest_G = "model_latest.pth"
    save_filename_latest_D = "D_latest.pth"

    # save a copy of weights every x epochs
    save_copy_every_epochs = 64

    # For model ensembling
    save_num_snapshots = 8

    # the number of iterations (default: 10) to print at
    log_interval = 25

    # run val or test only every x epochs
    val_test_epoch_interval = 10

    # ----------------------------------------------------------------------------  #
    # Val / Test Configs
    # ---------------------------------------------------------------------------- #

    # Self ensemble
    self_ensemble = False
    num_ensemble = len(ensemble_ops) + 1
    save_ensemble_channels = False
    save_train = False

    if save_ensemble_channels:
        self_ensemble = True

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

    pixelshuffle_ratio = 1
    use_residual = True
    use_gated = True
    use_FFA = True
    use_ECA = False
    use_atrous = True
    use_smooth_atrous = True

    # Guided map
    guided_map_kernel_size = 1
    guided_map_channels = 16
    guided_map_is_atrous_residual = False

    # Normalisation
    norm_layer = "adaptive-instance"
    # group norm with 8 num_groups
    assert norm_layer in [
        "adaptive-instance",
        "adaptive-batch",
        "instance",
        "batch",
        "group",
        "none",  # No normalization
    ]

    # Discriminator
    gan_type = "NSGAN"
    assert gan_type in ["NSGAN", "RAGAN"]
    use_patch_gan = False
    use_spectral_norm = False
    disc_normaliser = "group_norm"
    assert disc_normaliser in [
        "batch_norm",
        "instance_norm",
        "group_norm",
        "layer_norm",
    ]
    disc_num_groups = 8 if disc_normaliser == "group_norm" else None

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

    # Cloud TPU
    tpu_distributed = False


def dgf_poled():
    batch_size = 3
    num_epochs = 448

    CAN_layers = 9

    exp_name = "dgf-poled"
    model = "guided-filter"


def dgf_poled_pixelshuffle():
    batch_size = 4
    num_epochs = 960

    CAN_layers = 9

    exp_name = "dgf-poled-pixelshuffle"
    model = "guided-filter-pixelshuffle"
    pixelshuffle_ratio = 2


def dgf_toled():
    batch_size = 3
    num_epochs = 448
    exp_name = "dgf-toled"

    CAN_layers = 9

    model = "guided-filter"

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    image_dir = Path("data")
    train_source_dir = image_dir / "Toled_train" / "LQ"
    train_target_dir = image_dir / "Toled_train" / "HQ"

    val_source_dir = image_dir / "Toled_val" / "LQ"
    val_target_dir = image_dir / "Toled_val" / "HQ"

    test_source_dir = image_dir / "Toled_test" / "LQ"


def dgf_toled_pixelshuffle():
    batch_size = 3
    num_epochs = 960
    exp_name = "dgf-toled-pixelshuffle"

    CAN_layers = 9

    model = "guided-filter-pixelshuffle"
    pixelshuffle_ratio = 2

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    image_dir = Path("data")
    train_source_dir = image_dir / "Toled_train" / "LQ"
    train_target_dir = image_dir / "Toled_train" / "HQ"

    val_source_dir = image_dir / "Toled_val" / "LQ"
    val_target_dir = image_dir / "Toled_val" / "HQ"

    test_source_dir = image_dir / "Toled_test" / "LQ"


def FFA_poled():
    batch_size = 1
    num_epochs = 960
    do_augment = True
    num_threads = batch_size * 2

    exp_name = "FFA-deeper-poled"
    model = "FFA"

    use_patches = True  # Turn this off while using val.py
    crop_height = 256
    crop_width = 512


def FFA_toled():
    batch_size = 2
    num_epochs = 960
    do_augment = True
    num_threads = batch_size * 2

    exp_name = "FFA-deeper-toled"
    model = "FFA"

    use_patches = True  # Turn this off while using val.py
    crop_height = 256
    crop_width = 512

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    image_dir = Path("data")
    train_source_dir = image_dir / "Toled_train" / "LQ"
    train_target_dir = image_dir / "Toled_train" / "HQ"

    val_source_dir = image_dir / "Toled_val" / "LQ"
    val_target_dir = image_dir / "Toled_val" / "HQ"

    test_source_dir = image_dir / "Toled_test" / "LQ"


def unet_poled():
    batch_size = 10
    num_epochs = 448
    do_augment = True
    num_threads = batch_size * 2

    exp_name = "unet-poled"
    model = "Unet"

    use_patches = True  # Turn this off while using val.py
    crop_height = 256
    crop_width = 512


def unet_toled():
    batch_size = 10
    num_epochs = 448
    do_augment = True
    num_threads = batch_size * 2

    exp_name = "unet-toled"
    model = "Unet"

    use_patches = True  # Turn this off while using val.py
    crop_height = 256
    crop_width = 512

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    image_dir = Path("data")
    train_source_dir = image_dir / "Toled_train" / "LQ"
    train_target_dir = image_dir / "Toled_train" / "HQ"

    val_source_dir = image_dir / "Toled_val" / "LQ"
    val_target_dir = image_dir / "Toled_val" / "HQ"

    test_source_dir = image_dir / "Toled_test" / "LQ"


def guided_filter_l1_tanh_pixelshuffle_gca_5x5_atrous():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-gca-5x5-atrous-deeper"

    batch_size = 1
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

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def guided_filter_l1_tanh_pixelshuffle_gca_5x5_atrous_sim():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-gca-5x5-atrous-deeper-sim"

    batch_size = 1
    do_augment = True
    num_epochs = 16 + 32 + 64

    # Model args
    model = "guided-filter-pixelshuffle-gca-atrous"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 5
    guided_map_channels = 24

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #
    image_dir = Path("data")
    train_source_dir = image_dir / "Sim_train" / "POLED"
    train_target_dir = image_dir / "Sim_train" / "Glass"

    val_source_dir = image_dir / "Sim_val" / "POLED"
    val_target_dir = image_dir / "Sim_val" / "Glass"

    test_source_dir = None

    # Cosine annealing
    T_0 = 64
    T_mult = 1

    learning_rate = 3e-4


def guided_filter_l1_tanh_pixelshuffle_gca_5x5_atrous_sim_actual():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-gca-5x5-atrous-deeper-sim-actual"

    batch_size = 1
    do_augment = True
    num_epochs = 448

    # Model args
    model = "guided-filter-pixelshuffle-gca-atrous"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 5
    guided_map_channels = 24

    num_threads = 8
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4

    image_height = 1024
    image_width = 2048


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

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    image_dir = Path("data")
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

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #
    image_dir = Path("data")
    train_source_dir = image_dir / "Sim_train" / "POLED"
    train_target_dir = image_dir / "Sim_train" / "Glass"

    val_source_dir = image_dir / "Sim_val" / "POLED"
    val_target_dir = image_dir / "Sim_val" / "Glass"

    test_source_dir = None

    # Cosine annealing
    T_0 = 16
    T_mult = 2

    learning_rate = 3e-4


def guided_filter_l1_tanh_pixelshuffle_gca_5x5_improved_FFA_sim_actual():
    exp_name = "guided-filter-l1-tanh-pixelshuffle-gca-5x5-improved-FFA-sim-actual"

    batch_size = 2
    do_augment = True
    num_epochs = 64 + 128 + 256

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

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    image_dir = Path("data")
    train_source_dir = image_dir / "DIV2K" / "GT_train_aligned"
    train_target_dir = image_dir / "Poled_train" / "HQ"

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

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    image_dir = Path("data")
    train_source_dir = image_dir / "DIV2K" / "GT_train_aligned"
    train_target_dir = image_dir / "Poled_train" / "LQ"

    val_source_dir = None
    val_target_dir = None

    test_source_dir = image_dir / "Sim_train" / "GT"
    # test_source_dir = image_dir / "Sim_val" / "GT"

    # Loss
    lambda_image = 0.0  # l1
    lambda_CoBi_RGB = 1.0


def guided_filter_l1_tanh_pixelshuffle_forward_toled():
    exp_name = "toled-guided-filter-l1-tanh-pixelshuffle-forward-poled-contextual-patch-8-stride-8"

    batch_size = 3
    CAN_layers = 7
    do_augment = True
    save_copy_every_epochs = 64

    model = "guided-filter-pixelshuffle"
    pixelshuffle_ratio = 2

    num_epochs = 192

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    image_dir = Path("data")
    train_source_dir = image_dir / "DIV2K" / "GT_train_aligned"
    train_target_dir = image_dir / "Toled_train" / "LQ"

    val_source_dir = None
    val_target_dir = None

    # test_source_dir = image_dir / "Sim_train" / "GT"
    test_source_dir = image_dir / "Sim_val" / "GT"

    # Loss
    lambda_image = 0.0  # l1
    lambda_CoBi_RGB = 1.0


def aug_stage_2():
    exp_name = "aug_stage_2"

    batch_size = 1
    do_augment = False
    num_epochs = 448

    # Data
    use_source_npy = True
    image_dir = Path("data")
    output_dir = Path("outputs")
    train_source_dir = (
        output_dir
        / "guided-filter-l1-tanh-pixelshuffle-gca-5x5-atrous-deeper-sim-actual"
        / "train_latest_epoch_447_self_ensemble"
    )
    train_target_dir = image_dir / "Poled_train" / "HQ"

    val_source_dir = None
    val_target_dir = None

    test_source_dir = (
        output_dir
        / "guided-filter-l1-tanh-pixelshuffle-gca-5x5-atrous-deeper-sim-actual"
        / "test_latest_epoch_447_self_ensemble"
    )

    # Model args
    model = "aug-stage-2"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16
    guided_map_is_atrous_residual = True

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def final_poled():
    exp_name = "final-poled"

    batch_size = 1
    do_augment = True
    num_epochs = 960

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16
    guided_map_is_atrous_residual = True

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def final_poled_sim():
    exp_name = "final-poled-sim"

    batch_size = 1
    do_augment = True
    num_epochs = 16 + 32 + 64

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16
    guided_map_is_atrous_residual = True

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 3
    save_copy_every_epochs = 16

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #
    image_dir = Path("data")
    train_source_dir = image_dir / "Sim_train" / "POLED"
    train_target_dir = image_dir / "Sim_train" / "Glass"

    val_source_dir = image_dir / "Sim_val" / "POLED"
    val_target_dir = image_dir / "Sim_val" / "Glass"

    test_source_dir = None

    # Cosine annealing
    T_0 = 16
    T_mult = 2

    learning_rate = 3e-4


def final_poled_sim_actual():
    exp_name = "final-poled-sim-actual"

    batch_size = 1
    do_augment = True
    num_epochs = 960

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16
    guided_map_is_atrous_residual = True

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def final_poled_sim_actual_aug():
    exp_name = "final-poled-sim-actual-aug"

    batch_size = 1
    do_augment = False
    num_epochs = 960

    # Data
    use_source_npy = True
    image_dir = Path("data")
    train_source_dir = (
        Path("outputs") / "final-poled-sim-actual" / "train_latest_epoch_447_self_ensemble"
    )
    train_target_dir = image_dir / "Poled_train" / "HQ"

    val_source_dir = (
        Path("outputs") / "final-poled-sim-actual" / "val_latest_epoch_447_self_ensemble"
    )
    val_target_dir = image_dir / "Poled_val" / "HQ"

    test_source_dir = (
        Path("outputs") / "final-poled-sim-actual" / "test_latest_epoch_447_self_ensemble"
    )

    output_dir = Path("outputs") / exp_name
    # Model args
    model = "aug-stage-2"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16
    guided_map_is_atrous_residual = True

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64
    use_FFA = False
    use_ECA = True

    # # Loss
    # lambda_image = 1 - 0.84
    # lambda_ms_ssim = 0.84

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def final_toled():
    exp_name = "final-toled"

    batch_size = 1
    do_augment = True
    num_epochs = 960

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16
    guided_map_is_atrous_residual = True

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    image_dir = Path("data")
    train_source_dir = image_dir / "Toled_train" / "LQ"
    train_target_dir = image_dir / "Toled_train" / "HQ"

    val_source_dir = image_dir / "Toled_val" / "LQ"
    val_target_dir = image_dir / "Toled_val" / "HQ"

    test_source_dir = image_dir / "Toled_test" / "LQ"


def final_toled_sim():
    exp_name = "final-toled-sim"

    batch_size = 1
    do_augment = True
    num_epochs = 16 + 32 + 64

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16
    guided_map_is_atrous_residual = True

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 16

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #
    image_dir = Path("data")
    train_source_dir = image_dir / "Sim_train" / "TOLED"
    train_target_dir = image_dir / "Sim_train" / "Glass"

    val_source_dir = image_dir / "Sim_val" / "TOLED"
    val_target_dir = image_dir / "Sim_val" / "Glass"

    test_source_dir = None

    # Cosine annealing
    T_0 = 16
    T_mult = 2

    learning_rate = 3e-4


def final_toled_sim_actual():
    exp_name = "final-toled-sim-actual"

    batch_size = 1
    do_augment = True
    num_epochs = 960

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16
    guided_map_is_atrous_residual = True

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 10
    save_copy_every_epochs = 64

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    image_dir = Path("data")
    train_source_dir = image_dir / "Toled_train" / "LQ"
    train_target_dir = image_dir / "Toled_train" / "HQ"

    val_source_dir = image_dir / "Toled_val" / "LQ"
    val_target_dir = image_dir / "Toled_val" / "HQ"

    test_source_dir = image_dir / "Toled_test" / "LQ"


def final_toled_sim_actual_aug():
    exp_name = "final-toled-sim-actual-aug"

    batch_size = 1
    do_augment = False
    num_epochs = 960

    # Data
    use_source_npy = True
    image_dir = Path("data")
    train_source_dir = (
        Path("outputs") / "final-toled-sim-actual" / "train_latest_epoch_447_self_ensemble"
    )
    train_target_dir = image_dir / "Toled_train" / "HQ"

    val_source_dir = (
        Path("outputs") / "final-toled-sim-actual" / "val_latest_epoch_447_self_ensemble"
    )
    val_target_dir = image_dir / "Toled_val" / "HQ"

    test_source_dir = (
        Path("outputs") / "final-toled-sim-actual" / "test_latest_epoch_447_self_ensemble"
    )

    output_dir = Path("outputs") / exp_name

    # Model args
    model = "aug-stage-2"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16
    guided_map_is_atrous_residual = True
    use_FFA = False
    use_ECA = True

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    # Loss
    # lambda_image = 1 - 0.84
    # lambda_ms_ssim = 0.84

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


named_configs = [
    dgf_poled,
    dgf_poled_pixelshuffle,
    dgf_toled,
    dgf_toled_pixelshuffle,
    FFA_poled,
    FFA_toled,
    unet_poled,
    unet_toled,
    guided_filter_l1_tanh_pixelshuffle_gca_5x5_atrous,
    guided_filter_l1_tanh_pixelshuffle_gca_5x5_atrous_sim,
    guided_filter_l1_tanh_pixelshuffle_gca_5x5_atrous_sim_actual,
    guided_filter_l1_tanh_pixelshuffle_gca_5x5_improved,
    guided_filter_l1_tanh_pixelshuffle_gca_5x5_improved_ms_ssim_perceptual,
    guided_filter_l1_tanh_pixelshuffle_gca_5x5_improved_sim,
    guided_filter_l1_tanh_pixelshuffle_gca_5x5_improved_sim_actual,
    guided_filter_l1_tanh_pixelshuffle_gca_5x5_improved_FFA,
    guided_filter_l1_tanh_pixelshuffle_gca_5x5_improved_FFA_sim,
    guided_filter_l1_tanh_pixelshuffle_gca_5x5_improved_FFA_sim_actual,
    guided_filter_l1_tanh_pixelshuffle_forward_glass,
    guided_filter_l1_tanh_pixelshuffle_forward_poled,
    guided_filter_l1_tanh_pixelshuffle_forward_toled,
    aug_stage_2,
    final_poled,
    final_poled_sim,
    final_poled_sim_actual,
    final_poled_sim_actual_aug,
    final_toled,
    final_toled_sim,
    final_toled_sim_actual,
    final_toled_sim_actual_aug,
]

named_configs += ablative_configs


def initialise(ex):
    ex.config(base_config)
    for named_config in named_configs:
        ex.named_config(named_config)
    return ex
