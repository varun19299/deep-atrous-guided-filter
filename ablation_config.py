from pathlib import Path
import torch


def ours_poled():
    exp_name = "ours-poled-512"

    batch_size = 4
    do_augment = True
    num_epochs = 448

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    image_height = 512
    image_width = 1024

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def ours_toled():
    exp_name = "ours-toled-512"

    batch_size = 4
    do_augment = True
    num_epochs = 448

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    image_height = 512
    image_width = 1024

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    image_dir = Path("data")
    train_source_dir = image_dir / "Toled_train" / "LQ"
    train_target_dir = image_dir / "Toled_train" / "HQ"

    val_source_dir = None
    val_target_dir = None

    test_source_dir = image_dir / "Toled_val" / "LQ"

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


# Smooth dialated convs
def ours_poled_normal_conv():
    exp_name = "ours-poled-512-normal-conv"

    batch_size = 4
    do_augment = True
    num_epochs = 448

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16
    use_atrous = False
    use_smooth_atrous = False

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    image_height = 512
    image_width = 1024

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def ours_poled_no_smooth():
    exp_name = "ours-poled-512-no-smooth"

    batch_size = 4
    do_augment = True
    num_epochs = 448

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16
    use_atrous = True
    use_smooth_atrous = False

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    image_height = 512
    image_width = 1024

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def ours_toled_normal_conv():
    exp_name = "ours-toled-512-normal-conv"

    batch_size = 4
    do_augment = True
    num_epochs = 448

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16
    use_atrous = False
    use_smooth_atrous = False

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    image_height = 512
    image_width = 1024

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    image_dir = Path("data")
    train_source_dir = image_dir / "Toled_train" / "LQ"
    train_target_dir = image_dir / "Toled_train" / "HQ"

    val_source_dir = None
    val_target_dir = None

    test_source_dir = image_dir / "Toled_val" / "LQ"

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def ours_toled_no_smooth():
    exp_name = "ours-toled-512-no-smooth"

    batch_size = 4
    do_augment = True
    num_epochs = 448

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16
    use_atrous = True
    use_smooth_atrous = False

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    image_height = 512
    image_width = 1024

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    image_dir = Path("data")
    train_source_dir = image_dir / "Toled_train" / "LQ"
    train_target_dir = image_dir / "Toled_train" / "HQ"

    val_source_dir = None
    val_target_dir = None

    test_source_dir = image_dir / "Toled_val" / "LQ"

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


# Residual and Gated
def ours_poled_no_residual_no_gated():
    exp_name = "ours-poled-512-no-residual-no-gated"

    batch_size = 4
    do_augment = True
    num_epochs = 448

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16
    use_residual = False
    use_gated = False

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    image_height = 512
    image_width = 1024

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def ours_poled_no_gated():
    exp_name = "ours-poled-512-no-gated"

    batch_size = 4
    do_augment = True
    num_epochs = 448

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16
    use_residual = True
    use_gated = False

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    image_height = 512
    image_width = 1024

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def ours_toled_no_residual_no_gated():
    exp_name = "ours-toled-512-no-residual-no-gated"

    batch_size = 4
    do_augment = True
    num_epochs = 448

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16
    use_residual = False
    use_gated = False

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    image_height = 512
    image_width = 1024

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    image_dir = Path("data")
    train_source_dir = image_dir / "Toled_train" / "LQ"
    train_target_dir = image_dir / "Toled_train" / "HQ"

    val_source_dir = None
    val_target_dir = None

    test_source_dir = image_dir / "Toled_val" / "LQ"

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def ours_toled_no_gated():
    exp_name = "ours-toled-512-no-gated"

    batch_size = 4
    do_augment = True
    num_epochs = 448

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16
    use_residual = True
    use_gated = False

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    image_height = 512
    image_width = 1024

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    image_dir = Path("data")
    train_source_dir = image_dir / "Toled_train" / "LQ"
    train_target_dir = image_dir / "Toled_train" / "HQ"

    val_source_dir = None
    val_target_dir = None

    test_source_dir = image_dir / "Toled_val" / "LQ"

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


# Normalisation


def ours_poled_no_norm():
    exp_name = "ours-poled-512-no-norm"

    batch_size = 4
    do_augment = True
    num_epochs = 448

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16
    norm_layer = "none"

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    image_height = 512
    image_width = 1024

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def ours_poled_batch_norm():
    exp_name = "ours-poled-512-batch-norm"

    batch_size = 4
    do_augment = True
    num_epochs = 448

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16
    norm_layer = "batch"

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    image_height = 512
    image_width = 1024

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def ours_poled_instance_norm():
    exp_name = "ours-poled-512-instance-norm"

    batch_size = 4
    do_augment = True
    num_epochs = 448

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16
    norm_layer = "instance"

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    image_height = 512
    image_width = 1024

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def ours_poled_adaptive_batch_norm():
    exp_name = "ours-poled-512-adaptive-batch-norm"

    batch_size = 4
    do_augment = True
    num_epochs = 448

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16
    norm_layer = "adaptive-batch"

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    image_height = 512
    image_width = 1024

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def ours_toled_no_norm():
    exp_name = "ours-toled-512-no-norm"

    batch_size = 4
    do_augment = True
    num_epochs = 448

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16
    norm_layer = "none"

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    image_height = 512
    image_width = 1024

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    image_dir = Path("data")
    train_source_dir = image_dir / "Toled_train" / "LQ"
    train_target_dir = image_dir / "Toled_train" / "HQ"

    val_source_dir = None
    val_target_dir = None

    test_source_dir = image_dir / "Toled_val" / "LQ"

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def ours_toled_batch_norm():
    exp_name = "ours-toled-512-batch-norm"

    batch_size = 4
    do_augment = True
    num_epochs = 448

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16
    norm_layer = "batch"

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    image_height = 512
    image_width = 1024

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    image_dir = Path("data")
    train_source_dir = image_dir / "Toled_train" / "LQ"
    train_target_dir = image_dir / "Toled_train" / "HQ"

    val_source_dir = None
    val_target_dir = None

    test_source_dir = image_dir / "Toled_val" / "LQ"

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def ours_toled_instance_norm():
    exp_name = "ours-toled-512-instance-norm"

    batch_size = 4
    do_augment = True
    num_epochs = 448

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16
    norm_layer = "instance"

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    image_height = 512
    image_width = 1024

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    image_dir = Path("data")
    train_source_dir = image_dir / "Toled_train" / "LQ"
    train_target_dir = image_dir / "Toled_train" / "HQ"

    val_source_dir = None
    val_target_dir = None

    test_source_dir = image_dir / "Toled_val" / "LQ"

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def ours_toled_adaptive_batch_norm():
    exp_name = "ours-toled-512-adaptive-batch-norm"

    batch_size = 4
    do_augment = True
    num_epochs = 448

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16
    norm_layer = "adaptive-batch"

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    image_height = 512
    image_width = 1024

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    image_dir = Path("data")
    train_source_dir = image_dir / "Toled_train" / "LQ"
    train_target_dir = image_dir / "Toled_train" / "HQ"

    val_source_dir = None
    val_target_dir = None

    test_source_dir = image_dir / "Toled_val" / "LQ"

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


# Channel Attention


def ours_poled_no_CA():
    exp_name = "ours-poled-512-no-CA"

    batch_size = 4
    do_augment = True
    num_epochs = 448

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16
    use_FFA = False
    use_ECA = False

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    image_height = 512
    image_width = 1024

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def ours_poled_ECA():
    exp_name = "ours-poled-512-ECA"

    batch_size = 4
    do_augment = True
    num_epochs = 448

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16
    use_FFA = False
    use_ECA = True

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    image_height = 512
    image_width = 1024

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def ours_toled_no_CA():
    exp_name = "ours-toled-512-no-CA"

    batch_size = 4
    do_augment = True
    num_epochs = 448

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16
    use_FFA = False
    use_ECA = False

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    image_height = 512
    image_width = 1024

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    image_dir = Path("data")
    train_source_dir = image_dir / "Toled_train" / "LQ"
    train_target_dir = image_dir / "Toled_train" / "HQ"

    val_source_dir = None
    val_target_dir = None

    test_source_dir = image_dir / "Toled_val" / "LQ"

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def ours_toled_ECA():
    exp_name = "ours-toled-512-ECA"

    batch_size = 4
    do_augment = True
    num_epochs = 448

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16
    use_FFA = False
    use_ECA = True

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    image_height = 512
    image_width = 1024

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    image_dir = Path("data")
    train_source_dir = image_dir / "Toled_train" / "LQ"
    train_target_dir = image_dir / "Toled_train" / "HQ"

    val_source_dir = None
    val_target_dir = None

    test_source_dir = image_dir / "Toled_val" / "LQ"

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


# Loss functions
def ours_poled_l1_ms_ssim():
    exp_name = "ours-poled-512-l1-ms-ssim"

    batch_size = 4
    do_augment = True
    num_epochs = 448

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    # Loss
    lambda_image = 1 - 0.84
    lambda_ms_ssim = 0.84

    image_height = 512
    image_width = 1024

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def ours_poled_l1_percep():
    exp_name = "ours-poled-512-l1-percep"

    batch_size = 4
    do_augment = True
    num_epochs = 448

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    # Loss
    lambda_perception = 1.2

    image_height = 512
    image_width = 1024

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def ours_poled_l1_percep_adv():
    exp_name = "ours-poled-512-l1-percep-adv"

    batch_size = 4
    do_augment = True
    num_epochs = 448

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    # Loss
    lambda_perception = 1.2
    lambda_adversarial = 0.6

    image_height = 512
    image_width = 1024

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def ours_toled_l1_ms_ssim():
    exp_name = "ours-toled-512-l1-ms-ssim"

    batch_size = 4
    do_augment = True
    num_epochs = 448

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    image_height = 512
    image_width = 1024

    # Loss
    lambda_image = 1 - 0.84
    lambda_ms_ssim = 0.84

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    image_dir = Path("data")
    train_source_dir = image_dir / "Toled_train" / "LQ"
    train_target_dir = image_dir / "Toled_train" / "HQ"

    val_source_dir = None
    val_target_dir = None

    test_source_dir = image_dir / "Toled_val" / "LQ"

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def ours_toled_l1_percep():
    exp_name = "ours-toled-512-l1-percep"

    batch_size = 4
    do_augment = True
    num_epochs = 448

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    image_height = 512
    image_width = 1024

    # Loss
    lambda_perception = 1.2

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    image_dir = Path("data")
    train_source_dir = image_dir / "Toled_train" / "LQ"
    train_target_dir = image_dir / "Toled_train" / "HQ"

    val_source_dir = None
    val_target_dir = None

    test_source_dir = image_dir / "Toled_val" / "LQ"

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


def ours_toled_l1_percep_adv():
    exp_name = "ours-toled-512-l1-percep-adv"

    batch_size = 4
    do_augment = True
    num_epochs = 448

    # Model args
    model = "atrous-guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 3
    guided_map_channels = 16

    num_threads = batch_size * 2
    log_interval = 25
    val_test_epoch_interval = 6
    save_copy_every_epochs = 64

    image_height = 512
    image_width = 1024

    # Loss
    lambda_perception = 1.2
    lambda_adversarial = 0.6

    # ---------------------------------------------------------------------------- #
    # Data
    # ---------------------------------------------------------------------------- #

    image_dir = Path("data")
    train_source_dir = image_dir / "Toled_train" / "LQ"
    train_target_dir = image_dir / "Toled_train" / "HQ"

    val_source_dir = None
    val_target_dir = None

    test_source_dir = image_dir / "Toled_val" / "LQ"

    # Cosine annealing
    T_0 = 64
    T_mult = 2

    learning_rate = 3e-4


ablative_configs = [
    ours_poled,
    ours_toled,
    # Smooth atrous ablative
    ours_poled_normal_conv,
    ours_poled_no_smooth,
    ours_toled_normal_conv,
    ours_toled_no_smooth,
    # Residual and Gated
    ours_poled_no_residual_no_gated,
    ours_poled_no_gated,
    ours_toled_no_residual_no_gated,
    ours_toled_no_gated,
    # normalisation
    ours_poled_no_norm,
    ours_poled_batch_norm,
    ours_poled_instance_norm,
    ours_poled_adaptive_batch_norm,
    ours_toled_no_norm,
    ours_toled_batch_norm,
    ours_toled_instance_norm,
    ours_toled_adaptive_batch_norm,
    # channel attention
    ours_poled_no_CA,
    ours_poled_ECA,
    ours_toled_no_CA,
    ours_toled_ECA,
    # loss functions
    ours_poled_l1_ms_ssim,
    ours_poled_l1_percep,
    ours_poled_l1_percep_adv,
    ours_toled_l1_ms_ssim,
    ours_toled_l1_percep,
    ours_toled_l1_percep_adv,
]
