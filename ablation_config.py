from pathlib import Path
import torch


def ours_poled():
    exp_name = "ours-poled-512"

    batch_size = 1
    do_augment = True
    num_epochs = 448

    # Model args
    model = "guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 5
    guided_map_channels = 24

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

    batch_size = 1
    do_augment = True
    num_epochs = 448

    # Model args
    model = "guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 5
    guided_map_channels = 24

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

    batch_size = 1
    do_augment = True
    num_epochs = 448

    # Model args
    model = "guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 5
    guided_map_channels = 24
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

    batch_size = 1
    do_augment = True
    num_epochs = 448

    # Model args
    model = "guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 5
    guided_map_channels = 24
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

    batch_size = 1
    do_augment = True
    num_epochs = 448

    # Model args
    model = "guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 5
    guided_map_channels = 24
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

    batch_size = 1
    do_augment = True
    num_epochs = 448

    # Model args
    model = "guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 5
    guided_map_channels = 24
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

    batch_size = 1
    do_augment = True
    num_epochs = 448

    # Model args
    model = "guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 5
    guided_map_channels = 24
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

    batch_size = 1
    do_augment = True
    num_epochs = 448

    # Model args
    model = "guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 5
    guided_map_channels = 24
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

    batch_size = 1
    do_augment = True
    num_epochs = 448

    # Model args
    model = "guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 5
    guided_map_channels = 24
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

    batch_size = 1
    do_augment = True
    num_epochs = 448

    # Model args
    model = "guided-filter-pixelshuffle-gca-atrous-corrected"
    pixelshuffle_ratio = 2
    guided_map_kernel_size = 5
    guided_map_channels = 24
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

# Channel Attention

# Loss functions

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
]
