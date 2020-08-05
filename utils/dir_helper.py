"""
Helper to create directories
"""

import logging


def dir_init(args, is_local_rank_0: bool = True):
    """
    Creates paths for 

    : ckpt dir
    : run dir
    """
    if is_local_rank_0:
        logging.info("Initialising folders ...")

        for dir in [args.ckpt_dir, args.run_dir]:
            if not dir.is_dir():
                logging.info(f"Creating {dir.resolve()}")
                dir.mkdir(parents=True, exist_ok=True)
