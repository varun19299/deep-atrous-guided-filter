"""
Val Script for Phase/Amp mask
"""
# Libraries
import torch
import streamlit as st
from sacred import Experiment
from tqdm import tqdm
import numpy as np
import logging
import cv2

# Torch Libs
from torch.nn import functional as F

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
# Modules
from dataloader import get_dataloaders
from utils.dir_helper import dir_init
from utils.tupperware import tupperware
from models import get_model
from dataloader import PhaseMaskTestDataset, PhaseMaskADMMTestDataset

# Typing
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.typing_alias import *

# Import all configs
from config import *

# Train helpers
from utils.ops import unpixel_shuffle
from utils.train_helper import set_device, load_models

# Experiment, add any observers by command line
ex = Experiment("streamlit", interactive=True)
ex = initialise(ex)

# To prevent "RuntimeError: received 0 items of ancdata"
torch.multiprocessing.set_sharing_strategy("file_system")

# Create header
st.sidebar.markdown(r"# Model Configuration")
config_dict = {
    "ours": ours_fft_1280_1408_learn_1280_1408_meas_1280_1408_config,
    "ours-sim": ours_fft_1280_1408_learn_1280_1408_meas_1280_1408_simulated_config,
    "ours-finetuned": ours_fft_1280_1408_learn_1280_1408_meas_1280_1408_finetune_dualcam_1cap_config,
    "ours-finetuned-crop-608": ours_fft_1280_1408_learn_1280_1408_meas_608_864_finetune_dualcam_1cap_config,
    "ours-finetuned-crop-990": ours_fft_1280_1408_learn_1280_1408_meas_990_1254_finetune_dualcam_1cap_config,
    "naive": naive_fft_1280_1408_learn_1280_1408_meas_1280_1408_config,
    "naive-sim": naive_fft_1280_1408_learn_1280_1408_meas_1280_1408_simulated_config,
    "le-admm": le_admm_fft_1280_1408_learn_1280_1408_meas_1280_1408_config,
    "le-admm-sim": le_admm_fft_1280_1408_learn_1280_1408_meas_1280_1408_simulated_config,
    "ours-crop-608": ours_fft_1280_1408_learn_1280_1408_meas_608_864_config,
    "ours-crop-608-sim": ours_fft_1280_1408_learn_1280_1408_meas_608_864_simulated_config,
    "ours-crop-990": ours_fft_1280_1408_learn_1280_1408_meas_990_1254_config,
    "naive-crop-608": naive_fft_1280_1408_learn_1280_1408_meas_608_864_config,
    "naive-crop-608-sim": naive_fft_1280_1408_learn_1280_1408_meas_608_864_simulated_config,
    "le-admm-crop-608": le_admm_fft_1280_1408_learn_1280_1408_meas_608_864_config,
    "le-admm-crop-608-sim": le_admm_fft_1280_1408_learn_1280_1408_meas_608_864_simulated_config,
}

exp = st.sidebar.selectbox("Experiment", tuple(config_dict.keys()))
exp_config = config_dict[exp]

# Choose config
ex.config(exp_config)


@ex.main
def main():
    pass


if __name__ == "__main__":
    run_obj = ex.run_commandline()

    args = tupperware(run_obj.config)
    args.batch_size = 1

    # Set device, init dirs
    device, source_device = set_device(args)

    # Choose output directory
    output_dir = st.sidebar.text_input("Output Directory", "dumps")
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Choose fft gain
    source_number = st.sidebar.number_input(
        "Source Gain Box", min_value=0.0, max_value=5.0, value=1.0
    )
    source_gain = st.sidebar.slider("Source Gain", 0.0, 5.0, source_number)
    fft_number = st.sidebar.number_input(
        "FFT Gain Box", min_value=0.0, max_value=5.0, value=1.0
    )
    fft_gain = st.sidebar.slider("FFT Gain", 0.0, 5.0, fft_number)

    # Colour correction
    red_gain = st.sidebar.number_input(
        "R Gain", min_value=0.0, max_value=5.0, value=1.0
    )
    green_gain = st.sidebar.number_input(
        "G Gain", min_value=0.0, max_value=5.0, value=1.0
    )
    blue_gain = st.sidebar.number_input(
        "B Gain", min_value=0.0, max_value=5.0, value=1.0
    )
    colour_gain_vec = (
        torch.tensor([red_gain, green_gain, blue_gain]).unsqueeze(0).unsqueeze(0)
    )

    # Choose apply test gain
    args.test_apply_gain = st.sidebar.checkbox("Apply test gain (ie... 400/img.max())")

    # Admm vs FFT
    if ("admm" in args.exp_name) and ("pixelshuffle" in args.model):
        is_admm_pixel_shuffle = True
        is_admm = True
        is_pixelshuffle = True

    elif ("admm" in args.exp_name) and not ("pixelshuffle" in args.model):
        is_admm = True
        is_admm_pixel_shuffle = False
        is_pixelshuffle = False

    elif "pixelshuffle" in args.model:
        is_pixelshuffle = True
        is_admm = False
        is_admm_pixel_shuffle = False

    # Choose output filename
    outname = st.sidebar.text_input("Output Name (without extension)", "trial")
    output_path = output_dir / outname
    st.sidebar.markdown("### Recons name")
    st.sidebar.markdown(f"out_{args.exp_name}_{outname}.png")

    st.sidebar.markdown("### Intermediate name")
    fft_prefix = "fft" if not is_admm else "admm"
    st.sidebar.markdown(f"{fft_prefix}_{args.exp_name}_{outname}.png")

    # Model
    G, FFT, _ = get_model.model(args)
    G = G.to(device)
    FFT = FFT.to(device)

    st.sidebar.markdown(
        f"""### Is ADMM 
{is_admm}"""
    )
    st.sidebar.markdown(
        f"""### Is Pixelshuffle
{is_pixelshuffle}"""
    )

    dataset = PhaseMaskTestDataset if not is_admm else PhaseMaskADMMTestDataset
    dataset = dataset(args)
    img_loader = dataset._img_load

    # Load Models
    (G, FFT, _), _, global_step, start_epoch, loss = load_models(
        G,
        FFT,
        D=None,
        g_optimizer=None,
        fft_optimizer=None,
        d_optimizer=None,
        args=args,
        tag=args.inference_mode,
    )

    logging.info(
        f"Loaded experiment {args.exp_name}, dataset {args.dataset_name}, trained for {start_epoch} epochs."
    )

    # Upload file
    uploaded_file = st.file_uploader("Choose a png file", type="png")

    if uploaded_file:
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), dtype="int8"), -1)
        source = img_loader(raw=img).float()

        if not is_admm:
            source_rgb = np.zeros_like(source)[:3]
            source_rgb[0] = source[0]
            source_rgb[1] = 0.5 * (source[1] + source[2])
            source_rgb[2] = source[3]
        else:
            source_rgb = source

        source_rgb = torch.tensor(source_rgb).permute(1, 2, 0)
        source_rgb = (source_rgb - source_rgb.min()) / (
            source_rgb.max() - source_rgb.min()
        )
        if args.use_mask:
            source_rgb = source_rgb * np.expand_dims(np.array(FFT.mask), axis=2)
        source_rgb[source_rgb <= 1e-7] = 0.0

        st.markdown("## Source Image")
        st.image(source_rgb.numpy(), width=400)

        # Batch size 1
        source = source.unsqueeze(0).to(device)

        with st.spinner("Running model"):
            with torch.no_grad():
                G.eval()
                FFT.eval()

                # Check for pixelshuffle
                if is_admm_pixel_shuffle:
                    fft_output = FFT(source * source_gain)
                    fft_output = F.interpolate(
                        fft_output, scale_factor=4, mode="nearest"
                    )
                    fft_output_16_channel = unpixel_shuffle(
                        fft_output, args.pixelshuffle_ratio
                    )
                    output_12_channel = G(fft_output_16_channel * fft_gain)[-1]
                    output = F.pixel_shuffle(output_12_channel, args.pixelshuffle_ratio)

                elif is_pixelshuffle:
                    fft_output = FFT(source * source_gain)
                    fft_output_16_channel = unpixel_shuffle(
                        fft_output, args.pixelshuffle_ratio
                    )
                    output_12_channel = G(fft_output_16_channel * fft_gain)[-1]
                    output = F.pixel_shuffle(output_12_channel, args.pixelshuffle_ratio)

                else:
                    fft_output = FFT(source * source_gain)
                    output = G(fft_output * fft_gain)[-1]

                fft_output = fft_output[-1]
                fft_output = fft_output.cpu().detach()

                if not is_admm:
                    fft_output_vis_rgb = torch.zeros_like(fft_output)[:3]
                    fft_output_vis_rgb[0] = fft_output[0]
                    fft_output_vis_rgb[1] = (fft_output[1] + fft_output[2]) * 0.5
                    fft_output[2] = fft_output[3]
                else:
                    fft_output_vis_rgb = fft_output

                fft_output_vis_rgb = (fft_output_vis_rgb - fft_output_vis_rgb.min()) / (
                    fft_output_vis_rgb.max() - fft_output_vis_rgb.min()
                )

                fft_output_vis_rgb = fft_output_vis_rgb.permute(1, 2, 0).numpy()

                # Plot intermediate
                st.markdown("## Intermediate")
                st.image(fft_output_vis_rgb, width=400)

                # Plot output
                st.markdown("## Output Reconstruction")

                output_numpy = (
                    output[0].mul(0.5).add(0.5).permute(1, 2, 0).cpu().detach().numpy()
                )
                # HWC
                output_numpy = output_numpy * colour_gain_vec.numpy()

                st.image(output_numpy, width=400)

                if st.button("Save recons"):

                    path_fft = (
                        output_dir / f"{fft_prefix}_{args.exp_name}_{outname}.png"
                    )
                    path_output = output_dir / f"out_{args.exp_name}_{outname}.png"

                    cv2.imwrite(
                        str(path_output),
                        (output_numpy[:, :, ::-1] * 255.0).astype(np.int),
                    )
                    cv2.imwrite(
                        str(path_fft),
                        (fft_output_vis_rgb[:, :, ::-1] * 255.0).astype(np.int),
                    )
                    st.balloons()
                    st.success("Saved!")
