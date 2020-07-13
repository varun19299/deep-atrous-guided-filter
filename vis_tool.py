from sacred import Experiment
import cv2
from pathlib import Path
import numpy as np

from utils.tupperware import tupperware
from matplotlib import pyplot as plt

ex = Experiment("enlarged-crops")

colour_ll = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # RGB
colour_name = ["red", "green", "blue"]


def transform_img(img, roi_ll, name, inset_dir):
    img_h, img_w, _ = img.shape
    box_width = img_w // 3

    crop_ll = []

    for i, roi in enumerate(roi_ll):
        crop = img[
            int(roi[1]) : int(roi[1] + roi[3]), int(roi[0]) : int(roi[0] + roi[2])
        ].copy()

        # Resize crop to img_w/n_roi
        if i == len(roi) - 1 and img_w % len(roi):
            crop = cv2.resize(
                crop, (img_w // len(roi_ll) + img_w % len(roi), img_w // len(roi_ll))
            )
        else:
            crop = cv2.resize(crop, (img_w // len(roi_ll), img_w // len(roi_ll)))
        crop_h, crop_w, _ = crop.shape

        # Mark roi
        cv2.rectangle(
            img,
            (int(roi[0]), int(roi[1])),
            (int(roi[0] + roi[2]), int(roi[1] + roi[3])),
            colour_ll[i],
            15,
        )

        # Draw boundary for crop
        cv2.rectangle(crop, (0, 0), (crop_h, crop_w), colour_ll[i], 64)

        # Check size before and after
        # breakpoint()

        crop_ll.append(crop)

        cv2.imwrite(f"{inset_dir}/crop_{name}_{colour_name[i]}.png", crop)

    # Put in the insets
    out = np.zeros((img_h + crop_h, img_w, 3))
    out[:img_h] = img

    for i, crop in enumerate(crop_ll):
        crop_h, crop_w, _ = crop.shape
        out[img_h:, i * crop_w : (i + 1) * crop_w] = crop

    return out


@ex.automain
def main(_run):
    # Which dataset
    dataset = "poled"
    set = "val"

    data_dir = Path("../data/")
    output_dir = Path("../outputs/")
    out_name = "8.png"

    inset_dir = Path(f"../insets/{out_name.replace('.png','')}_{set}")
    inset_dir.mkdir(exist_ok=True, parents=True)

    meas_dir = data_dir / f"{dataset.capitalize()}_{set}" / "LQ"
    dgf_dir = output_dir / f"dgf-{dataset}" / f"{set}_latest_epoch_447"
    rnan_dir = output_dir / f"rnan-{dataset}" / f"{set}_latest_epoch_447"
    ours_dir = output_dir / f"final-{dataset}" / f"{set}_latest_epoch_959"
    gt_dir = data_dir / f"{dataset.capitalize()}_{set}" / "HQ"

    dir_ll = [meas_dir, dgf_dir, rnan_dir, ours_dir, gt_dir]

    img_h, img_w = 1024, 2048
    view_img_h, view_img_w = 192, 384
    view_t_img_h, view_t_img_w = 256, 384

    num_roi = 3

    # Method names
    ours_name = "ours"
    name_ll = ["meas", "dgf", "rnan", "ours", "gt"]
    path_ll = [folder / out_name for folder in dir_ll]
    position_ll = [(0, 0), (300, 0), (0, 400), (300, 400), (600, 0)]

    img_ll = {}
    img_transformed_ll = {}

    for img_path, name, position in zip(path_ll, name_ll, position_ll):
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        img_ll[name] = img

        cv2.namedWindow(name)
        cv2.moveWindow(name, position[0] - 1, position[1] - 1)
        cv2.imshow(name, cv2.resize(img, (view_img_w, view_img_h)))
        cv2.waitKey(1)

    while True:
        roi_ll = []
        img = img_ll[ours_name].copy()

        for i in range(num_roi):
            roi = cv2.selectROI(ours_name, img)

            if not (roi[3]):
                print("\n Invalid Crop")
                continue

            # Mark roi
            cv2.rectangle(
                img,
                (int(roi[0]), int(roi[1])),
                (int(roi[0] + roi[2]), int(roi[1] + roi[3])),
                colour_ll[i],
                2,
            )

            roi_ll.append(roi)

        for name in img_ll.keys():
            img_transformed_ll[name] = transform_img(
                img_ll[name], roi_ll, name, inset_dir
            )
            cv2.imshow(name, cv2.resize(img_ll[name], (view_img_w, view_img_h)))
            cv2.waitKey(1)

        if cv2.waitKey(0) & 0xFF == ord("p"):
            print("Quitting, not saving.")
            break

        if cv2.waitKey(0) & 0xFF == ord("q"):
            print("Saving crops.")

            for name in img_ll.keys():
                dump_name = f"{name}_{out_name}.png"
                cv2.imwrite(str(inset_dir / dump_name), img_transformed_ll[name])

                dump_name = f"marked_{name}_{out_name}.png"
                cv2.imwrite(str(inset_dir / dump_name), img_ll[name])

            break
