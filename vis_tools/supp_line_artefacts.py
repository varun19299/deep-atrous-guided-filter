from sacred import Experiment
import cv2
from pathlib import Path
import numpy as np

from utils.tupperware import tupperware
from matplotlib import pyplot as plt

ex = Experiment("enlarged-crops")


def transform_img(img, roi, name, inset_dir):
    img_h, img_w, _ = img.shape
    box_width = img_h // 2

    crop = img[
        int(roi[1]) : int(roi[1] + roi[3]), int(roi[0]) : int(roi[0] + roi[2])
    ].copy()

    crop = cv2.resize(crop, (box_width, box_width))
    crop_h, crop_w, _ = crop.shape

    # Mark roi
    cv2.rectangle(
        img,
        (int(roi[0]), int(roi[1])),
        (int(roi[0] + roi[2]), int(roi[1] + roi[3])),
        (0, 255, 255),
        10,
    )

    cv2.imwrite(f"{inset_dir}/crop_{name}.png", crop)


@ex.automain
def main(_run):
    set = "val"

    data_dir = Path("../data/")
    output_dir = Path("../outputs/")
    out_name = "5.png"

    inset_dir = Path(
        f"../insets/supplementary/line_artefacts_{out_name.replace('.png','')}_{set}"
    )
    inset_dir.mkdir(exist_ok=True, parents=True)

    panet_poled_dir = output_dir / "panet-poled" / f"{set}_latest_epoch_105"
    panet_toled_dir = output_dir / "panet-toled" / f"{set}_latest_epoch_105"

    ffa_poled_dir = output_dir / "FFA-poled" / f"{set}_latest_epoch_1983"
    ffa_toled_dir = output_dir / "FFA-toled" / f"{set}_latest_epoch_959"

    unet_poled_dir = output_dir / "unet-poled" / f"{set}_latest_epoch_3599"
    unet_toled_dir = output_dir / "unet-toled" / f"{set}_latest_epoch_1799"

    ours_poled_dir = output_dir / "final-poled-sim-actual" / f"{set}_latest_epoch_959"
    ours_toled_dir = output_dir / "final-toled-sim-actual" / f"{set}_latest_epoch_959"

    gt_dir = data_dir / f"Poled_{set}" / "HQ"

    dir_ll = [
        panet_poled_dir,
        panet_toled_dir,
        ffa_poled_dir,
        ffa_toled_dir,
        unet_poled_dir,
        unet_toled_dir,
        ours_poled_dir,
        ours_toled_dir,
        gt_dir,
    ]

    img_h, img_w = 1024, 2048
    view_img_h, view_img_w = 192, 384

    # Method names
    ours_name = "unet-poled"
    name_ll = [
        "panet-poled",
        "panet-toled",
        "ffa-poled",
        "ffa-toled",
        "unet-poled",
        "unet-toled",
        "ours-poled",
        "ours-toled",
        "gt",
    ]
    path_ll = [folder / out_name for folder in dir_ll]
    position_ll = [
        (0, 0),
        (480, 0),
        (960, 0),
        (1440, 0),
        (0, 540),
        (480, 540),
        (960, 540),
        (1440, 540),
        (0, 1080),
    ][::-1]
    img_ll = {}
    img_transformed_ll = {}

    for img_path, name, position in zip(path_ll, name_ll, position_ll):
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (img_w, img_h))
        img_ll[name] = img

        cv2.namedWindow(name)
        cv2.moveWindow(name, position[0] - 1, position[1] - 1)
        cv2.imshow(name, cv2.resize(img, (view_img_w, view_img_h)))
        cv2.waitKey(1)

    while True:
        img = img_ll[ours_name].copy()
        roi = cv2.selectROI(ours_name, img)

        if not (roi[3]):
            print("\n Invalid Crop")
            continue

        # Mark roi
        cv2.rectangle(
            img,
            (int(roi[0]), int(roi[1])),
            (int(roi[0] + roi[2]), int(roi[1] + roi[3])),
            (0, 255, 255),
            2,
        )

        for name in img_ll.keys():
            transform_img(img_ll[name], roi, name, inset_dir)
            cv2.imshow(name, cv2.resize(img_ll[name], (view_img_w, view_img_h)))
            cv2.waitKey(1)

        if cv2.waitKey(0) & 0xFF == ord("p"):
            print("Quitting, not saving.")
            break

        if cv2.waitKey(0) & 0xFF == ord("q"):
            print("Saving crops.")

            for name in img_ll.keys():
                dump_name = f"{name}_{out_name}.png"
                cv2.imwrite(str(inset_dir / dump_name), img_ll[name])

            break
