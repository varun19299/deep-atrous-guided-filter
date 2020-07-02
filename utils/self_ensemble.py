import cv2
import torch
from matplotlib import pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import ImageGrid


def flip_horizontal(img):
    """
    :param img: [1, C, H, W]
    :return:
    """
    return torch.flip(img, dims=(3,))


def flip_vertical(img):
    """
    :param img: [1, C, H, W]
    :return:
    """
    return torch.flip(img, dims=(2,))


def rotate_180(img):
    """
    :param img: [1, C, H, W]
    :return:
    """
    return torch.flip(img, dims=(2, 3))


def rotate_90_clock(img):
    """
    :param img: [1, C, H, W]
    :return:
    """
    img_cv = cv2.rotate(img[0].permute(1, 2, 0).cpu().numpy(), cv2.ROTATE_90_CLOCKWISE)
    return torch.tensor(img_cv).float().permute(2, 0, 1).unsqueeze(0).to(img.device)


def rotate_90_counterclock(img):
    """
    :param img: [1, C, H, W]
    :return:
    """
    img_cv = cv2.rotate(
        img[0].permute(1, 2, 0).cpu().numpy(), cv2.ROTATE_90_COUNTERCLOCKWISE
    )
    return torch.tensor(img_cv).float().permute(2, 0, 1).unsqueeze(0).to(img.device)


def flip_vertical_rotate_90_clock(img):
    return rotate_90_clock(flip_vertical(img))


def flip_vertical_rotate_90_clock_inverse(img):
    return flip_vertical(rotate_90_counterclock(img))


def flip_vertical_rotate_90_counterclock(img):
    return rotate_90_counterclock(flip_vertical(img))


def flip_vertical_rotate_90_counterclock_inverse(img):
    return flip_vertical(rotate_90_clock(img))


def flip_horizontal_rotate_90_clock(img):
    return rotate_90_clock(flip_horizontal(img))


def flip_horizontal_rotate_90_clock_inverse(img):
    return flip_horizontal(rotate_90_counterclock(img))


def flip_horizontal_rotate_90_counterclock(img):
    return rotate_90_counterclock(flip_horizontal(img))


def flip_horizontal_rotate_90_counterclock_inverse(img):
    return flip_horizontal(rotate_90_clock(img))


def _to_tensor(img):
    return torch.tensor(img.copy()).permute(2, 0, 1).unsqueeze(0)


def plot(img1, img2, img3):
    """
    :param img: [1, C, H, W]
    """
    fig = plt.figure(figsize=(12.0, 8.0))
    grid = ImageGrid(
        fig,
        111,  # similar to subplot(111)
        nrows_ncols=(1, 3),  # creates 2x2 grid of axes
        axes_pad=0.1,  # pad between axes in inch.
    )

    img1 = img1[0].permute(1, 2, 0).cpu().numpy()
    img2 = img2[0].permute(1, 2, 0).cpu().numpy()
    img3 = img3[0].permute(1, 2, 0).cpu().numpy()

    for ax, im in zip(grid, [img1, img2, img3]):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)

    plt.show()


ensemble_ops = {
    "flip_horz": (flip_horizontal, flip_horizontal),
    "flip_vert": (flip_vertical, flip_vertical),
    "rotate_180": (rotate_180, rotate_180),
    # "rotate_90_counter": (rotate_90_counterclock, rotate_90_clock),
    # "rotate_90_clock": (rotate_90_clock, rotate_90_counterclock),
    "flip_vert_rotate_90_counter": (
        flip_vertical_rotate_90_counterclock,
        flip_vertical_rotate_90_counterclock_inverse,
    ),
    "flip_vert_rotate_90_clock": (
        flip_vertical_rotate_90_clock,
        flip_vertical_rotate_90_clock_inverse,
    ),
    "flip_horz_rotate_90_counter": (
        flip_horizontal_rotate_90_counterclock,
        flip_horizontal_rotate_90_counterclock_inverse,
    ),
    "flip_horz_rotate_90_clock": (
        flip_horizontal_rotate_90_clock,
        flip_horizontal_rotate_90_clock_inverse,
    ),
}
