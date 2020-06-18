import cv2


def flip_horizontal(img):
    """
    :param img: [1, C, H, W]
    :return:
    """
    return img[:, :, :, ::-1]


def flip_vertical(img):
    """
    :param img: [1, C, H, W]
    :return:
    """
    return img[:, :, ::-1]


def rotate_180(img):
    """
    :param img: [1, C, H, W]
    :return:
    """
    return img[:, :, ::-1, ::-1]


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
    return flip_vertical(rotate_90_clock(img))


ensemble_ops = {
    "flip_horz": (flip_horizontal, flip_horizontal),
    "flip_vert": (flip_vertical, flip_vertical),
    "rotate_180": (rotate_180, rotate_180),
    "rotate_90_counter": (rotate_90_counterclock, rotate_90_clock),
    "rotate_90_clock": (rotate_90_clock, rotate_90_counterclock),
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
