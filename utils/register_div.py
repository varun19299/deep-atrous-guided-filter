import cv2
import numpy as np
import os
from os.path import exists, join
import sys
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.measure import compare_psnr


parser = argparse.ArgumentParser()
parser.add_argument(
    "--div", help="DIV2K folder", default="./udc_net/GT_train", type=str
)
parser.add_argument(
    "--hq", help="Corresponding HQ folder", default="./udc_net/HQ", type=str
)
parser.add_argument(
    "--save", help="save  folder", default="./udc_net/GT_align", type=str
)
parser.add_argument("--sanity", default=False, type=bool)

args = parser.parse_args()

if not exists(args.save):
    os.mkdir(args.save)


def getMatches(div, hq, ratio=0.8):
    # SIFT and FLANN matching
    sift = cv2.xfeatures2d.SIFT_create()
    kp_div, des_div = sift.detectAndCompute(div, None)
    kp_hq, des_hq = sift.detectAndCompute(hq, None)

    pts1 = []  # hq
    pts2 = []  # div
    good = []  # good matches
    # ratio = 0.8
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_hq, des_div, k=2)
    # ratio test as per Lowe's paper : https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
    for i, (m, n) in enumerate(matches):
        if m.distance < ratio * n.distance:
            good.append(m)
            pts2.append(kp_div[m.trainIdx].pt)
            pts1.append(kp_hq[m.queryIdx].pt)
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    return pts1, pts2, good


pbar = tqdm(range(len(os.listdir(args.div))), desc="Images done:")

for imgpath in os.listdir(args.div):
    div_path = join(args.div, imgpath)
    hq_path = join(args.hq, imgpath)
    div_col = cv2.imread(div_path)
    hq_col = cv2.imread(hq_path)

    div = cv2.cvtColor(div_col, cv2.COLOR_BGR2GRAY)
    hq = cv2.cvtColor(hq_col, cv2.COLOR_BGR2GRAY)

    # hq, div, matches
    pts1, pts2, good = getMatches(div, hq, ratio=0.9)
    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)  # find homography from div to hq

    div_align = cv2.warpPerspective(div_col, H, (hq.shape[1], hq.shape[0]))
    # print('PSNR :', compare_psnr(div_align, hq_col))
    pbar.update(1)
    # pbar.set_description(desc='Current psnr {compare_psnr(div_align )}')

    cv2.imwrite(join(args.save, imgpath), div_align)

pbar.close()


# # display images
# plt.figure(0)
# plt.subplot(1, 2, 1)
# plt.imshow(div_col[:,:,::-1])
# plt.title('Original DIV image')
# plt.subplot(1, 2, 2)
# plt.imshow(hq_col[:,:,::-1])
# plt.title('Original HQ image')


# plt.figure(1)
# plt.subplot(2, 1, 1)
# plt.imshow(div_align[:,:,::-1])
# plt.title('Transformed div image')
# plt.subplot(2, 1, 2)
# plt.imshow(hq_col[:,:,::-1])
# plt.show()

# # cv2.imwrite('div1_align.png', div_align)

# # if args.sanity:
# #     print('homography for the div image is :', H)
# #     div_align_gray = cv2.cvtColor(div_align, cv2.COLOR_BGR2GRAY)
# #     pts1, pts2, good= getMatches(div_align_gray, hq)
# #     H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)
# #     print('homography for the aligned image is :', H)
