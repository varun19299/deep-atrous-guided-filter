#!/usr/bin/env python
import sys

import numpy as np
import os.path
from scipy.io.matlab.mio import loadmat
#from skimage.measure import compare_ssim as ssim
from utils.myssim import compare_ssim as ssim

def list_files_walk_subdirs(path, ext):
    return [os.path.join(pt, name)
            for pt, dirs, files in os.walk(path)
            for name in files
            if name.lower().endswith(ext)]


def output_psnr_mse(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr


def mean_psnr_srgb(ref_mat, res_mat):
    n_blk, h, w, c = ref_mat.shape
    mean_psnr = 0
    for b in range(n_blk):
        ref_block = ref_mat[b, :, :, :]
        res_block = res_mat[b, :, :, :]
        ref_block = np.reshape(ref_block, (h, w, c))
        res_block = np.reshape(res_block, (h, w, c))
        psnr = output_psnr_mse(ref_block, res_block)
        mean_psnr += psnr
    return mean_psnr / n_blk


def mean_ssim_srgb(ref_mat, res_mat):
    n_blk, h, w, c = ref_mat.shape
    mean_ssim = 0
    for b in range(n_blk):
        ref_block = ref_mat[b, :, :, :]
        res_block = res_mat[b, :, :, :]
        ref_block = np.reshape(ref_block, (h, w, c))
        res_block = np.reshape(res_block, (h, w, c))
        ssim1 = ssim(ref_block, res_block, gaussian_weights=True, use_sample_covariance=False,
                     multichannel=True)
        mean_ssim += ssim1
    return mean_ssim / n_blk


if __name__=="__main__":
    # as per the metadata file, input and output directories are the arguments
    [_, input_dir, output_dir] = sys.argv

    ref_dir = os.path.join(input_dir, 'ref/')
    res_dir = os.path.join(input_dir, 'res/')
    print("REF DIR")
    print(ref_dir)
    print("RES DIR")
    print(res_dir)

    runtime = -1
    cpu = -1
    data = -1
    other = ""
    readme_fnames = [os.path.join(root, name)
                     for root, dirs, files in os.walk(res_dir)
                     for name in files
                     if name.lower().startswith('readme')]
    try:
        readme_fname = readme_fnames[0]
        print("Parsing extra information from %s" % readme_fname)
        with open(os.path.join(input_dir, 'res', readme_fname)) as readme_file:
            readme = readme_file.readlines()
            lines = [l.strip() for l in readme if l.find(":") >= 0]
            runtime = float(":".join(lines[0].split(":")[1:]))
            cpu = int(":".join(lines[1].split(":")[1:]))
            data = int(":".join(lines[2].split(":")[1:]))
            other = ":".join(lines[3].split(":")[1:])
    except:
        print("Error occured while parsing readme.txt")
        print("Please make sure you have a line for runtime, cpu/gpu, use of meatadata, and other (4 lines in total).")
    print("Parsed information:")
    print("Runtime: %f" % runtime)
    print("CPU/GPU: %d" % cpu)
    print("Method: %d" % data)
    print("Other: %s" % other)

    # ref_mat_fns = sorted([p for p in os.listdir(ref_dir) if p.lower().endswith('mat')])
    ref_mat_fns = list_files_walk_subdirs(ref_dir, 'mat')
    # res_mat_fns = sorted([p for p in os.listdir(res_dir) if p.lower().endswith('mat')])
    res_mat_fns = list_files_walk_subdirs(res_dir, 'mat')
    print('ref_mat_fns:')
    print(ref_mat_fns)
    print('res_mat_fns:')
    print(res_mat_fns)
    if len(res_mat_fns) < 1:
        raise Exception('MAT file not found. ')

    # ref_mat_path = os.path.join(ref_dir, ref_mat_fns[0])
    # res_mat_path = os.path.join(res_dir, res_mat_fns[0])
    res_mat_path = res_mat_fns[0]
    ref_mat_path = ref_mat_fns[0]

    print('ref_mat_path:')
    print(ref_mat_path)
    print('res_mat_path:')
    print(res_mat_path)

    # load reference and result mat files
    res_dict_key = 'results'
    ref_mat = loadmat(ref_mat_path)['val_gt']
    res_mat = loadmat(res_mat_path)[res_dict_key]

    print('type(ref_mat):')
    print(type(ref_mat))
    print('type(res_mat):')
    print(type(res_mat))

    print('ref_mat.shape:')
    print(ref_mat.shape)
    print('res_mat.shape:')
    print(res_mat.shape)


    # for sRGB images
    ref_mat = ref_mat.astype('float') / 255.0
    res_mat = res_mat.astype('float') / 255.0

    # PSNR
    mean_psnr = mean_psnr_srgb(ref_mat, res_mat)
    print('mean_psnr:')
    print(mean_psnr)

    # SSIM
    mean_ssim = mean_ssim_srgb(ref_mat, res_mat)
    print('mean_ssim:')
    print(mean_ssim)

    # the scores for the leaderboard must be in a file named "scores.txt"
    # https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
    with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
        output_file.write("PSNR:%f\n" % mean_psnr)
        output_file.write("SSIM:%f\n" % mean_ssim)
        output_file.write("RT:%f\n" % runtime)
        output_file.write("DEVICE:%d\n" % cpu)
        output_file.write("METHOD:%d\n" % data)
