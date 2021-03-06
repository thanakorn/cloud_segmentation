import os
import cv2 as cv
import random
import numpy as np
import yaml
from preprocessing.image_splitter import ImageSplitter
from os.path import join
from argparse import ArgumentParser

random.seed(42)

def main(params):
    img_dir = 'data/images'
    gt_dir = 'data/gt'
    out_dir = 'data/preprocessed'
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(join(out_dir, 'img'), exist_ok=True)
    os.makedirs(join(out_dir, 'gt'), exist_ok=True)

    patch_height = params['patch_height']
    patch_width = params['patch_width']
    img_files = os.listdir(img_dir)
    for file in img_files:
        img = cv.imread(join(img_dir, file))
        gt = cv.imread(join(gt_dir, '%s_GT.jpg' % file.split('.')[0]), cv.IMREAD_GRAYSCALE)
        gt = (gt > 127).astype(int)
        img_patches = ImageSplitter.split_image(img, patch_height, patch_width)
        gt_patches = ImageSplitter.split_image(gt, patch_height, patch_width)

        for img_patch, gt_patch in zip(img_patches, gt_patches):
            if gt_patch.shape != (patch_height, patch_width):
                continue
            
            filename = '%x' % random.getrandbits(16)
            cv.imwrite(join(out_dir, 'img', '%s.jpg' % filename), img_patch)
            cv.imwrite(join(out_dir, 'gt', '%s_gt.jpg' % filename), gt_patch)

if __name__=='__main__':
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    main(params['preprocessing'])