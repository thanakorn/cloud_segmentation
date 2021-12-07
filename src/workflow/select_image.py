import numpy as np
import os
import cv2 as cv
from os.path import join
from argparse import ArgumentParser

def main(args):
    data_dir = 'data/preprocessed'
    gt_files = os.listdir(join(data_dir, 'gt'))

    min_cloud_ratio = args.min_cloud_ratio
    dataset = []
    for file in gt_files:
        gt = cv.imread(join(data_dir, 'gt', file), cv.IMREAD_GRAYSCALE)
        if np.count_nonzero(gt) / gt.size > min_cloud_ratio:
            dataset.append(file.split('_')[0])
    
    with open(join(data_dir,'dataset.txt'), 'w') as f:
        f.write('\n'.join(dataset))


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('--min_cloud_ratio', type=int, default=0.05)
    args = parser.parse_args()
    main(args)