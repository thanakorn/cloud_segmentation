import numpy as np
import os
import cv2 as cv
import yaml
from os.path import join
from argparse import ArgumentParser

def main(params):
    data_dir = 'data/preprocessed'
    gt_files = os.listdir(join(data_dir, 'gt'))

    min_cloud_ratio = params['min_cloud_ratio']
    dataset = []
    for file in gt_files:
        gt = cv.imread(join(data_dir, 'gt', file), cv.IMREAD_GRAYSCALE)
        if np.count_nonzero(gt) / gt.size > min_cloud_ratio:
            dataset.append(file.split('_')[0])
    
    with open(join(data_dir,'dataset.txt'), 'w') as f:
        f.write('\n'.join(dataset))


if __name__=='__main__':
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    main(params['preprocessing'])