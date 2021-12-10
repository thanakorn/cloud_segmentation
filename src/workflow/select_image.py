import numpy as np
import os
import cv2 as cv
import yaml
import random
from os.path import join
from math import ceil

random.seed(42)

def main(params):
    min_cloud_ratio = params['min_cloud_ratio']
    val_ratio = params['val_ratio']
    test_ratio = params['test_ratio']
    train_ratio = 1.0 - (val_ratio + test_ratio)

    data_dir = 'data/preprocessed'
    gt_files = os.listdir(join(data_dir, 'gt'))

    dataset = []
    for file in gt_files:
        gt = cv.imread(join(data_dir, 'gt', file), cv.IMREAD_GRAYSCALE)
        if np.count_nonzero(gt) / gt.size > min_cloud_ratio:
            dataset.append(file.split('_')[0])

    random.shuffle(dataset)
    num_img = len(dataset)
    num_train, num_val = ceil(num_img * train_ratio), int(num_img * val_ratio)
    train_set = dataset[:num_train]
    val_set = dataset[num_train + 1: num_train + num_val]
    test_set = dataset[num_train + num_val + 1:]
    
    with open(join(data_dir,'train.txt'), 'w') as f:
        f.write('\n'.join(train_set))

    with open(join(data_dir,'val.txt'), 'w') as f:
        f.write('\n'.join(val_set))

    with open(join(data_dir,'test.txt'), 'w') as f:
        f.write('\n'.join(test_set))


if __name__=='__main__':
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    main(params['preprocessing'])