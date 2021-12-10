import torch
import cv2 as cv
from os.path import join
from torch.utils.data import Dataset

class CloudDataset(Dataset):
    def __init__(self, img_dir, gt_dir, img_list) -> None:
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.img_list = img_list

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_id = self.img_list[index]
        img = cv.cvtColor(cv.imread(join(self.img_dir, '%s.jpg' % img_id)), cv.COLOR_BGR2RGB).transpose(2,0,1)
        gt = cv.imread(join(self.gt_dir, '%s_gt.jpg' % img_id), cv.IMREAD_GRAYSCALE)
        gt = (gt > 0).astype(int)
        return torch.tensor(img).float(), torch.tensor(gt).float()