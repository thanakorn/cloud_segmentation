import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import json
from torch.utils.data import dataloader, random_split, DataLoader
from model.unet import UNet
from data.dataset import CloudDataset
from math import ceil
from torch import sigmoid
from sklearn.metrics import accuracy_score, precision_score, recall_score, jaccard_score, confusion_matrix

torch.random.manual_seed(10)

def main(params):
    with open('data/preprocessed/test.txt') as f:
        img_list = [img_id.strip() for img_id in f.readlines()]

    dataset = CloudDataset('data/preprocessed/img', 'data/preprocessed/gt', img_list)
    dataloader = DataLoader(dataset, batch_size=16)
    model = UNet(
        n_classes=params['n_classes'],
        in_channel=params['in_channels']
    )
    model.load_state_dict(torch.load('model/model.pth'))
    model.eval()

    acc = []
    precision = []
    recall = []
    iou = []
    cm = []
    for input, target in dataloader:
        out = model(input)
        predict = (sigmoid(out).detach().numpy() > 0.5).astype(int)
        y_true = target.detach().numpy().astype(int)
        acc.append(accuracy_score(y_true.flatten(), predict.flatten()))
        precision.append(precision_score(y_true.flatten(), predict.flatten()))
        recall.append(recall_score(y_true.flatten(), predict.flatten()))
        iou.append(jaccard_score(y_true.flatten(), predict.flatten()))
        cm.append(confusion_matrix(y_true.flatten(), predict.flatten(), normalize='true'))

    _, axes = plt.subplots(nrows=min(input.shape[0], 5), ncols=3, figsize=(10,10))
    for i, ax in enumerate(axes):
        ax[0].imshow(input[i].detach().numpy().astype(int).transpose(1,2,0))
        ax[0].set_axis_off()
        ax[1].imshow(y_true[i], cmap='binary', vmin=0., vmax=1.)
        ax[1].set_axis_off()
        ax[2].imshow(predict[i], cmap='binary', vmin=0., vmax=1.)
        ax[2].set_axis_off()
    plt.savefig('figures/sample_test_output.jpg', bbox_inches='tight')

    accuracy = sum(acc) / len(acc)
    precision = sum(precision) / len(precision)
    recall = sum(recall) / len(recall)
    iou = sum(iou) / len(iou)
    cm = np.stack(cm).mean(axis=0)
    plt.figure()
    seaborn.heatmap(cm, annot=True, fmt='.2f', cbar=False, xticklabels=['Not Cloud', 'Cloud'], yticklabels=['Not Cloud', 'Cloud'])
    plt.savefig('figures/confusion_matrix.jpg')

    with open('model/performance.json', 'w') as f:
        f.write(json.dumps(
            dict(accuracy=accuracy, precision=precision, recall=recall, iou=iou),
            indent=4
        ))

if __name__=='__main__':
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    main(params['model'])