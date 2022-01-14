import torch
import yaml
import matplotlib.pyplot as plt
import os
from torch.nn.modules.loss import BCEWithLogitsLoss
from math import ceil
from model.unet import UNet
from model.segmentation_module import SegmentationModule
from data.dataset import CloudDataset
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from torch.nn.functional import sigmoid

torch.random.manual_seed(10)

def get_optimizer(optimizer: str):
    module = __import__('torch.optim', fromlist=[optimizer])
    return getattr(module, optimizer)

def main(model_params, training_params):
    batch_size = training_params['batch_size']
    optimizer = training_params['optimizer']
    epochs = training_params['n_epochs']

    with open('data/preprocessed/train.txt') as f1, open('data/preprocessed/val.txt') as f2:
        train_list = [img_id.strip() for img_id in f1.readlines()]
        val_list = [img_id.strip() for img_id in f2.readlines()]

    train = CloudDataset('data/preprocessed/img', 'data/preprocessed/gt', train_list)
    val = CloudDataset('data/preprocessed/img', 'data/preprocessed/gt', val_list)
    
    train_loader = DataLoader(train, batch_size=batch_size)
    val_loader = DataLoader(val, batch_size=batch_size, drop_last=True)

    model = UNet(
        n_classes=model_params['n_classes'],
        in_channel=model_params['in_channels']
    )
    optim = get_optimizer(optimizer)(params=model.parameters(), **dict(lr=0.001))
    loss_fn = BCEWithLogitsLoss()
    module = SegmentationModule(model, optim, loss_fn)
    logger = CSVLogger('logs', name='cloud_segmentation')
    trainer = Trainer(
        max_epochs=epochs, 
        check_val_every_n_epoch=1,
        logger=logger
    )
    trainer.fit(model=module, train_dataloader=train_loader, val_dataloaders=val_loader)
    torch.save(model.state_dict(), 'model/model.pth')

    # Generate sample results
    model.eval()
    iterator = iter(val_loader)
    next(iterator)
    x, y = next(iterator)
    sample_outs = sigmoid(model(x))
    sample_outs = (sample_outs.detach().numpy() > 0.5).astype(int)
    _, axes = plt.subplots(nrows=5, ncols=3, figsize=(10,10))
    for i, ax in enumerate(axes):
        ax[0].imshow(x[i].detach().numpy().astype(int).transpose(1,2,0))
        ax[0].set_axis_off()
        ax[1].imshow(y[i].detach().numpy().astype(int), cmap='binary', vmin=0., vmax=1.)
        ax[1].set_axis_off()
        ax[2].imshow(sample_outs[i], cmap='binary', vmin=0., vmax=1.)
        ax[2].set_axis_off()

    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/sample_output.jpg', bbox_inches='tight')

if __name__=='__main__':
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    main(params['model'], params['training'])
