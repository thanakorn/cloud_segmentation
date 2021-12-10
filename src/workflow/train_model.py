import torch
from torch.nn.modules.loss import BCEWithLogitsLoss
import yaml
from math import ceil
from model.unet import UNet
from model.segmentation_module import SegmentationModule
from data.dataset import CloudDataset
from torch.utils.data import random_split, DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger

torch.random.manual_seed(10)

def get_optimizer(optimizer: str):
    module = __import__('torch.optim', fromlist=[optimizer])
    return getattr(module, optimizer)

def main(model_params, training_params):

    val_ratio = training_params['val_ratio']
    test_ratio = training_params['test_ratio']
    batch_size = training_params['batch_size']
    optimizer = training_params['optimizer']
    epochs = training_params['n_epochs']

    with open('data/preprocessed/dataset.txt') as f:
        img_list = [img_id.strip() for img_id in f.readlines()]

    dataset = CloudDataset('data/preprocessed/img', 'data/preprocessed/gt', img_list)
    train, val, test = random_split(dataset, [
        ceil((1.0 - (val_ratio + test_ratio)) * len(dataset)), 
        int(val_ratio * len(dataset)),
        int(test_ratio * len(dataset)),
    ])
    
    train_loader = DataLoader(train, batch_size=batch_size)
    val_loader = DataLoader(val, batch_size=batch_size)
    test_loader = DataLoader(test, batch_size=batch_size)

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
    trainer.test(module, test_loader)
    torch.save(model.state_dict(), 'model/model.pth')

if __name__=='__main__':
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)

    main(params['model'], params['training'])
