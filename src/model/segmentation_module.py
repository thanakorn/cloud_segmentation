from torch.nn import Module
from torch.optim import Optimizer
from pytorch_lightning import LightningModule

class SegmentationModule(LightningModule):
    def __init__(self, model: Module, optimizer: Optimizer, criteria: Module) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.criteria = criteria

    def configure_optimizers(self):
        return self.optimizer

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        input, target = batch
        predict = self.model(input)
        loss = self.criteria(predict, target)
        return loss

    def training_epoch_end(self, outputs):
        self.log('train_loss', sum([out['loss'].item() for out in outputs]))

    def validation_step(self, batch, batch_idx):
        input, target = batch
        predict = self.model(input)
        loss = self.criteria(predict, target)
        return loss

    def validation_epoch_end(self, outputs):
        self.log('val_loss', sum([out.item() for out in outputs]))

    def test_step(self, batch, batch_idx):
        input, target = batch
        predict = self.model(input)
        loss = self.criteria(predict, target)
        return loss

    def test_epoch_end(self, outputs):
        self.log('test_loss', sum([out.item() for out in outputs]))