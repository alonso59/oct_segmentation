from networks.swin_unet import SwinUnet
from pytorch_lightning import LightningModule, Trainer
from src.loss import *
from src.metrics import mIoU

class ModelSegmentation(LightningModule):
    def __init__(self, device, n_classes=1, img_size=224, pretrain=True) -> None:
        super().__init__()
        self.model = SwinUnet(self, in_chans=3, n_classes=n_classes, img_size=img_size, pretrain=pretrain)
        if pretrain:
            self.model.state_dict()
            self.model.load_from("pretrained/swin_tiny_patch4_window7_224.pth", self.device)
        self.device = device

    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss_fn = WeightedCrossEntropyDice(class_weights=[0.7,1,1,1], device=self.device, activation='softmax')
        metric_fn = mIoU(self.device)
        loss = loss_fn(y_hat, y)
        metric = metric_fn(y_hat, y)
        tensorboard_logs = {'train_loss': loss, f'train_{metric_fn.__name__}': metric}
        total=len(y)

        return {'loss': loss, 'correct':metric, 'total': total, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self.forward(x)
        loss_fn = WeightedCrossEntropyDice(class_weights=[0.7,1,1,1], device=self.device, activation='softmax')
        metric_fn = mIoU(self.device)
        loss = loss_fn(y_hat, y)
        metric = metric_fn(y_hat, y)
        tensorboard_logs = {'val_loss': loss, f'val_{metric_fn.__name__}': metric}
        self.log("val_loss", loss)
        return {'val_loss': loss, f'val_{metric_fn.__name__}':metric, 'log': tensorboard_logs}

    def training_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        # calculating average loss  
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # calculating correect and total predictions
        correct=sum([x["correct"] for  x in outputs])
        total=sum([x["total"] for  x in outputs])

        # creating log dictionary
        tensorboard_logs = {'Loss': avg_loss, "Metric": correct / total}

        epoch_dictionary={
            # required
            'loss': avg_loss,
            
            # for logging purposes
            'log': tensorboard_logs
            }

        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Train",
                                            avg_loss,
                                            self.current_epoch)
        
        self.logger.experiment.add_scalar("Accuracy/Train",
                                            correct/total,
                                            self.current_epoch)
        return epoch_dictionary

    def validation_end(self, outputs):
        #  the function is called after every epoch is completed

        # calculating average loss  
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # calculating correect and total predictions
        correct=sum([x["correct"] for  x in outputs])
        total=sum([x["total"] for  x in outputs])

        # creating log dictionary
        tensorboard_logs = {'Loss': avg_loss, "Metric": correct / total}

        epoch_dictionary={
            # required
            'loss': avg_loss,
            
            # for logging purposes
            'log': tensorboard_logs
            }
        # logging using tensorboard logger
        self.logger.experiment.add_scalar("Loss/Val",
                                            avg_loss,
                                            self.current_epoch)
        
        self.logger.experiment.add_scalar("Metric/Val",
                                            correct/total,
                                            self.current_epoch)
        return epoch_dictionary

    def configure_optimizers(self, lr, weight_decay, B1, B2):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay, betas=(B1, B2))