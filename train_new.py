import os
import sys
from tabnanny import verbose
import torch
import datetime
import numpy as np
import torch.nn as nn
import json
from src.loss import *
from src.metrics import mIoU
from src.dataset import loaders
from src.utils import create_dir, seeding
from scheduler import CyclicCosineDecayLR
from models import ModelSegmentation
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR, ExponentialLR
# import torch.utils.tensorboard
from torchsummary import summary
import configparser

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import LightningModule, Trainer
from model_ligh import ModelSegmentation
from pytorch_lightning.loggers import TensorBoardLogger


def main():
    config = configparser.ConfigParser()
    config.read("configs/swin_config.ini")
    paths = config['PATHS']
    hyperparameters = config['HYPERPARAMETERS']
    general = config['GENERAL']
    """ 
    Seeding 
    """
    seeding(42)  # 42
    """ 
    Hyperparameters 
    """
    batch_size = hyperparameters.getint('batch_size'),
    num_epochs = hyperparameters.getint('num_epochs'),
    lr = hyperparameters.getfloat('lr'),
    B1 = hyperparameters.getfloat('B1'),
    B2 = hyperparameters.getfloat('B2'),
    weight_decay = hyperparameters.getfloat('weight_decay'),
    class_weights = [0.3, 5, 8]
    gpus_ids = [0, 1, 2, 3]
    """
    Paths
    """
    train_imgdir = paths.get('train_imgdir')
    train_maskdir = paths.get('train_maskdir')
    val_imgdir = paths.get('val_imgdir')
    val_maskdir = paths.get('val_maskdir')
    """
    General settings
    """
    num_workers = os.cpu_count()
    pin_memory = general.getboolean('pin_memory')
    n_classes = general.getint('n_classes')
    img_size = general.getint('img_size')
    pretrain = general.getboolean('pretrain')
    device = torch.device(f"cuda" if torch.cuda.is_available() else 'cpu')
    """ 
    Getting loader
    """
    model = ModelSegmentation(n_classes=n_classes, img_size=img_size, pretrain=pretrain)

    """ Loaders """
    train_loader, val_loader = loaders(train_imgdir=train_imgdir,
                                    train_maskdir=train_maskdir,
                                    val_imgdir=val_imgdir,
                                    val_maskdir=val_maskdir,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    pin_memory=pin_memory
                                    )
    """
    Directories
    """
    checkpoint_path = "checkpoints/" + datetime.datetime.now().strftime("%d%H%M%S_") + '/'
    create_dir("checkpoints")
    create_dir(checkpoint_path)
    with open(checkpoint_path + 'experiment.ini', 'w') as configfile:
        config.write(configfile)
    sys.stdout = open(checkpoint_path + 'stdout.txt', 'w')

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(checkpoint_path, 'checkpoints'),
        filename="{epoch}-{step}-{val_loss:.3f}",
        verbose=True,
        monitor="val_loss",
        mode="min",
    )
    stop_callback = EarlyStopping(
        monitor='val_loss',
        mode='auto',
        patience=50,
        verbose=True,
    )

    logger = TensorBoardLogger(save_dir='runs', name=datetime.datetime.now().strftime("%d%H%M%S_"))

    trainer = Trainer(
        gpus=1,
        checkpoint_callback=checkpoint_callback,
        early_stop_callback=stop_callback,
        logger=logger,
    )

    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
if __name__ == '__main__':
    main()
    sys.stdout.close()
