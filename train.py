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
from src.trainer import trainer
from src.dataset import loaders
from src.utils import create_dir, seeding

from scheduler import CyclicCosineDecayLR

from models import ModelSegmentation

from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR, ExponentialLR

# import torch.utils.tensorboard
from torchsummary import summary
import configparser


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
    batch_size = hyperparameters.getint('batch_size')
    num_epochs = hyperparameters.getint('num_epochs')
    lr = hyperparameters.getfloat('lr')
    B1 = hyperparameters.getfloat('B1')
    B2 = hyperparameters.getfloat('B2')
    weight_decay = hyperparameters.getfloat('weight_decay')
    class_weights = [0.3, 1, 1, 1]
    gpus_ids = [0]
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
    train_loader, val_loader = loaders(train_imgdir=train_imgdir,
                                       train_maskdir=train_maskdir,
                                       val_imgdir=val_imgdir,
                                       val_maskdir=val_maskdir,
                                       batch_size=batch_size,
                                       num_workers=num_workers,
                                       pin_memory=pin_memory
                                       )
    iter_plot_img = len(val_loader) * 5
    """ 
    Building model 
    """
    models_class = ModelSegmentation(device)
    model = models_class.swin_unet(n_classes=n_classes, img_size=img_size, pretrain=pretrain)
    # model = models_class.unet(in_channels=1, n_classes=n_classes, img_size=img_size, feature_start=16,
    #                           layers=4, bilinear=False, dropout=0.0, kernel_size=3, stride=1, padding=1)
    name_model = model.__name__
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    if len(gpus_ids) > 1:
        print("Data parallel...")
        model = nn.DataParallel(model, device_ids=gpus_ids)
    """ 
    Prepare training 
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(B1, B2))
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=B1)
    loss_fn = WeightedCrossEntropyDice(class_weights=class_weights, device=device)
    # loss_fn = DiceLoss(device=device)
    metrics = mIoU(device)
    # scheduler = StepLR(optimizer=optimizer, step_size=60, gamma=0.8)
    scheduler = CyclicCosineDecayLR(optimizer,
                                    init_decay_epochs=400,
                                    min_decay_lr=1e-5,
                                    restart_interval=100,
                                    restart_lr=1e-4)
    """
    Directories
    """
    checkpoint_path = "checkpoints/" + datetime.datetime.now().strftime("%d%H%M%S_") + name_model + '/'
    create_dir("checkpoints")
    create_dir(checkpoint_path)
    with open(checkpoint_path + 'experiment.ini', 'w') as configfile:
        config.write(configfile)
    sys.stdout = open(checkpoint_path + 'stdout.txt', 'w')
    # summary(model, input_size=(1, img_size, img_size), batch_size=-1)
    print(f'Total_params:{pytorch_total_params}')
    """ 
    Trainer
    """
    trainer(num_epochs=num_epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            metric=metrics,
            device=device,
            checkpoint_path=checkpoint_path,
            scheduler=scheduler,
            iter_plot_img=iter_plot_img,
            name_model=name_model,
            base_lr=lr
            )
if __name__ == '__main__':
    main()
    sys.stdout.close()
