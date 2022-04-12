import os
import sys
import torch
import datetime
import numpy as np
import torch.nn as nn

from src.loss import *
from src.metrics import mIoU
from src.trainer import trainer
from src.dataset import loaders
from src.utils import create_dir, seeding

from models import ModelSegmentation

from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR, ExponentialLR

from torch.utils.tensorboard import SummaryWriter

def main():
    """ 
    Seeding 
    """
    seeding(42) #42
    """ 
    Hyperparameters 
    """
    batch_size = 128
    num_epochs = 100
    lr = 0.001
    B1 = 0.9
    B2 = 0.999
    weight_decay = 1e-4
    class_weights = [1, 1, 1, 1]
    gpus_ids = [0]
    """
    Settings
    """
    train_imgdir='dataset/data5/train_images'
    train_maskdir='dataset/data5/train_masks'
    val_imgdir='dataset/data5/val_images'
    val_maskdir='dataset/data5/val_masks'
    num_workers=os.cpu_count()
    pin_memory=True
    channels=3
    iter_plot_img=10
    # device
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
                                       pin_memory=pin_memory,
                                       channels=channels
                                       )
    """ 
    Building model 
    """
    models_class = ModelSegmentation(device)
    model = models_class.swin_unet(in_channels=3, n_classes=4, img_size=224, pretrain=True)

    if len(gpus_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpus_ids)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    """ 
    Prepare training 
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(B1, B2))
    
    loss_fn = WeightedCrossEntropyDice(class_weights=class_weights, device=device)
    metrics = mIoU(device)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.8)
    """ 
    Directories 
    """
    checkpoint_path = "checkpoints/" + datetime.datetime.now().strftime("%d_%H_%M_%S_") + model.__name__ + '/'
    create_dir("checkpoints")
    create_dir(checkpoint_path)
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
        iter_plot_img=iter_plot_img
        )


if __name__ == '__main__':
    main()
