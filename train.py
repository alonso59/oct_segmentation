import os
import sys
from tabnanny import verbose
import torch
import datetime
import numpy as np
import torch.nn as nn
import logging

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
    Directories
    """
    ver_ = 0
    while(os.path.exists(f"logs/version{ver_}/")):
        ver_ += 1
    version = f"logs/version{ver_}/"
    checkpoint_path = version + "checkpoints/"
    create_dir(checkpoint_path)
    with open(version + 'config.txt', 'w') as configfile:
        config.write(configfile)
    """
    logging
    """
    logging.basicConfig(filename=version + "log.log",
                        filemode='a',
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger()
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stdout_handler)
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
    class_weights = [0.2644706,  12.33872479, 12.23935952, 17.82146076]
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
    iter_plot_img = len(val_loader) * 10
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
    loss_fn = WCEGeneralizedDiceLoss(class_weights=class_weights, device=device)
    # loss_fn = DiceLoss(device=device)
    metrics = mIoU(device)
    # scheduler = StepLR(optimizer=optimizer, step_size=60, gamma=0.8)
    scheduler = CyclicCosineDecayLR(optimizer,
                                    init_decay_epochs=400,
                                    min_decay_lr=1e-5,
                                    restart_interval=100,
                                    restart_lr=1e-4)

    # summary(model, input_size=(1, img_size, img_size), batch_size=-1)
    logger.info(f'Total_params:{pytorch_total_params}')
    """ 
    Trainer
    """
    logger.info('**********************************************************')
    logger.info('**************** Initialization sucessful ****************')
    logger.info('**********************************************************')
    logger.info('--------------------- Start training ---------------------')
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
            base_lr=lr, 
            callback_stop_value=40,
            tb_dir = version,
            logger=logger
            )
    logger.info('-------------------- Finished Train ---------------------')
    logger.info('******************* Start evaluation  *******************')
    load_best_model = torch.load(checkpoint_path + 'model.pth')
    loss_eval = eval(load_best_model, val_loader, loss_fn, metrics, device)
if __name__ == '__main__':
    main()
