import os
import sys
from tabnanny import verbose
import torch
import datetime
import numpy as np
import torch.nn as nn
import logging
import src.settings as cfg
from src.loss import *
from src.metrics import mIoU
from src.trainer import trainer, eval
from src.dataset import loaders
from src.utils import create_dir, seeding
from scheduler import CyclicCosineDecayLR
from models import ModelSegmentation
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR, ExponentialLR
import segmentation_models_pytorch as smp
from pytorch_model_summary import summary as sm
# import torch.utils.tensorboard
from torchsummary import summary
import configparser


def main():
    logger, checkpoint_path, version = initialize()
    """ 
    Hyperparameters 
    """
    num_epochs = cfg.EPOCHS
    lr = cfg.LEARNING_RATE
    B1 = cfg.BETA1
    B2 = cfg.BETA2
    weight_decay = cfg.WEIGHT_DECAY
    # class_weights = [0.2644706,  12.33872479, 12.23935952, 17.82146076]
    class_weights = [1, 1, 1, 1]
    gpus_ids = cfg.GPUS_ID
    """
    General settings
    """
    n_classes = cfg.CLASSES
    img_size = cfg.IMAGE_SIZE
    pretrain = cfg.PRETRAIN
    device = torch.device(f"cuda" if torch.cuda.is_available() else 'cpu')
    """ 
    Getting loader
    """
    # model, preprocess_input = smp_model()
    # model = model.to(device)
    train_loader, val_loader = loaders(train_imgdir=cfg.TRAIN_IMAGES,
                                       train_maskdir=cfg.TRAIN_MASKS,
                                       val_imgdir=cfg.VAL_IMAGES,
                                       val_maskdir=cfg.VAL_MASKS,
                                       batch_size=cfg.BATCH_SIZE,
                                       num_workers=os.cpu_count(),
                                       pin_memory=True,
                                       preprocess_input=None#preprocess_input
                                       )
    iter_plot_img = cfg.EPOCHS // 80
    """ 
    Building model 
    """
    models_class = ModelSegmentation(device)
    model = models_class.swin_unet(
        n_classes=n_classes, 
        img_size=img_size, 
        pretrain=pretrain,
        embed_dim=cfg.embed_dim,
        depths=cfg.depths,
        num_heads=cfg.num_heads,
        window_size=cfg.window_size,
        drop_path_rate=cfg.dropout,
    )
    
    # model = models_class.unet(in_channels=1, n_classes=n_classes, img_size=img_size, feature_start=16,
    #                           layers=5, bilinear=False, dropout=0.1, kernel_size=3, stride=1, padding=1)
    name_model = model.__name__
    # name_model = 'FPN_resnet18'
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    if len(gpus_ids) > 1:
        print("Data parallel...")
        model = nn.DataParallel(model, device_ids=gpus_ids)
    """ 
    Prepare training 
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(B1, B2))
    # loss_fn = WeightedCrossEntropyDice(class_weights=class_weights, device=device)
    loss_fn = DiceLoss(device=device)
    metrics = mIoU(device)
    if cfg.SCHEDULER == 'step':
        scheduler = StepLR(optimizer=optimizer, step_size=cfg.STEP_SIZE, gamma=cfg.GAMMA)
    if cfg.SCHEDULER == 'cosine':
        scheduler = CyclicCosineDecayLR(optimizer,
                                    init_decay_epochs=num_epochs // 3,
                                    min_decay_lr=lr / 10,
                                    restart_interval=num_epochs // 10,
                                    restart_lr=lr / 5)
    logger.info(sm(model, torch.zeros((1, 1, img_size, img_size)).to(device), show_input=False))
    # summary(model, input_size=(3, img_size, img_size), batch_size=cfg.BATCH_SIZE)
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
            callback_stop_value=num_epochs,# // 8,
            tb_dir = version,
            logger=logger
            )
    logger.info('-------------------- Finished Train ---------------------')
    logger.info('******************* Start evaluation  *******************')
    load_best_model = torch.load(checkpoint_path + 'model.pth')
    loss_eval = eval(load_best_model, val_loader, loss_fn, device)
    print([loss_eval])

def initialize():
    """
    Directories
    """
    ver_ = 0
    while(os.path.exists(f"logs/version{ver_}/")):
        ver_ += 1
    version = f"logs/version{ver_}/"
    checkpoint_path = version + "checkpoints/"
    create_dir(checkpoint_path)
    with open(version + 'config.txt', 'w') as text_file:
        text_file.write(f"*** Hyperparameters ***\n")
        text_file.write(f"Loss function: {cfg.LOSS_FN}\n")
        text_file.write(f"Learning rate: {cfg.LEARNING_RATE}\n")
        text_file.write(f"weight_decay: {cfg.WEIGHT_DECAY}\n")
        text_file.write(f"BETA1, BETA2: {cfg.BETA1, cfg.BETA2}\n")
        text_file.write(f"Batch size: {cfg.BATCH_SIZE}\n")
        text_file.write(f"Epochs: {cfg.EPOCHS}\n")
        text_file.write(f"*** Scheduler LR ***\n")
        text_file.write(f"Schaduler Type: {cfg.SCHEDULER}\n")
        text_file.write(f"Gamma: {cfg.GAMMA}\n")
        text_file.write(f"Step size: {cfg.STEP_SIZE}\n")
        text_file.write(f"*** Gerneral settings ***\n")
        text_file.write(f"Image Size: {cfg.IMAGE_SIZE}\n")
        text_file.write(f"Pretrain: {cfg.PRETRAIN}\n")
        text_file.write(f"Num classes: {cfg.CLASSES}\n")
        text_file.write(f"No. of GPUs: {len(cfg.GPUS_ID)}\n")
        text_file.close()
    """
    logging
    """
    logging.basicConfig(filename=version + "info.log",
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
    return logger, checkpoint_path, version

def smp_model():
    from segmentation_models_pytorch.encoders import get_preprocessing_fn
    model = smp.FPN(
    encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=cfg.CLASSES,                      # model output channels (number of classes in your dataset)
    )
    preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')
    return model, preprocess_input



if __name__ == '__main__':
    main()
