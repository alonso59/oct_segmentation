import os
import sys
import logging
import torch
import torch.nn as nn
import settings as cfg
import segmentation_models_pytorch as smp
from metrics import mIoU, IoU_binary
from metric import SegmentationMetrics
from trainer import trainer, eval
from dataset import loaders
from utils import create_dir, seeding
from models import SegmentationModels
from matplotlib import pyplot as plt
from scheduler import CyclicCosineDecayLR
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from loss import *
import datetime
# import torch.utils.tensorboard

def main():
    logger, checkpoint_path, version = initialize()
    """ Hyperparameters """
    num_epochs = cfg.EPOCHS
    lr = cfg.LEARNING_RATE
    B1 = cfg.BETA1
    B2 = cfg.BETA2
    weight_decay = cfg.WEIGHT_DECAY
    class_weights = cfg.CLASS_WEIGHTS
    gpus_ids = cfg.GPUS_ID
    """General settings"""
    n_classes = cfg.CLASSES
    img_size = cfg.IMAGE_SIZE
    device = torch.device(f"cuda" if torch.cuda.is_available() else 'cpu')
    iter_plot_img = cfg.EPOCHS // 20
    """ Building model """
    models_class = SegmentationModels(device, in_channels=3, img_size=img_size, n_classes=n_classes)
    # model, preprocess_input = models_class.UNet(feature_start=16, layers=4, kernel_size=5, padding=2, dropout=0.0)
    model, preprocess_input = models_class.SwinUnet(
        pretrain=True,
        embed_dim=cfg.EMBED_DIM,
        depths=cfg.DEPTHS,
        num_heads=cfg.NUM_HEADS,
        window_size=cfg.WINDOW_SIZE,
        drop_path_rate=0.0,
        mlp_ratio=4.
    )
    # model, preprocess_input = models_class.UNet_imagenet(encoder_name='resnet18')
    # model, preprocess_input = models_class.sunet()
    try:
        name_model = model.__name__
    except:
        name_model = 'UNet_imagenet'
        pass

    models_class.summary(logger=logger)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Total_params:{pytorch_total_params}')

    """ Getting loader """ 
    train_loader, val_loader = loaders(train_imgdir=cfg.TRAIN_IMAGES,
                                       train_maskdir=cfg.TRAIN_MASKS,
                                       val_imgdir=cfg.VAL_IMAGES,
                                       val_maskdir=cfg.VAL_MASKS,
                                       batch_size=cfg.BATCH_SIZE,
                                       num_workers=cfg.NUM_WORKERS,
                                       pin_memory=True,
                                       preprocess_input=preprocess_input
                                       )

    if len(gpus_ids) > 1:
        print("Data parallel...")
        model = nn.DataParallel(model, device_ids=gpus_ids)
    
    """ Prepare training """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(B1, B2))
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    loss_fn = WeightedCrossEntropyDice(class_weights=class_weights, device=device)
    # loss_fn = DiceLoss(device=device)
    # loss_fn = CrossEntropyLoss(class_weights, device=device)
    # loss_fn = BinaryCrossEntropyLoss(device)
    # metrics = mIoU(device=device)
    metrics = SegmentationMetrics()
    # metrics = IoU_binary(device)
    if cfg.SCHEDULER == 'step':
        scheduler = StepLR(optimizer=optimizer, step_size=cfg.STEP_SIZE, gamma=cfg.GAMMA)
    if cfg.SCHEDULER == 'cosine':
        scheduler = CyclicCosineDecayLR(optimizer,
                                        init_decay_epochs=num_epochs // 10,
                                        min_decay_lr=lr / 20,
                                        restart_interval=num_epochs // 20,
                                        restart_lr=lr / 5)

    """ Trainer """
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
            iter_plot_img=10,
            name_model=name_model,
            tb_dir=version,
            logger=logger,
            callback_stop_value=num_epochs // 2,
            )
    logger.info('-------------------- Finished Train ---------------------')
    logger.info('******************* Start evaluation  *******************')
    load_best_model = torch.load(checkpoint_path + 'model.pth')
    loss_eval = eval(load_best_model, val_loader, loss_fn, device)
    print([loss_eval])


def initialize():
    """Directories"""
    ver_ = 0
    while(os.path.exists(f"logs/version{ver_}/")):
        ver_ += 1
    version = f"logs/version{ver_}/"
    version = datetime.datetime.now()
    version = 'logs/' + str(version.strftime("%Y-%m-%d_%H_%M_%S")) + '/'
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

    """logging"""
    logging.basicConfig(filename=version + "info.log",
                        filemode='a',
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger()
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stdout_handler)

    """ Seeding """
    seeding(42)  # 42
    return logger, checkpoint_path, version


if __name__ == '__main__':
    main()
