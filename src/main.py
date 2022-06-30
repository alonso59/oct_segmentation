import os
import yaml
import torch
import torch.nn as nn

from torch.optim.lr_scheduler import *

from models import SegmentationModels

from training.loss import *
from training.scheduler import *
from training.dataset import loaders
from training.trainer import trainer, eval

from initialize import initialize as init
from training.metric import SegmentationMetrics


# import torch.utils.tensorboard

def train(cfg):
    logger, checkpoint_path, version = init(cfg)
    paths = cfg['paths']
    hyper = cfg['hyperparameters']
    general = cfg['general']
    """ 
    Hyperparameters 
    """
    batch_size = hyper['batch_size']
    num_epochs = hyper['num_epochs']
    lr = hyper['lr']
    B1 = hyper['b1']
    B2 = hyper['b2']
    weight_decay = hyper['weight_decay']
    n_gpus = cfg['hyperparameters']['n_gpus']
    """
    Paths
    """
    train_imgdir = paths['train_imgdir']
    train_mskdir = paths['train_mskdir']
    val_imgdir = paths['val_imgdir']
    val_mskdir = paths['val_mskdir']
    """
    General settings
    """
    n_classes = general['n_classes']
    img_size = general['img_size']
    name_model = cfg['model_name']
    device = torch.device("cuda")
    """ 
    Getting loader
    """
    train_loader, val_loader = loaders(train_imgdir=train_imgdir,
                                       train_maskdir=train_mskdir,
                                       val_imgdir=val_imgdir,
                                       val_maskdir=val_mskdir,
                                       batch_size=batch_size,
                                       num_workers=os.cpu_count(),
                                       pin_memory=True,
                                       preprocess_input=None,
                                       )

    logger.info(f'Training items: {len(train_loader) * batch_size}')
    logger.info(f'Validation items: {len(val_loader) * batch_size}')

    """ 
    Building model 
    """
    models_class = SegmentationModels(device, config_file=cfg, in_channels=3, img_size=img_size, n_classes=n_classes)
    model, name_model = models_class.model_building(name_model=name_model)
    models_class.summary(logger=logger)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Total_params:{pytorch_total_params}')
    if n_gpus > 1:
        print("Data parallel...")
        model = nn.DataParallel(model, device_ids=[x for x in range(n_gpus)])
    """ 
    Prepare training 
    """
    if hyper['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(B1, B2))
    elif hyper['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=B1)
    else:
        raise AssertionError('Optimizer not implemented')
    assert hyper['loss_fn'] == 'dice_loss' or hyper['loss_fn'] == 'wce_dice', "Loss function not implemented"
    if hyper['loss_fn'] == 'dice_loss':
        loss_fn = DiceLoss(device)
    elif hyper['loss_fn'] == 'wce_dice':
        class_weights = [1, 1, 1]
        loss_fn = WeightedCrossEntropyDice(device, class_weights=class_weights)

    metrics = SegmentationMetrics()
    scheduler = StepLR(optimizer=optimizer, step_size=cfg['hyperparameters']['scheduler']['step'], gamma=cfg['hyperparameters']['scheduler']['gamma'])

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
            iter_plot_img=int(num_epochs * 0.1),
            name_model=name_model,
            callback_stop_value=int(num_epochs * 0.15),
            tb_dir = version,
            logger=logger
            )
    logger.info('-------------------- Finished Train ---------------------')
    logger.info('******************* Start evaluation  *******************')
    load_best_model = torch.load(checkpoint_path + 'model.pth')
    loss_eval = eval(load_best_model, val_loader, loss_fn, metrics, device)
    logger.info([loss_eval])


if __name__ == '__main__':
    with open('configs/oct.yaml', "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    train(cfg)