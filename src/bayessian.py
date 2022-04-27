import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import sys

from ray import tune
from ray.tune.suggest.ax import AxSearch
from ax.service.ax_client import AxClient

from train import *
from src.trainer import eval

def experiment(parameters):
    logger, checkpoint_path, version = initialize()
    """ 
    Hyperparameters 
    """
    batch_size = parameters.get("batch_size", 128)
    num_epochs = 500
    lr = parameters.get("lr", 0.001)
    B1 = parameters.get("beta1", 0.9)
    B2 = parameters.get("beta2", 0.999)
    weight_decay = parameters.get("weight_decay", 0)
    # class_weights = [0.2644706,  12.33872479, 12.23935952, 17.82146076]
    class_weights = [0.6, 1, 1, 1]
    """
    General settings
    """
    n_classes = cfg.CLASSES
    img_size = cfg.IMAGE_SIZE
    pretrain = True
    device = torch.device(f"cuda" if torch.cuda.is_available() else 'cpu')
    """
    Paths
    """
    train_imgdir = '/home/alonso/Documents/torch_segmentation/dataset/224_3C/train_images'
    train_maskdir ='/home/alonso/Documents/torch_segmentation/dataset/224_3C/train_masks'
    val_imgdir = '/home/alonso/Documents/torch_segmentation/dataset/224_3C/val_images'
    val_maskdir = '/home/alonso/Documents/torch_segmentation/dataset/224_3C/val_masks'
    """ 
    Getting loader
    """
    train_loader, val_loader = loaders(train_imgdir=train_imgdir,
                                       train_maskdir=train_maskdir,
                                       val_imgdir=val_imgdir,
                                       val_maskdir=val_maskdir,
                                       batch_size=batch_size,
                                       num_workers=os.cpu_count(),
                                       pin_memory=True
                                       )
    iter_plot_img = int(cfg.EPOCHS * 0.1)
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
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    """ 
    Prepare training 
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(B1, B2))
    # loss_fn = WeightedCrossEntropyDice(class_weights=class_weights, device=device)
    loss_fn = DiceLoss(device=device)
    metrics = mIoU(device)
    if cfg.SCHEDULER == 'step':
        scheduler = StepLR(optimizer=optimizer, step_size=cfg.STEP_SIZE, gamma=cfg.GAMMA)
    elif cfg.SCHEDULER == 'cosine':
        scheduler = CyclicCosineDecayLR(optimizer,
                                    init_decay_epochs=num_epochs // 3,
                                    min_decay_lr=lr / 10,
                                    restart_interval=num_epochs // 10,
                                    restart_lr=lr / 5)
    else:
        print('Invaild loss function!')
        sys.exit()

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
            callback_stop_value=num_epochs // 5,
            tb_dir = version,
            logger=logger
            )
    logger.info('-------------------- Finished Train ---------------------')
    logger.info('******************* Best checkpoint evaluation  *******************')
    load_best_model = torch.load(checkpoint_path + 'model.pth')
    logger.info('************************* Report to Ray  **************************')
    tune.report(loss_eval=eval(load_best_model, val_loader, loss_fn, device))
    # loss_eval = eval(load_best_model, val_loader, loss_fn, device)
    # logger.info([loss_eval])
    # logger.info('******************* Last weights evaluation  **********************')
    # load_last_model = torch.load(checkpoint_path + 'model_last.pth')
    # loss_eval = eval(load_last_model, val_loader, loss_fn, device)
    # logger.info([loss_eval])
    
    

def initialize():
    """
    Directories
    """
    trial = 0
    while(os.path.exists(f"/home/alonso/Documents/torch_segmentation/logs/bayessian224" + f"/trial{trial}/")):
        trial += 1
    version = f"/home/alonso/Documents/torch_segmentation/logs/bayessian224" + f"/trial{trial}/"
    checkpoint_path = version + "checkpoints/"
    create_dir(checkpoint_path)
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


def bayessian():
    logging.basicConfig(filename="bayessian1.log",
                        filemode='a',
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    loggerB = logging.getLogger()
    stdout_handler = logging.StreamHandler(sys.stdout)
    loggerB.addHandler(stdout_handler)
    ax = AxClient(enforce_sequential_optimization=False)
    
    ax.create_experiment(name="swin_experiment",
                         parameters=[
                            {"name": "lr", "type": "range", "bounds": [5e-4, 5e-3], "log_scale": True},
                            {"name": "batch_size", "type": "range", "bounds": [64, 100]},
                            {"name": "weight_decay", "type": "range", "bounds": [1e-5, 1e-3]},
                            {"name": "beta1", "type": "range", "bounds": [0.5, 0.9]},
                            {"name": "beta2", "type": "range", "bounds": [0.5, 0.999]}
                         ],
                         objective_name="loss_eval",
                         minimize=True
                        )
    # Set up AxSearcher in RayTune
    algo = AxSearch(ax_client=ax)
    # Wrap AxSearcher in a concurrently limiter, to ensure that Bayesian optimization receives the
    # data for completed trials before creating more trials
    algo = tune.suggest.ConcurrencyLimiter(algo, max_concurrent=4)
    tune.run(experiment,
            num_samples=30,
            search_alg=algo,
            resources_per_trial={"gpu": 1},
            verbose=2,  # Set this level to 1 to see status updates and to 2 to also see trial results.
            # To use GPU, specify: resources_per_trial={"gpu": 1}.
            )
    best_parameters, values = ax.get_best_parameters()
    means, covariances = values
    loggerB.info(best_parameters)
    loggerB.info(means)
    loggerB.info(covariances)

if __name__ == '__main__':
    bayessian()