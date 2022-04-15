import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from src.trainer import eval

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from ax.utils.tutorials.cnn_utils import train, evaluate
import sys

from train import *


def experiment(parameters):
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
    batch_size = parameters.get("batch_size", 128)
    num_epochs = hyperparameters.getint('num_epochs')
    lr = parameters.get("lr", 0.001)
    B1 = parameters.get("beta1", 0.9)
    B2 = parameters.get("beta1", 0.999)
    weight_decay = parameters.get("weight_decay", 0)
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
    scheduler = StepLR(optimizer=optimizer, step_size=60, gamma=0.8)
    # scheduler = CyclicCosineDecayLR(optimizer,
    #                                 init_decay_epochs=500,
    #                                 min_decay_lr=5e-6,
    #                                 restart_interval=50,
    #                                 restart_lr=5e-6,
    #                                 warmup_epochs=200,
    #                                 warmup_start_lr=0.0005)
    """
    Directories
    """
    checkpoint_path = "checkpoints/" + datetime.datetime.now().strftime("%d%H%M%S_") + name_model + '/'
    create_dir("checkpoints")
    create_dir(checkpoint_path)
    with open(checkpoint_path + 'experiment.ini', 'w') as configfile:
        config.write(configfile)
    with open(checkpoint_path + 'bayessian.ini', 'w') as text_file:
        text_file.write(f"Learning rate: {lr}\n")
        text_file.write(f"weight_decay: {weight_decay}\n")
        text_file.write(f"BETA1, BETA2: {B1, B2}\n")
        text_file.write(f"Epochs: {num_epochs}\n")
        text_file.write(f"Batch size: {batch_size}\n")
        text_file.close()
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
    load_best_model = torch.load(checkpoint_path + 'model.pth')
    _, iou_eval = eval(load_best_model, val_loader, loss_fn, metrics, device)
    return iou_eval

def bayessian():
    best_parameters, values, exp, model = optimize(
        parameters=[
            {"name": "lr", "type": "range", "bounds": [5e-4, 5e-3], "log_scale": True},
            {"name": "batch_size", "type": "range", "bounds": [64, 128]},
            {"name": "weight_decay", "type": "range", "bounds": [0.0, 1e-3]},
            {"name": "beta1", "type": "range", "bounds": [0.5, 0.9]},
            {"name": "beta2", "type": "range", "bounds": [0.5, 0.999]}
            # {"name": "num_epochs", "type": "range", "bounds": [300, 500]},
        ],
        total_trials=20,
        evaluation_function=experiment,
        objective_name='iou',
    )

    print(best_parameters)
    print(exp)
    means, covariances = values
    print(means)
    print(covariances)

    best_objectives = np.array([[trial.objective_mean*100 for trial in exp.trials.values()]])

    best_objective_plot = optimization_trace_single_method(
        y=np.maximum.accumulate(best_objectives, axis=1),
        title="Model performance vs. # of iterations",
        ylabel="Classification Accuracy, %",
    )
    render(best_objective_plot)

    render(plot_contour(model=model, param_x='batchsize', param_y='lr', metric_name='accuracy'))

if __name__ == '__main__':
    bayessian()