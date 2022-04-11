import datetime
import numpy as np
from src.trainer import trainer
from src.dataset import loaders
import os
from matplotlib import pyplot as plt
from src.utils import create_dir, seeding
from src.loss import *
from src.metrics import mIoU
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from networks.swin_unet import SwinUnet
from networks.unet import Unet
import sys
import torch
from torchsummary import summary


def main():
    """ Seeding """
    seeding(42) #42

    """ Configuration parameters """
    lr = 0.001
    batch_size = 4
    num_epochs = 100
    model_name = 'swin'
    if model_name == 'unet':
        checkpoint_path = "checkpoints/" + \
            datetime.datetime.now().strftime("%d_%H-%M_UNET_OCT/")
    elif model_name == 'swin':
        checkpoint_path = "checkpoints/" + \
            datetime.datetime.now().strftime("%d_%H-%M_SWIN_OCT/")
    else:
        print('Invalid model!')
        sys.exit()
    gpus_ids = [0]

    """ CUDA device """
    device = torch.device(f"cuda")

    """ Dataset and loader """
    train_loader, val_loader = loaders(train_imgdir='data4/train_images',
                                       train_maskdir='data4/train_masks',
                                       val_imgdir='data4/val_images',
                                       val_maskdir='data4/val_masks',
                                       batch_size=batch_size,
                                       num_workers=os.cpu_count(),
                                       pin_memory=True,
                                       channels=3
                                       )

    """ Building model """
    model = SwinUnet(in_chans=3, img_size=224, num_classes=4).to(device)
    model.state_dict()
    model.load_from("pretrained/swin_tiny_patch4_window7_224.pth", device)

    summary(model, input_size=(3, 224, 224), batch_size=-1)

    if len(gpus_ids) > 1:
        model = nn.DataParallel(model, device_ids=gpus_ids)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    """ Prepare training """

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4,
                                 betas=(0.7, 0.999))
    class_weights = [1, 1, 1, 1]
    loss_fn = WeightedCrossEntropyDice(
        class_weights=class_weights, device=device)
    metrics = mIoU(device)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.8)
    """ Directories """
    create_dir("checkpoints")
    create_dir(checkpoint_path)

    """ Training the model """
    trainer(num_epochs=num_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        metric=metrics,
        device=device,
        checkpoint_path=checkpoint_path,
        base_lr=lr,
        scheduler=scheduler,
        iter_plot_img=10
        )


if __name__ == '__main__':
    main()
