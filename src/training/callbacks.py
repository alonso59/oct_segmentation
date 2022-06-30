from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
from torchvision.utils import draw_segmentation_masks as drawer 
import sys
import numpy as np
from torchvision import transforms
from torch.nn import functional as F
import albumentations as T

class TensorboardWriter(object):

    def __init__(self, metric, name_dir):

        super(TensorboardWriter).__init__()
        self.writer = SummaryWriter(log_dir=name_dir)
        self.metric = metric

    def loss_epoch(self, train_loss, val_loss, step):
        results_loss = {'Train': train_loss, 'Val': val_loss}
        self.writer.add_scalars("Loss", results_loss, step)
    
    def metrics_epoch(self, train_metric, val_metric, step, metric_name):
        results_metric = {'Train'+'/'+metric_name: train_metric, 'Val'+'/'+metric_name: val_metric}
        self.writer.add_scalars(metric_name, results_metric, step)

    def metric_iter(self, metric, step, stage, metric_name):
        self.writer.add_scalar(stage + '/' + metric_name, metric, step)

    def loss_iter(self, loss, step, stage: str):
        self.writer.add_scalar(stage + '/Loss', loss, step)

    def learning_rate(self, lr_, step):
        self.writer.add_scalar("lr", lr_, step)

    def save_graph(self, model, loader):
        self.writer.add_graph(model, loader)

    def save_text(self, tag, text_string):
        self.writer.add_text(tag=tag, text_string=text_string)

    def save_images(self, x, y, y_pred, step, device, tag):
        gt = image_tensorboard(y[:4, :, :], device)
        if y_pred.shape[1] == 1:
            pred = torch.sigmoid(y_pred[:4, :, :, :])
            pred = torch.round(pred)
        else:
            pred = torch.softmax(y_pred[:4, :, :, :], dim=1)
            pred = torch.argmax(pred, dim=1).unsqueeze(1)
        pred = image_tensorboard(pred, device)
        x1 = denormalize_vis(x[:4, :, :, :])
        self.writer.add_images(f'Data_'+tag, x1[:4, :, :, :], step, dataformats='NCHW')
        self.writer.add_images(f'Ground truth_'+tag, gt, step, dataformats='NCHW')
        self.writer.add_images(f'Prediction_'+tag, pred.squeeze(1), step, dataformats='NCHW')

def image_tensorboard(img, device):
    img_rgb = torch.zeros((img.size(0), 3, img.size(2), img.size(3))).float().to(device)
    img_rgb[:, 0, :, :] = torch.where(img.squeeze(1) == 1, 1, 0)
    img_rgb[:, 1, :, :] = torch.where(img.squeeze(1) == 2, 1, 0)
    img_rgb[:, 2, :, :] = torch.where(img.squeeze(1) == 3, 1, 0)
    return img_rgb

def denormalize_vis(tensor):
    invTrans = transforms.Normalize(mean=(-0.1338, -0.1338, -0.1338), std=(1/0.1466, 1/0.1466, 1/0.1466))
    return torch.clamp(invTrans(tensor), 0, 1)