from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
from torchvision.utils import draw_segmentation_masks as drawer 
import sys


class TensorboardWriter():

    def __init__(self, metric, name_dir):

        super().__init__()
        self.writer = SummaryWriter(log_dir=name_dir)
        self.metric = metric

    def per_epoch(self, train_loss, val_loss, train_metric, val_metric, step):
        results_loss = {'Train': train_loss, 'Val': val_loss}
        results_metric = {'Train': train_metric, 'Val': val_metric}
        self.writer.add_scalars("Loss", results_loss, step)
        self.writer.add_scalars(
            f'{self.metric.__name__}', results_metric, step)

    def per_iter(self, loss, metric, step, name):
        self.writer.add_scalar(f"{name}/Loss", loss, step)
        self.writer.add_scalar(f'{name}/{self.metric.__name__}', metric, step)

    def learning_rate(self, lr_, step):
        self.writer.add_scalar("lr", lr_, step)

    def save_graph(self, model, loader):
        self.writer.add_graph(model, loader)

    def save_text(self, tag, text_string):
        self.writer.add_text(tag=tag, text_string=text_string)

    def save_images(self, x, y, y_pred, step, device):
        
        gt = image_tensorboard(y[:4, :, :], device)
        pred = F.softmax(y_pred[:4, :, :, :], dim=1)
        pred = torch.argmax(pred, dim=1)
        pred = image_tensorboard(pred, device)
        self.writer.add_images(f'Data', x[:4, :, :, :], step)
        self.writer.add_images(f'Ground truth', gt.unsqueeze(1), step)
        self.writer.add_images(f'Prediction', pred.unsqueeze(1), step)

def image_tensorboard(img, device):
    img_rgb = torch.zeros(img.size(), device=device)
    img_rgb = torch.div(img, img.max().item())
    return img_rgb