from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
from torchvision.utils import draw_segmentation_masks as drawer 
import sys
import numpy as np
from torchvision import transforms
from torch.nn import functional as F
class TensorboardWriter(object):

    def __init__(self, metric, name_dir):

        super(TensorboardWriter).__init__()
        self.writer = SummaryWriter(log_dir=name_dir)
        self.metric = metric

    def per_iter(self, loss, metric, step, name):
        self.writer.add_scalar(f"{name}/Loss", loss, step)
        self.writer.add_scalar(f'{name}/IoU', metric, step)

    def learning_rate(self, lr_, step):
        self.writer.add_scalar("lr", lr_, step)