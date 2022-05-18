import sys

import torch
import torch.nn.functional as F
import torch.nn as nn


class Accuracy(nn.Module):

    def __init__(self, class_index):
        super().__init__()
        self.class_index = class_index
    @property
    def __name__(self):
        return "accuracy"

    def forward(self, y_pr, y_gt):
        num_classes = y_pr.shape[1]
        true_1_hot = torch.eye(num_classes)[y_gt.squeeze(1).long()]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot = true_1_hot.type(y_pr.type())
        dims = (0,) + tuple(range(2, y_gt.ndimension()))
        # Getting probabilities
        y_pr = F.softmax(y_pr, dim=1).unsqueeze(1)
        y_pr = y_pr.reshape(-1)
        true_1_hot = true_1_hot.reshape(-1)
        tp = torch.sum(true_1_hot == y_pr)
        score = tp / true_1_hot.shape[0]
        return score


class mIoU(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device
    @property
    def __name__(self):
        return "mIoU"

    def forward(self, logits, true, eps=1e-5):
        num_classes = logits.shape[1]
        # print(true.squeeze(1).size())
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot = true_1_hot.type(true.type()).to(self.device)
        probas = F.softmax(logits, dim=1).to(self.device)
        dims = (0, 2, 3)
        mult = (probas * true_1_hot).to(self.device)
        sum = (probas + true_1_hot).to(self.device)
        intersection = torch.sum(mult, dim=dims)
        cardinality = torch.sum(sum, dim=dims)
        union = cardinality - intersection
        iou_score = (intersection / (union + eps))
        # print(iou_score)
        return iou_score.cpu().detach().numpy()

class IoU_binary(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device
    @property
    def __name__(self):
        return "mIoU"

    def forward(self, logits, true, eps=1e-5):
        true_1_hot = torch.eye(2, device=self.device)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
        dims = (0,) + tuple(range(2, true.ndimension()))
        mult = (probas * true_1_hot)
        sum = (probas + true_1_hot)
        intersection = torch.sum(mult, dim=dims)
        cardinality = torch.sum(sum, dim=dims)
        union = cardinality - intersection
        iou_score = (intersection / (union + eps))
        # print(iou_score)
        return iou_score.cpu().detach().numpy()