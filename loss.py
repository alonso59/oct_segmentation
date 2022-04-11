import torch
import torch.nn.functional as F
import torch.nn as nn


class WeightedCrossEntropyDice(nn.Module):
    def __init__(self, class_weights, device):
        super(WeightedCrossEntropyDice, self).__init__()
        self.device = device
        self.class_weights = torch.tensor(class_weights).float().to(device)
        self.CE = nn.CrossEntropyLoss(weight=self.class_weights)

    @property
    def __name__(self):
        return "weigthed_entropy_dice"

    def forward(self, inputs, targets, eps=1e-7):
        num_classes = inputs.shape[1]
        w = torch.ones(inputs.shape).type(inputs.type()).to(self.device)
        for c in range(inputs.shape[1]):
            w[:, c, :, :] = self.class_weights[c]

        # One Hot ground truth
        true_1_hot = torch.eye(num_classes)[targets.squeeze(1).long()]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float().to(self.device)
        true_1_hot = true_1_hot.type(inputs.type())

        # Getting probabilities
        probas = F.softmax(inputs, dim=1)

        # Compute DiceLoss
        mult = (probas * true_1_hot).to(self.device)
        dims = (0, 2, 3)
        intersection = 2 * torch.sum(mult, dim=(0, 2, 3)) + eps
        cardinality = torch.sum(probas, dim=dims) + torch.sum(true_1_hot, dim=dims) + eps
        dice_loss = 1 - (intersection / cardinality).mean()

        # Compute categorical cross entropy
        target1 = targets.squeeze(1).long().to(self.device)
        cross = self.CE(inputs.to(self.device), target1)

        return dice_loss * 0.5 + cross * 0.5


class WeightedCrossEntropyGDL(nn.Module):
    def __init__(self, class_weights, device):
        super(WeightedCrossEntropyGDL, self).__init__()
        self.device = device
        self.class_weights = torch.tensor(class_weights).float().to(device)
        self.CE = nn.CrossEntropyLoss(weight=self.class_weights)

    @property
    def __name__(self):
        return "weighted_entropy_generalized_dice"

    def forward(self, inputs, targets, eps=1e-7):
        num_classes = inputs.shape[1]
        w = torch.ones(inputs.shape).type(inputs.type()).to(self.device)
        for c in range(inputs.shape[1]):
            w[:, c, :, :] = self.class_weights[c]

        # One Hot ground truth
        true_1_hot = torch.eye(num_classes)[targets.squeeze(1).long()]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float().to(self.device)
        true_1_hot = true_1_hot.type(inputs.type())

        # Getting probabilities
        probas = F.softmax(inputs, dim=1)

        # Compute DiceLoss
        mult = (probas * true_1_hot).to(self.device)
        sum_w = torch.sum(w, dim=(0, 2, 3))
        dims = (0, 2, 3)
        intersection = 2 * torch.pow(sum_w, 2) * torch.sum(mult, dim=(0, 2, 3)) + eps
        cardinality = torch.pow(sum_w, 2) * (torch.sum(probas, dim=dims) + torch.sum(true_1_hot, dim=dims)) + eps
        dice_loss = 1 - (intersection / cardinality).mean()

        # Compute categorical cross entropy
        target1 = targets.squeeze(1).long().to(self.device)
        cross = self.CE(inputs.to(self.device), target1)

        return dice_loss * 0.5 + cross * 0.5


class DiceLoss(nn.Module):
    def __init__(self, device):
        super(DiceLoss, self).__init__()
        self.device = device
    @property
    def __name__(self):
        return "dice_loss"

    def forward(self, inputs, targets, eps=1e-7):
        num_classes = inputs.shape[1]
        # One Hot ground thrut
        true_1_hot = torch.eye(num_classes)[targets.squeeze(1).long()]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot = true_1_hot.type(inputs.type()).to(self.device)
        # Getting probabilities
        probas = F.softmax(inputs, dim=1).to(self.device)
        # Compute DiceLoss
        mult = (probas * true_1_hot).to(self.device)

        dims = (0, 2, 3)
        intersection = 2 * torch.sum(mult, dim=(0, 2, 3)) + eps
        cardinality = torch.sum(probas, dim=dims) + torch.sum(true_1_hot, dim=dims) + eps
        dice_score = 1 - (intersection / cardinality).mean()
        return dice_score


class CrossEntropyLoss(nn.Module):
    def __init__(self, class_weights, device):
        super(CrossEntropyLoss, self).__init__()
        self.device = device
        self.class_weights = torch.tensor(class_weights).float().to(device)
        self.CE = nn.CrossEntropyLoss(weight=self.class_weights)

    @property
    def __name__(self):
        return "cross_entropy"

    def forward(self, inputs, targets):
        target1 = targets.squeeze(1).long().to(self.device)
        cross_entropy = self.CE(inputs.to(self.device), target1)
        return cross_entropy
