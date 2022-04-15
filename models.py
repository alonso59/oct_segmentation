from networks.swin_unet import SwinUnet
from networks.unet import Unet
import torch.nn as nn
from torchsummary import summary


class ModelSegmentation(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.device = device

    def unet(self, in_channels=3, n_classes=1, img_size=512, feature_start=64,
             layers=4, bilinear=False, dropout=0.0, kernel_size=3, stride=1, padding=1):

        model = Unet(
            num_classes=n_classes,
            input_channels=in_channels,
            num_layers=layers,
            features_start=feature_start,
            bilinear=bilinear,
            dp=dropout,
            kernel_size=(kernel_size, kernel_size),
            padding=padding,
            stride=stride
        ).to(self.device)

        summary(model, input_size=(in_channels, img_size, img_size), batch_size=-1)

        return model

    def swin_unet(self, in_channels=3, n_classes=1, img_size=224, pretrain=True):

        model = SwinUnet(in_chans=in_channels, img_size=img_size, num_classes=n_classes).to(self.device)
        
        if pretrain:
            model.state_dict()
            model.load_from("pretrained/swin_tiny_patch4_window7_224.pth", self.device)
        summary(model, input_size=(in_channels, img_size, img_size), batch_size=-1)
        return model

    """
    you can add your own network here
    .
    .
    .
    """
