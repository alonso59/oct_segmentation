import os
import numpy as np
import albumentations as T
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from albumentations.pytorch import ToTensorV2
from src.utils import visualize

class ImagesFromFolder(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, channels=3):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.channels = channels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        if self.channels == 3:
            image = np.array(Image.open(img_path).convert('RGB')) / 255.
        else:
            image = np.array(Image.open(img_path)) / 255.

        mask = np.array(Image.open(mask_path))

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        return image, mask


def loaders(train_imgdir,
            train_maskdir,
            val_imgdir,
            val_maskdir,
            batch_size,
            num_workers=4,
            pin_memory=True,
            channels=3
            ):

    train_transforms = T.Compose(
        [

            T.Rotate(limit=(-10, 10), p=1.0),
            T.HorizontalFlip(p=0.5),
            ToTensorV2(),
        ]
    )

    val_transforms = T.Compose(
        [
            ToTensorV2(),
        ]
    )

    train_ds = ImagesFromFolder(image_dir=train_imgdir,
                                mask_dir=train_maskdir,
                                transform=train_transforms,
                                channels=channels
                                )

    val_ds = ImagesFromFolder(image_dir=val_imgdir,
                              mask_dir=val_maskdir,
                              transform=val_transforms,
                              channels=channels
                              )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return train_loader, val_loader


def test():
    train_transforms = T.Compose(
        [
            T.Rotate(limit=(-15, 15), p=1.0),
            T.HorizontalFlip(p=0.5),
            T.Affine(scale=(0.9, 1.1), p=0.5),
            # T.CLAHE(clip_limit=2.0, tile_grid_size=(3, 3), p=1.0)
        ]
    )
    train_ds = ImagesFromFolder(image_dir='data4/train_images',
                                mask_dir='data4/train_masks',
                                transform=train_transforms,
                                )
    randint = np.random.randint(low=0, high=len(train_ds))
    imgs=[]
    msks=[]
    for i in range(3):
        image, mask = train_ds[randint]
        imgs.append(image)
        msks.append(mask)
    visualize(len(imgs), np.array(imgs), np.array(msks), pr_mask=None, path_save=None, metric_dict=None)


if __name__ == "__main__":
    test()  # test
