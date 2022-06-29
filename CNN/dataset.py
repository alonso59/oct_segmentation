import os
import numpy as np
import albumentations as T
import cv2
from torch.utils.data import DataLoader, Dataset
from skimage.restoration import denoise_tv_chambolle
from albumentations.core.transforms_interface import ImageOnlyTransform
from utils import visualize
import sys


class ImagesFromFolder(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, preprocess_input=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        self.preprocess_input = preprocess_input

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        # read data
        # try:
        # image = np.array(cv2.imread(img_path, cv2.IMREAD_COLOR))
        # mask = np.array(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
        # except:
        image = np.load(img_path)
        mask = np.load(mask_path)

        image = np.expand_dims(image, axis=-1)
        image = np.repeat(image, 3, axis=-1).astype('uint8')

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        # if self.preprocess_input is not None and False:
        #     preprocessing = get_preprocessing(self.preprocess_input)
        #     augmentations = preprocessing(image=image)
        #     image = augmentations["image"]
        image = image.transpose(2, 0, 1)
        mask = np.expand_dims(mask, axis=0)
        return image, mask


def loaders(train_imgdir,
            train_maskdir,
            val_imgdir,
            val_maskdir,
            batch_size,
            num_workers=4,
            pin_memory=True,
            preprocess_input=None
            ):

    train_transforms = T.Compose(
        [   
            GrayGammaTransform(p=0.5),
            T.Rotate(limit=(-20, 20), p=1.0, border_mode=cv2.BORDER_CONSTANT),
            T.HorizontalFlip(p=0.5),
            # T.RandomBrightnessContrast(p=0.3),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, p=0.5),
            T.Affine(scale=(0.95, 1.05), p=0.5),
            T.CLAHE(clip_limit=2.0, tile_grid_size=(3, 3), p=0.5),
            # TVDenoising(p=0.3),
            T.Normalize(mean=(0.1338, 0.1338, 0.1338), std=(0.1466, 0.1466, 0.1466))
        ]
    )
    val_transforms = T.Compose(
        [
            T.Normalize(mean=(0.1338, 0.1338, 0.1338), std=(0.1466, 0.1466, 0.1466))
        ]
    )
    train_ds = ImagesFromFolder(image_dir=train_imgdir,
                                mask_dir=train_maskdir,
                                transform=train_transforms,
                                preprocess_input=preprocess_input
                                )
    val_ds = ImagesFromFolder(image_dir=val_imgdir,
                              mask_dir=val_maskdir,
                              transform=val_transforms,
                              preprocess_input=preprocess_input
                              )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )
    return train_loader, val_loader


def get_preprocessing(preprocessing_fn):
    _transform = [
        T.Lambda(image=preprocessing_fn),
    ]
    return T.Compose(_transform)


class GrayLogTransform(ImageOnlyTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        return gray_log(img)

class GrayGammaTransform(ImageOnlyTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        return gray_gamma(img)

class TVDenoising(ImageOnlyTransform):
    def __init__(self, always_apply: bool = False, p: float = 0.5):
        super().__init__(always_apply, p)
    def apply(self, img, **params):
        return tv_denoising(img)


def gray_log(img):
    gray = img / 255.
    c = np.log10(1 + np.max(gray))
    out = c * np.log(1 + gray)
    return out


def gray_gamma(img):
    gamma = np.random.uniform(0.5, 2)
    gray = img / 255.
    out = np.array(gray ** gamma)
    out = 255*out
    return out.astype('uint8')


def tv_denoising(img):
    w = np.random.uniform(0, 0.1)
    gray = img / 255.
    out = denoise_tv_chambolle(gray, weight=w)
    out = out * 255
    return out.astype('uint8')


def test():
    train_transforms = T.Compose(
        [
            GrayGammaTransform(p=1.0),
            T.Rotate(limit=(-40, 40), p=1.0, border_mode=cv2.BORDER_CONSTANT),
            T.HorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, p=1.0),
            T.Affine(scale=(0.95, 1.05), p=1.0),
            T.CLAHE(clip_limit=2.0, tile_grid_size=(3, 3), p=1.0),
            TVDenoising(p=1.0, w=0.03)
        ]
    )
    train_ds = ImagesFromFolder(image_dir='dataset/new_128_3C/Images_train_large',
                                mask_dir='dataset/new_128_3C/Masks_train_large',
                                transform=train_transforms,
                                )
    randint = np.random.randint(low=0, high=len(train_ds))
    imgs = []
    msks = []
    for i in range(10):
        image, mask = train_ds[randint]
        imgs.append(image)
        msks.append(mask)
    visualize(len(imgs), np.array(imgs), np.array(msks), pr_mask=None, path_save='', metric_dict=None)


if __name__ == "__main__":
    test()  # test
