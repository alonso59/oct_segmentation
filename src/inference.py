import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np
import albumentations as T
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns

from PIL import Image
from utils import get_filenames, create_dir
from patchify import patchify, unpatchify

from training.loss import *
from training.trainer import eval
from training.metrics import mIoU
from training.dataset import ImagesFromFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix




def inference_path(model, image, mask, pred_imgdir, pred_mskdir, pred_predsdir):
    save_image_filename = pred_imgdir + os.path.splitext(os.path.split(image)[1])[0] + '.png'
    save_mask_filename = pred_mskdir + os.path.splitext(os.path.split(image)[1])[0] + '.png'
    save_pred_filename = pred_predsdir + os.path.splitext(os.path.split(image)[1])[0] + '.png'
    imgzs = 128
    large_image = np.load(image)
    large_mask = np.load(mask)

    patches_images = patchify(large_image, (imgzs, imgzs), step=imgzs)
    patches_masks = patchify(large_mask, (imgzs, imgzs), step=imgzs)

    preds = []
    iou = []
    y_trues = []
    y_preds = []

    for i in range(patches_images.shape[0]):
        for j in range(patches_images.shape[1]):
            image_x = patches_images[i, j, :, :]
            mask_y = patches_masks[i, j, :, :]
            pred, mean_iou = predict_w_gt(model, image_x, mask_y)
            y_trues.append(mask_y.reshape(-1))
            y_preds.append(pred.reshape(-1))
            iou.append(mean_iou)
            preds.append(pred)

    print(np.array(iou).mean())
    preds = np.reshape(preds, patches_images.shape)
    preds = np.array(preds)

    
    rec_img = unpatchify(patches=patches_images, imsize=(imgzs * preds.shape[0], imgzs * preds.shape[1]))
    rec_msk = unpatchify(patches=patches_masks, imsize=(imgzs * preds.shape[0], imgzs * preds.shape[1]))
    rec_pred = unpatchify(patches=preds, imsize=(imgzs * preds.shape[0], imgzs * preds.shape[1]))

    shape_1 = (rec_pred.shape[0], rec_pred.shape[1], 3)

    rec_pred_rgb = np.zeros(shape=shape_1, dtype='uint8')

    rec_pred_rgb[:, :, 0] = np.where(rec_pred == 1, 1, rec_pred_rgb[:, :, 0])
    rec_pred_rgb[:, :, 1] = np.where(rec_pred == 2, 1, rec_pred_rgb[:, :, 1])
    rec_pred_rgb[:, :, 2] = np.where(rec_pred == 3, 1, rec_pred_rgb[:, :, 2])

    rec_pred_rgb = Image.fromarray(rec_pred_rgb * 255)
    rec_img = Image.fromarray((rec_img))
    rec_msk = Image.fromarray(((rec_msk / 3) * 255).astype('uint8'))

    rec_img.save(save_image_filename)
    rec_msk.save(save_mask_filename)
    rec_pred_rgb.save(save_pred_filename)
    
    return np.array(iou).mean()


def evaluation(model, images, masks):

    device = torch.device("cuda")

    loss_fn = DiceLoss(device)

    val_transforms = T.Compose([T.Normalize(mean=(0.1338, 0.1338, 0.1338), std=(0.1466, 0.1466, 0.1466))])

    val_ds = ImagesFromFolder(image_dir=images,
                              mask_dir=masks,
                              transform=val_transforms,
                              preprocess_input=None
                              )

    val_loader = DataLoader(val_ds,
                            batch_size=64,
                            num_workers=12,
                            pin_memory=True,
                            shuffle=False
                            )

    loss_eval = eval(model, val_loader, loss_fn, device)

    print([loss_eval])


def predict_w_gt(model, x_image, y_mask):
    device = torch.device("cuda")
    metric = mIoU(device)

    norm = T.Normalize(mean=(0.1338, 0.1338, 0.1338), std=(0.1466, 0.1466, 0.1466))

    image = np.expand_dims(x_image, axis=-1)
    image = np.repeat(image, 3, axis=-1)
    image = norm(image=image)
    image = image['image'].transpose((2, 0, 1))
    image = torch.tensor(image, dtype=torch.float, device=device)
    image = image.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        y_pred = model(image)

    y_mask = torch.tensor(y_mask, device=device, dtype=torch.long).unsqueeze(0).unsqueeze(0)

    iou = metric(y_pred, y_mask)
    iou = np.where(iou == 0, 1, iou)
    y_pred = F.softmax(y_pred, dim=1)
    y_pred = torch.argmax(y_pred, dim=1)
    y_pred = y_pred.squeeze(0).detach().cpu().numpy()

    return y_pred, iou.mean()


def main(cfg):
    model_path = 'logs/2022-07-06_11_08_39/checkpoints/model.pth'
    paths = cfg['paths']
    base = paths['base']

    val_imgdir = os.path.join(base, paths['val_imgdir'])
    val_mskdir = os.path.join(base, paths['val_mskdir'])

    test_imgdir = os.path.join(base, paths['test_imgdir'])
    test_mskdir = os.path.join(base, paths['test_mskdir'])

    pred_imgdir = os.path.join(base, paths['save_testimg'])
    pred_mskdir = os.path.join(base, paths['save_testmsk'])
    pred_predsdir = os.path.join(base, paths['save_testpred'])

    create_dir(pred_imgdir)
    create_dir(pred_mskdir)
    create_dir(pred_predsdir)

    files = get_filenames(test_imgdir, 'npy')
    filesM = get_filenames(test_mskdir, 'npy')
    
    model = torch.load(model_path)

    evaluation(model, val_imgdir, val_mskdir)

    iou = []
    fileImage = []

    for im, mk in zip(files, filesM):
        iou_item= inference_path(model, im, mk, pred_imgdir, pred_mskdir, pred_predsdir)
        iou.append(iou_item)
        fileImage.append(os.path.split(im)[1])

    iou = np.array(iou)

    df2 = pd.DataFrame({'file': fileImage,
                        'iou': iou,
                        })
    df2.to_csv('predictions.csv')
    iou.sort()
    hmean = np.mean(iou)
    hstd = np.std(iou)
    # _, q1, q3 = np.percentile(iou, 50), np.percentile(iou, 25), np.percentile(iou, 75)
    # sigma = hstd
    # mu = hmean
    # iqr = 1.5 * (q3 - q1)
    # x1 = np.linspace(q1 - iqr, q1)
    # x2 = np.linspace(q3, q3 + iqr)
    # pdf1 = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x1 - mu)**2 / (2 * sigma**2))
    # pdf2 = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x2 - mu)**2 / (2 * sigma**2))

    print(f'Mean:{hmean}, Std:{hstd}')

    # pdf = stats.norm.pdf(iou, hmean, hstd)
    # pl.plot(iou, pdf, '-o', label=f'Mean:{hmean:0.3f}, Std:{hstd:0.3f}, Q1:{q1:0.3f}, Q3:{q3:0.3f}')

    arran = np.linspace(0.5, 1, num=(len(iou)))
    plt.hist(iou, bins=arran, edgecolor='black')
    # pl.fill_between(x1, pdf1, 0, alpha=.6, color='green')
    # pl.fill_between(x2, pdf2, 0, alpha=.6, color='green')
    plt.xlim([0.4, 1.1])
    plt.xlabel('IoU', fontsize=18, fontweight='bold')
    plt.ylabel('No. Images', fontsize=18, fontweight='bold')
    # plt.legend(loc='best')
    plt.savefig('train.png')


if __name__ == '__main__':
    with open('configs/oct.yaml', "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    main(cfg)
