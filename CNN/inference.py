import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from patchify import patchify
import settings as cfg
from torchsummary import summary
import torch.nn.functional as F
import os
import multiprocessing as mp
from metrics import Accuracy, mIoU
from dataset import ImagesFromFolder
from torch.utils.data import DataLoader, Dataset
from trainer import eval
from loss import *
import scipy.stats as stats
import pylab as pl
import pandas as pd


def get_filenames(path, ext):
    X0 = []
    for i in sorted(os.listdir(path)):
        if i.endswith(ext):
            X0.append(os.path.join(path, i))
    return X0


def patching(img, size, shiftx, shifty):
    for i in range(size, img.shape[0], shifty):
        for j in range(size, img.shape[1], shiftx):
            pass


def plot_inference():
    files_images = get_filenames(cfg.VAL_IMAGES, 'tiff')
    files_masks = get_filenames(cfg.VAL_MASKS, 'tiff')

    np.random.seed(30)  # 2, 10, 20, 30 seed randome
    name = 'unet'
    x = np.random.randint(0, len(files_images))
    pick_image = files_images[x]

    large_image = np.array(Image.open(pick_image))
    patches_images = patchify(np.array(large_image), (128, 128), step=128)

    pick_mask = files_masks[x]
    large_mask = np.array(Image.open(pick_mask).convert('L'))
    patches_mask = patchify(np.array(large_mask), (128, 128), step=128)

    pred = np.zeros(patches_images.shape)

    for i in range(patches_images.shape[0]):
        for j in range(patches_images.shape[1]):
            single_image = patches_images[i, j, :, :]
            pred[i, j, :, :] = prediction(single_image, name)
    img1 = patches_images[0, :, :, :]
    img2 = patches_images[1, :, :, :]
    imgh1 = np.hstack((img1))
    imgh2 = np.hstack((img2))
    img = np.vstack((imgh1, imgh2))
    msk1 = patches_mask[0, :, :, :]
    msk2 = patches_mask[1, :, :, :]
    mskh1 = np.hstack((msk1))
    mskh2 = np.hstack((msk2))
    msk = np.vstack((mskh1, mskh2))
    pred1 = pred[0, :, :, :]
    pred2 = pred[1, :, :, :]
    predh1 = np.hstack((pred1))
    predh2 = np.hstack((pred2))
    predict = np.vstack((predh1, predh2))
    figure, ax = plt.subplots(nrows=3, ncols=1)
    figure.set_figwidth(12)
    figure.set_figheight(12)
    ax[0].imshow(img, cmap='gray')
    ax[0].title.set_text('Test image')
    ax[1].imshow(msk, cmap='jet')
    ax[1].title.set_text('Ground Truth')
    ax[2].imshow(predict, cmap='jet')
    ax[2].title.set_text('Prediction')
    figure.suptitle(name, fontsize=24)
    plt.savefig('dataset/' + os.path.splitext(os.path.split(pick_image)[1])[0] + '_' + name + '.tiff')
    # plt.show()

def inference_simple_path(image):
    name = 'unet'
    large_image = np.array(Image.open(image))
    patches_images = patchify(np.array(large_image), (224, 224), step=224)


    # print(patches_images.shape)
    iou = 0.0
    pred = np.zeros(
        (patches_images.shape[0],
         patches_images.shape[1],
         patches_images.shape[2],
         patches_images.shape[3],
         3))
    for i in range(patches_images.shape[0]):
        single_image = patches_images[i, :, :, :]
        pred[i, :, :, :, :]= prediction_simple(single_image)
    
    img1 = patches_images[0, :, :, :]
    img2 = patches_images[1, :, :, :]
    # img3 = patches_images[2, :, :, :]
    imgh1 = np.hstack((img1))
    imgh2 = np.hstack((img2))
    # imgh3 = np.hstack((img3))
    # img = np.vstack((imgh1, imgh2, imgh3))
    img = np.vstack((imgh1, imgh2))
    img = Image.fromarray(img)
    img.save('dataset/DUKE_preds/Images/' + os.path.splitext(os.path.split(image)[1])[0] + '_' + name + '.png')

    pred1 = pred[0, :, :, :]
    pred2 = pred[1, :, :, :]
    # pred3 = pred[2, :, :, :]
    predh1 = np.hstack((pred1))
    predh2 = np.hstack((pred2))
    # predh3 = np.hstack((pred3))
    # pred = np.vstack((predh1, predh2, predh3)) * 255
    pred = np.vstack((predh1, predh2)) * 255
    predict = pred.astype('uint8')
    predict = Image.fromarray(predict).convert('RGB')
    predict.save('dataset/DUKE_preds/Preds/' + os.path.splitext(os.path.split(image)[1])[0] + '_' + name + '.png')
    return iou / patches_images.shape[0]


def prediction_simple(image):
    """ CUDA device """
    device = torch.device("cuda")
    PATH = 'logs/version1/checkpoints/model.pth'
    image1 = image / 255.
    # image1 = np.expand_dims(image1, axis=0)
    image1 = np.expand_dims(image1, axis=1)
    image1 = np.repeat(image1, 3, axis=1)
    image1 = torch.tensor(image1, dtype=torch.float, device=device)
    best_model = torch.load(PATH)
    pr_mask = best_model(image1)

    pr_mask = F.softmax(pr_mask, dim=1)
    pr_mask = torch.argmax(pr_mask, dim=1)

    img_rgb = torch.zeros((pr_mask.size(0), 3, pr_mask.size(1), pr_mask.size(2)), dtype=torch.float, device=device)

    img_rgb[:, 0, :, :] = torch.where(pr_mask == 1, 1, 0)
    img_rgb[:, 1, :, :] = torch.where(pr_mask == 2, 1, 0)
    img_rgb[:, 2, :, :] = torch.where(pr_mask == 3, 1, 0)
    single_mask = (img_rgb.squeeze().cpu().long().detach().numpy())

    return single_mask.transpose(0, 2, 3, 1)



def inference_path_compared(image, mask):
    name = 'unet'
    large_image = np.array(Image.open(image))
    patches_images = patchify(np.array(large_image), (224, 224), step=224)

    large_mask = np.array(Image.open(mask))
    patches_mask = patchify(np.array(large_mask), (224, 224), step=224)

    # print(patches_images.shape, patches_mask.shape)
    iou = 0.0
    pred = np.zeros(
        (patches_images.shape[0],
         patches_images.shape[1],
         patches_images.shape[2],
         patches_images.shape[3],
         3))
    for i in range(patches_images.shape[0]):
        single_image = patches_images[i, :, :, :]
        single_mask = patches_mask[i, :, :, :]
        pred[i, :, :, :, :], iou1 = prediction(single_image, single_mask)
        iou += iou1

    print(iou / patches_images.shape[0])
    img1 = patches_images[0, :, :, :]
    img2 = patches_images[1, :, :, :]
    # img3 = patches_images[2, :, :, :]
    imgh1 = np.hstack((img1))
    imgh2 = np.hstack((img2))
    # imgh3 = np.hstack((img3))
    # img = np.vstack((imgh1, imgh2, imgh3))
    img = np.vstack((imgh1, imgh2))
    img = Image.fromarray(img)
    img.save('dataset/preds/Images/' + os.path.splitext(os.path.split(image)[1])[0] + '_' + name + '.png')

    img1 = patches_mask[0, :, :, :]
    img2 = patches_mask[1, :, :, :]
    # img3 = patches_mask[2, :, :, :]
    imgh1 = np.hstack((img1))
    imgh2 = np.hstack((img2))
    # imgh3 = np.hstack((img3))
    # imgx = np.vstack((imgh1, imgh2, imgh3))
    imgx = np.vstack((imgh1, imgh2))
    # imgx = ((imgx / 3) * 255).astype(np.uint8)
    img = Image.fromarray(imgx)
    img.save('dataset/preds/Masks/' + os.path.splitext(os.path.split(image)[1])[0] + '_' + name + '.png')

    pred1 = pred[0, :, :, :]
    pred2 = pred[1, :, :, :]
    # pred3 = pred[2, :, :, :]
    predh1 = np.hstack((pred1))
    predh2 = np.hstack((pred2))
    # predh3 = np.hstack((pred3))
    # pred = np.vstack((predh1, predh2, predh3)) * 255
    pred = np.vstack((predh1, predh2)) * 255
    predict = pred.astype('uint8')
    predict = Image.fromarray(predict).convert('RGB')
    predict.save('dataset/preds/pr/' + os.path.splitext(os.path.split(image)[1])[0] + '_' + name + '.png')
    return iou / patches_images.shape[0]


def inference_patches(images, masks):
    PATH = 'logs/version1/checkpoints/model.pth'
    device = torch.device("cuda")
    metric = mIoU(device)
    loss_fn = DiceLoss(device)
    model = torch.load(PATH)

    val_ds = ImagesFromFolder(image_dir=images,
                              mask_dir=masks,
                              transform=None,
                              preprocess_input=None
                              )
    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        num_workers=4,
        pin_memory=True,
        shuffle=False
    )
    loss_eval, iou_eval = eval(model, val_loader, loss_fn, metric, device)
    print([loss_eval, iou_eval])



def prediction(image, mask):
    """ CUDA device """
    device = torch.device("cuda")
    metric = mIoU(device)
    PATH = 'logs/version1/checkpoints/model.pth'
    image1 = image / 255.
    # image1 = np.expand_dims(image1, axis=0)
    image1 = np.expand_dims(image1, axis=1)
    image1 = np.repeat(image1, 3, axis=1)
    mask = torch.from_numpy(mask).unsqueeze(1)
    image1 = torch.tensor(image1, dtype=torch.float, device=device)
    best_model = torch.load(PATH)
    pr_mask = best_model(image1)

    iou = metric(pr_mask, mask.long())
    iou = np.where(iou == 0, 1, iou)

    pr_mask = F.softmax(pr_mask, dim=1)
    pr_mask = torch.argmax(pr_mask, dim=1)

    img_rgb = torch.zeros((pr_mask.size(0), 3, pr_mask.size(1), pr_mask.size(2)), dtype=torch.float, device=device)

    img_rgb[:, 0, :, :] = torch.where(pr_mask == 1, 1, 0)
    img_rgb[:, 1, :, :] = torch.where(pr_mask == 2, 1, 0)
    img_rgb[:, 2, :, :] = torch.where(pr_mask == 3, 1, 0)
    single_mask = (img_rgb.squeeze().cpu().long().detach().numpy())
    # print(iou)
    return single_mask.transpose(0, 2, 3, 1), iou.mean()


def main():
    files = get_filenames('OCT_DATASET/dataset/224_3C_May/Images_val_large/', 'tiff')
    filesM = get_filenames('OCT_DATASET/dataset/224_3C_May/Masks_val_large/', 'tiff')
    duke = get_filenames('OCT_DATASET/dataset/DUKE_bscans/AMD/', 'png')
    for I in duke:
        inference_simple_path(I)

    # inference_patches('OCT_DATASET/dataset/224_3C_May/val_images', 'OCT_DATASET/dataset/224_3C_May/val_masks')
    # plot_inference()
    # iou1 = []
    # fileImage = []
    # for I, M in zip(files, filesM):
    #     iou1.append(inference_path_compared(I, M))
    #     fileImage.append(os.path.split(I)[1])
    # iou = np.array(iou1)
    # print(iou.mean())

    # df2 = pd.DataFrame({'file': fileImage,
    #                     'iou': iou1,
    #                     })
    # df2.to_csv('predictions.csv')
    # iou.sort()
    # hmean = np.mean(iou)
    # hstd = np.std(iou)
    # median, q1, q3 = np.percentile(iou, 50), np.percentile(iou, 25), np.percentile(iou, 75)
    # sigma = hstd
    # mu = hmean
    # iqr = 1.5 * (q3 - q1)
    # x1 = np.linspace(q1 - iqr, q1)
    # x2 = np.linspace(q3, q3 + iqr)
    # pdf1 = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x1 - mu)**2 / (2 * sigma**2))
    # pdf2 = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x2 - mu)**2 / (2 * sigma**2))

    # print(f'Mean:{hmean}, Std:{hstd}')
    # print(len(iou))
    # pdf = stats.norm.pdf(iou, hmean, hstd)
    # pl.plot(iou, pdf, '-o', label=f'Mean:{hmean:0.3f}, Std:{hstd:0.3f}, Q1:{q1:0.3f}, Q3:{q3:0.3f}')

    # arran = np.linspace(0.5, 1, num=(len(iou)))
    # pl.hist(iou, bins=arran, edgecolor='black')
    # pl.fill_between(x1, pdf1, 0, alpha=.6, color='green')
    # pl.fill_between(x2, pdf2, 0, alpha=.6, color='green')
    # plt.xlim([0.4, 1.1])
    # plt.xlabel('IoU')
    # plt.ylabel('No. Images')
    # plt.legend(loc='best')
    # plt.savefig('test_best.png')


if __name__ == '__main__':
    # plot_inference()
    main()
