import os
import sys
import shutil
import numpy as np
import configparser
import multiprocessing as mp
import matplotlib.pyplot as plt

from PIL import Image
from eyepy.core.base import Oct
from sklearn.model_selection import train_test_split

config = configparser.ConfigParser()
config.read("config.ini")
paths = config['PATHS']
settings = config['SETTINGS']


def vol_files(name):
    # print(name)
    vol_filename = os.path.join(paths.get('OCT_PATH'), name + '.vol')
    oct_read = Oct.from_heyex_vol(vol_filename)
    return oct_read


def get_annotations(oct_read):
    # Outer Plexiform Layer: OPL
    OPL = np.round(
        oct_read.bscans[0].annotation["layers"]['OPL']).astype('uint16')
    INL = np.round(
        oct_read.bscans[0].annotation["layers"]['INL']).astype('uint16')
    # Ellipsoide Zone: EZ
    PR2 = np.round(
        oct_read.bscans[0].annotation["layers"]['PR2']).astype('uint16')
    PR1 = np.round(
        oct_read.bscans[0].annotation["layers"]['PR1']).astype('uint16')
    # BM
    try:
        BM = np.round(
            oct_read.bscans[0].annotation["layers"]['BM']).astype('uint16')
    except:
        BM = np.zeros((PR1.shape[0])).astype('uint16')
        pass
    # ELM
    try:
        ELM = np.round(
            oct_read.bscans[0].annotation["layers"]['ELM']).astype('uint16')
    except:
        ELM = np.zeros((PR1.shape[0])).astype('uint16')
        pass
    return OPL, INL, PR2, PR1, BM, ELM


def crop_overlap(file, img, msk):
    oct_read = vol_files(file)
    OPL, INL, PR2, PR1, BM, ELM = get_annotations(oct_read)
    j = 1
    k = 0

    image = np.array(Image.open(img))
    mask = np.array(Image.open(msk))
    size = 64
    shift = 64
    for i in range(size, image.shape[1], 32):
        min_pixel = np.max(INL[k:i])
        max_pixel = np.max(PR1[k:i])
        print(np.max(mask))
        if min_pixel != 0 and max_pixel != 0 and max_pixel > min_pixel:
            delta1 = min_pixel - 64 if min_pixel - 64 > 0 else 0
            delta2 = max_pixel + 64 if max_pixel + 64 < 496 else 496
            img_save = image[delta1:delta2, i - size:i]
            msk_save = mask[delta1:delta2, i - size:i] / np.max(mask) * 255.
            img1 = Image.fromarray(img_save).convert('L')
            img1.save("testI.bmp")
            msk1 = Image.fromarray(msk_save).convert('L')
            msk1.save("testM.bmp")
            break
            j += 1
        k = i


def read_images():
    filname = 'Anonym_1_1896'
    img_file = 'dataset/data_224_3C/Images_train_large/Anonym_1_1896.bmp'
    msk_file = 'dataset/data_224_3C/Masks_train_large/Anonym_1_1896.bmp'
    crop_overlap(filname, img_file, msk_file)


def get_histo(img):
    image = np.array(Image.open(img))

    x = []
    for i in range(image.shape[0]):
        x.append(image[i, :].mean())
    
    print(np.array(x).shape)
    print(np.array(x))
    plt.figure()
    plt.plot(-np.arange(image.shape[0]), np.array(x))
    plt.show()
if __name__ == "__main__":
    # read_images()
    image = 'testI.bmp'
    get_histo(image)