import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from eyepy.core.base import Oct

def vol_files(name):
    vol_filename = os.path.join('dataset/OCT3', name + '.vol')
    oct_read = Oct.from_heyex_vol(vol_filename)
    return oct_read

def get_filenames(path, ext):
    X0 = []
    for i in sorted(os.listdir(path)):
        if i.endswith(ext):
            X0.append(os.path.join(path, i))
    return X0

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

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

def get_images_masks(file):
    try:
        name = os.path.splitext(os.path.split(file)[1])[0]

        oct_read = vol_files(name)

        data = oct_read.bscans[0].scan
        data = np.expand_dims(data, axis=-1)
        dat1 = Image.fromarray(data.squeeze(2))

        zeros = np.zeros((data.shape[0], data.shape[1], 3)).astype('uint8')
        data1 = np.add(data, zeros)

        OPL, INL, PR2, PR1, BM, ELM = get_annotations(oct_read)

        # Generate ground truth
        mask = np.zeros((data.shape[0], data.shape[1])).astype('uint8')
        for i in range(OPL.shape[0]):
            data1[INL[i], i, 0] = 255
            data1[OPL[i], i, 1] = 220
            data1[PR2[i], i, 2] = 255
            data1[PR1[i], i, :] = [255, 255, 0]
            data1[ELM[i], i, :] = [255, 0, 255]
            data1[BM[i], i, :] = [0, 255, 255]

            # OPL
            mask[INL[i]:OPL[i], i] = 2 if INL[i] <= OPL[i] and INL[i] > 0 and OPL[i] > 0 else mask[INL[i]:OPL[i], i]
            # ELM
            # mask[ELM[i]:PR1[i], i] = 3 if ELM[i] <= PR1[i] and ELM[i] > 0 and PR1[i] > 0 else mask[ELM[i]:PR1[i], i]
            # EZ
            mask[PR1[i]:PR2[i], i] = 1 if PR1[i] <= PR2[i] and PR1[i] > 0 and PR2[i] > 0 else mask[PR1[i]:PR2[i], i]
            # BM
            # mask[PR2[i]:BM[i], i] = 4 if PR2[i] <= BM[i] and PR2[i] > 0 and BM[i] > 0 else mask[PR2[i]:BM[i], i]
            # RPE
            # mask[ELM[i]:BM[i], i] = 1 if ELM[i] <= BM[i] and ELM[i] > 0 and BM[i] > 0 else mask[ELM[i]:BM[i], i]
        # mask1 = Image.fromarray(mask)
        # ann1 = Image.fromarray(data1)
        # dat1 = Image.fromarray(data.squeeze(axis=2))
        # name_file = name + ".tiff"
        # dat1.save(cfg.IMAGES_PATH + name_file)
        # mask1.save(cfg.MASK_PATH + name_file)
        # ann1.save(cfg.ANN_PATH + name_file)
    except Exception as exc:
        print(exc)
        print(file)
    return mask


def crop_overlap(file, image, mask, path_img, path_msk, patient_id, size=128, shift=64):
    """
    file: OCT file extention .vol
    image: Numpy array
    mask: Numpy array
    path_img: path to save patches
    path_msk: path to save patches
    size: image size
    """
    name = os.path.splitext(os.path.split(file)[1])[0]
    oct_read = vol_files(name)
    OPL, INL, PR2, PR1, BM, ELM = get_annotations(oct_read)
    j = 1
    k = 0
    for i in range(size, image.shape[1], shift):
        min_pixel = np.max(INL[k:i])
        max_pixel = np.max(PR2[k:i])
        if min_pixel != 0 and max_pixel != 0 and max_pixel > min_pixel:
            delta1 = max_pixel - min_pixel
            delta2 = size - delta1
            delta3 = delta2 // 2
            delta4 = min_pixel - delta3
            delta5 = max_pixel + delta3
            if delta2 % 2 != 0:
                delta5 += 1
            if delta4 < 0:
                delta4 = 0
                delta5 = size
            if delta5 > image.shape[0]:
                delta5 = image.shape[0]
                delta4 = delta5 - size
            img_save = image[delta4:delta5, i - size:i]
            msk_save = mask[delta4:delta5, i - size:i]
            np.save(path_img + patient_id + f"_{j}.npy", img_save)
            np.save(path_msk + patient_id + f"_{j}.npy", msk_save)
            # img = Image.fromarray(img_save)
            # msk = Image.fromarray(msk_save)
            # img.save(path_img + patient_id + f"_{j}.png")
            # msk.save(path_msk + patient_id + f"_{j}.png")
            j += 1
        k = i


def main():
    id_train = ['8837',  '48104', '35282', 'IRD_RPE65_13', '35281',  'IRD_RPE65_22', '10162', '29331',  '51208',
                '20952', '52372', 'IRD_RPE65_02', '52374', 'IRD_RPE65_10',  'IRD_RPE65_07', 'IRD_RPE65_16', '52113', '49031', '52065',
                'IRD_RPE65_01', '49905', '35821', '42979', '43781', '52037', 'IRD_RPE65_06',  'IRD_RPE65_15', '16834',
                 'IRD_RPE65_18', '35897', 'IRD_RPE65_08', 'IRD_RPE65_09',  '52051', 'IRD_RPE65_04','34571',
                '30518', '6627', '48782', 'IRD_RPE65_05', 'IRD_RPE65_12', 'IRD_RPE65_03', '51021', '49699', '28008', '26472',
                '51886', '52009', 'IRD_RPE65_17',  '52386', 'IRD_RPE65_23', '49919', '49885','51812', 
                'IRD_RPE65_11', 'IRD_RPE65_21',  '49759', 'IRD_RPE65_19', 'IRD_RPE65_20']

    id_val = ['52025', '40300', '15313', '51013', '23028', '51155', '52484', '44872', '51904', '35277', '45954',
              '33996', '35328', '21509', '51038', '51939', '51870', '35280', '48731', '16831', '49883', '52458',
             ]

    IDs = []
    filenames_oct = get_filenames('dataset/OCT3/', 'vol')
    sum = 0.0
    squared_sum = 0.0
    for idx, f1 in enumerate(tqdm(filenames_oct)):
        name = os.path.splitext(os.path.split(f1)[1])[0]
        oct_read = vol_files(name)
        meta = oct_read.meta
        IDs.append(str(meta['PatientID']))

        data = oct_read.bscans[0].scan
        mask = get_images_masks(f1)
        # data = Image.fromarray(data)
        base_path = 'dataset/128_OPL_EZ/'
        train_path_images = base_path + 'train/Images/'
        train_path_masks = base_path + 'train/Masks/'
        val_path_images = base_path + 'val/Images/'
        val_path_masks = base_path + 'val/Masks/'
        patches_images_train = train_path_images + 'images_patches/'
        patches_images_val = val_path_images + 'images_patches/'
        patches_masks_train = train_path_masks + 'masks_patches/'
        patches_masks_val = val_path_masks + 'masks_patches/'

        create_dir(train_path_images)
        create_dir(val_path_images)
        create_dir(train_path_masks)
        create_dir(val_path_masks)

        create_dir(patches_images_train)
        create_dir(patches_images_val)
        create_dir(patches_masks_train)
        create_dir(patches_masks_val)
        
        filename_save = str(meta['VisitDate']).replace(":", "_").replace(" ", "_").replace("-", "_")
        try:
            id_train.index(meta['PatientID'])
            np.save(train_path_images + str(meta['PatientID']) + '_' + filename_save + f"{idx}.npy", data)
            np.save(train_path_masks + str(meta['PatientID']) + '_' + filename_save + f"{idx}.npy", mask)
            sum += np.mean(data / 255.)
            squared_sum += np.mean((data / 255.)**2)
            crop_overlap(f1, data, mask, patches_images_train, patches_masks_train,
                         str(meta['PatientID']) + '_' + filename_save, size=128, shift=64)
        except:
            id_val.index(meta['PatientID'])
            np.save(val_path_images + str(meta['PatientID']) + '_' + filename_save + f"{idx}.npy", data)
            np.save(val_path_masks + str(meta['PatientID']) + '_' + filename_save + f"{idx}.npy", mask)
            # print(mask.shape, data.shape)
            sum += np.mean(data / 255.)
            squared_sum += np.mean((data / 255.)**2)
            crop_overlap(f1, data, mask, patches_images_val, patches_masks_val, str(
                meta['PatientID']) + '_' + filename_save, size=128, shift=64)
    
    mean = sum / len(filenames_oct)
    std = (squared_sum / len(filenames_oct) - mean**2)**0.5
    print(f'Mean: {mean}, Std: {std}')
if __name__ == "__main__":
    main()
