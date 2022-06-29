import os
""" Dataset directories """
DATASET_DIR = 'dataset/128_2C/'

# TRAIN_IMAGES = DATASET_DIR + 'train_images/'
# TRAIN_MASKS = DATASET_DIR + 'train_masks/'
# VAL_IMAGES = DATASET_DIR + 'val_images/'
# VAL_MASKS = DATASET_DIR + 'val_masks/'

TRAIN_IMAGES = 'OCT_DATASET/dataset/PatientIDs128_2C/train/Images/images_patches/'
TRAIN_MASKS = 'OCT_DATASET/dataset/PatientIDs128_2C/train/Masks/masks_patches/'
VAL_IMAGES = 'OCT_DATASET/dataset/PatientIDs128_2C/val/Images/images_patches/'
VAL_MASKS = 'OCT_DATASET/dataset/PatientIDs128_2C/val/Masks/masks_patches/'
""" HYPER-PARAMETERS """
LOSS_FN = 'dice loss'
EPOCHS = 800
BATCH_SIZE = 128
LEARNING_RATE = 0.001
BETA1 = 0.9
BETA2 = 0.999
WEIGHT_DECAY = 1e-6
CLASS_WEIGHTS = [1, 1, 1]
SCHEDULER = 'step' #step, cosine
GAMMA = 0.8
STEP_SIZE = EPOCHS * 0.05
GPUS_ID = [0]
PRETRAIN = True

""" GERNERAL SETTINGS """
IMAGE_SIZE = 128
CLASSES = 3
NUM_WORKERS = 12

EMBED_DIM = 48
DEPTHS = [2, 2, 2, 2]
NUM_HEADS = [2, 4, 8, 16]
WINDOW_SIZE = 8
DROPOUT = 0.1
