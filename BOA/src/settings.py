import os
""" Dataset directories """
DATASET_DIR = 'dataset/'

TRAIN_IMAGES = DATASET_DIR + 'new_128_2C/train_images/'
TRAIN_MASKS = DATASET_DIR + 'new_128_2C/train_masks/'
VAL_IMAGES = DATASET_DIR + 'new_128_2C/val_images/'
VAL_MASKS = DATASET_DIR + 'new_128_2C/val_masks/'

""" HYPER-PARAMETERS """
LOSS_FN = 'dice loss'
EPOCHS = 800
BATCH_SIZE = 64
LEARNING_RATE = 0.001
BETA1 = 0.9
BETA2 = 0.999
WEIGHT_DECAY = 1e-5
CLASS_WEIGHTS = [1, 1, 1, 1]
SCHEDULER = 'cosine' #step, cosine
GAMMA = 0.8
STEP_SIZE = EPOCHS * 0.1
GPUS_ID = [0]
PRETRAIN = True

""" GERNERAL SETTINGS """
IMAGE_SIZE = 128
CLASSES = 3
NUM_WORKERS = os.cpu_count()

embed_dim = 96
depths = [2, 2, 6, 2]
num_heads = [3, 6, 12, 24]
window_size = 7
dropout = 0.1