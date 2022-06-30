import logging
import datetime
import os
import sys
import numpy as np
import torch


def initialize(cfg):
    """Directories"""
    now = datetime.datetime.now()
    version = 'logs/' + str(now.strftime("%Y-%m-%d_%H_%M_%S")) + '/'
    checkpoint_path = version + "checkpoints/"
    create_dir(checkpoint_path)
    with open(version + 'config.txt', 'w') as text_file:
        text_file.write(f"*** Hyperparameters ***\n")
        hyp = cfg['hyperparameters']
        text_file.write(f"Loss function: {hyp['loss_fn']}\n")
        text_file.write(f"Learning rate: {hyp['lr']}\n")
        text_file.write(f"weight_decay: {hyp['weight_decay']}\n")
        text_file.write(f"BETA1, BETA2: {hyp['b1'], hyp['b2']}\n")
        text_file.write(f"Batch size: {hyp['batch_size']}\n")
        text_file.write(f"Epochs: {hyp['num_epochs']}\n")
        text_file.write(f"*** Scheduler LR ***\n")
        sch = hyp['scheduler']
        text_file.write(f"Schaduler Type: {sch['type']}\n")
        text_file.write(f"Gamma: {sch['gamma']}\n")
        text_file.write(f"Step size: {sch['step']}\n")
        gen = cfg['general']
        text_file.write(f"*** Gerneral settings ***\n")
        text_file.write(f"Image Size: {gen['img_size']}\n")
        text_file.write(f"Pretrain: {gen['pretrain']}\n")
        text_file.write(f"Num classes: {gen['n_classes']}\n")
        text_file.write(f"Model name: {cfg['model_name']}\n")
        text_file.close()

    """logging"""

    logging.basicConfig(filename=version + "info.log",
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))

    """ Seeding """
    seeding(42)  # 42
    return logger, checkpoint_path, version

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def seeding(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True