import tensorflow as tf
import random
import numpy as np


class Config:
    # directory with train data saved in your device locally
    data_dir = '/path/to/train_data/'
    mask_dir = '/path/to/masks'
    augmented_dir = '/path/to/augmented_images'
    aug_masks_dir = '/path/to/new_masks'
    logdir = '/path/to/logdir'

    validation_fraction = 0.15
    test_fraction = 0.10
    train_batch = 16
    valid_batch = 32
    test_batch = 32

    input_dim = 256
    input_ch = 3
    output_dim = 256
    output_ch = 3

    seed = 42
    lr = 0.01
    epochs = 50
    device = tf.device('/device:GPU:0') if tf.config.list_physical_devices('GPU') else tf.device('/CPU:0')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)