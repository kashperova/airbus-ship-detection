import tensorflow as tf
import random
import numpy as np


class Config:
    # directory with train data saved in my google drive (unavailable for you)
    data_dir = '/content/gdrive/MyDrive/AirBusDataset/train_v2/'
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
    device = tf.device('/device:GPU:0') if tf.test.is_gpu_available() else tf.device('/CPU:0')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)