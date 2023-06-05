import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os
from ..config import Config
from scipy import ndimage


class ShipDataGenerator(tf.keras.utils.Sequence):
    """
        Dataset class for retrieving custom data

        Attributes
        ----------
            path_df: pd.DataFrame
                images and masks

            transform: A.Compose
                Compose transforms and handle all transformations
    """

    def __init__(self, path_df: pd.DataFrame, transform: A.Compose):
        self.path_df = path_df
        self.transform = transform

    def __len__(self):
        return len(self.path_df)

    def __getitem__(self, index):
        """
        Loading image and mask and applying transofrmations
        and Sobel edge detection filter for image
        """

        img_id = self.path_df.iloc[index]['ImageId']
        img_path = os.path.join(Config.data_dir, img_id)
        mask_id = self.path_df.iloc[index]['Mask']
        mask_path = os.path.join(Config.mask_dir, mask_id)

        # open image
        image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)

        # create empty mask
        if mask_id == "empty":
            mask = np.zeros((Config.input_dim, Config.input_dim))
            cv2.imwrite(Config.mask_dir + img_id[:-4] + '.png', mask)
            mask_path = os.path.join(Config.mask_dir, img_id[:-4] + '.png')
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # apply transformation
        if self.transform is not None:
            image = self.transform(image=image)
            mask = self.transform(image=mask)

        # apply Sobel edge detection filter
        image = ndimage.sobel(image['image'])
        # reshape (add one dimension)
        mask = mask['image']
        mask = mask.reshape((1, Config.input_dim, Config.input_dim))
        image = image.reshape((1, Config.input_dim, Config.input_dim, 3))

        # normalize
        image = tf.cast(image, tf.float32) / 255.0
        mask -= 1
        return image, mask


