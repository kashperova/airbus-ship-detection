import tensorflow as tf
import pandas as pd
import cv2
import os
from ..config import Config
from scipy import ndimage


class ShipDataGenerator(tf.keras.utils.Sequence):
    """
        Dataset class for retrieving custom data

        Attributes
        ----------
            data: pd.DataFrame
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
        mask_id = self.path_df.iloc[index]['MaskId']
        mask_path = os.path.join(Config.data_dir, mask_id)
        image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)
        image = ndimage.sobel(image)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return image, mask