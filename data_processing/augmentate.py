"""
Script for data augmentation.
Creating new images and masks from existing ones
will help to prevent imbalancing of dataset.
"""

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from ..config import set_seed, Config
from .rle_mask import rle2mask


class AugmentationData:
    """
    Class for data augmentation. Some methods of transformation
    will applied to exsiting images with detected ships. After that new samples
    will be saved in train directory.

        Attributes
        ----------
        train_dir: str - directory with train images
        mask_dir: str - directory with train masks
        output_dir: str - directory with output augmented images
        new_mask_dir: str - directory with output changed masks
        non_zero_ships: list - list of images id with detected ships
        image_masks: pd.DataFrame - dataframe with masks in rle-format

        Methods
        ----------
        __init__():
          initialize paths, dataframe and list for further processing

        __save_masks():
          private method for saving initial masks in defined directory

        augmentate():
          public method for creating new images and masks
          returns list of new images/masks id
    """

    def __init__(self, non_zero_ships: list, image_masks: pd.DataFrame):
        # set random seed for reproducibility
        set_seed(42)

        # define data directories
        self.train_dir = Config.data_dir
        self.mask_dir = Config.mask_dir
        self.output_dir = Config.augmented_dir
        self.new_mask_dir = Config.mask_dir
        self.non_zero_ships = non_zero_ships
        self.image_masks = image_masks
        self.__save_masks()

    def __save_masks(self):
        for img_id in self.non_zero_ships:
            img_masks = self.image_masks.loc[self.image_masks['ImageId'] == img_id, 'EncodedPixels'].tolist()
            # take the individual ship masks and create a single mask array for all ships
            all_masks = np.zeros((768, 768))
            for mask in img_masks:
                all_masks += rle2mask(mask)
            mask = Image.fromarray(all_masks)
            mask = mask.convert('L')
            mask.save(Config.mask_dir + "/" + img_id[:-4] + ".png")

    def augmentate(self) -> list:
        # create empty list for new images_id
        new_images_ids = []
        for image_id in self.non_zero_ships:
            mask_id = image_id[:-4] + ".png"

            # Load image, mask from file
            image = tf.io.read_file(self.train_dir + image_id)
            mask = tf.io.read_file(self.mask_dir + mask_id)

            # Decode JPEG image to tensor
            image = tf.image.decode_jpeg(image, channels=3)
            mask = tf.image.decode_png(mask, channels=1)

            # Resize image
            image = tf.image.resize(image, [768, 768])
            mask = tf.image.resize(mask, [768, 768])

            # Flip image, mask horizontally
            image = tf.image.flip_left_right(image)
            mask = tf.image.flip_left_right(mask)

            # Flip image, mask vertically
            image = tf.image.flip_up_down(image)
            mask = tf.image.flip_up_down(mask)

            # Rotate image by 90 degrees clockwise
            image = tf.image.rot90(image, k=1)
            mask = tf.image.rot90(mask, k=1)

            # Convert tensor back to JPEG image and save to file
            image = tf.cast(image, tf.uint8)
            image = tf.image.encode_jpeg(image)
            tf.io.write_file(self.output_dir + image_id[:-4] + '_aug.jpg', image)

            mask = tf.image.encode_png(mask)
            tf.io.write_file(self.new_mask_dir + image_id[:-4] + '_aug.png', mask)
            new_images_ids.append(image_id[:-4] + '_aug.jpg')
        return new_images_ids