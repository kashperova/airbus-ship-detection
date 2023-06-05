"""
Script for data augmentation.
Creating new images and masks from existing ones
will help to prevent imbalancing of dataset.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from ..config import set_seed, Config
from .rle_mask import rle2mask


class DataAugmentation:
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
        images_ids = []

        # define image data generator for training data
        data_generator = ImageDataGenerator(
            rescale=1. / 255,  # rescale pixel values to [0,1]
            rotation_range=15,  # rotate images randomly up to 15 degrees
            width_shift_range=0.2,  # shift images horizontally up to 20% of the width
            height_shift_range=0.2,  # shift images vertically up to 20% of the height
            shear_range=0.1,  # apply shear transformation up to 10%
            zoom_range=0.1,  # zoom in/out up to 10%
            horizontal_flip=True,  # flip images horizontally
            fill_mode='nearest'  # fill any gaps caused by the above transformations with the nearest pixel value
        )

        # define batch size
        batch_size = 32

        # generate training data batches
        train_generator = data_generator.flow_from_directory(
            self.train_dir,  # directory containing the training data
            target_size=(256, 256),  # resize images to 256x256 pixels
            batch_size=batch_size,  # number of images in each batch
            class_mode='binary',  # binary classification problem (ship vs. no ship)
            shuffle=True  # shuffle the data before each epoch
        )

        # generate mask data batches
        mask_generator = data_generator.flow_from_directory(
            self.mask_dir,  # directory containing the mask data
            target_size=(256, 256),  # resize masks to 256x256 pixels
            batch_size=batch_size,  # number of masks in each batch
            class_mode='binary',  # binary classification problem (ship vs. no ship)
            shuffle=True  # shuffle the data before each epoch
        )

        # create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # create output directory for new masks if it doesn't exist
        if not os.path.exists(self.new_mask_dir):
            os.makedirs(self.new_mask_dir)

        # generate new images and masks and save them to the output directory
        for i in range(1000):  # generate 1000 new images
            batch = next(train_generator)
            images = batch[0]
            filenames = batch[1]
            mask_batch = next(mask_generator)
            masks = mask_batch[0]
            for j in range(batch_size):
                image = images[j]
                mask = masks[j]
                filename = filenames[j]
                new_filename = filename.split('.')[0] + '_' + str(i) + '.jpg'  # add suffix to filename for each new image
                new_filepath = os.path.join(self.output_dir, new_filename)
                new_mask_filename = filename.split('.')[0] + '_' + str(i) + '.png'  # add suffix to filename for each new mask
                new_mask_filepath = os.path.join(self.new_mask_dir, new_mask_filename)
                plt.imsave(new_filepath, image)
                plt.imsave(new_mask_filepath, mask, cmap='gray')
                images_ids.append(filename.split('.')[0] + '_' + str(i) + '.jpg')

        return images_ids
