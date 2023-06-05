import pandas as pd
import albumentations as A
from sklearn.model_selection import train_test_split
from augmentate import AugmentationData
from ..config import Config
from .data_generator import ShipDataGenerator


def split_data_for_training(non_zeros_ships: list, zero_ships_data: list, image_masks: pd.DataFrame):
    """
    Method for splitting data on train, validation and test dataframes

    :param non_zeros_ships: list of images ids with detected ships
    :param zero_ships_data: list of images ids with no detected ships
    :param image_masks: pd.DataFrame with all data
    :return: train_df, valid_df, test_df (dataframes for training)
    """

    # increase samples with ships using augmentation techniques
    aug = AugmentationData(non_zeros_ships, image_masks)
    augmented_ids = aug.augmentate()

    # join new ids with old ones
    augmented_mask_ids = [i[:-4] + ".png" for i in augmented_ids]
    non_zeros_ships_masks = [i[:-4] + ".png" for i in non_zeros_ships]
    zero_ids_images = zero_ships_data
    zero_ids_masks = ["empty" for i in range(len(zero_ids_images))]

    new_data = {'ImageId': augmented_ids + non_zeros_ships + zero_ids_images,
                'Mask': augmented_mask_ids + non_zeros_ships_masks + zero_ids_masks}

    final_dfrm = pd.DataFrame.from_dict(new_data)

    # split data on train, validation and test
    train2rest = Config.validation_fraction + Config.test_fraction
    test2valid = Config.validation_fraction / train2rest

    train_df, rest = train_test_split(
        final_dfrm, random_state=Config.seed,
        test_size=train2rest
    )

    test_df, valid_df = train_test_split(
        rest, random_state=Config.seed,
        test_size=test2valid
    )

    # reset indexes
    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    return train_df, valid_df, test_df


def generate_datasets(non_zeros_ships: list, zero_ships_data: list, image_masks: pd.DataFrame):
    """
    Wrapper for split_data_for_training

    :param non_zeros_ships: list of images ids with detected ships
    :param zero_ships_data: list of images ids with no detected ships
    :param image_masks: pd.DataFrame with all data
    :return: train_data, valid_data, test_data : ShipDataGenerator - data generator of sample(image, mask)
    """
    train_df, valid_df, test_df = split_data_for_training(non_zeros_ships, zero_ships_data, image_masks)

    train_transforms = A.Compose([A.Resize(height=Config.input_dim, width=Config.input_dim, p=1),
                                  A.HorizontalFlip(p=0.5),
                                  A.augmentations.geometric.transforms.Affine(scale=1.2, always_apply=True)])
    eval_transforms = None

    train_data = ShipDataGenerator(path_df=train_df, transform=train_transforms)
    valid_data = ShipDataGenerator(path_df=valid_df, transform=eval_transforms)
    test_data = ShipDataGenerator(path_df=test_df, transform=eval_transforms)
    return train_data, valid_data, test_data

