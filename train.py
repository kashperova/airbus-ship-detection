"""
Script for model training.
Here were implemented training and evaluating functions.
Also some function for metrics were developed.
"""
import os
import pandas as pd
import keras.backend as K
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
import argparse
from tensorflow.python.estimator import keras
from .config import Config
from model.model import UNet
from data_processing.data_split import generate_datasets
import matplotlib.pyplot as plt


def DiceLoss(targets, inputs):
    """ Method for counting Dice coefficient"""

    smooth = 1e-6

    inputs = K.cast(inputs, 'float32')
    targets = K.cast(targets, 'float32')

    intersection = K.sum(K.dot(targets, inputs))
    dice = (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice


def DiceBCELoss(targets, inputs):
    """ Method for counting Dice-BCE loss. This loss combines Dice loss
    with the standard binary cross-entropy (BCE) loss """

    smooth = 1e-6

    inputs = K.cast(inputs, 'float32')
    targets = K.cast(targets, 'float32')

    BCE = binary_crossentropy(targets, inputs)
    intersection = K.sum(K.dot(targets, inputs))
    dice_loss = 1 - (2 * intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)

    Dice_BCE = BCE + dice_loss
    return Dice_BCE


def parameters_for_train():
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_dir', type=str, default=Config.data_dir,  help='directory path to images')
    parser.add_argument('--image_csv', type=str, default=Config.data_dir, help='path to csv with ids')
    parser.add_argument('--mask_dir', type=str, default=Config.mask_dir, help='directory path to masks')
    parser.add_argument('--model_dir', type=str, default=os.getcwd(), help='directory path for saving checkpoint of model')
    parser.add_argument('--logdir', type=str, default=Config.logdir, help='path to tensorboard logs')
    parser.add_argument('--epochs', type=int, default=Config.epochs, help='amount of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='training batch size')
    parser.add_argument('--lr', type=float, default=Config.lr, help='learning rate')
    args = parser.parse_args()
    return args


def train(images_csv, logdir, lr, epochs, batch_size, model_dir, mask_dir, image_dir):
    # callbacks autosave
    mode_autosave = ModelCheckpoint(model_dir,
                                    monitor='val_dice_coef',
                                    mode='max', save_best_only=True)
    # learning rate reducer
    lr_reducer = ReduceLROnPlateau(factor=0.1,
                                   cooldown=1,
                                   patience=3,
                                   min_lr=0.01)

    tensorboard = TensorBoard(log_dir=logdir, histogram_freq=0,
                              write_graph=True)

    # early stopping if validation set have big losses
    early_stopping = EarlyStopping(patience=3, verbose=1)

    callbacks = [mode_autosave, lr_reducer, tensorboard, early_stopping]

    model = UNet(Config.input_ch, Config.output_ch)

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer,
                  loss=DiceBCELoss, metrics=[DiceLoss])

    image_masks = pd.read_csv(images_csv)
    nan_rows = image_masks.isna().any(axis=1)
    no_nan_rows = image_masks.notna().all(axis=1)

    non_zeros_ships = list(image_masks[nan_rows]['ImageId'].unique())
    zero_ships_data = list(image_masks[no_nan_rows]['ImageId'].unique())

    Config.mask_dir = mask_dir
    Config.data_dir = image_dir

    train_data, valid_data, test_data = generate_datasets(non_zeros_ships, zero_ships_data, image_masks)
    results = model.fit(train_data, batch_size=batch_size, epochs=epochs, validation_data=valid_data, callbacks=callbacks)
    plot_metrics(results)


def plot_metrics(results):
    """Method for visualization results of training"""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(20, 10))
    ax_loss.plot(results.epoch, results.history["loss"], label="Training losses")
    ax_loss.plot(results.epoch, results.history["val_loss"], label="Validation losses")
    ax_loss.legend()
    ax_acc.plot(results.epoch, results.history["dice_coef"], label="Training Dice Coef")
    ax_acc.plot(results.epoch, results.history["val_dice_coef"], label="Validation Dice Coef")
    ax_acc.legend()


if __name__ == "main":
    args = parameters_for_train()
    train(args.images_csv, args.logdir, args.lr, args.epochs, args.batch_size, args.model_dir, args.mask_dir, args.image_dir)
