import keras
import argparse
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import cv2
from .config import Config


def get_segmentation(test_dir, img, model):
    img_path = os.path.join(test_dir, img)
    img = cv2.imread(img_path)
    transform = A.Resize(height=Config.input_dim, width=Config.input_dim, p=1)
    img = transform(image=img)
    img = img['image']
    img = img.reshape((1, Config.input_dim, Config.input_dim, 3))
    pred = model.predict(img)
    pred = np.squeeze(pred, axis=0)
    return cv2.imread(img_path), pred


def inference(args):
    model = keras.models.load_model(args.model_dir, compile=False)
    test_images = []
    files = os.listdir(args.test_dir)
    for img in files:
        test_images.append(img)

    r, c = 1, 2
    for i in range(len(test_images)):
        img, pred = get_segmentation(args.test_dir, test_images[i], model)
        fig = plt.figure(figsize=(12, 10))
        fig.add_subplot(r, c, 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title("Image")
        fig.add_subplot(r, c, 2)
        plt.imshow(pred, interpolation=None)
        plt.axis('off')
        plt.title("Segmentation")


if __name__ == "main":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=Config.model_dir,  help='path to model checkpoint')
    parser.add_argument('--test_dir', type=str, default=Config.test_dir, help='path to test images')
    args = parser.parse_args()
    inference(args)
