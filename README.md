# Airbus ship detection challenge
Test task based on data from Kaggle competition Airbus Ship Detection for applying on intern position on R&D Winstars Center


## Configuration and data loading
Before you start to explore solution and run scripts, you should to define some necessary parameters
in ```Config``` class which point to directories with training images and masks, 
as well as directories for augmented data (located in ```config.py```).

In <a href = "">EDA notebook</a> I propose 2 ways for downloading data using Kaggle API. Firstly, you should upload
your kaggle.json with API key and run such command:

```!kaggle competitions download -c airbus-ship-detection```

Next you can save  initial dataset locally (Google Colab or Jupyter Notebook):
```
!mkdir /content/AirBusDataset
!cp /content/airbus-ship-detection.zip /content/AirBusDataset/airbus-ship-detection.zip
!unzip -q /content/AirBusDataset/airbus-ship-detection.zip -d /content/AirBusDataset/
!rm /content/AirBusDataset/airbus-ship-detection.zip
```

Or you can save data in your Google Drive (more comfortable for frequent running):
```
from google.colab import drive
drive.mount('/content/gdrive')
```
and after that running previous 4 commands changing file paths to your google drive

<br>After unpacking zip file you should define parameters in ```config.py```.</br>

```
├── data_dir : path to training images (train_v2 directory in Kaggle)
├── mask_dir : path to training masks (converted from rle format and saved to this directory)
├── augmented_dir : path to augmented images (transformed images with detected ships to prevent imbalancing of intiail dataset)
├── aug_masks_dir : path to augmented masks (transformed masks from initial dataset)
```

At this class you can also change important parameters for model training such as
learning rate, batch size, numbers of epochs, device, etc.

## EDA

## Data augmentation

## Building a model

## Training and evaluating

## How to run

If you want to run ```train.py``` 

## Other ways for solving / improving accuracy

When I started doing exploratory data analysis, I realized that ships in segmentation masks 
are more like bonding boxes and, accordingly, this competition can be considered as an object detection task.
After reading several papers, I came to the conclusion that in this case
(satellite images with only 3 channels, rather small objects on training data), 
the architecture SWIN Transformer with YOLOv5 should work well. 

I tried some experiments with model based on this architecture in <a href="">this repository</a>. Model was implemented
in PyTorch, as I more experienced at this framework for DL. And also I want to mention 

As for methods

## References