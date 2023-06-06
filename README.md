# Airbus ship detection challenge
Test task based on data from Kaggle competition Airbus Ship Detection for applying on intern position on R&D Winstars Center
-  [Project Structure](#project-structure)
-  [Configuration and data loading](#configuration-and-data-loading)
-  [EDA](#eda)
-  [Data augmentation](#data-augmentation)
- [Building a model](#building-a-model)
- [Training and evaluation](#training-and-evaluation)
- [Other ways for solving / improving accuracy](#other-ways-for-solving--improving-accuracy)
- [How to run](#how-to-run)
  - [Train](#train)
  - [Inference](#inference)
- [References](#references)

## Project structure
Here is structure of solution.
```
   solution
    ├── model                  - this folder model files 
    │  ├── model.py            - script of implementation U-net architecture
    │  └── model.h5            - saved checkpoint of model (used in inference.py)
    │      
    ├── notebooks              - this folder contains notebooks (.ipynb files) 
    │  ├── eda.ipynb           - notebook with eda[EDA_AirbusShipDetection_(1).ipynb]
    │  └── evaluate.ipynb      - notebook with train and model evaluation    
    │    
    ├── data_processing        - this folder consits scripts for data preprocessing 
    │  ├── augmentate.py        - script for data augmentation
    │  ├── rle_mask.py          - script for converting run-length encode format to binary mask 
    │  ├── data_split.py        - script for splitting data on train, valid and test datasets       
    │  └── data_generator.py    - script for generating sample (image, mask) during training   
    │
    ├── config.py              - script for saving configuration setting and constants
    ├── train.py               - script for training and saving model
    ├── inference.py           - script for running model on test images
    ├── README.md              - current file that descibes used data, methods and ideas.
    ├── /screenshots           - directory with visualization, screenshots and metrics plots.   
    └── requirements.txt       - list of libraries for the project
```

## Configuration and data loading
Before you start to explore solution and run scripts, you should to define some necessary parameters
in ```Config``` class which point to directories with training images and masks, 
as well as directories for augmented and testing data (located in ```config.py```).

In <a href = "notebooks/eda.ipynb">EDA notebook</a> I propose 2 ways for downloading data using Kaggle API. Firstly, you should upload
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
├── test_dir : path to images for testing
├── logdir : path to directory for saving logs during training
├── images_csv : path to file with images masks (train_ship_segmentations_v2.csv)
```

At this class you can also change important parameters for model training such as
learning rate, batch size, numbers of epochs, device, etc.

## EDA

Exploratory data analysis is saved in <a href="notebooks/eda.ipynb">this notebook</a>.
Below are the main insights from the analysis:

How many ships detected on images             |  Distribution of areas occupied by ships in images 
:-------------------------:|:-------------------------:
![eda_1.png](screenshots%2Feda_1.png)  |    ![eda_2.png](screenshots%2Feda_2.png)


<ul>
<li>There are over 25 000 samples with only one detected ship and over 7000 samples with two ships. </li>

<li>The maximum amount of detected ships on samples is 15.</li>
<li>Initial dataset so imbalanced. For better model training we need oversample dataset with images of detected ships. </li>
<li>Among the images, the area of the identified boats is very small. This complicates the task of segmentation.</li>
</ul>

## Data augmentation
As mention before, initial dataset is so imbalanced. That's why I decided to use some transformations
to augmentate data. To be honest, this process took me the most time, and now, when I write this documentation on the last day of the deadline, 
I understand that I should have spent more time on tuning the model, rather than preprocessing the data. 

During augmentation I used ```HorizontalFlip```, ```VerticalFlip```, ```Resize```, ```Rotation``` and ```Affine Tranformation```.

After the augmentation, the number of samples with detected ships doubled.
Next, I removed every fifth sample without ships, thus the ratio in the dataset became 2:3.

Additionaly, after first version of model, in ```data_generator.py``` I used ```Sobel filter``` 
because I read that its use improves the performance of convolutional networks 
and performs well in small object segmentation tasks  [2] . 

## Building a model

As mention in test task description, prefered model for solving this task is U-net. 
My implementation of model is saved in ```models/model.py```. 

I used architecture from original paper [3]. 

## Training and evaluation

As I said earlier, I did not correctly allocate time during the execution of the test task, 
so I had little time to train the model, so I used a small number of epochs (10-15).




## How to run

### Train
If you want to run ```train.py```, you should define working directories in ```config.py``` (with downloaded data).
After that you need to install virtual environment with necessary libs using ```requirements.txt```. 

#### Requirements
1. Check your python version: 

    ```python3 --version```

2. Install virtual environment in solution directory: 

    ```python3 -m venv venv```<br>
```source venv/bin/activate```
4. Install all necessary libraries to your virtual environment using such command as: </br>
```pip install -r requirements.txt```

If you defined all parameters in ```config.py```, you don't need to specify anything to run train
script. 

```python train.py```

If you want to change parameters though command line:

```python train.py --image_csv new_dir_to_csv --model_dir new_model_dir --epochs 30 --lr 0.001```

### Inference

For testing model you should define 2 important parameters:
- model_dir - path to model checkpoint
- test_dir - path to test images

As in the case of the training script, you can initially specify the parameters
in the configuration class, or through the command line.

```python inference.py --model_dir path_to_checkpoint --test_dir path_to_test_images```



## Other ways for solving / improving accuracy

When I started doing exploratory data analysis, I realized that ships in segmentation masks 
are more like bounding boxes and, accordingly, this competition can be considered as an object detection task.
After reading several papers, I came to the conclusion that in this case
(satellite images with only 3 channels, rather small objects on training data), 
the architecture SWIN Transformer with YOLOv5 should work well [5]. 

I did not have time to implement this solution, as I spent a lot of time building a pipeline for data augmentation. At the beginning, I decided to use 
albumentations and opencv libraries,
but there were encoding problems that I tried to solve for a long time (I usually work with PyTorch). In the end, due to the deadline, I decided to use the built-in methods
of Tensorflow to work with images.
But I plan to implement this model, so maybe in future there is something 
worthwhile in <a href="">this repository</a>.
<br></br>
<b> As for methods of improving current model (U-net for segmentation task): </b>
1. More data. Initial dataset was so imbalanced, approximately 70% of image have no detected ships.
If I had more time, I would experiment with other ways to reason and transform the dataset, 
I would try to use GAN for generating new training samples.
2. More time for model training and hyperparameter optimization  (experiments with learning rate on
constant epochs number)
3. Don't resize of the image during preprocessing, 
but divide the image into windows (for example, into 9 windows sized 256 by 256) and after 
that concatenating results of model segmentation.
4. Also we can use pretrained weights for downsampling block (```Down``` class in ```model.py```).
At last day of deadline I read paper [4] where described U-net for road segmentation on satellite images.
The authors used for the encoder pretrained weights on a simple classification of roads in the images.

## References
[1] - <a href="https://www.mdpi.com/2313-433X/9/2/46/pdf">Data Augmentation in Classification and Segmentation:
A Survey and New Strategies</a>

[2] - <a href="https://arxiv.org/pdf/1910.00138.pdf">Custom Extended Sobel Filters</a>

[3] - <a href="https://arxiv.org/pdf/1505.04597.pdf">U-Net: Convolutional Networks for Biomedical
Image Segmentation</a>

[4] - <a href="https://arxiv.org/pdf/2109.14671.pdf">Segmentation of Roads in Satellite Images
using specially modified U-Net CNNs</a>

[5] - <a href="https://www.mdpi.com/2072-4292/14/12/2861">Swin-Transformer-Enabled YOLOv5 with Attention Mechanism for Small Object Detection on Satellite Images</a>