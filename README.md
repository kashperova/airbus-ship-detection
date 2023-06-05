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
    ├── Dockerfile             - prepare environment 
    ├── model.pt               - model saved in .pt format (PyTorch)
    │
    ├── notebooks              - this folder contains notebooks (.ipynb files) 
    │  ├── EDA              - dir with images
    │  └── labels.csv          - target values    
    │    
    ├── prepare_data           - this folder consits script for data preprocessing 
    │  ├── dataset.py          - [implmentation of HandWrittenCharsDataset class] retrieve image with label
    │  └── load_data.py        - [implmentation of DataLoading class] load data and split it to train, valid and test   
    │
    ├── config.py              - script for testing on your own images
    ├── model_cnn.py           - [implmentation of CNN class] model for image classification in PyTorch    
    ├── train.py               - [implmentation of TrainAndEvaluateModel class] train and save model
    ├── eval_plots.py          - implmentation of methods for metrics visualization
    ├── inference.py           - script for testing on your own images
    ├── Documentation.md       - current file that descibes used data, methods, ideas and reports accuracy.
    ├── /reports               - directory with metrics plots.
    └── requirements.txt       - list of libraries for the project
```

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

Exploratory data analysis is saved in <a href="">this notebook</a>.
Below are the main insights from the analysis:

How mant ships detected on images             |  Distribution of areas occupied by ships in images 
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
to augmentate data. To be honest, this process spent the biggest time 

## Building a model

## Training and evaluation

## How to run

### Train
If you want to run ```train.py```, you should define working directories in ```config.py``` (with downloaded data).
After that you need to install virtual environment with necessary libs using ```requirements.txt```. 

### Inference


## Other ways for solving / improving accuracy

When I started doing exploratory data analysis, I realized that ships in segmentation masks 
are more like bounding boxes and, accordingly, this competition can be considered as an object detection task.
After reading several papers, I came to the conclusion that in this case
(satellite images with only 3 channels, rather small objects on training data), 
the architecture SWIN Transformer with YOLOv5 should work well [5]. 

Some experiments with model based on this architecture <i>will be</i> saved in <a href="https://github.com/kashperova/swin-yolov5-airbus-ship-detection">this repository</a>.

I did not have time to implement this solution, as I spent a lot of time building a pipeline for data augmentation. At the beginning, I decided to use 
albumentations and opencv libraries,
but there were encoding problems that I tried to solve for a long time (I usually work with PyTorch). In the end, due to the deadline, I decided to use the built-in methods
of Tensorflow to work with images.
But I plan to implement this model, so maybe when you read this documentation, there is something 
worthwhile in <a href="https://github.com/kashperova/swin-yolov5-airbus-ship-detection">this repository</a>.
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