'''
library 모음
'''

# 기본 lib
import argparse  # 아직 사용하지는 않음
import copy
import datetime
import gc
import json # for coco format 
import os
import random
import time
import tqdm 

# data science lib
import numpy as np
import pandas as pd

# gp lib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# import seaborn as sns

# cv lib
import cv2
import imageio
import PIL.Image as Image

# pytorch lib
# from torch.optim import lr_scheduler  # https://gaussian37.github.io/dl-pytorch-lr_scheduler/
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# torchvision lib
import torchvision
from torchvision import datasets, models, transforms


# augmentation tool
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

# segment lib
# pip install segmentation_models_pytorch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import losses