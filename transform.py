from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
import torch
import os

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

import torchvision

from data_helper import UnlabeledDataset, LabeledDataset
import torch

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
%matplotlib inline

H_transforms = {
    "L" : {
        "src" : np.float32([[100, 110], [200, 110], [0, 225], [306, 225]]),
        "dst" : np.float32([[-125, -475], [425, -475], [50, 250], [200, 250]]),
        "output_size" : (250,250),
        "cropping" : (0,0)
    },

    "E" : {
        "src" : np.float32([[0, 256], [306, 256], [0, 0], [306, 0]]),
        "dst" : np.float32([[144, 256], [180, 256], [0, 0], [306, 0]]),
        "output_size" : (306, 256),
        "cropping" : (140,0)
    }

}

def transform_image(img, transform_dict):
    """
    img : A single image of shape: (height, width, channels)
    transform_dict : Dict of the transform parameters:
        Take from the H_transform dict
        eg:
        "src" : # 4 points in orginal
        "dst" : # 4  poitns in the new mapped img
        "output_size" : # Size of the output warped img
        "cropping" :  # simple cropping #Todo: make this more robust

    Returns:
        img : the transformed img
        Minv : the reverse transform matrix
    """
    src = transform_dict["src"]
    dst = transform_dict["dst"]

    crop = transform_dict["cropping"]
    output_size = transform_dict["output_size"]

    IMAGE_H, IMAGE_W = img.shape[0], img.shape[1]

    M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
    Minv = cv2.getPerspectiveTransform(src, dst) # Inverse transformation

    img2 = img[crop[0]:(crop[0]+IMAGE_H), crop[1]:crop[1] + IMAGE_W] # Apply np slicing for ROI crop
    warped_img = cv2.warpPerspective(img2, M, (250, 250)) # Image warping

    return warped_img, Minv


def plot_transform(sample, H_transform):
    """
        Will take in a sample of 6 images transform them and then plot them

        No return
    """

    fig, axs = plt.subplots(2, 3,figsize=(35,20))
    for i in range(6):
        img = sample[0][i].numpy().transpose(1, 2, 0)
        trans_img, _ = transform_image(img, H_transform)
        if i < 3:
            axs[0,i].imshow(trans_img)
            plt.axis('off')
        else:
            axs[1,i-3].imshow(trans_img)
            plt.axis('off')

def plot_original(sample):
    """
        Plots a grid of the 6 images in a sample

    """
    plt.figure(figsize=(15,20))
    plt.imshow(torchvision.utils.make_grid(sample[0], nrow=3).numpy().transpose(1, 2, 0))
    plt.axis('off')
