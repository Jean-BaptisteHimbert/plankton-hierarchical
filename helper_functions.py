import numpy as np
import matplotlib.pyplot as plt
import logging
import seaborn as so
import pickle 
from pathlib import Path
from sklearn import preprocessing

import torch
from torch.utils.data import DataLoader
import torchvision


logger = logging.getLogger(__file__)

def imshow(mean, std, inp, title=None): # Example from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    # Imshow for Tensor
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    plt.axis('off')
    plt.imshow(inp, aspect='auto')

def class_histogram(dataset, log_scale: bool = False):
    fig = plt.figure(figsize=(11, 0.29*len(dataset.classes)))
    if log_scale:
        plt.xscale('log')
    ax= so.histplot(y=[dataset.classes[idx] for idx in dataset.labels], discrete=True)

    for patch in ax.patches:
        ax.annotate(patch.get_width(), (patch.get_width()+0.029, patch.get_y() + patch.get_height()/2 ), verticalalignment='center', fontsize='x-small')

    plt.tight_layout()

def sample_images(datamodule, loader, num_images):
    
    mean = datamodule.mean
    std = datamodule.std
    if loader == 'train':
        data_iterator = iter(datamodule.train_dataloader())
    elif loader == 'val':
        data_iterator = iter(datamodule.val_dataloader())
    elif loader == 'test':
        data_iterator = iter(datamodule.test_dataloader())

    images, class_name = next(data_iterator)
    input_shape = images[0].shape

    # Make a grid from a subset of the images
    images_subset = images[:num_images]
    grid = torchvision.utils.make_grid(images_subset)
    grid = grid.numpy().transpose((1, 2, 0))
    grid = std * grid + mean
    grid = np.clip(grid, 0, 1)

    # Display the grid using matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(grid)

    # Set the spacing between subplots
    fig.subplots_adjust(hspace=0.5)

    # Remove the axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    plt.show

    return input_shape