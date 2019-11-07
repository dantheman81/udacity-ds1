import argparse
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import PIL
import json
import modelhelper
import utilityfunctions
import workspace_utils

from tabulate import tabulate

parser = argparse.ArgumentParser()

parser.add_argument('image_path', action='store')
parser.add_argument('checkpoint', action='store')
parser.add_argument('--topk', action='store', dest='topk')
parser.add_argument('--category_names', action='store', dest='category_names')
parser.add_argument('--gpu', action='store_true', default=False,dest='boolean_gpu')


results = parser.parse_args()

# default values
image_path = results.image_path
chechpoint = results.checkpoint
arch = 'vgg16'
topk = 5
category_names = ''
device = 'cpu'

if results.boolean_gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
if results.checkpoint:
    checkpoint = results.checkpoint
    
if results.topk:
    topk = int(results.topk)
    
if results.category_names:
    category_names = results.category_names
    
model, optimizer, epochs = utilityfunctions.load_checkpoint(checkpoint)

top_probabilities, top_classes, top_flowers = utilityfunctions.predict(image_path, model, topk, category_names, device)

print(tabulate({"Flower name": top_flowers, "Class": np.transpose(top_classes), "Probability": np.transpose(top_probabilities)}, headers="keys"))