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

parser = argparse.ArgumentParser()

parser.add_argument('data_directory', action='store')
parser.add_argument('--save_dir', action='store', dest='save_directory')
parser.add_argument('--arch', action='store', dest='arch')
parser.add_argument('--learning_rate', action='store', dest='learning_rate')
parser.add_argument('--hidden_units', action='store', dest='hidden_units')
parser.add_argument('--epochs', action='store', dest='epochs')
parser.add_argument('--gpu', action='store_true', default=False,dest='boolean_gpu')

results = parser.parse_args()

# default values
arch = 'vgg16'
save_directory = './'
learning_rate = 0.001
hidden_units = 4096
epochs = 5
device = 'cpu'

if results.boolean_gpu:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
if results.arch:
    arch = results.arch
    
if results.save_directory:
    save_directory = results.save_directory
    
if results.learning_rate:
    learning_rate = results.learning_rate
    
if results.hidden_units:
    hidden_units = results.hidden_units

if results.epochs:
    epochs = int(results.epochs)

# load data from data_directory and create loaders
trainloader, validloader, testloader, train_data = utilityfunctions.load_data(results.data_directory)

# create model
if arch == 'vgg16':
    model = models.vgg16(pretrained=True)
    in_units = 25088
    for param in model.parameters():
        param.requires_grad = False
elif arch == 'alexnet':
    model = models.alexnet(pretrained=True)
    in_units = 9216
    for param in model.parameters():
        param.requires_grad = False
else:
    print('Architecture not supported - supported: VGG16, alexnet')
    exit()
            
criterion, optimizer = modelhelper.create_nn(model, learning_rate, in_units, hidden_units)

# train the classifier layers using backpropagation using the pre-trained network to get the features
from workspace_utils import active_session

with active_session():
    modelhelper.do_deep_learning(model, trainloader, validloader, epochs, 40, criterion, optimizer, device)
    
modelhelper.check_accuracy_on_test(testloader, model, device)

utilityfunctions.save_checkpoint(model, optimizer, train_data, epochs, arch, learning_rate, save_directory)
