##########################################################################################################
# Argparser#############################################################

import argparse

parser = argparse.ArgumentParser(description='Simple AI APP')
"""
parser.add_argument('-learning_rate','--learning', action="store_true", default= 0.001)
parser.add_argument('-hidden_units', '--hidden', action="store_true", default= 512)
parser.add_argument('-epochs', '--epochs', action="store_true", default= 3)
parser.add_argument('-arch', '--archic', action="store_true", default= "vgg19")
"""

parser.add_argument('--data_dir', default='flowers')
parser.add_argument('--save_directory', default='./')
parser.add_argument('--learning_rate', default=0.001,type=float)
parser.add_argument('--hidden_units', type=int, default= 512)
parser.add_argument('--epochs', type=int, default= 3)
parser.add_argument('--arch',  default= "vgg19")
parser.add_argument('--gpu',  default= 'cuda')

arguments=parser.parse_args()

data_dir=arguments.data_dir
save_directory=arguments.save_directory
lr=arguments.learning_rate
hu=arguments.hidden_units
eps=arguments.epochs
arch=arguments.arch
gpu=arguments.gpu





#print(lr,hu,eps)



##### IMPORTS######################

# Imports here

# importing visualization packages

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

# importing matplotlib
import matplotlib.pyplot as plt

from collections import OrderedDict 

# importing torch modules
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.models as models
from torch.autograd import Variable
from torch.optim import lr_scheduler

# Pil libraries

import PIL
from PIL import Image

# import numpy

import numpy as np
import numpy

#importing seaborn

import seaborn as sns


#importing time

import time
import os
import copy


if(torch.cuda.is_available()==False):
	dev='cpu'


##################################################
### DIRECTORIES####

#data_dir = 'flowers'
data_dir = data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

##################################################
#########################################################################################################

# FROM THE TRANSFER LEARNING TUTORIAL OF PYTORCH
#https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#load-data

# THE DATA TRANSFORMS

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]),
}


# TODO: Load the datasets with ImageFolder
##image_datasets = 




dirs = {'train': train_dir, 
        'valid': valid_dir, 
        'test' : test_dir}
image_datasets = {x: datasets.ImageFolder(dirs[x],   transform=data_transforms[x]) for x in ['train', 'valid', 'test']}

# TODO: Using the image datasets and the trainforms, define the dataloaders
##dataloaders = 



bs=64

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=bs, shuffle=True) for x in ['train', 'valid', 'test']}
dataset_sizes = {x: len(image_datasets[x]) 
                              for x in ['train', 'valid', 'test']}
class_names = image_datasets['train'].classes

#########################################################################################################
#########################################################################################################
#########################################################################################################

# LABEL MAPPING#
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
##########################################################################################################33
##########################################################################################################33
# SETTING THE MODEL
if(arch=='vgg19'):
    model=models.vgg19(pretrained=True)
elif(arch=='vgg16'):
    model=models.vgg16(pretrained=True)

    #model=models.vgg16(pretrained=True)

#########################################################################################################
##########################################################################################################
# SETTING THE CLASSIFIER!


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

"""
model.classifier = nn.Sequential(nn.Linear(25088, 4096),
#                                 nn.ReLU(),
 #                                nn.Dropout(0.2),
  #                               nn.Linear(16384, 8192),
#                                 nn.ReLU(),
 #                                nn.Dropout(0.2),
  #                               nn.Linear(8192, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(4096, 512),
#                                 nn.ReLU(),
 #                                nn.Dropout(0.2),
  #                               nn.Linear(512, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 102),
                                 nn.LogSoftmax(dim=1))
"""
classifier = nn.Sequential(OrderedDict([
                         ('fc1',   nn.Linear(25088, 1024)),
                         ('drop',  nn.Dropout(p=0.2)),
                         ('relu',  nn.ReLU()),
#                         ('fc2',   nn.Linear(1024, 512)),
                         ('fc2',   nn.Linear(1024, hu)),
                         ('drop',  nn.Dropout(p=0.2)),
                         ('relu',  nn.ReLU()),
 #                        ('fc3',   nn.Linear(512, 256)),
                         ('fc3',   nn.Linear(hu, 256)),
                         ('drop',  nn.Dropout(p=0.2)),
                         ('relu',  nn.ReLU()),
                         ('fc4',   nn.Linear(256, 102)),
                         ('drop',  nn.Dropout(p=0.2)),
                         ('relu',  nn.ReLU()),
#                         ('fc5',   nn.Linear(64, 8)),
#                         ('drop',  nn.Dropout(p=0.2)),
#                         ('relu',  nn.ReLU()),
                         ('output', nn.LogSoftmax(dim=1))
                         ]))

model.classifier = classifier

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
#optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)
#optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
optimizer = optim.Adam(model.classifier.parameters(), lr=lr)

model.to(device);

#########################################################################################################
##########################################################################################################
# PARAMETERS FOR THE NETWORK!

# Criteria NLLLoss which is recommended with Softmax final layer
criteria = nn.NLLLoss()
# Observe that all parameters are being optimized
#optim = optim.Adam(model.classifier.parameters(), lr=0.001)
optim = optim.Adam(model.classifier.parameters(), lr=lr)
# Decay LR by a factor of 0.1 every 4 epochs
sched = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
# Number of epochs
#eps=10


#########################################################################################################
##########################################################################################################
# TRAINING THE MODEL

#def train_model(model, criteria, optimizer, scheduler,    
#                                      num_epochs=25, device='cuda'):
def train_model(model, criteria, optimizer, scheduler,    
                                      num_epochs=25, device=gpu):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model

# TRAINING THE MODEL
#eps=3
#model_tr = train_model(model, criteria, optim, sched, eps, 'cuda')
model_tr = train_model(model, criteria, optim, sched, eps, gpu)


#########################################################################################################
##########################################################################################################
# TESTING THE MODEL

def calc_accuracy(model, data, cuda=False):
    model.eval()
    model.to(device='cuda')    
    
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloaders[data]):
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            # obtain the outputs from the model
            outputs = model.forward(inputs)
            # max provides the (maximum probability, max value)
            _, predicted = outputs.max(dim=1)
            # check the 
#            if idx == 0:
#                print(predicted) #the predicted class
#                print(torch.exp(_)) # the predicted probability
            equals = predicted == labels.data
#            if idx == 0:
#                print(equals)
            print(equals.float().mean())

# CALCULATING THE ACCURACY!

calc_accuracy(model,'test',True)

########################################################################################################
########################################################################################################
# TODO: Save the checkpoint 

checkpoint = {
#              'arch':'vgg19',
              'arch': arch,
              'input_size': 25088,
              'output_size': 102,
              'hidden_units': hu,
#              'hidden_layers': [each.out_features for each in model.hidden_layers],
              'epoch':eps,
#              'class_to_idx':model.class_to_idx = image_datasets['train'].class_to_idx,
              'class_to_idx': image_datasets['train'].class_to_idx,
              'model_state_dict': model.state_dict(),
              'opt_state_dict': optimizer.state_dict()
             }

torch.save(checkpoint, save_directory+'checkpoint.pth')





