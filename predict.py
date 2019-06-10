##########################################################################################################
# Argparser#############################################################

import argparse

parser = argparse.ArgumentParser(description='Simple AI APP')

parser.add_argument('--top_k', default=5,type=int)
parser.add_argument('--category_names', default='cat_to_name.json')
parser.add_argument('--gpu', default='cuda')
parser.add_argument('--path_to_image', default='flowers/test/12/image_04014.jpg')
parser.add_argument('--checkpoint', default='checkpoint')
#image_path = 'flowers/test/12/image_04014.jpg'
#
arguments=parser.parse_args()

topk=arguments.top_k
catnames=arguments.category_names
dev=arguments.gpu
image_path=arguments.path_to_image
checkpoint=arguments.checkpoint


########################################################################################################################
# IMPORTS

# Imports here

# importing visualization packages
#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'

# importing matplotlib
import matplotlib
matplotlib.use('Agg')
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


#############################################################################
#############################################################################
# label mapping

import json

#with open('cat_to_name.json', 'r') as f:
with open(catnames, 'r') as f:
    cat_to_name = json.load(f)
    
#############################################################################
#############################################################################
# TODO: Write a function that loads a checkpoint and rebuilds the model



def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if(checkpoint['arch']=='vgg19'):
        model = models.vgg19(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False        
    elif(checkpoint['arch']=='vgg16'):
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False        
    hu=checkpoint['hidden_units']
#    model=models.vgg19(pretrained=True)
#    model.classifier=classifier
    classifier = nn.Sequential(OrderedDict([
                             ('fc1',   nn.Linear(25088, 1024)),
                             ('drop',  nn.Dropout(p=0.2)),
                             ('relu',  nn.ReLU()),
#                             ('fc2',   nn.Linear(1024, 512)),
                             ('fc2',   nn.Linear(1024, hu)),
                             ('drop',  nn.Dropout(p=0.2)),
                             ('relu',  nn.ReLU()),
#                             ('fc3',   nn.Linear(512, 256)),
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


#    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

#    '''
#    model = model(checkpoint['input_size'],
#                             checkpoint['output_size'],
#                             checkpoint['epoch'],
#                             checkpoint['class_to_idx'])
#    '''
    model.class_to_index = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['model_state_dict'])
#    optimizer.load_state_dict(checkpoint['opt_state_dict'])
    return model

#############################################################################
#############################################################################

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model    

    im = Image.open(image)
    '''
    width, height = im.size
    print(width,height)
    if(width<height):
        widthnew=256
        heightnew=height*widthnew/width
    else:
        heightnew=256
        widthnew=width*heightnew/height
#    x,y = 256,256
    print(widthnew,heightnew)
#    im.thumbnail(,Image.ANTIALIAS)
    im.thumbnail(,Image.ANTIALIAS)
#    im.crop(box=(32,32,32,32))
    im.crop(box=(224,224,224,224))
    '''
    if im.size[0] > im.size[1]: #(if the width > height)
        im.thumbnail((1000000, 256)) #constrain the height to be 256
    else:
        img.thumbnail((256, 200000)) #otherwise constrain the width

    left_margin = (im.width-224)/2
    bottom_margin = (im.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    
    im = im.crop((left_margin, bottom_margin, right_margin,    
                   top_margin))

                      
    np_image=numpy.array(im)/255
    np_image=(np_image -numpy.array([0.485,0.456,0.406]))/numpy.array([0.229,0.224,0.225])
    np_image=np_image.transpose((2, 0, 1))

    return np_image
    
#############################################################################
#############################################################################
# ImShow

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
#    image = image.numpy().transpose((1, 2, 0))
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
#    model=model.to(device)
    inputs=process_image(image_path)
#    print(inputs)
    
#    inputs = torch.from_numpy(inputs).float().to(device)
#    model=model.to('cuda')
    inputs = torch.from_numpy(inputs).type(torch.FloatTensor)
    inputs=inputs.to(device)
    inputs=inputs.unsqueeze(0)
    model=model.to(device)
#    ''
    # inverting the class dictionary
    
    dict_cx=model.class_to_index
    inverted_cx = dict([v,k] for k,v in dict_cx.items())
    
    logps = model.forward(inputs)
    #                batch_loss = criterion(logps, labels)
                    
    #                test_loss += batch_loss.item()
                    
    probs = torch.exp(logps)
                    # Calculate accuracy

    # Top probs
    top_probs, top_labs = probs.topk(topk)
    top_probs = top_probs.detach().cpu().numpy().tolist()[0] 
    top_labs = top_labs.detach().cpu().numpy().tolist()[0]
                
    ps = torch.exp(logps)
    top_p, top_class = ps.topk(topk, dim=1)

    top_labels = [inverted_cx[lab] for lab in top_labs]
    top_flowers = [cat_to_name[inverted_cx[lab]] for lab in top_labs]


    return top_probs, top_labels, top_flowers

#############################################################################################################
#############################################################################################################

# TODO: Display an image along with the top 5 classes

def plot_testing(model,image_path):
    # Setting up the plot
    plt.figure(figsize = (10,10))
    ax = plt.subplot(2,1,1)
#    print('aqui')
    # Setting up the title
    # taking the third element from the splitting of the path
    flower_num = image_path.split('/')[2]
    # using the json from the beginning!
    title_ = cat_to_name[flower_num]
    # Plotting the flower
    # you've got to use the imshow (Image Show). this is also in the documentation!
    image = process_image(image_path)
#    print(image)
    imshow(image, ax, title = title_);
    # Make prediction
    # taking the top 5 probabilities, labels and flower names
    # use the function predict created above
#    probs, labs, flowers = predict(image_path, model) 
    probs, labs, flowers = predict(image_path, model,topk) 
    # Plotting th bar chart
    # another subplot, but in another place
    print(probs)
    plt.subplot(2,1,2)
    # a barplot with seaborn library!
#    plt.show()
    sns.barplot(x=probs, y=flowers, color=sns.color_palette()[0]);
    plt.savefig('predicting.png')
    plt.show()

###########################################################################################
# Plotting the Image

#modelnew=load_checkpoint('checkpoint.pth')
modelnew=load_checkpoint(checkpoint+'.pth')
device=dev
#image_path = 'flowers/test/100/image_07897.jpg'
#image_path = 'flowers/test/12/image_04014.jpg'
#print(image_path)
plot_testing(modelnew,image_path)









