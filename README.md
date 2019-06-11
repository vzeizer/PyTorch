# Project Deep Learning with PyTorch from Udacity Nanodegree AI fundamenetals with Python


## Project Instructions

### Part 1 Developing an Image Classifier with Deep Learning

In this first part of the project, you'll work through a Jupyter notebook to implement an image classifier with PyTorch. 
We'll provide some tips and guide you, but for the most part the code is left up to you. 
As you work through this project, please refer to the rubric for guidance towards a successful submission.

Remember that your code should be your own, please do not plagiarize (see here for more information).

This notebook will be required as part of the project submission. 
After you finish it, make sure you download it as an HTML file and include it with the files you write in the next part of the project.

We've provided you a workspace with a GPU for working on this project. 
If you'd instead prefer to work on your local machine, you can find the files on GitHub here.

If you are using the workspace, be aware that saving large files can create issues with backing up your work. 
You'll be saving a model checkpoint in Part 1 of this project which can be multiple GBs in size if you use a large classifier network. 
Dense networks can get large very fast since you are creating N x M weight matrices for each new layer. 
If you're using VGGnets, the input to the dense classifier is something like 20000 units. 
Adding a layer of 1024 units (of 32-bit floating point values) after that leads to 20000 * 1024 * 32 \,\mathrm{bits} \approx 82 \, \mathrm{MB}20000∗1024∗32bits≈82MB just for that one weight matrix. 
In general, it's better to avoid wide layers and instead use more hidden layers, this will save a lot of space. 
Keep an eye on the size of the checkpoint you create. 
You can open a terminal and enter ls -lh to see the sizes of the files. 
If your checkpoint is greater than 1 GB, reduce the size of your classifier network and resave the checkpoint.

### Part 2 - Building the command line application

Now that you've built and trained a deep neural network on the flower data set, it's time to convert it into an application that others can use. 
Your application should be a pair of Python scripts that run from the command line. 
For testing, you should use the checkpoint you saved in the first part.

Specifications

The project submission must include at least two files train.py and predict.py. 
The first file, train.py, will train a new network on a dataset and save the model as a checkpoint. 
The second file, predict.py, uses a trained network to predict the class for an input image. 
Feel free to create as many other files as you need. 
Our suggestion is to create a file just for functions and classes relating to the model and another one for utility functions like loading data and preprocessing images. 
Make sure to include all files necessary to run train.py and predict.py in your submission.


Train a new network on a data set with train.py

Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains

Options:

-Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
-Choose architecture: python train.py data_dir --arch "vgg13"
-Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
-Use GPU for training: python train.py data_dir --gpu


Predict flower name from an image with predict.py along with the probability of that name. 
That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint

Options:

-Return top KK most likely classes: python predict.py input checkpoint --top_k 3
-Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
-Use GPU for inference: python predict.py input checkpoint --gpu

## What to Install?

1. Python
2. NumPy
3. matplotlib
4. pytorch
5. PIL
6. seaborn


## Content of the files: 

1. Image Classifier Project (1).html: the html of the Jupyter notebook file of the project;
2. predict.py: predicts a flower based on a given architecture;
3. train.py: train a model to classify flowers;
4. Rubrics_project2.pdf: rubrics of the project according to Udacity.

## MIT License

Copyright (c) 2019 Vagner Zeizer C. Paes

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.




