from fastai.vision.all import *
from fastai.callback.hook import *
import os
import cv2
from torch.nn import functional
from torch.nn import Sequential, Conv2d
from torch.optim import SGD, Adam
from torchvision import models

# loading data
def load_data(folder_name):
    training_tensor = [tensor(Image.open(i)) for i in folder_name]
    training_stack = ((torch.stack(training_tensor)).float())
    return training_stack

# transforming data
def training_data(*args):
    training = (torch.cat(args))
    return training

# data information
def size(training_stack):
    size = ((training_stack.shape)[1]) * (training_stack.shape[2])
    return size

# creating data 
def init_weights(size):
    weights = (torch.randn(size)).requires_grad_()
    return weights

# creating data 
def bias():
    bias = torch.randn(1)
    return bias

# transforming data 
def transform_data_for_model(training_stack):
    result = training_stack[1] * training_stack[2]
    return result

# transforming data
def matrix_multiply(training_stack):
    new_training_stack = (training_stack).view(-1,784)
    pred = ((new_training_stack) @ weights) + bias
    return pred

# metric
def loss(pred,target):
    loss = (pred-target).abs().mean()
    return loss

# train
def update(lr):
    new_weights -= weights.grad * lr
    return new_weights

# data information
def size_of_image(image):
    image_size = image.shape
    return image_size

# data transformation
def apply_kernel(row,col,kernel):
    convolution = (img[row-1:row+2,col-1:col+2] * kernel).sum()
    return convolution

# transformation
def convolution_top():
    rng = (1,27)
    top_edge = tensor([[apply_kernel(i,j,top_edge) for j in rng] for i in rng])
    return top_edge

# information
def row(padding, stride, height):
    new_row = (height + padding) // stride
    return new_row

# information
def column(padding,stride,height):
    new_column = (height + padding) // stride
    return new_column

# information
def output_shape(w,n,p,f):
    output = int((W - K + (2*P))/(S + 1))
    new_output = (w - n + (2*p) - f) + 1
    return new_output

# creating kernels
def top_edge():
    top_edge = (tensor([1,1,1],[0,0,0],[-1,-1,-1])).float()
    return top_edge

# creating kernels
def bottom_edge():
    bottom_edge = (tensor([-1,-1,-1],[0,0,0],[1,1,1])).float()
    return bottom_edge

# creating kernels
def right_edge():
    right_edge = (tensor([-1,0,1],[-1,0,1],[-1,0,1])).float()
    return right_edge

# creating kernels
def left_edge():
    left_edge = (tensor([1,0,-1],[1,0,-1],[1,0,-1])).float()
    return left_edge

# creating kernels
def diag1_edge():
    diag1_edge = (tensor([1,0,-1],[0,1,0],[-1,0,1])).float()
    return diag1_edge
