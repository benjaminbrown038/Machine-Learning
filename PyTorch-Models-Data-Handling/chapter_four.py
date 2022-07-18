'''
Chapter 4 of Deep Learning for Coders

Most of the data used for these functions are regarding tensors that represents 3 and 7.
Some of these functions are specific for this data.
'''

import matplotlib.pyplot as plt
from fastai.vision.all import *
from torch.nn import functional
import numpy

'''
Data
'''
path = untar_data(URLs.MNIST_SAMPLE)
time = torch.arange(0,20)
params = torch.randn(3).requires_grad_()

'''
Creating a simple model with parameters and applying gradient to weight to minimize loss

parameters: <tensor> parameters for model
returns: <float> prediction of model
'''
def apply_step(params,prn=True):
    speed = time*3 + (time-9.5)**2 + 1
    a,b,c = params
    pred = a*(time**2) + b*time + c
    loss = ((pred - speed)**2).mean()
    loss.backward()
    lr = 1e-5
    # becomes a tensor that computes
    params.grad
    params.data -= lr * params.grad.data
    params.grad=None
    if prn: print(loss.item())
    return pred

# loss
def L1_loss(average,real):
    result = (average - real).abs().mean()
    return result

# loss 
def mean_sq_error_loss(average,real):
    result = ((average-real)**2).sqrt().mean()
    return result

# weigths
def init_params(size,std=1.0):
    params = (torch.randn(size)*std).requires_grad_()
    return params

# train
def linear1(xb):
    weights = xb@weights + bias
    return weights

# activation
def sigmoid(x):
    sig = 1/(1+torch.exp(-x))
    return sig

# loss
def mnist_loss(predictions,targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1,1-predictions,predictions).mean()

# train
def calc_grad(xb,yb,model):
    preds = model(xb)
    loss = mnist_loss(preds,yb)
    loss.backward()

# metrics
def batch_accuracy(xb,yb):
    preds = xb.sigmoid()
    correct = (preds > .5) == yb
    result = correct.float().mean()
    return result

# metrics
def validate_epoch(model):
    accs = [batch_accuracy(model(xb),yb) for xb,yb in valid_dl]
    result = round(torch.stack(accs).mean().item(),4)
    return result

# train
def train_epoch(model,dl,opt):
    for xb,yb in dl:
        calc_grad(xb,yb,model)
        opt.step()
        opt.zero_grad()
      
# train
def train_model(model,epochs):
    for i in range(epochs):
        train_epoch(model)
        print(validate_epoch(model),end=' ')

# train
def simple_net(xb):
    res = xb@w1 + b1
    res = res.max(tensor(0.0))
    res = res@w2 + b2
    return res

# train
class BasicOptim:

    def __init__(self,params,lr):
        self.params,self.lr = list(params),lr

    def step(self,*args,**kwargs):
        for p in self.params:
            p.data-=p.grad.data *self.lr

    def zero_grad(self,*args,**kwargs):
        for p in self.params:

