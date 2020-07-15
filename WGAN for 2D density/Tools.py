import numpy as np
import scipy.io as sio
import pandas as pd
import math 
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns 
import sys 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
import iisignature as signature
from scipy import stats 

def generateur_donnees(sample_size=10, variance=0.5):
    
    uniform = 9*torch.rand(sample_size)
    
    ordo = torch.sin(uniform) + math.sqrt(variance)*torch.randn(sample_size)
    
    points = torch.stack((uniform,ordo)).T
    
    return points

def generate_noise(batch, noise_size):
    
    return torch.normal(mean = torch.zeros(size=(batch,noise_size))).float()

def gradient_penalty(discriminateur,true_data,fake_data,batch_size,penalty_param=1):
    
    length = true_data.shape[1]
    
    eta = torch.FloatTensor(batch_size,length).uniform_(0,1)
    
    interpolated = (1-eta)*true_data + eta*fake_data
    
    interpolated = Variable(interpolated, requires_grad=True)
    
    discrim_interpolated = discriminateur(interpolated)
    
    gradients = autograd.grad(outputs=discrim_interpolated, inputs=interpolated, grad_outputs=torch.ones(discrim_interpolated.size()),create_graph=True, retain_graph=True)[0]
    
    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * penalty_param
    
    gradient_norm = (gradients.norm(2, dim=1)).mean()
    
    return grad_penalty, gradient_norm

def perform_ks_test(truedata,fakedata, results=False):
    
    dim1 = stats.ks_2samp(truedata[:,1], fakedata[:,1])

    dim2 = stats.ks_2samp(truedata[:,0], fakedata[:,0])
    
    print('Result for dimension 1 : ', dim1)
    print('Result for dimension 2 : ', dim2)
    
    if results == True:
        return dim1, dim2
    