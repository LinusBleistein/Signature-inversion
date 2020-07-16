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
import signatory


def generateur_donnees(sample_size=10,taille=2, order = 4, dimension=2):
    
    data = torch.zeros(size=(sample_size,dimension,taille))
    
    noise = torch.rand(size=(sample_size,dimension,1))
    
    #print(noise)
    
    for i in np.arange(sample_size):
        data[i,:,:] = torch.arange(1,taille+1)
        
    for i in np.arange(sample_size):
        for dim in np.arange(dimension):
            data[i,dim,:] *= noise[i,dim,0]
    
    signature_size = signature.siglength(dimension,order)
    
    signature_true_data = torch.zeros(size=(sample_size,signature_size))
    
    for data_point in np.arange(sample_size):
        path = data[data_point,:,:]
        path = path.numpy()
        path = path.T
        signature_true_data[data_point,:] = torch.tensor(signature.sig(path,order))
    
    return signature_true_data

def generate_noise(batch, noise_size):
    
    return torch.normal(mean = torch.zeros(size=(batch,noise_size))).float()

def extrapolate_signature(sig, dimension, order):
    
    first_coefficients = sig[0:dimension]
    
    sig_size = signature.siglength(dimension,order)
    
    extrapolated_path = torch.zeros(size=(dimension,2))
    
    for i in np.arange(dimension):
        
        extrapolated_path[i,:] = torch.tensor(np.arange(0,2,1)*first_coefficients[i])
        
    extrapolated_path = extrapolated_path.T
    
    extrapolated_signature = signature.sig(extrapolated_path, order)
    
    return extrapolated_signature 

def extrapolate_multiple_signatures(sigs,dimension,order):
    
    data_size = sigs.shape[0]
    
    sig_size = signature.siglength(dimension,order)
    
    extrapolated_data = np.zeros(shape=(data_size, sig_size))
    
    for row in np.arange(data_size):
        signature_to_extrapolate = sigs[row,:]
        extrapolated = extrapolate_signature(signature_to_extrapolate,dimension,order)
        extrapolated_data[row,:] = extrapolated
        
    return extrapolated_data

def gradient_penalty(discriminateur,true_data,fake_data,batch_size,penalty_param=10):
    
    length = true_data.shape[1]
    
    eta = torch.FloatTensor(batch_size,length).uniform_(0,1)
    
    interpolated = (1-eta)*true_data + eta*fake_data
    
    interpolated = Variable(interpolated, requires_grad=True)
    
    discrim_interpolated = discriminateur(interpolated)
    
    gradients = autograd.grad(outputs=discrim_interpolated, inputs=interpolated, grad_outputs=torch.ones(discrim_interpolated.size()),create_graph=True, retain_graph=True)[0]
    
    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * penalty_param
    
    gradient_norm = (gradients.norm(2, dim=1)).mean()
    
    return grad_penalty, gradient_norm



def get_signature_as_tensor(sig,dimension,order):
    
    ''' This function transforms a signature given as a vector into a series of tensors, 
    stored in a dictionnary. It uses the method extract_signature_term() of Signatory. 
    
    For instance, for a path in R^d, the key called 'Depth z' is a Pytorch tensor 
    of size (R^d)^{\otimes z}.
    
    Arguments that need to be specified are:
        - sig : the original signature
        - dimension : the dimension of the original path
        - order : the order (or total depth) at which the original signature is calculated.
    
    '''
    
    total_tensor = {}
    
    for depth in np.arange(1,order+1):
        
        signature_at_depth = signatory.extract_signature_term(sig,dimension,int(depth))
        
        tensor_size = [dimension]*int(depth)
        
        signature_at_depth = signature_at_depth.view(tensor_size)
        
        total_tensor["Depth " + str(depth)] = signature_at_depth
        
    return total_tensor      




def get_length(signature,dimension,order):
    
    '''This function approximates the length of the path through the signature.
    
        Arguments are : 
            - signature : the full signature. 
            - order : the order at which the signature was truncated. 
    
    '''
    
    last_signature_term = signatory.extract_signature_term(signature,dimension,int(order))
    
    return torch.norm(math.factorial(order)*last_signature_term,2)**(1/order)
    