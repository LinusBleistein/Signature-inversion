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
    