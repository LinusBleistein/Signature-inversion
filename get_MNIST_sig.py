from get_data import get_pendigits
import signatory
import iisignature
import torch
import numpy as np 

import matplotlib.pyplot as plt 

def MNIST_sigs(integer,order):
    
    X,y=get_pendigits()
    
    y = y.astype(np.float)
    
    selected_digits = (y == integer)
    
    n_digits = selected_digits.sum()
    
    training_data = X[(y==integer),:,:]/100
    
    sigsize = iisignature.siglength(2,order)
    
    sigs = torch.zeros(size=(n_digits,sigsize))
    
    for i in np.arange(n_digits):
    
        path = torch.tensor(training_data[i,:,:])

        sigs[i,:] = signatory.signature(path.unsqueeze(0),order)
        
    return sigs,n_digits
    
    