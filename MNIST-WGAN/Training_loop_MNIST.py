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

from Tools import generateur_donnees, generate_noise, gradient_penalty, extrapolate_signature,extrapolate_multiple_signatures

from get_data import get_pendigits

from get_MNIST_sig import MNIST_sigs



def train_MNIST(iterations,noise_size, generateur, discriminateur, optigen, optidiscri, sigorder,integer):
    
    erreur_discrim_sur_fake = [0]*iterations
    erreur_discrim_sur_true = [0]*iterations
    gradient = [0]*iterations
    
    training_data,training_size = MNIST_sigs(integer,sigorder)
    
    batch_size = 100 #Nombre de données sur lesquelles sont entraînées les générateurs et discriminateurs 
    
    one = torch.tensor(1, dtype=torch.float)
    mone = one * -1

    for it in np.arange(iterations):

        ##############################
        # Entraînement du générateur #
        ##############################

        if it%2 == 0:

            optigen.zero_grad()
            optidiscri.zero_grad()

            noise = generate_noise(batch_size,noise_size)

            fake_data = generateur(noise)

            generateur_discriminateur_out = discriminateur(fake_data) #Evaluation des fausses données 
            generateur_loss = -generateur_discriminateur_out.mean() #Loss

            generateur_loss.backward()
            optigen.step()

        ##################################
        # Entraînement du discriminateur #
        ##################################

        if it%1 ==0: 
            
            optidiscri.zero_grad()
            optigen.zero_grad()
            
            batch_id = np.random.choice(np.arange(0,training_size),batch_size)
            
            true_datas = training_data[batch_id,:]

            d_loss_real = discriminateur(true_datas) 
            d_loss_real  = d_loss_real.mean()
            d_loss_real.backward(mone,retain_graph=True)

            fake_data = generateur(generate_noise(batch_size,noise_size))

            d_loss_fake = discriminateur(fake_data)
            d_loss_fake = d_loss_fake.mean()
            d_loss_fake.backward(one, retain_graph=True)

            gradient_pen, gradient[it] = gradient_penalty(true_data = true_datas,fake_data = fake_data,batch_size = batch_size,discriminateur = discriminateur)

            gradient_pen.backward()

            d_loss = d_loss_real - d_loss_fake + gradient_pen

            erreur_discrim_sur_true[it] = d_loss_real - d_loss_fake

            optidiscri.step()
        
        if it%100 == 0:
            print('Iteration {} done...'.format(it))
            
        if it == iterations-1:
            print('Tadaaa !')
    

    return  erreur_discrim_sur_fake,erreur_discrim_sur_true, gradient