# -*- coding: utf-8 -*-

import numpy as np
import scipy.io as sio
import math 
import matplotlib.pyplot as plt
import matplotlib
import sys 
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
import iisignature as signature

import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

from tools import generateur_donnees, generate_noise, extrapolate_signature, gradient_penalty, extrapolate_multiple_signatures
from WGAN import *
from Training_loop_MNIST import train_MNIST
from MNIST_Plots import plot_training_results

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


############
# Settings #
############

noise_size = 8 #Size of the latent space of the generator, ie size of the input noise 

n = 100 #Number of training iterations

signature_order = 10 #Order of the generated signatures 

integer = 3 #MNIST digit you want to generate 

#########################
# Training of the model #
#########################

signature_length = signature.siglength(2,signature_order)

#gen =generateur(noise_size,signature_length*2,signature_length)
gen =generateur_small(noise_size,248,signature_length)

discrim = discriminateur_small(signature_length,2048)

optimisateur_generateur = optim.Adam(gen.parameters(),lr = 0.0001, betas = (0.5,0.999))
optimisateur_discriminateur = optim.Adam(discrim.parameters(),lr=0.0001,betas = (0.5,0.999))

erreur_discrim_sur_fake,erreur_discrim_sur_true, gradient = train_MNIST(iterations = n, noise_size = noise_size,generateur=gen,discriminateur=discrim, optigen=optimisateur_generateur,optidiscri=optimisateur_discriminateur,sigorder=signature_order,integer=integer)

##############
# Save model #
##############

path = 'trained_models/generator_int_' + str(integer) + '_sigorder_'+ str(signature_order) + '_iterations_' +str(n)

torch.save(gen.state_dict(),path)