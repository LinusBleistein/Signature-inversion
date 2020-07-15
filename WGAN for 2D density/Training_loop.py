import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
import iisignature as signature

from Tools import *

def train(variance,iterations, noise_size, discriminator,generator,optidiscri, optigen):
    
    erreur_discrim_sur_fake = [0]*iterations
    erreur_discrim_sur_true = [0]*iterations
    gradient = [0]*iterations
    
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

            noise = generate_noise(batch_size, noise_size)

            fake_data = generator(noise)

            generateur_discriminateur_out = discriminator(fake_data) #Evaluation des fausses données 
            generateur_loss = -generateur_discriminateur_out.mean() #Loss

            erreur_discrim_sur_fake[it] = -generateur_loss #-Loss sur les fausses données

            generateur_loss.backward()
            optigen.step()

        ##################################
        # Entraînement du discriminateur #
        ##################################

        if it%1 ==0:
            
            optidiscri.zero_grad()
            optigen.zero_grad()

            true_datas = generateur_donnees(sample_size = batch_size, variance = variance)

            d_loss_real = discriminator(true_datas) 
            d_loss_real  = d_loss_real.mean()
            d_loss_real.backward(mone,retain_graph=True)

            fake_datas = generator(generate_noise(batch_size, noise_size))

            d_loss_fake = discriminator(fake_datas)
            d_loss_fake = d_loss_fake.mean()
            d_loss_fake.backward(one, retain_graph=True)

            gradient_pen, gradient[it] = gradient_penalty(discriminateur = discriminator,true_data = true_datas,fake_data = fake_datas,batch_size = batch_size)

            gradient_pen.backward()

            d_loss = d_loss_real - d_loss_fake + gradient_pen

            erreur_discrim_sur_true[it] = d_loss_real - d_loss_fake

            optidiscri.step()

            
        ##########
        # Autres #
        ##########
        
        if it%100 == 0:
            print('Iteration {} done...'.format(it))
            
        if it == iterations-1:
            print('Tadaaa !')
    

    return  erreur_discrim_sur_fake,erreur_discrim_sur_true, gradient