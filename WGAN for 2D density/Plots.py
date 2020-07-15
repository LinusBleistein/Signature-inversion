import seaborn as sns 
import numpy as np 
import matplotlib.pyplot as plt 

def plot_training_results(fakedata,truedata,w1_distance,gradient):
    
    f,ax = plt.subplots(figsize=(25,10))

    ax.scatter(fakedata[:,0],fakedata[:,1], c= 'red', label='Fake data')
    ax.scatter(truedata[:,0],truedata[:,1],c = 'blue',label='True data')
    ax.plot(np.arange(0,10,0.01),np.sin(np.arange(0,10,0.01)))
    ax.set_title('True vs. generated fake data', fontsize = 'xx-large',fontweight='bold')
    ax.legend(fontsize='xx-large')
    plt.show()
    
    n = len(gradient)

    points = np.arange(0,n)

    w1_distance = np.array(w1_distance)

    gradient = np.array(gradient)

    f,ax = plt.subplots(figsize=(25,10))
    ax.scatter(points, w1_distance, alpha=0.5)
    ax.set_title('Wasserstein distance, as calculated by the discriminator', fontsize = 'xx-large',fontweight='bold')
    plt.show()

    f,ax = plt.subplots(figsize=(25,10))
    ax.scatter(points,gradient,alpha=0.5)
    ax.set_title('Norme du gradient',fontsize = 'xx-large',fontweight='bold')
    plt.show()
    
def plot_densities(fakedata,truedata): 
    
    f,ax = plt.subplots(1,2,figsize=(20,10), sharex='row', sharey='row')

    ax[1].set_title('Density of fake data', fontsize='xx-large', fontweight='bold',y=1.01)

    ax[0].set_title('Density of true data', fontsize='xx-large', fontweight='bold', y=1.01)

    sns.kdeplot(fakedata[:,0], fakedata[:,1], cmap="Reds", shade=True, shade_lowest=False, ax = ax[1])

    sns.kdeplot(truedata[:,0], truedata[:,1], cmap="Blues", shade=True, shade_lowest=False, ax = ax[0])

    plt.show()
    
    
def plot_cdf(fakedata,truedata):
    
    sns.set_style('whitegrid')
    sns.set_context('talk')

    f,ax = plt.subplots(2,figsize=(15,10))
    f.tight_layout(pad=3)

    f.suptitle('Empirical CDF for true and generated data for both dimensions', fontweight= 'bold')

    ax[0].plot(np.sort(fakedata[:,0]), np.linspace(0, 1, 10000, endpoint=False), label = 'Generated data')
    ax[0].plot(np.sort(truedata[:,0]), np.linspace(0, 1, 10000, endpoint=False), label = 'True data', linestyle='-.')
    ax[0].set_title('Dimension 1', fontweight='bold')
    ax[0].legend(fontsize='large')

    ax[1].plot(np.sort(fakedata[:,1]), np.linspace(0, 1, 10000, endpoint=False), label = 'Generated data')
    ax[1].plot(np.sort(truedata[:,1]), np.linspace(0, 1, 10000, endpoint=False), label = 'True data', linestyle='-.')
    ax[1].set_title('Dimension 2', fontweight='bold')
    ax[1].legend(fontsize='large')

    sns.despine(offset=20)

    plt.show()