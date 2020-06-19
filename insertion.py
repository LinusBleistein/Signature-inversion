import numpy as np
import math 
import torch 
import seaborn as sns

import signatory
import itertools



from tools import get_signature_as_tensor,get_length

def Insertion(p,n,x,signature,dimension):
    
    '''This function computes the Insertion operator taken at x, for parameters p and n.
    
    Arguments : 
    
        - x : vector in R^dimension at which the operator should be evaluated.
        - p : insertion spot. 
        - n : total depth of the signature.
        - signature : total signature of a path
        - dimension : dimension of the path.
    '''
    
    #Get the last (n-th) term of the signature as a tensor of size (R^dimension)^{\otimes n}
    
    last_signature_term = get_signature_as_tensor(signature,dimension,n)["Depth "+ str(n)]
    
    #Add a dimension, since the output of the operator should be in (R^d)^{\otimes (n+1)}
    
    new_tensor = new_tensor=torch.empty([dimension]*(n+1))
    
    #Creates a list containing all possible coordinates of the big tensor.
    #This list is of length dimension**n. 
    
    tensor_size = np.arange(dimension)
    
    coordinates_list = list(itertools.product(tensor_size,repeat= n+1))
    
    
    #Every element of coordinates_list is a coordinate of the new tensor, which has d^(n+1) coordinates.
    
    #Computes the Insertion operator.
    
    for coordinate in coordinates_list:
        
        coordinate = list(coordinate)
                
        x_coordinate = x[coordinate[p-1]]
        
        new_coordinate = coordinate.copy()
        
        del new_coordinate[p-1]
        
        new_tensor[tuple(coordinate)] = last_signature_term[tuple(new_coordinate)]*x_coordinate
        
        
    return math.factorial(n)*new_tensor

def get_A_matrix(p,signature,order,dimension):
    
    '''This function creates the matrix lenght_of_path*A, used in the optimization problem.
    
        Arguments: 
            - p : insertion spot.
            - signature : full original signature.
            - order : order of truncation of signature. 
            - dimension : dimension of the path. 
    
    '''
    
    #Create basis of R^d
    
    basis = np.eye(dimension,dimension)
    
    #Evaluate the insertion operator on the basis
    
    total_tensor = np.zeros(shape=(dimension**(order + 1),dimension))
    for row in np.arange(dimension):
        base_vector = basis[row,:]
        insertion_tensor = Insertion(p,order,base_vector,signature,dimension)
        #insertion_tensor = insertion_tensor.reshape(-1,dimension)

        insertion_tensor = insertion_tensor.flatten()  
        #total_tensor[row*(dimension**(order)):(row+1)*(dimension**(order)),:] = insertion_tensor
        total_tensor[:,row]=insertion_tensor
    
    total_tensor = total_tensor.reshape(dimension**(order+1),dimension)
    
    length = get_length(signature,dimension,order)
    #print(length)
    #return length*total_tensor
    return(total_tensor)
    
def solve_optimization_problem(signature,signature_next_step,p,n,dimension):
    
    '''This function solves the optimization problem that allows to approximate the signature 
        of the path. 
        
        Arguments:
            - n : order at which the signature is taken.
            - signature : full signature at order n.
            - signature_next_step : full signature at order n+1 (!).
            - p : insertion spot, determinates the time interval on which the derivative is approximated. 
            - dimension : dimension of the path.
    '''
    
    #Create A matrix and b vector used in the optimization problem. 
    
    A_matrix = np.array(get_A_matrix(p,signature,n,dimension))
    
    b_vector = math.factorial(n+1)*np.array(signatory.extract_signature_term(signature_next_step,dimension,int(n+1)))
    
    b_vector = b_vector.flatten()
    
    #SVD 
    
    U,Sigma,V = np.linalg.svd(A_matrix,full_matrices=True)
    
    Y = (U.T)@b_vector 
    #Only take the d-first values of Y
    
    Y = Y[0:dimension]
    
    #Compute optimal x
    
    x = (1/np.linalg.norm(Y))*Y
    
    x_optimal = V@x
    
    return x_optimal  




