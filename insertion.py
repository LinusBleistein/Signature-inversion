import numpy as np
import math
import torch

import signatory
import itertools


def get_length(signature,n,d): 
    '''This function approximates the length of the path through the signature.
    
    Arguments
    ---------
        signature : torch.Tensor, shape (1,(d^(n+1)-d^n)/(d-1),)
            Signature of a path truncated at order n. Its length is the sum from k=1 to n of d^k.

        n : int
            Depth of the signature.
        
        d : int
            Dimension of the underlying path.
    
    '''
    
    last_signature_term = signatory.extract_signature_term(signature,d,int(n))
    
    return torch.norm(math.factorial(n)*last_signature_term,2)**(1/n)



def Insertion(signature,x,p,n,d):
    '''This function computes the Insertion operator taken at x, for parameters p and n.

    Arguments
    ---------
        signature : torch.Tensor, shape (1,(d^(n+1)-d^n)/(d-1),)
            Signature of a path truncated at order n. Its length is the sum from k=1 to n of d^k.


        x : array, shape (d,)
            Vector in R^d at which the operator is evaluated.
        
        p : int
            Insertion spot.

        n : int
            Depth of the signature.
        
        d : int
            Dimension of the underlying path.

    Returns
    -------
        new_tensor:torch.Tensor, shape (d,...,d) (where d is repeated n+1 times)
            A (n+1)-dimensional tensor in R^d corresponding to the insertion of x in the order n signature.
    '''

    #Get the last (n-th) term of the signature as a tensor of size (R^d)^{\otimes n}

    last_signature_term = signatory.extract_signature_term(signature,d,n)
    last_signature_term=last_signature_term.view([d]*int(n))

    #Add a dimension, since the output of the operator should be in (R^d)^{\otimes (n+1)}
    new_tensor=torch.empty([d]*(n+1))

    #Creates a list containing all possible coordinates of the big tensor.
    #This list is of length d**n.

    coordinates_list = list(itertools.product(torch.arange(d),repeat= n+1))

    #Every element of coordinates_list is a coordinate of the new tensor, which has d^(n+1) coordinates.

    #Computes the Insertion operator.
    for coordinate in coordinates_list:
        coordinate = list(coordinate)

        new_coordinate = coordinate.copy()
        del new_coordinate[p-1]

        new_tensor[tuple(coordinate)] = last_signature_term[tuple(new_coordinate)]*x[coordinate[p-1]]

    return math.factorial(n)*new_tensor

def get_A_matrix(signature,p,n,d):
    '''This function creates the matrix lenght_of_path*A, used in the optimization problem.

     Arguments
    ---------
        signature : torch.Tensor, shape (1,(d^(n+1)-d^n)/(d-1),)
            Signature of a path truncated at order n. Its length is the sum from k=1 to n of d^k.
        
        p : int
            Insertion spot.

        n : int
            Depth of the signature.
        
        d : int
            Dimension of the underlying path.

    Returns
    -------
        A: torch.tensor, shape (d^(n+1),d)
            Matrix A representing the linear insertion map.
    '''

    #Create basis of R^d

    basis = torch.diag(torch.ones(d))

    #Evaluate the insertion operator on the basis

    A = torch.zeros([d**(n + 1),d])
    for row in np.arange(d):
        base_vector = basis[row,:]
        insertion_tensor = Insertion(signature,base_vector,p,n,d)
        A[:,row]=insertion_tensor.flatten()

    A = A.reshape(d**(n+1),d)

    length = get_length(signature,n,d)
    return length*A

def solve_optimization_problem(signature,p,n,d):

    '''This function solves the optimization problem that allows to approximate the derivatives of the path.
        of the path.

    Arguments
    ---------
        signature : torch.Tensor, shape (1,(d^(n+1)-d^n)/(d-1),)
            Signature of a path truncated at order n. Its length is the sum from k=1 to n of d^k.
        
        p : int
            Insertion spot.

        n : int
            Depth of the signature.
        
        d : int
            Dimension of the underlying path.

    Returns
    -------
        x_optimal: array, shape (d,)
            Solution of the optimization problem, approximation of the derivative of the path on the pth time interval.
    '''

    #Create A matrix and b vector used in the optimization problem.
    A_matrix = np.array(get_A_matrix(signature[:,:-d**(n)],p,n-1,d))

    b_vector = math.factorial(n+1)*np.array(signatory.extract_signature_term(signature,d,n))

    b_vector = b_vector.flatten()

    #SVD
    U,Sigma,V = np.linalg.svd(A_matrix,full_matrices=True)

    Y = (U.T)@b_vector
    #Only take the d-first values of Y

    Y = Y[0:d]

    #Compute optimal x

    x = (1/np.linalg.norm(Y))*Y

    x_optimal = V@x

    return x_optimal


def invert_signature(signature,n,d,first_point=None):
    """Recontruct the path from its signature

    Arguments
    ---------
        signature : torch.Tensor, shape (1,(d^(n+1)-d^n)/(d-1),)
            Signature of a path truncated at order n. Its length is the sum from k=1 to n of d^k.

        n : int
            Depth of the signature.
        
        d : int
            Dimension of the underlying path.

        first_point: optional, array, shape (d,)
            Initial point of the path. If None, the reconstructed path is set to begin at zero.

    Returns
    -------
        reconstructed_path: array, shape (n+1,d,)
            The inverted path. 
    """

    reconstructed_path_derivatives = np.zeros((n,d))

    reconstructed_path = np.zeros((n+1,d))
    
    if first_point is not None:
        reconstructed_path[0,:]=first_point

    for p in np.arange(1,n+1):

        x_optimal = solve_optimization_problem(signature,p,n,d)

        reconstructed_path_derivatives[p-1,:] = x_optimal

        reconstructed_path[p,:] = reconstructed_path[p-1,:] + reconstructed_path_derivatives[p-1,:]*(1/n)

    return(reconstructed_path)


if __name__ == '__main__':
    n=2
    d=5
    test_path = torch.rand((1,10,d))
    signature_test = signatory.signature(test_path,n)
    print(signature_test.shape)
    x=[1,2,3,4,5]
    p=3
    #A_matrix=get_A_matrix(signature_test,p,n,d)
    #signature = signatory.signature(test_path, 2)
    signature_next = signatory.signature(test_path,n+1)

    solve_optimization_problem(signature_next,p,n+1,d)
    invert_signature(signature_next,n+1,d,first_point=torch.zeros(d))




