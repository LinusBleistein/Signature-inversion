import math
import torch

import signatory
import itertools
import time


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
    factorial_n=torch.tensor(n+1).float().lgamma().exp()
    return torch.norm(factorial_n*last_signature_term,2)**(1/n)


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

    B=torch.eye(d)
    new_shape=[d]+[1]*(p-1)+[d]+[1]*(n+1-p)
    repeat_points=[1]+[d]*(p-1)+[1]+[d]*(n+1-p)
    new_B=B.view(new_shape)
    new_B=new_B.repeat(repeat_points)

    last_signature_term = signatory.extract_signature_term(signature,d,n)
    last_signature_term=last_signature_term.view([d]*int(n)).unsqueeze_(p-1)
    sig_new_tensor=last_signature_term.expand([d]*(n+2))
    
    factorial_n=torch.tensor(n+1).float().lgamma().exp()
    A=(factorial_n*new_B*sig_new_tensor).flatten(start_dim=1)

    
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

    A_matrix = get_A_matrix(signature[:,:-d**(n)],p,n-1,d)
    factorial_n=torch.tensor(n+1).float().lgamma().exp()
    b_vector =factorial_n*signatory.extract_signature_term(signature,d,n)

    b_vector =b_vector.flatten()


    # QR decomposition
    Q,R=torch.qr(A_matrix.T.double(),some=True)

    z=(R.T)@(Q.T)@b_vector
    x_optimal=z/torch.norm(z)

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

    reconstructed_path_derivatives = torch.zeros((n,d))

    reconstructed_path = torch.zeros((n+1,d))
    
    if first_point is not None:
        reconstructed_path[0,:]=first_point

    for p in torch.arange(1,n+1):
        x_optimal = solve_optimization_problem(signature,p,n,d)

        reconstructed_path_derivatives[p-1,:] = x_optimal

        reconstructed_path[p,:] = reconstructed_path[p-1,:] + reconstructed_path_derivatives[p-1,:]*(1/(n))

    return(reconstructed_path)


if __name__ == '__main__':
    n=10
    d=5
    p=2
    
    test_path = torch.rand((1,100,d))
    signature_test = signatory.signature(test_path,n)
    print(signature_test.shape)
    invert_signature(signature_test,n,d)

