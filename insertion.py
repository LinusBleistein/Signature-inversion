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
    last_signature_term=last_signature_term.view([d]*int(n)).unsqueeze_(p-1)

    sig_new_tensor=last_signature_term.expand([d]*(n+1))

    new_shape=[1]*(p-1)+[d]+[1]*(n+1-p)
    repeat_points=[d]*(p-1)+[1]+[d]*(n+1-p)

    x_new_tensor=torch.tensor(x).view(new_shape)
    x_new_tensor=x_new_tensor.repeat(repeat_points)


    return math.factorial(n)*sig_new_tensor*x_new_tensor


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

    basis = torch.eye(d)

    #Evaluate the insertion operator on the basis

    A = torch.zeros([d**(n + 1),d])
    for row in torch.arange(d):
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
    time1=time.time()

    A_matrix = get_A_matrix(signature[:,:-d**(n)],p,n-1,d)

    time2=time.time()
    #print("Get A matrix____",time1-time2)

    b_vector = math.factorial(n)*signatory.extract_signature_term(signature,d,n)

    b_vector = b_vector.flatten()

    #SVD
    device = A_matrix.device
    A_matrix = A_matrix.cpu()

    U,Sigma,V = torch.svd(A_matrix,some=True)

    U = U.to(device)
    Sigma = Sigma.to(device)
    V = V.to(device)

    time3=time.time()
    #print('Solve SVD_____',time3-time2)

    Y = (U.T)@b_vector.float()
    #Only take the d-first values of Y
    Y = Y[0:d]

    #Compute optimal x

    x = (1/torch.norm(Y))*Y

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

    reconstructed_path_derivatives = torch.zeros((n,d))

    reconstructed_path = torch.zeros((n+1,d))
    
    if first_point is not None:
        reconstructed_path[0,:]=first_point

    for p in torch.arange(1,n+1):
        x_optimal = solve_optimization_problem(signature,p,n,d)

        reconstructed_path_derivatives[p-1,:] = torch.tensor(x_optimal)

        reconstructed_path[p,:] = reconstructed_path[p-1,:] + reconstructed_path_derivatives[p-1,:]*(1/(n))

    return(reconstructed_path)


if __name__ == '__main__':
    n=11
    d=3
    p=3
    
    test_path = torch.rand((1,100,d))
    signature_test = signatory.signature(test_path,n)
    print(signature_test.shape)
    invert_signature(signature_test,n,d,first_point=torch.zeros(d))




