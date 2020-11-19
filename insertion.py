import torch
import signatory



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

    A=(new_B*sig_new_tensor).flatten(start_dim=1)

    return A.T

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

    A_matrix = get_A_matrix(signature[:,:-d**(n)],p,n-1,d)
    b_vector = n*signatory.extract_signature_term(signature,d,n)

    b_vector = b_vector.flatten()

    x_optimal=(A_matrix.T@b_vector)

    sign_1 = signatory.extract_signature_term(signature, d, n - 1)

    return x_optimal/(torch.norm(sign_1)**2)


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
    n=8
    d=5
    p=2
    
    test_path = torch.rand((1,100,d))
    signature_test = signatory.signature(test_path,n)
    print(signature_test.shape)
    invert_signature(signature_test,n,d)

