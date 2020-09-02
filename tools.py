import math 
import torch 
import signatory 


def get_length(signature,dimension,order):
    
    '''This function approximates the length of the path through the signature.
    
        Arguments are : 
            - signature : the full signature. 
            - order : the order at which the signature was truncated. 
    
    '''
    
    last_signature_term = signatory.extract_signature_term(signature,dimension,int(order))
    
    return torch.norm(math.factorial(order)*last_signature_term,2)**(1/order)
    