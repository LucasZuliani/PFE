import torch

def RMSE(v1:torch.FloatTensor, v2:torch.FloatTensor)->torch.FloatTensor:
    """
    Compute the Root Mean Squared Error between two tensors.
    
    Args:
        v1 : First tensor
        v2 : Second tensor
        
    Returns:
        Root Mean Squared Error

    >>> RMSE(torch.FloatTensor([1,2,3]), torch.FloatTensor([1,2,3]))
    tensor(0.)
    >>> RMSE(torch.FloatTensor([1,2,3]), torch.FloatTensor([1,2,3+torch.sqrt(torch.tensor(3.))]))
    tensor(1.)
    """
    return torch.sqrt(torch.mean((v1-v2)**2))