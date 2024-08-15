import torch


def dataset_normalisation(x,region=[-1,1]):
    x = x.flatten()
    x_min = torch.min(x)
    x_max = torch.max(x)
    
    output = (x -x_min)*(region[1]-region[0])/(x_max-x_min)
    return output, x_min, x_max

def dataset_renormalisation(x,region=[-1,1],x_min,x_max):
    x*(x_max-x_min)
     

    
    