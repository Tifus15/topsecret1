import torch

dataset = torch.rand(100,3,1,3)*100


def minmax(dataset):
    T = dataset.shape[0]
    B = dataset.shape[1]
    maxim=torch.max(torch.max(dataset, dim=0)[0],dim=0)[0]
    minim=torch.min(torch.min(dataset, dim=0)[0],dim=0)[0]
    print(maxim.shape)
    return (dataset - minim)/(maxim-minim), maxim, minim
"""    
out,max_key,min_key = minmax(dataset)
print(out.shape)

def inv_minmax(dataset,min_key,max_key):
    return (dataset*(max_key-min_key))+min_key

inv = inv_minmax(out,min_key,max_key)
print(torch.mean(dataset-inv)) 
"""    
    
