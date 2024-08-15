import torch
import random
from torch.utils.data import Dataset


def edges_to_A(src,dst, no_self_loops=False):
    src_nodes = max(src)+1
    dst_nodes = max(dst)+1
    nodes = max(src_nodes,dst_nodes)
    if len(src) != len(dst):
        print("wrong sizes")
        return torch.zeros(nodes,nodes)
    else:
        A = torch.zeros(nodes,nodes)
        for in_n, out_n in zip(src,dst):
            if in_n == out_n and no_self_loops:
                continue
            else:
                A[in_n,out_n] = 1
        return A
    
def graph_pack(adj,x,dx,H):
    data_bib=hamDataset()
    keys = ["x","dx","H","nx","A"]
    data_bib.dataset_init(keys)
    
    
    for i in range(x.shape[0]-1):
        for j in range(x.shape[1]):
            data_bib.ord["x"].append(x[i,j,:,:])
            data_bib.ord["dx"].append(dx[i,j,:,:])
            data_bib.ord["H"].append(H[i,j].reshape(-1))
            data_bib.ord["nx"].append(x[i+1,j,:,:])
            data_bib.ord["A"].append(adj)
    return data_bib
"""
class Data(object):
    def __init__(self):
        self.data_list=[]#dicts
        self.keys = []
    def append_data_dict(self,dict):
        
        self.data_list.append(dict)
        if len(self.keys)==0:
            self.get_keys()
    def shuffle(self):
        random.shuffle(self.data_list)
    def get_all(self):
        out = {}
        for key in self.keys:
            #print(key)
            #print(self.keys)
            #print(len(self.keys))
            
            out[key]=self.get(key)
        return out
    
    def get(self,str):
        l=[]
        for i in range(len(self.data_list)):
            l.append(self.data_list[i][str])
        return l  
    def get_keys(self):
        self.keys = list(self.data_list[0].keys())
        #print(self.keys) 
        
    def print_keys(self):
        print(self.keys) 
            
"""
class hamDataset(Dataset):
    def __init__(self):    
        self.keys = []
        self.ord={}
        
    def print_keys(self):
        print(self.keys)  
    def dataset_init(self,keys):
        for key in keys:
            self.ord[key] = []   
        self.keys = keys
    def __len__(self):
        return len(self.ord["x"])   
    
    def __getitem__(self, idx):
        return self.ord["x"][idx], self.ord["dx"][idx],self.ord["H"][idx],self.ord["nx"][idx],self.ord["A"][idx]
            