import numpy as np
import torch
import dgl
def dst_list(nodes,start=0):

    base = list(np.arange(start,start+nodes))
    out=[]
    for i in range(nodes):
        out = out + base
    return out

def src_list(nodes,start=0):
    out=[]
    for i in range(nodes):
        out = out +list(np.zeros((nodes),dtype=int)+start+i)
    return out

def del_loops(src,dst):
    for i, (val1,val2) in enumerate(zip(src,dst)):
        
        if val1 == val2:
            print("{} == {}".format(val1,val2))
            print("{} popped".format(i))
            src.pop(i)
            dst.pop(i)
    return src, dst

def make_graph_no_loops(nodes,start):
    src = src_list(nodes,start)
    dst = dst_list(nodes,start)
    return del_loops(src,dst)


def load_dataset(file,hfile):
    data = torch.load(file)
    H = torch.load(hfile)
    return data, H
def make_simple_snapshots(data,H):
    xlist=[]
    xnext_list=[]
    Hlist=[]
    N = data.shape[1]
    print(N)
    T = data.shape[0]
    for i in range(N):
        for j in range(T-1):
            temp = data[j,i,:,:]
            tempH = H[j,i,:]
            tempx = data[j+1,i,:,:]
            #print("simp x:{}".format(temp.shape))
            #print("simp nx:{}".format(tempx.shape))
            xlist.append(temp)
            Hlist.append(tempH)
            xnext_list.append(tempx)
    return xlist, Hlist

def transform_simple_dgl(src,dst,snaps,hs):
    gs = []
    for snap,h in zip(snaps,hs):
        #print("nx:{}".format(nx.shape))
        #print("x:{}".format(snap.shape))
        g = dgl.graph((src,dst))
        g.ndata["x"] = snap[:,0:6]
        g.ndata["dx"] = snap[:,6:]
        g.ndata["H"] = h[:,0:1]
        gs.append(g)
    return gs

def make_snapshots(data,H,timesize):
    xlist=[]
    Hlist=[]
    
    
    N = data.shape[1]
    print(N)
    T = data.shape[0]
    for i in range(N):
        for j in range(T-timesize-1):
            temp = data[j:timesize+j,i,:,:]
           #print("temp {}".format(temp.shape))
            #print(i)
            tempH = H[j:timesize+j,i,:,:]
            xlist.append(temp)
            Hlist.append(tempH)
    return xlist, Hlist

def transform_dgl(src,dst,snaps,hs):
    gs = []
    for snap,h in zip(snaps,hs):
        g = dgl.graph((src,dst))
        g.ndata["x"] = snap[:,:,0:6].transpose(0,1)
        g.ndata["dx"] = snap[:,:,6:].transpose(0,1)
        g.ndata["H"] = h[:,:,0:1].transpose(0,1)
        gs.append(g)
    return gs
def get_d_dx_H(sample):
    gs = dgl.unbatch(sample)
    H = []
    for g in gs:
        h_raw = g.ndata["H"].transpose(0,1)
        H.append(h_raw[:,0,0:1])
    H_out = torch.cat((H),dim=-1)
    x_out = sample.ndata["x"].transpose(0,1)
    dx_out = sample.ndata["dx"].transpose(0,1)
    
    return x_out, dx_out, H_out

def get_simple_elements(sample):
    gs = dgl.unbatch(sample)
    H = []
    for g in gs:
        h_raw = g.ndata["H"].transpose(0,1)
        H.append(h_raw[0:1,0:1])
    H_out = torch.cat((H),dim=-1)
    x_out = sample.ndata["x"]
    dx_out = sample.ndata["dx"]
   
    
    return x_out, dx_out, H_out
    
    

def dataset_loader(first=["4_x.pt","4_dx.pt","4_h.pt"],second=["5_x.pt","5_dx.pt","5_h.pt"]):
    x1 = torch.load(first[0])
    dx1= torch.load(first[1])
    H1 = torch.load(first[2])
    x2 = torch.load(second[0])
    dx2= torch.load(second[1])
    H2 = torch.load(second[2])
    
    
    dxx1 = torch.cat((x1,dx1),dim=-1)
    dxx2 = torch.cat((x2,dx2),dim=-1)
    
    return dxx1, H1 ,dxx2, H2

def minmax(dataset):
    T = dataset.shape[0]
    B = dataset.shape[1]
    maxim=torch.max(dataset.flatten())
    minim=torch.min(dataset.flatten())
    #print(maxim.shape)
    return (dataset - minim)/(maxim-minim), maxim, minim


def inv_minmax(dataset,min_key,max_key):
    return (dataset*(max_key-min_key))+min_key


def minimax_test(dataset):
    d , maxim, minim = minmax(dataset)
    rec = inv_minmax(d,minim,maxim)
    return torch.mean(rec.flatten() - dataset.flatten())


    


