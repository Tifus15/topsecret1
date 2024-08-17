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
            #print("{} == {}".format(val1,val2))
            #print("{} popped".format(i))
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
    
    

def dataset_loader(first=["nbody_12_traj.pt","nbody_12_H.pt"],second=["nbody_13_traj.pt","nbody_13_H.pt"]):
    x1 = torch.load(first[0])
    H1 = torch.load(first[1])
    x2 = torch.load(second[0])
    H2 = torch.load(second[1])
    
    
    #dxx1 = torch.cat((x1,dx1),dim=-1)
    #dxx2 = torch.cat((x2,dx2),dim=-1)
    
    return x1, H1 ,x2, H2

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

def loss_reader(str):
    if str == "MSE":
        return torch.nn.MSELoss()
    elif str == "HUB":
        return torch.nn.HuberLoss()
    else:
        return torch.nn.MSELoss()
    

def RKroll_for_learning(model,x0,t):
    def evaluate_model(model,x):
        h_pred = model(x)
        H_l = torch.split(h_pred,1,dim=1)
        dHdx = torch.autograd.grad(H_l,x,retain_graph=True, create_graph=True)[0] 
        dqdp_s = torch.split(dHdx,3,dim=-1)
        dx_pred = torch.cat((dqdp_s[1],-dqdp_s[0]),dim=-1)
        return dx_pred
    out_l = []
    out_l.append(x0.unsqueeze(0))
    #print(out_l[0].shape)
    for i in range(1,len(t)):
        dt=t[i]-t[i-1]
        K1 = evaluate_model(model,out_l[i-1].squeeze())
        K2 = evaluate_model(model,out_l[i-1].squeeze()+dt*K1/2)
        K3 = evaluate_model(model,out_l[i-1].squeeze()+dt*K2/2)
        K4 = evaluate_model(model,out_l[i-1].squeeze()+dt*K3)
        rk4=out_l[i-1].squeeze()+dt*(K1+2*K2+2*K3+K4)/6
        out_l.append(rk4.unsqueeze(0))
        #print(out_l[i].shape)
    
    return torch.cat((out_l),dim=0)


def Euler_for_learning(model,x0,t):
    def evaluate_model(model,x):
        h_pred = model(x)
        H_l = torch.split(h_pred,1,dim=1)
        dHdx = torch.autograd.grad(H_l,x,retain_graph=True, create_graph=True)[0] 
        dqdp_s = torch.split(dHdx,3,dim=-1)
        dx_pred = torch.cat((dqdp_s[1],-dqdp_s[0]),dim=-1)
        return dx_pred
    out_l = []
    out_l.append(x0.unsqueeze(0))
    #print(out_l[0].shape)
    for i in range(1,len(t)):
        dt=t[i]-t[i-1]
        K1 = evaluate_model(model,out_l[i-1].squeeze())
        rk4=out_l[i-1].squeeze()+dt*K1
        out_l.append(rk4.unsqueeze(0))
        #print(out_l[i].shape)
    
    return torch.cat((out_l),dim=0)


    


