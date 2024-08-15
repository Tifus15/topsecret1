import numpy as np
from tqdm import tqdm
import dgl
import wandb
import torch
import matplotlib.pyplot as plt

def angle_transformer(data):
    x = torch.cos(data)
    y = torch.sin(data)
    out = torch.atan2(y,x)
    return out
    
    
def check_angle_transformer():
    x = -100 + torch.rand(128,100,2)*200
    test = torch.cos(x)
    y = angle_transformer(x)
    print(torch.max(y.flatten()))# pi
    print(torch.min(y.flatten()))# -pi
    y_test = torch.cos(y)
    print(torch.mean(torch.abs(y_test-test).flatten())) # minimal mistake
    

def load_model(model, loc_w):
     # we do not specify ``weights``, i.e. create untrained model
    model.load_state_dict(torch.load(loc_w))
    return model

def dst_list(nodes):

    base = list(np.arange(nodes))
    out=[]
    for i in range(nodes):
        out = out + base
    return out

def src_list(nodes):
    out=[]
    for i in range(nodes):
        out = out +list(np.zeros((nodes),dtype=int)+i)
    return out
"""
configs = {"model": "GCN",
           "sob": True,
           "sob_a": 0.01,
           "a" : 0.01,
           "opti" : "adamW",
           "reg" : "none",
           "loss": "Huber",
           "acts" : ["tanh",""],
           "epochs": 100,
           "modelsize" : 128,
           "batchsize" : 32,
           "timesize" : 32,
           "lr": 5e-4,
           "split": 0.9,
           "device":"cuda",
           "single":True}
"""
def create_pend1dof_graph_snapshots(snaps,src,dst):
    graphs=[]
    for snap in snaps:
        #print(snap.shape) 
        g = dgl.graph((src,dst))
        g.ndata["xfeat"] = snap[:,:,0:2].transpose(0,1) 
        g.ndata["dxfeat"] = snap[:,:,2:4].transpose(0,1)
        g.ndata["hfeat"] =  snap[:,:,-1].transpose(0,1)
        graphs.append(g)
    return graphs
def correct_ham_data(g):
    gs = dgl.unbatch(g)
    l = []
    for s in gs:
        temp = g.ndata["hfeat"] 
        l.append(temp[0,:,:])
    return torch.cat((l),dim=-1)
def create_pend2dof_graph_snapshots(snaps,src,dst):
    graphs=[]
    nodes = 2
    for snap in snaps:
        #print(snap.shape)
        tempx=torch.zeros(2,snap.shape[0],2) 
        tempx[0,:,0] = snap[:,0,0]
        tempx[1,:,0] = snap[:,0,1]
        tempx[0,:,1] = snap[:,0,2]
        tempx[1,:,1] = snap[:,0,3]
        tempdx=torch.zeros(2,snap.shape[0],2) 
        tempdx[0,:,0] = snap[:,0,4]
        tempdx[1,:,0] = snap[:,0,5]
        tempdx[0,:,1] = snap[:,0,6]
        tempdx[1,:,1] = snap[:,0,7]
        tempH=torch.zeros(2,snap.shape[0],1)
        tempH[0,:,0] = snap[:,0,8]
        tempH[1,:,0] = snap[:,0,8]
        
        g = dgl.graph((src,dst))
        g.ndata["xfeat"] = tempx
        g.ndata["dxfeat"] = tempdx
        g.ndata["hfeat"] =  tempH
        graphs.append(g)
    return graphs
       
def create_pend3dof_graph_snapshots(snaps,src,dst):
    graphs=[]
    nodes = 3
    for snap in snaps:
        #print(snap.shape)
        tempx=torch.zeros(3,snap.shape[0],2) 
        tempx[0,:,0] = snap[:,0,0]
        tempx[1,:,0] = snap[:,0,1]
        tempx[2,:,0] = snap[:,0,2]
        tempx[0,:,1] = snap[:,0,3]
        tempx[1,:,1] = snap[:,0,4]
        tempx[2,:,1] = snap[:,0,5]
        tempdx=torch.zeros(3,snap.shape[0],2) 
        tempdx[0,:,0] = snap[:,0,6]
        tempdx[1,:,0] = snap[:,0,7]
        tempdx[2,:,0] = snap[:,0,8]
        tempdx[0,:,1] = snap[:,0,9]
        tempdx[1,:,1] = snap[:,0,10]
        tempdx[2,:,1] = snap[:,0,11]
        tempH=torch.zeros(3,snap.shape[0],1)
        tempH[0,:,0] = snap[:,0,12]
        tempH[1,:,0] = snap[:,0,12]
        tempH[2,:,0] = snap[:,0,12]
        
        g = dgl.graph((src,dst))
        g.ndata["xfeat"] = tempx
        g.ndata["dxfeat"] = tempdx
        g.ndata["hfeat"] =  tempH
        graphs.append(g)
    return graphs 

def create_pend4dof_graph_snapshots(snaps,src,dst):
    graphs=[]
    nodes = 4
    for snap in snaps:
        #print(snap.shape)
        tempx=torch.zeros(4,snap.shape[0],2) 
        tempx[0,:,0] = snap[:,0,0]
        tempx[1,:,0] = snap[:,0,1]
        tempx[2,:,0] = snap[:,0,2]
        tempx[3,:,0] = snap[:,0,3]
        tempx[0,:,1] = snap[:,0,4]
        tempx[1,:,1] = snap[:,0,5]
        tempx[2,:,1] = snap[:,0,6]
        tempx[3,:,1] = snap[:,0,7]
        tempdx=torch.zeros(4,snap.shape[0],2) 
        tempdx[0,:,0] = snap[:,0,8]
        tempdx[1,:,0] = snap[:,0,9]
        tempdx[2,:,0] = snap[:,0,10]
        tempdx[3,:,0] = snap[:,0,11]
        tempdx[0,:,1] = snap[:,0,12]
        tempdx[1,:,1] = snap[:,0,13]
        tempdx[2,:,1] = snap[:,0,14]
        tempdx[3,:,1] = snap[:,0,15]
        tempH=torch.zeros(4,snap.shape[0],1)
        tempH[0,:,0] = snap[:,0,16]
        tempH[1,:,0] = snap[:,0,16]
        tempH[2,:,0] = snap[:,0,16]
        tempH[3,:,0] = snap[:,0,16]
        
        g = dgl.graph((src,dst))
        g.ndata["xfeat"] = tempx
        g.ndata["dxfeat"] = tempdx
        g.ndata["hfeat"] =  tempH
        graphs.append(g)
    return graphs
        
        
def convert2dgl_snapshots(g_snaps,src,dst):
    graphs=[]
    whole = g_snaps[0].shape[-1]-1
    print("converting")
    for snap in tqdm(g_snaps):
        
        temp =dgl.graph((src,dst))
        temp.ndata["x"]= snap[:,:,0:int(whole/2)].transpose(0,1)
        temp.ndata["dx"] = snap[:,:,int(whole/2):whole].transpose(0,1)
        temp.ndata["H"] = snap[:,:,-1].transpose(0,1)
        graphs.append(temp)
    return graphs

def make_snapshots(data, TIME_SIZE):
    time_size =TIME_SIZE
    points = data.shape[0]
    print("{} snapshots".format((points-time_size)*data.shape[1]))
    list = []
    
    
    for i in range(data.shape[1]):
        for j in range(points-time_size):
            list.append(data[j:j+time_size,i,:,:])
                
    return list

def visualize_loss(title, loss_container):
    if loss_container.shape[0]==2:
        t = torch.linspace(0,loss_container.shape[1],loss_container.shape[1])
        fig = plt.figure()
        plt.title("{} logy".format(title))
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.semilogy(t,loss_container[0,:].detgach().numpy(),c="r")
        plt.semilogy(t,loss_container[1,:].detach().numpy(),c="b")
        plt.legend(["train","test"])
        fig.savefig("loss_"+title+".png")
    else:
        t = torch.linspace(0,loss_container.shape[1],loss_container.shape[1])
        fig,ax = plt.subplots(1,3)
        ax[0].set_title("{} loss".format(title))
        ax[0].set_xlabel("epochs")
        ax[0].set_ylabel("loss")
        ax[0].semilogy(t,loss_container[0,:].detach().numpy(),c="r")
        ax[0].semilogy(t,loss_container[3,:].detach().numpy(),c="b")
        ax[0].legend(["train","test"])
        ax[1].set_title("{} gradient loss".format(title))
        ax[1].set_xlabel("epochs")
        ax[1].set_ylabel("loss")
        ax[1].semilogy(t,loss_container[1,:].detach().numpy(),c="r")
        ax[1].semilogy(t,loss_container[4,:].detach().numpy(),c="b")
        ax[1].legend(["train grad","test grad"]) 
        ax[2].set_title("{} hamiltonian loss".format(title))
        ax[2].set_xlabel("epochs")
        ax[2].set_ylabel("loss")
        ax[2].semilogy(t,loss_container[2,:].detach().numpy(),c="r")
        ax[2].semilogy(t,loss_container[5,:].detach().numpy(),c="b")
        ax[2].legend(["train ham","test ham"]) 
        fig.savefig("loss_"+title+".png")
    

#check_angle_transformer()
