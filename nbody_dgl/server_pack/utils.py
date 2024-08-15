import numpy as np
from tqdm import tqdm
import dgl
import wandb
import torch
import matplotlib.pyplot as plt

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
        plt.show()
    else:
        t = torch.linspace(0,loss_container.shape[1],loss_container.shape[1])
        fig,ax = plt.subplots(1,2)
        ax[0].set_title("{} loss".format(title))
        ax[0].set_xlabel("epochs")
        ax[0].set_ylabel("loss")
        ax[0].semilogy(t,loss_container[0,:].detach().numpy(),c="r")
        ax[0].semilogy(t,loss_container[2,:].detach().numpy(),c="b")
        ax[0].legend(["train","test"])
        ax[1].set_title("{} gradient loss".format(title))
        ax[1].set_xlabel("epochs")
        ax[1].set_ylabel("loss")
        ax[1].semilogy(t,loss_container[1,:].detach().numpy(),c="r")
        ax[1].semilogy(t,loss_container[3,:].detach().numpy(),c="b")
        ax[1].legend(["train grad","test grad"]) 
    plt.show()

