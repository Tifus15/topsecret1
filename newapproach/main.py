import torch
import torch.nn as nn
from nbody_loader import load_Nbody_data
from device_util import ROOT_PATH
import os
import sys
import dgl
from model_cart import HNN,rollout_HNN
from dataset import create_data_Nbody
import random
import torch.optim as optim 
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm


filename = "nbody_10_05222024103501.npy"

import dgl.data
import matplotlib.pyplot as plt
import networkx as nx

def visualize_loss(cont):
    fig = plt.figure()
    plt.semilogy(list(range(1,cont.shape[0]+1)),cont[:,0].cpu())
    plt.semilogy(list(range(1,cont.shape[0]+1)),cont[:,1].cpu())
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(["train","test"])
    plt.show()


#50 samples got me 4h of waiting time- ALWAYS SAVE THE SAMPLES!!!!!
######################################################## 
if not os.path.isfile(ROOT_PATH + "/"+ filename):
    print("file doesn't exist here: {}".format(ROOT_PATH + "/"+ filename))
    sys.exit()
else:
    pos,vel,t,mass,K,P = load_Nbody_data(filename)
########################################################
### make torch
xyz = torch.Tensor(pos)
vxyz = torch.Tensor(vel)
t = torch.Tensor(t)
## mass is 1 for every body means v = p
qp = torch.cat((xyz,vxyz),dim=-1)[:-1,:,:]# without last for better formating
print(qp.shape) # allready as a graph data

N= qp.shape[1]
bodies = list(range(N))
print(bodies)
src=[]
dst=[]
for i in bodies:
    for j in range(0,N):
        src.append(i)
        dst.append(j)
loop_graph = dgl.graph((bodies,bodies))
no_loop_graph = dgl.graph((src,dst))
no_loop_graph = dgl.remove_self_loop(no_loop_graph)

model =HNN(loop_graph,no_loop_graph,6,100)
t_batch = 20
dataset = create_data_Nbody(qp.unsqueeze(1),20)
print(len(dataset))
random.shuffle(dataset)
split=0.9
border = int(len(dataset)*0.9)

train = dataset[0:border]
test = dataset[border:]
print(len(train))
print(len(test))

EPOCHS = 10
OPTI = "AdamW"
LR = 1e-5
LOSS="Huber"

loss_container = torch.zeros(EPOCHS,2)
    

if OPTI== "AdamW":
    print("AdamW as optimizer")
    optimizer = optim.AdamW(model.parameters(),LR)
       
elif OPTI=="RMS":
    print("RMSprop as optimizer")
    optimizer = optim.RMSprop(model.parameters(),LR)
      
else:
    print("SGD as optimizer")
    optimizer = optim.SGD(model.parameters(),LR)
    

s_train = len(train)
s_test = len(test)

print("train samples: {}, test samples: {}".format(s_train,s_test))
if LOSS=="MSE":
    print("MSELoss")
    loss_fn = nn.MSELoss()
elif LOSS =="Huber":
    print("HuberLoss")
    loss_fn = nn.HuberLoss()
else:
    print("MAELoss")
    loss_fn = nn.L1Loss()
dt = t[1]-t[0]
for epoch in tqdm(range(EPOCHS)):
    loaded = GraphDataLoader(train,batch_size=1,shuffle=True)
    n_train = len(loaded)
    model.train()
    print("{} : TRAIN batches".format(n_train))
    for sample in tqdm(loaded):
        loss_train=0
        optimizer.zero_grad()
            
        x, y = sample.ndata["x"].requires_grad_() , sample.ndata["y"].requires_grad_()
        #print(x.shape)
        #print(y.shape)
        #rollout
        #model.get_mass(x[:,3:6])
    
        for i in range(t_batch):
            M=model.get_mass(x[:,3:6])
            K1 = model.dHdx(x)
            K2 = model.dHdx(x+dt*K1/2)
            K3 = model.dHdx(x+dt*K2/2)
            K4 = model.dHdx(x+dt*K3)
            x_new = x + 1/6 * dt* (K1+2*K2+2*K3+K4)
            M_new=model.get_mass(x_new[:,3:6])
            loss_train += (loss_fn(x_new,y[:,i,:])+0.1*loss_fn(torch.sum(M-M_new),torch.tensor(0)))
            x=x_new
        ## end rollout
        loss_train/=t_batch
        loss_container[epoch,0] += loss_train.item()
        loss_train.backward()
        optimizer.step() 
            
        # testing
    loss_container[epoch,0]/=n_train
    model.eval()
    loaded_t= GraphDataLoader(test,batch_size=1,shuffle=True)
    n_test = len(loaded_t)
    print("{} : TEST batches".format(n_test))
    for sample in tqdm(loaded_t):
        loss_test=0
        x, y = sample.ndata["x"].requires_grad_() , sample.ndata["y"].requires_grad_()
        #print(x.shape)
        #print(y.shape)
        #model.get_mass(x[:,3:6])
        for i in range(t_batch):
            M = model.get_mass(x[:,3:6])
            K1 = model.dHdx(x)
            K2 = model.dHdx(x+dt*K1/2)
            K3 = model.dHdx(x+dt*K2/2)
            K4 = model.dHdx(x+dt*K3)
            x_new = x + 1/6 * dt* (K1+2*K2+2*K3+K4)
            M_new = model.get_mass(x_new[:,3:6])
            loss_test += (loss_fn(x_new,y[:,i,:])+0.1*loss_fn(torch.sum(M-M_new),torch.tensor(0)))
            x=x_new
        loss_test/=t_batch
        loss_container[epoch,1] += loss_test.item()
        loss_container[epoch,1]/=n_test
       
            
    print("E: {} HNN:: TRAIN: {} TEST: {}\n".format(epoch, loss_container[epoch,0], loss_container[epoch,1]))  
    print("M:{}".format(model.no_loop.ndata["M"]))   
visualize_loss(loss_container)     
    














