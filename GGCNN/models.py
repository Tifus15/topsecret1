import torch
import torch.nn as nn
#from torch_geometric_temporal.nn import *
import torch.functional as F
from device_util import DEVICE
from dgl.nn.pytorch.conv import GraphConv as dgl_GraphConv
import dgl
#Some kind of Gated Graph convolution network
class GGCNN_HNN(torch.nn.Module):
    def __init__(self, node_features,grad=True):
        super(GGCNN_HNN, self).__init__()
        self.grad=grad
        self.recurrent1 = GConvGRU(node_features, 32, 1)# main feature
        self.reccurent2 = GConvGRU(32, 128, 1) # main feature
        self.linear = torch.nn.Linear(128, 1)
        self.grad = torch.nn.Linear(1, 2)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.Tanh()
        self.node_features = node_features
    
    def H(self,x,edge_index):
        H = self.recurrent1(x, edge_index, torch.ones(edge_index.shape[1]).to(DEVICE))
        #print("1.."+str(H.shape))
        H = self.act1(H)
        #print("2.. "+str(H.shape))
        H = self.reccurent2(H, edge_index, torch.ones(edge_index.shape[1]).to(DEVICE))
        #print("2.. "+str(H.shape))
        H = self.act2(H)
        #print("4.. "+str(H.shape))
        H = self.linear(H)
        #print("3.. "+str(H.shape))
        return H
    

    def dhdx(self,x,edge_index):
        H  = self.H(x,edge_index)
        #if self.grad:
        #    dh = torch.autograd.grad(torch.mean(H),x,retain_graph=True)[0]
        #else:
        dh = self.grad(H)
        return dh
    
    def forward(self, x, edge_index):
        dh = self.dhdx(x,edge_index)
        return dh @ self.J().transpose(0,1)
    
    def J(self):
        dim = self.node_features
        M = torch.eye(dim)
        M = torch.cat([M[dim//2:], -M[:dim//2]]).to(DEVICE)  
        return M 
    

class dgl_HNN(torch.nn.Module):
    def __init__(self,graph,in_feats,hid_feats,act=nn.Tanh(),grad=True):
        super(dgl_HNN,self).__init__()
        src = graph[0]
        dst = graph[1]
        self.graph = dgl.graph((src,dst))
        self.layer1 = dgl_GraphConv(in_feats,hid_feats)
        self.act = act
        self.layer2 = dgl_GraphConv(hid_feats,in_feats)
        self.indim = in_feats

    def change_graph(self,g):
        #print(g)
        self.graph = g

    def J(self):
        dim = self.node_features
        M = torch.eye(dim)
        M = torch.cat([M[dim//2:], -M[:dim//2]]).to(DEVICE)  
        return M 

    def dHdx(self,x):
        #print("DGL: input shape: {}".format(x.shape))
        y = self.layer1(self.graph,x)
        #print("DGL: after first layer: {}".format(y.shape))
        y = self.act(y)
        y = self.layer2(self.graph,y)
        #print("DGL: after second layer: {}".format(y.shape))
        return y

    def forward(self,x):
        y = self.dHdx(x)
        out = y @ self.J().transpose(0,1).to(DEVICE) 
        return out
    
def roll(model,x0, t,edges,method="euler"):
    ### batch : [ N F]
    list = []
    list.append(x0.view(x0.shape[0],x0.shape[1], 1))
    if method == "rk4":
        for i in range(1,len(t)):
            K1 = model(list[-1].squeeze(),edges)
            K2 = model(K1+ (t[i]-t[i-1])*list[-1].squeeze()/2,edges)
            K3 = model(K2+ (t[i]-t[i-1])*list[-1].squeeze()/2,edges)
            K4 = model(K3+ (t[i]-t[i-1])*list[-1].squeeze(),edges)

            list.append(list[-1] + (t[i]-t[i-1])*(K1 + 2*K2 + 2*K3 + K4).view(x0.shape[0],x0.shape[1],1)/6)
    else:
        for i in range(1,len(t)):
            f = model(list[-1].squeeze(),edges)
            list.append(list[-1]+(t[i]-t[i-1])*f.view(x0.shape[0],x0.shape[1],1))
    return torch.cat((list),dim=-1).to(DEVICE)


class PortHamModel(torch.nn.Module):
    def __init__(self, HNN_model):
        super(GGCNN_HNN, self).__init__()
        self.HNN = HNN_model
        self.RHNN_1 = torch.nn.linear(2,20)
        self.RHNN_2 = torch.nn.linear(20,2)
        #self.cont1 = torch.nn.linear(2,20)
        #self.cont2 = torch.nn.linear(2,20)

    def dhJ(self,x,edge_idx):
        out = self.HNN(x,edge_idx) # J @ dHdx
        return out
    
    def dhR(self,x,edge_idx):
        dhdx = self.HNN.dhdx(x,edge_idx)
        out = self.RHNN_1(dhdx)
        out = self.RHNN_2(out)
        return out

    def forward(self,x,edge_idx):
        dJ = self.dhJ(x,edge_idx)
        dR = self.dhR(x, edge_idx)
        return dJ - dR




