import dgl
import torch.nn as nn
import torch
import torch.nn.functional as F
import dgl.function as fn
from model import *
from model_gnn import *
"""
class H_layer(nn.Module):
    def __init__(self,g,in_dim,h_dim,bias=True):
        super().__init__()
        self.h_net = nn.Linear(in_dim,h_dim,bias=bias)
        self.g = g
    def change_graph(self,g):
        self.g = g
    def forward(self,x):
        g_c = dgl.add_self_loop(self.g)
        g_c.ndata["h"] = self.h_net(x)
        g_c.update_all(fn.copy_u(u="h", out="m"),fn.sum(msg="m", out="h"))
        H = g_c.ndata["h"]
        return H
"""

class HGNN(nn.Module):
    def __init__(self,g,acts, in_dim=2, h_dim=2, hidden=[64, 64],bias=True):
        super().__init__() 
        self.base = GATGNN(g,acts,in_dim,h_dim,hidden,bias=bias)
        #self.h_l = GATLayer(out_dim,h_dim,bias)
        self.g = g
        M = torch.eye(in_dim)
        M = torch.cat([M[in_dim//2:], -M[:in_dim//2]])
        self.J = M
        
    def change_graph(self,g):
        self.base.change_graph(g)
        self.g = g
        
    def forward(self,t,x):
        #print("x is leaf {}".format(x.is_leaf))
        y = self.base(t,x)
        self.g.ndata["h"] = y
        gs = dgl.unbatch(self.g)
        h=[]
        for g in gs:
            out = g.ndata["h"]
            temp = out.sum()/g.num_nodes()
            h.append(temp.unsqueeze(0))
        
        #print("H_val: {}".format(H_val.shape))    
        out = torch.autograd.grad(h,x,create_graph=True)[0]
      #  print("shape in {}".format(x.shape))
      #  print("shape out {}".format(out.shape))
        return out
        
        #print("form forward : {}".format(x.shape))
        #return self.dHdx(x) @ self.J.transpose(0,1)
        
         
def rolloutdxHGN(model,x):
    l = []
    for i in range(x.shape[0]):
        temp = model(0,x[i,:,:]).unsqueeze(0)
        l.append(temp)
    out = torch.cat((l),dim=0)
    return out 

def rk4(x,dt,model):
   # print("x {}".format(x.shape))
    k1 = model(0,x)
   # print("k1 {}".format(k1.shape))
    k2 = model(0,x + dt*k1/2)
   # print("k2 {}".format(k2.shape))
    k3 = model(0,x + dt*k2/2)
   # print("k3 {}".format(k3.shape))
    k4 = model(0,x + dt*k3)
   # print("k4 {}".format(k4.shape))
    
    return (k1 + 2*k2 + 2*k3 + k4)/6
    

def rollout(x0,t,model, method = "rk4"):
    l = []
    l.append(x0.unsqueeze(0).detach().requires_grad_())
    for i in range(len(t)-1):
       # print("rollout {}".format(i))
        dt = t[i+1]-t[i]
        xi = l[-1].squeeze().detach().requires_grad_()
        if method == "rk4":
            xii = xi + dt * rk4(xi,dt,model)
        l.append(xii.unsqueeze(0).detach().requires_grad_())
        #print("l[{}] is leaf {}".format(i,l[i].is_leaf))
    #print("l[{}] is leaf {}".format(len(t)-1,l[len(t)-1].is_leaf))
    return torch.cat((l),dim=0)
      
        