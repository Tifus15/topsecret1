import torch
import torch.nn as nn
from GNN_model import *

class rollout_GNN_GRU(nn.Module):
    def __init__(self,g,in_dim,hid,gnn_hid):
        super().__init__()
        self.GRU = nn.GRU(in_dim,hid,1)
        self.GNN = GNN(g,hid,gnn_hid,in_dim)
        
        #self.NN = nn.Linear(hid,in_dim)
        nn.init.normal_(self.GRU.weight_ih_l0,mean=0.,std=0.1)
        nn.init.normal_(self.GRU.weight_hh_l0,mean=0.,std=0.1)
        nn.init.constant_(self.GRU.bias_ih_l0,val=0)
        nn.init.constant_(self.GRU.bias_hh_l0,val=0)
    
    def forward(self,t,x0):
        
        T = len(t)
        output = torch.zeros(x0.shape[0],T,x0.shape[1])
        doutput = torch.zeros(x0.shape[0],T,x0.shape[1])
        output[:,0,:] = x0
        xi, hidden = self.GRU(x0.unsqueeze(dim=0))
        #print("x0: {}".format(x0.shape))
        #print("xi: {}".format(xi.unsqueeze(dim=0).shape))
        xii = self.GNN(xi.squeeze())
        doutput[:,0,:] = xii
        #print("xii: {}".format(xii.shape))
        #print("hidden: {}".format(hidden.shape))
        #print(xii.shape)
        
        dt = t[1]-t[0]
        temp = x0 + dt*xii
        output[:,1,:] = temp
        for i in range(2,T):
            #print("xi: {}".format(xi.unsqueeze(dim=0).shape))
            xi, hidden = self.GRU(temp.unsqueeze(dim=1),hidden)
            #print("hidden: {}".format(hidden.shape))
            xii = self.GNN(xi.squeeze())
            doutput[:,i-1,:] = xii
            #print("xii: {}".format(xii.shape))
            
            dt = t[i]-t[i-1]
            temp = temp+ dt*xii
            output[:,i,:] = temp
        xi, hidden = self.GRU(temp.unsqueeze(dim=1),hidden)
        #print("hidden: {}".format(hidden.shape))
        xii = self.GNN(xi.squeeze())
        doutput[:,-1,:] = xii
        out = torch.cat((output,doutput),dim=-1)
        return out