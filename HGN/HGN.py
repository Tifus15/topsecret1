import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import GraphConv,GATv2Conv
import dgl
import dgl.function as fn
from torch.autograd.functional import jacobian
from dgl.nn.pytorch.utils import Sequential



def function_act(name):
    if name == "tanh":
        return nn.Tanh()
    if name == "relu":
        return nn.ReLU()
    if name == "sin":
        return Sin()
    if name == "softplus":
        return nn.Softplus()
    else:
        return nn.Identity()

"""sin activation function as torch module"""
    
class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(1.0 * x)

class HNN_maker(nn.Module):
    def __init__(self,g, in_dim,hid_dim, out_dim,bias=True,type="GCN"):
        super(HNN_maker, self).__init__()
        M = torch.eye(in_dim)
        M = torch.cat([M[in_dim//2:], -M[:in_dim//2]])
        self.J = M
        #self.q_net = nn.Sequential(nn.Linear(int(in_dim/2),hid_dim),nn.Linear(hid_dim,int(in_dim/2)))
        self.g = g
        if type == "GCN":
            self.net = Sequential(GraphConv(in_dim,hid_dim,"left",True,activation =function_act("tanh")),
                                     GraphConv(hid_dim,out_dim,"left",True,activation =function_act("")))
        elif type == "GAT":
            self.net = Sequential(GATv2Conv(in_dim,hid_dim,num_heads=1,activation=function_act("tanh")),
                                     GATv2Conv(hid_dim,out_dim,num_heads=1,activation=function_act("")))
        """
        self.modules = []
        if type == "GCN":
            self.modules.append(GraphConv(in_dim,hid_dim,"left",True,activation =function_act(acts[0])))
            for i in range(1,len(acts)-1):
                self.modules.append(GraphConv(hid_dim,hid_dim,"left",True,activation =function_act(acts[i])))
            self.modules.append(GraphConv(hid_dim,out_dim,"left",True,activation =function_act(acts[-1])))
        elif type == "GAT":
            self.modules.append(GATv2Conv(in_dim,hid_dim,num_heads=1,activation=function_act(acts[0])))
            for i in range(1,len(acts)-1):
                self.modules.append(GATv2Conv(hid_dim,hid_dim,num_heads=1,activation=function_act(acts[i])))
            self.modules.append(GATv2Conv(hid_dim,out_dim,num_heads=1,activation=function_act(acts[-1])))
            
        self.net = nn.Sequential(*self.modules)
        """
    def set_graph(self,g):
        self.g = g  
    
       
    def layer_eval(self,y):
        """
        #print(y.shape)
        y1 = y[:,0:int(y.shape[-1]/2)]
        #print(y1.shape)
        y2 = y[:,int(y.shape[-1]/2):]
        #print(y2.shape)
        x_1 = self.q_net(y1)
        
        #for layer in self.modules:
        #    print(x.shape)
        #    x = layer(self.g,x)
        #return x
        #print("in layer_eval")
        x = torch.cat((x_1,y2),dim=-1)
        """
        return self.net(self.g,y)
    
    def H(self,y):
        #print("going in layer_eval")
        x = self.layer_eval(y)
        #print("out of layer eval")
        self.g.ndata["x"] = x
        self.g.update_all(fn.copy_u('x', 'm'), fn.sum('m', 'h'))
        
        out = self.g.ndata["h"]
        #print(out.shape)
        gs = dgl.unbatch(self.g)
        h = []
        for g in gs:
            out = g.ndata["h"]
            temp = out.sum()
            h.append(temp.unsqueeze(0))
        H_val = torch.cat((h),dim=0)
        #print(H_val.shape)
        return H_val
    
    def dHdx(self,y):
        def undiag(x):
            #print(x.shape)
            h_dim = x.shape[0]
            nodes_dim = x.shape[1]
            feats_dim = x.shape[2]
            orig_nodes = int(nodes_dim/h_dim)
            jac_list = []
            for i in range(h_dim):
                jac_list.append(x[i,i:i+orig_nodes,:])
            h = torch.cat((jac_list),dim=0)
            return h  
        #print("in jacobi {} ".format(y.shape)) 
        out = jacobian(self.H,y).squeeze()
        #print(out)
        #print("out of jacobi {}".format(out.shape))
        #print(out)
        if out.dim() == 1:
            puffered = out.unsqueeze(0)
        elif out.shape == y.shape:
            puffered = y
        else:
            puffered = undiag(out)
        #print(puffered.shape)
        #print(puffered)
        #print(out)
        return puffered
    
    def forward(self,t,x):
        #print("form forward : {}".format(x.shape))
        return self.dHdx(x) @ self.J.transpose(0,1)
    
    def rollH(self,x):
        l=[]
        for i in range(x.shape[0]):
            temp =self.H(x[i,:,:]).unsqueeze(0)
            #print(temp.shape)
            l.append(temp)
        h=torch.cat((l),dim=0)
        return h
    def rolldx(self,x):
        l=[]
        for i in range(x.shape[0]):
            l.append(self.forward(0,x[i,:,:]).unsqueeze(0))
        dx=torch.cat((l),dim=0)
        return dx
    


        
    
if __name__ == "__main__":
    g=dgl.graph(([0,0,0,1,1,1,2,2,2],[0,1,2,0,1,2,0,1,2]))
    H_model = HNN_maker(g,2,512,5,["tanh",""],type="GAT")
    #print(H_model)
    
    feat = torch.rand(3,2)
    print(feat.shape)
    
    H=H_model.H(feat)
    dHdx = H_model.dHdx(feat)
    print(dHdx.shape)
    print(H.shape)
    
        
    
    
  
    




