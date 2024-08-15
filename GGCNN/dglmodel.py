import torch
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, Tanh, ReLU
import torch.functional as F
from device_util import DEVICE
from dgl.nn.pytorch.conv import GraphConv as dgl_GraphConv
import dgl
import dgl.function as fn


class portHNN_split_dgl(torch.nn.Module):
    def __init__(self,g, in_dim,enc_dim, Hout,init_w=0.1):
        super(portHNN_split_dgl,self).__init__()
        self.indim = in_dim
        self.dim_split = int(in_dim/2)
        if enc_dim == in_dim:
            self.Kfun = Seq(Linear(self.dim_split, Hout),
                        Tanh(),
                        Linear(Hout, Hout),
                        ReLU(),
                        Linear(Hout, Hout))
            self.Ufun = Seq(Linear(self.dim_split, Hout),
                        Tanh(),
                        Linear(Hout, Hout),
                        ReLU(),
                        Linear(Hout, Hout))
            self.encK = Linear(self.dim_split,self.dim_split)
            self.encP1 = Linear(self.dim_split,self.dim_split)
            self.encP2 = Linear(self.dim_split,self.dim_split)
        else:
            self.Kfun = Seq(Linear(enc_dim, Hout),
                        Tanh(),
                        Linear(Hout, Hout),
                        ReLU(),
                        Linear(Hout, Hout))
            self.Ufun = Seq(Linear(enc_dim, Hout),
                        Tanh(),
                        Linear(Hout, Hout),
                        ReLU(),
                        Linear(Hout, Hout))
            self.encK = Linear(self.dim_split,enc_dim)
            self.encP1 = Linear(self.dim_split,enc_dim)
            self.encP2 = Linear(self.dim_split,enc_dim)
        src = g[0]
        dst = g[1]
        self.g = dgl.graph((src,dst))
        self.H = Linear(Hout, in_dim)
        self.D = Linear(in_dim,in_dim)
        nn.init.normal_(self.Kfun[0].weight,mean=0.,std=init_w)
        nn.init.normal_(self.Kfun[2].weight,mean=0.,std=init_w)
        nn.init.normal_(self.Kfun[4].weight,mean=0.,std=init_w)
        nn.init.normal_(self.Ufun[0].weight,mean=0.,std=init_w)
        nn.init.normal_(self.Ufun[2].weight,mean=0.,std=init_w)
        nn.init.normal_(self.Ufun[4].weight,mean=0.,std=init_w)
        
        nn.init.normal_(self.encK.weight,mean=0.,std=init_w)
        nn.init.normal_(self.encP1.weight,mean=0.,std=init_w)
        nn.init.normal_(self.encP2.weight,mean=0.,std=init_w)
        nn.init.normal_(self.H.weight,mean=0.,std=init_w)
        nn.init.normal_(self.D.weight,mean=0.,std=init_w)
    def change_graph(self,g):
        self.g = g
        
    def getK(self,x):
        p = torch.split(x, int(self.indim/2),dim=-1)[1]
        self.g.ndata["h1"] = self.encK(p)
        self.g.update_all(fn.copy_u("h1", "m"), fn.sum("m", "h1"))
        self.g.ndata["E"] = self.Kfun(self.g.ndata["h1"])
        #print(K.shape)
        return self.g.ndata["E"]
    
    def getU(self,x):
        q = torch.split(x, int(self.indim/2),dim=-1)[0]
        self.g.ndata["h2_1"] = self.encP1(q)
        self.g.ndata["h2_2"] = self.encP2(q)
        self.g.apply_edges(fn.u_add_v('h2_1', 'h2_2', 'out'))
        #print("edges out size : {}".format(self.g.edata["out"].shape))
        self.g.edata["E"] = self.Ufun(self.g.edata["out"])
        
        #print(U.shape)
        return self.g.edata["E"]
    
    def getH(self,x):
        _ = self.getK(x)
        _ = self.getU(x)
        #K = self.g.ndata["E"]
        #print("K shape {}".format(self.g.ndata["E"].shape))
        #P = self.g.edata["E"]
        #print("p shape {}".format(self.g.edata["E"].shape))
        self.g.update_all(fn.u_mul_e('E', 'E', 'm'), fn.sum('m', 'E_new'))
        return self.g.ndata["E_new"]
    
    def dHdx(self,x):
        self.getH(x)
        return self.H(self.g.ndata["E_new"])
    
    def J(self):
        M = torch.eye(self.indim)
        M = torch.cat([M[self.indim//2:], -M[:self.indim//2]]).to(DEVICE)  
        return M 
    
    def get_D(self,x):
        dh = self.dHdx(x)
        p = torch.split(dh, int(self.indim/2),dim=-1)[1]
        D_dHdx = self.D(x)
        self.g.ndata["d"] = torch.split(D_dHdx, int(self.indim/2),dim=-1)[1]
        self.g.update_all(fn.copy_u("d", "m"), fn.sum("m", "d"))
        return torch.cat((torch.zeros(p.shape).to(DEVICE),self.g.ndata["d"]),dim=-1).to(DEVICE)
    
    def forward(self,x):
        dH = self.dHdx(x)
        d = self.get_D(x)
        #no autograd it destorys learning
        #dh = torch.autograd.grad(H,x,retain_graph=True)[0]
        return dH @ self.J().transpose(0,1) - d


class portHNN_dgl(torch.nn.Module):
    def __init__(self,g, in_dim,enc_dim, Hout,init_w=0.1):
        super(portHNN_dgl,self).__init__()
        self.indim=in_dim
        if enc_dim == in_dim:
            self.Kfun = Seq(Linear(in_dim, Hout),
                        Tanh(),
                        Linear(Hout, Hout),
                        ReLU(),
                        Linear(Hout, Hout))
            self.Ufun = Seq(Linear(in_dim, Hout),
                        Tanh(),
                        Linear(Hout, Hout),
                        ReLU(),
                        Linear(Hout, Hout))
            self.encK = Linear(in_dim,in_dim)
            self.encP1 = Linear(in_dim,in_dim)
            self.encP2 = Linear(in_dim,in_dim)
        else:
            self.Kfun = Seq(Linear(enc_dim, Hout),
                        Tanh(),
                        Linear(Hout, Hout),
                        ReLU(),
                        Linear(Hout, Hout))
            self.Ufun = Seq(Linear(enc_dim, Hout),
                        Tanh(),
                        Linear(Hout, Hout),
                        ReLU(),
                        Linear(Hout, Hout))
            self.encK = Linear(in_dim,enc_dim)
            self.encP1 = Linear(in_dim,enc_dim)
            self.encP2 = Linear(in_dim,enc_dim)
        src = g[0]
        dst = g[1]
        self.g = dgl.graph((src,dst))
        self.H = Linear(Hout, in_dim)
        self.D = Linear(in_dim,in_dim)
        nn.init.normal_(self.Kfun[0].weight,mean=0.,std=init_w)
        nn.init.normal_(self.Kfun[2].weight,mean=0.,std=init_w)
        nn.init.normal_(self.Kfun[4].weight,mean=0.,std=init_w)
        nn.init.normal_(self.Ufun[0].weight,mean=0.,std=init_w)
        nn.init.normal_(self.Ufun[2].weight,mean=0.,std=init_w)
        nn.init.normal_(self.Ufun[4].weight,mean=0.,std=init_w)
        
        nn.init.normal_(self.encK.weight,mean=0.,std=init_w)
        nn.init.normal_(self.encP1.weight,mean=0.,std=init_w)
        nn.init.normal_(self.encP2.weight,mean=0.,std=init_w)
        nn.init.normal_(self.H.weight,mean=0.,std=init_w)
        nn.init.normal_(self.D.weight,mean=0.,std=init_w)
    def change_graph(self,g):
        self.g = g
        
    def getK(self,x):
        self.g.ndata["h1"] = self.encK(x)
        self.g.update_all(fn.copy_u("h1", "m"), fn.sum("m", "h1"))
        self.g.ndata["E"] = self.Kfun(self.g.ndata["h1"])
        #print(K.shape)
        return self.g.ndata["E"]
    
    def get_D(self,x):
        D_dHdx = self.D(self.dHdx(x))
        self.g.ndata["d"] = D_dHdx
        self.g.update_all(fn.copy_u("d", "m"), fn.sum("m", "d"))
        return self.g.ndata["d"]
    
    def getU(self,x):
        self.g.ndata["h2_1"] = self.encP1(x)
        self.g.ndata["h2_2"] = self.encP2(x)
        self.g.apply_edges(fn.u_add_v('h2_1', 'h2_2', 'out'))
        #print("edges out size : {}".format(self.g.edata["out"].shape))
        self.g.edata["E"] = self.Ufun(self.g.edata["out"])
        
        #print(U.shape)
        return self.g.edata["E"]
    
    def getH(self,x):
        _ = self.getK(x)
        _ = self.getU(x)
        K = self.g.ndata["E"]
        #print("K shape {}".format(self.g.ndata["E"].shape))
        P = self.g.edata["E"]
        #print("p shape {}".format(self.g.edata["E"].shape))
        self.g.update_all(fn.u_mul_e('E', 'E', 'm'), fn.sum('m', 'E_new'))
        return self.g.ndata["E_new"]
    
    def dHdx(self,x):
        self.getH(x)
        return self.H(self.g.ndata["E_new"])
    
    def J(self):
        M = torch.eye(self.indim)
        M = torch.cat([M[self.indim//2:], -M[:self.indim//2]]).to(DEVICE)  
        return M 
    
    def forward(self,x):
        dH = self.dHdx(x)
        d = self.get_D(x)
        #no autograd it destorys learning
        #dh = torch.autograd.grad(H,x,retain_graph=True)[0]
        return dH @ self.J().transpose(0,1) - d

#### not important
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
        M = torch.eye(self.indim)
        M = torch.cat([M[self.indim//2:], -M[:self.indim//2]]).to(DEVICE)  
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
        d = self.get_D(x)
        out = y @ self.J().transpose(0,1).to(DEVICE) - d
        return out
    



def roll(model,x0, t,edges,method="euler"):
    ### batch : [ N F]
    list = []
    list.append(x0.view(x0.shape[0],x0.shape[1], 1))
    model.change_graph(edges)
    if method == "rk4":
        for i in range(1,len(t)):
            K1 = model(list[-1].squeeze())
            K2 = model(K1+ (t[i]-t[i-1])*list[-1].squeeze()/2)
            K3 = model(K2+ (t[i]-t[i-1])*list[-1].squeeze()/2)
            K4 = model(K3+ (t[i]-t[i-1])*list[-1].squeeze())

            list.append(list[-1] + (t[i]-t[i-1])*(K1 + 2*K2 + 2*K3 + K4).view(x0.shape[0],x0.shape[1],1)/6)
    else:
        for i in range(1,len(t)):
            f = model(list[-1].squeeze(),edges)
            list.append(list[-1]+(t[i]-t[i-1])*f.view(x0.shape[0],x0.shape[1],1))
    return torch.cat((list),dim=-1).to(DEVICE)

